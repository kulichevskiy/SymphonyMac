use crate::github::{self, PrCheck, PrCiStatus, PrComment};
use crate::SharedState;
use std::collections::HashMap;
use std::path::PathBuf;
use tauri::AppHandle;

use super::{AgentStatus, PipelineStage};

/// Maximum number of failing checks we surface in a single fix-run prompt.
/// Cap exists to keep prompts bounded — if more than this fail at once, the
/// agent fixes the first batch, pushes, and the next poll picks up whatever
/// remains.
const MAX_CI_FAILURES_IN_PROMPT: usize = 5;

/// Maximum chars of `gh run view --log-failed` output we paste per failing
/// check. The issue suggests `head -200` lines; this char budget is roughly
/// that order of magnitude and protects against pathological one-line logs.
const CI_LOG_EXCERPT_MAX_CHARS: usize = 4000;

/// Number of consecutive `pr_ci_status` fetch failures we tolerate on a run
/// that has Codex approval before failing the run terminally. Without this
/// cutoff, a permanently broken `gh` (e.g. missing checks-read scope) would
/// strand approved PRs in Review forever, since the approval-path falls
/// through to `WaitForCiUnknown` on every fetch failure.
///
/// Sized so users get ~5 minutes of transient retry headroom at the default
/// 60s poll interval — enough to absorb a brief network blip, short enough
/// that a permanent permissions issue surfaces visibly instead of silently
/// deadlocking the pipeline.
const CI_STATUS_FETCH_FAILURE_THRESHOLD: u32 = 5;

#[derive(Debug, Clone)]
struct ReviewRunSnapshot {
    run_id: String,
    repo: String,
    issue_number: u64,
    issue_title: String,
    issue_body: String,
    issue_labels: Vec<String>,
    workspace_path: String,
    last_review_request_at: Option<String>,
    last_pushed_sha: Option<String>,
    review_iteration: u32,
    stage_context: Option<crate::orchestrator::StageContext>,
    last_ci_failure_sha: Option<String>,
    /// Snapshot of the run's consecutive `pr_ci_status` failure counter at the
    /// start of this poll tick. Used to decide whether a fresh fetch failure
    /// pushes us over the terminal-failure threshold.
    ci_status_fetch_failure_count: u32,
    /// Signature of the trigger set used to spawn the most recent fix-run on
    /// this run. Stuck-loop escape compares the next would-be trigger
    /// signature to this value: equality means we're about to ask the agent to
    /// fix the same thing twice and should halt instead.
    last_trigger_signature: Option<String>,
    /// Cumulative cost (USD) summed across every run for the same
    /// (repo, issue), captured under-lock at the start of the poll tick. Used
    /// for the cost-cap escape.
    issue_cost_usd: f64,
    /// Configured iteration cap captured under-lock at the start of the tick.
    /// 0 disables the cap.
    max_review_iterations: u32,
    /// Configured cost cap captured under-lock at the start of the tick.
    /// 0.0 disables the cap.
    cost_cap_per_issue_usd: f64,
}

pub async fn poll_review_runs(app: &AppHandle, state: &SharedState) {
    let (snapshots, approve_patterns, feedback_marker) = {
        let s = state.lock().await;
        let max_review_iterations = s.config.max_review_iterations;
        let cost_cap_per_issue_usd = s.config.cost_cap_per_issue_usd;
        // Only the latest Running Review sentinel per (repo, issue) participates
        // in polling. Older Running sentinels (e.g. left over from a manual
        // re-launch) are silent. Without this dedupe, two sentinels for the
        // same issue would either deadlock on each other (if the duplicate
        // guard matched Running) or re-dispatch fix-runs against feedback the
        // latest sentinel already handled (if it didn't). A run in `Preparing`
        // (fix-run in flight) is not eligible to be the polling sentinel and
        // is filtered out by the helper.
        let snapshots: Vec<ReviewRunSnapshot> =
            latest_running_review_per_issue(s.runs.values())
                .into_iter()
                .map(|run| {
                    let issue_cost_usd =
                        crate::orchestrator::cumulative_cost_for_issue(&s, &run.repo, run.issue_number);
                    ReviewRunSnapshot {
                        run_id: run.id.clone(),
                        repo: run.repo.clone(),
                        issue_number: run.issue_number,
                        issue_title: run.issue_title.clone(),
                        issue_body: String::new(),
                        issue_labels: run.issue_labels.clone(),
                        workspace_path: run.workspace_path.clone(),
                        last_review_request_at: run.last_review_request_at.clone(),
                        last_pushed_sha: run.last_pushed_sha.clone(),
                        review_iteration: run.review_iteration,
                        stage_context: run.stage_context.clone(),
                        last_ci_failure_sha: run.last_ci_failure_sha.clone(),
                        ci_status_fetch_failure_count: run.ci_status_fetch_failure_count,
                        last_trigger_signature: run.last_trigger_signature.clone(),
                        issue_cost_usd,
                        max_review_iterations,
                        cost_cap_per_issue_usd,
                    }
                })
                .collect();
        (
            snapshots,
            s.config.codex_approve_patterns.clone(),
            s.config.codex_feedback_marker.clone(),
        )
    };

    for snapshot in snapshots {
        check_codex_activity(app, state, snapshot, &approve_patterns, &feedback_marker).await;
    }
}

/// Pure helper that selects the latest Running Review sentinel per
/// (repo, issue) from a flat list. Extracted so the dedupe logic is
/// directly unit-testable.
fn latest_running_review_per_issue<'a>(
    runs: impl IntoIterator<Item = &'a crate::orchestrator::AgentRun>,
) -> Vec<&'a crate::orchestrator::AgentRun> {
    let mut latest_by_issue: HashMap<(String, u64), &crate::orchestrator::AgentRun> =
        HashMap::new();
    for run in runs.into_iter().filter(|run| {
        run.stage == PipelineStage::Review && run.status == AgentStatus::Running
    }) {
        let key = (run.repo.clone(), run.issue_number);
        latest_by_issue
            .entry(key)
            .and_modify(|existing| {
                if run.started_at > existing.started_at {
                    *existing = run;
                }
            })
            .or_insert(run);
    }
    latest_by_issue.into_values().collect()
}

async fn check_codex_activity(
    app: &AppHandle,
    state: &SharedState,
    snapshot: ReviewRunSnapshot,
    approve_patterns: &[String],
    feedback_marker: &str,
) {
    let pr_state = match github::pr_full_state(&snapshot.repo, snapshot.issue_number).await {
        Ok(Some(pr)) => pr,
        Ok(None) => return,
        Err(error) => {
            crate::agent::runtime_helpers::append_log(
                state,
                &snapshot.run_id,
                format!("[review] Failed to fetch PR state: {}", error),
            )
            .await;
            return;
        }
    };

    // 0. DIRTY-state takes priority over approval, Codex feedback, AND CI gating —
    // a PR with hard conflicts cannot be merged, and applying any other fix on
    // top of an unmergeable branch would only churn. Resolve the rebase first;
    // the fix-run re-requests review on the rebased SHA, so an existing
    // approval (now stale because HEAD changed) will be re-issued by Codex
    // against the new state. CI re-runs against the new HEAD as well.
    if github::pr_is_dirty(pr_state.merge_state_status.as_deref()) {
        try_dispatch_rebase_fix_run(app, state, &snapshot, &pr_state).await;
        return;
    }

    // CI status fetch is best-effort per tick: a transient `gh` failure must
    // not bring down the rest of the polling loop. When unavailable, we treat
    // CI as "unknown" and refuse to advance the approval path — we'd rather
    // wait one more tick than ship without a CI verdict.
    //
    // BUT: we also track consecutive failures on the run, so a *permanent*
    // breakage (e.g. missing `checks:read` scope on `gh`) doesn't silently
    // deadlock approved PRs in Review forever. Once `WaitForCiUnknown` would
    // fire AND the counter is over the threshold, we surface a terminal error
    // on the run instead of looping silently. The counter resets to 0 on every
    // successful fetch.
    let ci_status: Option<PrCiStatus> =
        match github::pr_ci_status(&snapshot.repo, pr_state.number).await {
            Ok(status) => {
                if snapshot.ci_status_fetch_failure_count > 0 {
                    // Reset the counter so a future broken streak starts fresh.
                    let _ = crate::agent::pipeline_helpers::set_ci_status_fetch_failure_count(
                        state,
                        &snapshot.run_id,
                        0,
                    )
                    .await;
                }
                Some(status)
            }
            Err(error) => {
                let new_count = snapshot.ci_status_fetch_failure_count.saturating_add(1);
                let _ = crate::agent::pipeline_helpers::set_ci_status_fetch_failure_count(
                    state,
                    &snapshot.run_id,
                    new_count,
                )
                .await;
                crate::agent::runtime_helpers::append_log(
                    state,
                    &snapshot.run_id,
                    format!(
                        "[review] Failed to fetch CI status for PR #{}: {} — will retry next tick (failure {} of {}).",
                        pr_state.number,
                        error,
                        new_count,
                        CI_STATUS_FETCH_FAILURE_THRESHOLD
                    ),
                )
                .await;
                None
            }
        };

    // Effective counter: snapshot value + 1 if this tick's fetch failed.
    // We need the *post-update* value for the threshold check below.
    let effective_fetch_failure_count = if ci_status.is_some() {
        0
    } else {
        snapshot.ci_status_fetch_failure_count.saturating_add(1)
    };

    let head_match =
        head_sha_matches(snapshot.last_pushed_sha.as_deref(), pr_state.head_ref_oid.as_deref());

    // 1. Approval takes priority — if Codex says LGTM, advance to Merge once
    //    the CI gate (per issue #4) is also satisfied. If Codex approves but
    //    CI is failing, we fall through to the fix-run path below so the same
    //    poll cycle can spawn a CI fix-run instead of stalling.
    if let Some(comment) = find_codex_approval(
        &pr_state.comments,
        snapshot.last_review_request_at.as_deref(),
        approve_patterns,
        feedback_marker,
    ) {
        // Defense against changes pushed *after* `@codex review`: if HEAD moved,
        // the approval is stale and we wait for a fresh review.
        if !head_match {
            crate::agent::runtime_helpers::append_log(
                state,
                &snapshot.run_id,
                format!(
                    "[review] Ignoring Codex approval at {} — PR HEAD ({}) differs from last reviewed SHA ({}). Waiting for a fresh review on the new HEAD.",
                    comment.created_at,
                    pr_state.head_ref_oid.as_deref().unwrap_or("unknown"),
                    snapshot.last_pushed_sha.as_deref().unwrap_or("unknown"),
                ),
            )
            .await;
            return;
        }

        match decide_approval_outcome(ci_status.as_ref()) {
            ApprovalOutcome::Advance => {
                crate::agent::runtime_helpers::append_log(
                    state,
                    &snapshot.run_id,
                    format!(
                        "[review] Codex approval detected at {} and CI is green — advancing to Merge.",
                        comment.created_at
                    ),
                )
                .await;

                crate::agent::advance_review_to_merge(
                    app,
                    state,
                    crate::agent::ReviewAdvanceContext {
                        run_id: snapshot.run_id,
                        repo: snapshot.repo,
                        issue_number: snapshot.issue_number,
                        issue_title: snapshot.issue_title,
                        issue_body: snapshot.issue_body,
                        issue_labels: snapshot.issue_labels,
                        workspace_path: snapshot.workspace_path,
                    },
                )
                .await;
                return;
            }
            ApprovalOutcome::WaitForCi => {
                crate::agent::runtime_helpers::append_log(
                    state,
                    &snapshot.run_id,
                    format!(
                        "[review] Codex approval detected at {}, but required CI checks are still pending. Holding in Review.",
                        comment.created_at
                    ),
                )
                .await;
                return;
            }
            ApprovalOutcome::WaitForCiUnknown => {
                if ci_fetch_failure_should_terminate(effective_fetch_failure_count) {
                    let error = format!(
                        "Codex approved PR #{} but CI status fetch failed {} consecutive times. \
This usually means `gh pr view --json statusCheckRollup` cannot read the rollup \
(missing `checks:read` permission, repo restrictions, or a persistent gh outage). \
Failing the run terminally so the deadlock is visible — fix the underlying \
permissions/auth and resume manually.",
                        pr_state.number, effective_fetch_failure_count,
                    );
                    crate::agent::runtime_helpers::append_log(
                        state,
                        &snapshot.run_id,
                        format!("[review] {}", error),
                    )
                    .await;
                    crate::agent::pipeline_helpers::fail_review_run_with_error(
                        app,
                        state,
                        &snapshot.run_id,
                        error,
                    )
                    .await;
                    return;
                }
                crate::agent::runtime_helpers::append_log(
                    state,
                    &snapshot.run_id,
                    format!(
                        "[review] Codex approval detected at {}, but CI status is unknown this tick (failure {} of {}). Holding in Review until CI status can be fetched.",
                        comment.created_at,
                        effective_fetch_failure_count,
                        CI_STATUS_FETCH_FAILURE_THRESHOLD,
                    ),
                )
                .await;
                return;
            }
            ApprovalOutcome::CiFailing => {
                crate::agent::runtime_helpers::append_log(
                    state,
                    &snapshot.run_id,
                    format!(
                        "[review] Codex approval detected at {} but CI is failing — not advancing to Merge. Will spawn fix-run for the failing checks.",
                        comment.created_at
                    ),
                )
                .await;
                // Fall through into the fix-run path below; the CI-failure
                // collector will pick up the failing checks.
            }
        }
    }

    // 2. Determine fix-run signals. A fix-run can be triggered by Codex
    //    feedback comments, by failing CI, or both at once.

    // Codex feedback path requires a baseline timestamp — without one, we'd
    // treat *every* historical Codex comment as new and force-push fixes for
    // stale feedback. CI failures don't have this hazard (they're keyed by
    // SHA, not by comment time), so we still allow CI-only fix-runs when the
    // baseline is missing (e.g. legacy persisted Review runs).
    let feedback_comments = match snapshot.last_review_request_at.as_deref() {
        Some(baseline_ts) => collect_codex_feedback(
            &pr_state.comments,
            Some(baseline_ts),
            approve_patterns,
            feedback_marker,
        ),
        None => Vec::new(),
    };

    let failing_check_refs: Vec<&PrCheck> = ci_status
        .as_ref()
        .map(|s| s.failing_checks())
        .unwrap_or_default();

    let should_spawn_for_ci = should_trigger_ci_fix_run(
        &failing_check_refs,
        pr_state.head_ref_oid.as_deref(),
        snapshot.last_ci_failure_sha.as_deref(),
    );

    if feedback_comments.is_empty() && !should_spawn_for_ci {
        return;
    }

    // External-push guard: if HEAD moved since the last review request, a human
    // (or some other process) pushed to this branch. Don't fire a fix-run on
    // top — we don't know what that push contained, and force-pushing over it
    // could destroy work. Wait until the human investigates.
    if !head_match {
        crate::agent::runtime_helpers::append_log(
            state,
            &snapshot.run_id,
            format!(
                "[review] Detected fix-run trigger but PR HEAD ({}) differs from last reviewed SHA ({}). Skipping fix-run — external push detected.",
                pr_state.head_ref_oid.as_deref().unwrap_or("unknown"),
                snapshot.last_pushed_sha.as_deref().unwrap_or("unknown"),
            ),
        )
        .await;
        return;
    }

    // Double-spawn guard: another fix-run for the same issue is already
    // Preparing/Running. Even though we filtered out runs with active
    // subprocesses above, a sibling Review run for the same issue may exist
    // (e.g. a manual retry); skip if we'd start a duplicate.
    let already_active = {
        let s = state.lock().await;
        s.runs.values().any(|run| {
            another_review_active(
                &run.repo,
                run.issue_number,
                &run.id,
                &run.stage,
                &run.status,
                &snapshot.repo,
                snapshot.issue_number,
                &snapshot.run_id,
            )
        })
    };
    if already_active {
        crate::agent::runtime_helpers::append_log(
            state,
            &snapshot.run_id,
            "[review] Detected fix-run trigger but another Review run is already active for this issue — skipping fix-run spawn this tick."
                .to_string(),
        )
        .await;
        return;
    }

    // Build the per-check CI failure context now that we've cleared the gates.
    // Fetching log excerpts is best-effort and serial — we cap the number of
    // checks to keep the prompt bounded.
    let ci_failure_contexts = if should_spawn_for_ci {
        build_ci_failure_contexts(&snapshot.repo, &failing_check_refs).await
    } else {
        Vec::new()
    };

    let feedback_text = if feedback_comments.is_empty() {
        String::new()
    } else {
        format_feedback_for_prompt(&feedback_comments)
    };
    let pr_number = pr_state.number;
    let branch_name = pr_state.head_ref_name.clone();
    let next_iteration = snapshot.review_iteration.saturating_add(1);

    // Build the trigger signature + summary for this would-be fix-run. The
    // signature feeds the dedup escape; the summary is shown on the dashboard.
    let mut triggers: Vec<TriggerInput<'_>> = Vec::new();
    for comment in &feedback_comments {
        triggers.push(TriggerInput::CodexFeedback {
            body: comment.body.as_str(),
        });
    }
    for ctx in &ci_failure_contexts {
        triggers.push(TriggerInput::CiFailure {
            name: ctx.name.as_str(),
        });
    }
    let next_signature = compute_trigger_signature(&triggers);
    let next_summary = trigger_summary(&triggers);

    if let Some(reason) = decide_escape(
        snapshot.review_iteration,
        snapshot.max_review_iterations,
        snapshot.issue_cost_usd,
        snapshot.cost_cap_per_issue_usd,
        snapshot.last_trigger_signature.as_deref(),
        &next_signature,
        &next_summary,
    ) {
        // Persist the signature anyway so the dashboard can show what the
        // escape was about, then park the run.
        crate::agent::pipeline_helpers::set_last_trigger(
            state,
            &snapshot.run_id,
            Some(next_signature),
            Some(next_summary),
        )
        .await;
        crate::agent::pipeline_helpers::escape_review_run_to_awaiting_approval(
            app,
            state,
            &snapshot.run_id,
            reason.human_message(),
        )
        .await;
        return;
    }

    let log_message = format!(
        "[review] Spawning fix-run iteration {}: {} Codex feedback comment{}, {} failing CI check{}.",
        next_iteration,
        feedback_comments.len(),
        if feedback_comments.len() == 1 { "" } else { "s" },
        ci_failure_contexts.len(),
        if ci_failure_contexts.len() == 1 { "" } else { "s" },
    );

    // Increment review_iteration AND drop the run out of `Running` BEFORE we
    // tokio::spawn the fix-run agent. The poll loop filters by `status ==
    // Running`, so transitioning to `Preparing` synchronously here closes the
    // race Codex flagged: even if the next poll tick fires before the spawned
    // task registers a PID (e.g. while the `before_run` hook is still running),
    // it will see the run is no longer in `Running` and skip it. The fix-run
    // path inside `run_agent_process` keeps the run in `Preparing` until
    // `finalize_fix_run_success` flips it back.
    let _ = crate::agent::runtime_helpers::append_log(state, &snapshot.run_id, log_message).await;
    let _ = crate::agent::pipeline_helpers::set_review_iteration(
        state,
        &snapshot.run_id,
        next_iteration,
    )
    .await;
    crate::agent::pipeline_helpers::set_last_trigger(
        state,
        &snapshot.run_id,
        Some(next_signature),
        Some(next_summary),
    )
    .await;
    if should_spawn_for_ci {
        // Mark the SHA we're spawning a CI fix-run against so subsequent polls
        // on the same SHA don't re-spawn while the fix-run is in flight or
        // after it completes without pushing. A successful fix-run pushes a
        // new SHA, which naturally won't match this stored value.
        let new_sha = pr_state.head_ref_oid.clone();
        let _ = crate::agent::pipeline_helpers::set_last_ci_failure_sha(
            state,
            &snapshot.run_id,
            new_sha,
        )
        .await;
    }
    crate::agent::pipeline_helpers::mark_review_run_dispatching_fix_run(
        app,
        state,
        &snapshot.run_id,
    )
    .await;

    crate::agent::pipeline_helpers::spawn_fix_run(
        app,
        state,
        crate::agent::pipeline_helpers::FixRunSnapshot {
            run_id: snapshot.run_id,
            repo: snapshot.repo,
            issue_number: snapshot.issue_number,
            issue_title: snapshot.issue_title,
            issue_labels: snapshot.issue_labels,
            workspace_path: PathBuf::from(snapshot.workspace_path),
            previous_context: snapshot.stage_context,
            pr_number,
            branch_name,
            payload: crate::agent::pipeline_helpers::FixRunPayload::Feedback {
                feedback: feedback_text,
                ci_failures: ci_failure_contexts,
            },
        },
    );
}

/// Attempt to spawn a rebase fix-run when GitHub reports the PR as DIRTY.
/// Honors the same external-push and double-spawn guards as the feedback
/// path. Logs (but does not error) when a guard refuses the dispatch — the
/// next poll tick will retry.
async fn try_dispatch_rebase_fix_run(
    app: &AppHandle,
    state: &SharedState,
    snapshot: &ReviewRunSnapshot,
    pr_state: &github::PullRequestFullState,
) {
    // External-push guard: if HEAD moved since the last review request, a
    // human (or some other process) pushed to this branch. `git push
    // --force-with-lease` would already fail safely, but spawning a rebase
    // agent against an externally-rewritten branch is wasted work and can
    // confuse the operator.
    if !head_sha_matches(snapshot.last_pushed_sha.as_deref(), pr_state.head_ref_oid.as_deref()) {
        crate::agent::runtime_helpers::append_log(
            state,
            &snapshot.run_id,
            format!(
                "[review] PR is DIRTY but PR HEAD ({}) differs from last reviewed SHA ({}). Skipping rebase fix-run — external push detected.",
                pr_state.head_ref_oid.as_deref().unwrap_or("unknown"),
                snapshot.last_pushed_sha.as_deref().unwrap_or("unknown"),
            ),
        )
        .await;
        return;
    }

    let already_active = {
        let s = state.lock().await;
        s.runs.values().any(|run| {
            another_review_active(
                &run.repo,
                run.issue_number,
                &run.id,
                &run.stage,
                &run.status,
                &snapshot.repo,
                snapshot.issue_number,
                &snapshot.run_id,
            )
        })
    };
    if already_active {
        crate::agent::runtime_helpers::append_log(
            state,
            &snapshot.run_id,
            "[review] PR is DIRTY but another Review run is already active for this issue — skipping rebase fix-run spawn this tick.".to_string(),
        )
        .await;
        return;
    }

    let pr_number = pr_state.number;
    let branch_name = pr_state.head_ref_name.clone();
    let base_branch = pr_state
        .base_ref_name
        .clone()
        .unwrap_or_else(|| "main".to_string());
    let conflicting_files = pr_state.files.clone();
    let next_iteration = snapshot.review_iteration.saturating_add(1);

    let triggers = vec![TriggerInput::Rebase {
        base_branch: base_branch.as_str(),
        conflicting_files: &conflicting_files,
    }];
    let next_signature = compute_trigger_signature(&triggers);
    let next_summary = trigger_summary(&triggers);

    if let Some(reason) = decide_escape(
        snapshot.review_iteration,
        snapshot.max_review_iterations,
        snapshot.issue_cost_usd,
        snapshot.cost_cap_per_issue_usd,
        snapshot.last_trigger_signature.as_deref(),
        &next_signature,
        &next_summary,
    ) {
        crate::agent::pipeline_helpers::set_last_trigger(
            state,
            &snapshot.run_id,
            Some(next_signature),
            Some(next_summary),
        )
        .await;
        crate::agent::pipeline_helpers::escape_review_run_to_awaiting_approval(
            app,
            state,
            &snapshot.run_id,
            reason.human_message(),
        )
        .await;
        return;
    }

    let log_message = format!(
        "[review] PR mergeStateStatus=DIRTY ({} file{} touched). Spawning rebase fix-run iteration {} against base `{}`.",
        conflicting_files.len(),
        if conflicting_files.len() == 1 { "" } else { "s" },
        next_iteration,
        base_branch,
    );
    let _ = crate::agent::runtime_helpers::append_log(state, &snapshot.run_id, log_message).await;
    let _ = crate::agent::pipeline_helpers::set_review_iteration(
        state,
        &snapshot.run_id,
        next_iteration,
    )
    .await;
    crate::agent::pipeline_helpers::set_last_trigger(
        state,
        &snapshot.run_id,
        Some(next_signature),
        Some(next_summary),
    )
    .await;
    crate::agent::pipeline_helpers::mark_review_run_dispatching_fix_run(
        app,
        state,
        &snapshot.run_id,
    )
    .await;

    crate::agent::pipeline_helpers::spawn_fix_run(
        app,
        state,
        crate::agent::pipeline_helpers::FixRunSnapshot {
            run_id: snapshot.run_id.clone(),
            repo: snapshot.repo.clone(),
            issue_number: snapshot.issue_number,
            issue_title: snapshot.issue_title.clone(),
            issue_labels: snapshot.issue_labels.clone(),
            workspace_path: PathBuf::from(snapshot.workspace_path.clone()),
            previous_context: snapshot.stage_context.clone(),
            pr_number,
            branch_name,
            payload: crate::agent::pipeline_helpers::FixRunPayload::Rebase {
                base_branch,
                conflicting_files,
            },
        },
    );
}

/// What the orchestrator should do when Codex has approved but we still need
/// to consult the CI gate (per issue #4: approval alone does not advance to
/// Merge).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApprovalOutcome {
    /// CI is green (or there are no checks). Safe to advance to Merge.
    Advance,
    /// CI status fetch failed this tick. Hold and retry next tick.
    WaitForCiUnknown,
    /// At least one required check is still pending and nothing has failed.
    /// Hold in Review and wait for CI to settle.
    WaitForCi,
    /// At least one check is failing. Don't advance — let the fix-run path
    /// pick up the failing checks.
    CiFailing,
}

/// True when the consecutive CI-status-fetch failure count has reached the
/// terminal threshold. Extracted as a free function so the boundary is
/// directly unit-testable and the constant lives in one place.
fn ci_fetch_failure_should_terminate(failure_count: u32) -> bool {
    failure_count >= CI_STATUS_FETCH_FAILURE_THRESHOLD
}

fn decide_approval_outcome(ci_status: Option<&PrCiStatus>) -> ApprovalOutcome {
    let Some(status) = ci_status else {
        return ApprovalOutcome::WaitForCiUnknown;
    };
    if !status.failing_checks().is_empty() {
        return ApprovalOutcome::CiFailing;
    }
    if status.is_green() {
        return ApprovalOutcome::Advance;
    }
    // Not green and not failing — by construction (see PrCiStatus::is_green),
    // that means at least one required check is still pending.
    ApprovalOutcome::WaitForCi
}

/// Whether this poll tick should spawn a CI-failure fix-run.
///
/// Triggers exactly when:
/// - There is at least one failing check on the PR, AND
/// - We can identify the current PR HEAD SHA, AND
/// - That SHA differs from the SHA we last spawned a CI fix-run against.
///
/// The SHA dedupe is what stops infinite re-spawns when a fix-run finishes
/// without pushing (or pushes a fix that doesn't actually green CI).
fn should_trigger_ci_fix_run(
    failing_checks: &[&PrCheck],
    current_head: Option<&str>,
    last_ci_failure_sha: Option<&str>,
) -> bool {
    if failing_checks.is_empty() {
        return false;
    }
    let Some(current) = current_head else {
        return false;
    };
    match last_ci_failure_sha {
        Some(prev) if prev == current => false,
        _ => true,
    }
}

async fn build_ci_failure_contexts(
    repo: &str,
    failing_checks: &[&PrCheck],
) -> Vec<crate::agent::CiFailureContext> {
    let mut out = Vec::with_capacity(failing_checks.len().min(MAX_CI_FAILURES_IN_PROMPT));
    for check in failing_checks.iter().take(MAX_CI_FAILURES_IN_PROMPT) {
        let log_excerpt = match check.run_id.as_deref() {
            Some(run_id) => {
                github::run_failure_log_excerpt(repo, run_id, CI_LOG_EXCERPT_MAX_CHARS).await
            }
            None => None,
        };
        out.push(crate::agent::CiFailureContext {
            name: check.name.clone(),
            state: format!("{:?}", check.state).to_uppercase(),
            log_excerpt,
            details_url: check.details_url.clone(),
        });
    }
    out
}

/// Find the first Codex bot comment newer than `baseline_ts` that contains an
/// approve pattern.
pub(crate) fn find_codex_approval<'a>(
    comments: &'a [PrComment],
    baseline_ts: Option<&str>,
    approve_patterns: &[String],
    feedback_marker: &str,
) -> Option<&'a PrComment> {
    comments.iter().find(|comment| {
        is_codex_author(&comment.author)
            && newer_than(&comment.created_at, baseline_ts)
            && github::parse_codex_approval(&comment.body, approve_patterns, feedback_marker)
    })
}

/// Collect Codex bot comments newer than `baseline_ts` that look like review
/// feedback: contain the trailing feedback marker and do NOT match any approve
/// pattern. Returned in original (chronological) order so the agent prompt
/// preserves the order Codex wrote them.
pub(crate) fn collect_codex_feedback<'a>(
    comments: &'a [PrComment],
    baseline_ts: Option<&str>,
    approve_patterns: &[String],
    feedback_marker: &str,
) -> Vec<&'a PrComment> {
    comments
        .iter()
        .filter(|comment| {
            is_codex_author(&comment.author)
                && newer_than(&comment.created_at, baseline_ts)
                && comment_is_feedback(&comment.body, approve_patterns, feedback_marker)
        })
        .collect()
}

/// A Codex comment is treated as feedback when it carries the trailing footer
/// marker (so we know it's a review summary, not chatter) and it doesn't
/// already match an approval pattern.
pub(crate) fn comment_is_feedback(
    body: &str,
    approve_patterns: &[String],
    feedback_marker: &str,
) -> bool {
    if feedback_marker.is_empty() {
        return false;
    }
    if !body.contains(feedback_marker) {
        return false;
    }
    !github::parse_codex_approval(body, approve_patterns, feedback_marker)
}

fn format_feedback_for_prompt(comments: &[&PrComment]) -> String {
    let mut blocks = Vec::with_capacity(comments.len());
    for (index, comment) in comments.iter().enumerate() {
        blocks.push(format!(
            "Comment {} (posted {}):\n{}",
            index + 1,
            comment.created_at,
            comment.body.trim_end()
        ));
    }
    blocks.join("\n\n---\n\n")
}

/// Only the official Codex bot logins count as Codex reviewers — never a substring
/// match. A regular user named e.g. `my-codex-account` must NOT be able to fake
/// approvals via PR comments.
fn is_codex_author(author: &str) -> bool {
    matches!(
        author,
        "chatgpt-codex-connector[bot]" | "chatgpt-codex-connector"
    )
}

fn newer_than(comment_at: &str, baseline: Option<&str>) -> bool {
    let Some(baseline) = baseline else {
        return true;
    };
    comment_at > baseline
}

/// Double-spawn guard: returns true when `candidate_run` (a record from the
/// orchestrator state) represents an *in-flight fix-run* for `target_repo` /
/// `target_issue` that isn't the polling sentinel itself (`target_run_id`).
///
/// We deliberately match only `Preparing` here, not `Running`. The polling
/// sentinel itself stays in `Running`, so treating sibling Review runs in
/// `Running` as active siblings would deadlock when two sentinels exist for
/// the same issue (each sees the other as active and never spawns a fix-run).
/// Fix-runs flip the run to `Preparing` synchronously before `tokio::spawn`
/// and stay there for the entire subprocess lifetime — that's exactly the
/// state we want to guard against.
#[allow(clippy::too_many_arguments)]
fn another_review_active(
    candidate_repo: &str,
    candidate_issue: u64,
    candidate_run_id: &str,
    candidate_stage: &PipelineStage,
    candidate_status: &AgentStatus,
    target_repo: &str,
    target_issue: u64,
    target_run_id: &str,
) -> bool {
    candidate_repo == target_repo
        && candidate_issue == target_issue
        && candidate_stage == &PipelineStage::Review
        && candidate_run_id != target_run_id
        && matches!(candidate_status, AgentStatus::Preparing)
}

/// Trigger flavours that can fire a Review-stage fix-run. The escape logic
/// uses these to build a signature: if two consecutive fix-runs have the
/// identical (sorted) trigger set, the agent is going in circles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TriggerInput<'a> {
    /// Codex left review feedback. The body string is hashed (truncated) so
    /// the same comment text always produces the same signature.
    CodexFeedback { body: &'a str },
    /// A required CI check is failing. Keyed by check name — the same workflow
    /// failing again on a new SHA matches the prior signature, which is the
    /// circular-fix behavior the dedup is meant to catch.
    CiFailure { name: &'a str },
    /// PR is in DIRTY mergeStateStatus. Conflicting file set is used so that
    /// rebasing twice against the same conflict surface is recognised as a loop.
    Rebase {
        base_branch: &'a str,
        conflicting_files: &'a [String],
    },
}

/// Truncated FNV-1a hash of `body`. Stable, fast, and only used to
/// fingerprint comment bodies inside trigger signatures — not security-critical.
fn fingerprint(body: &str) -> String {
    // FNV-1a 64-bit. We render only the lower 8 hex digits to keep the
    // signature short; collisions on short hashes are acceptable here because
    // the only consequence is a false-positive escape, which the operator can
    // override.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in body.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:08x}", (hash & 0xffff_ffff) as u32)
}

/// Build a stable signature string from the next fix-run's trigger set. Used
/// to detect the "two consecutive fix-runs with identical triggers" loop.
pub(crate) fn compute_trigger_signature(triggers: &[TriggerInput<'_>]) -> String {
    let mut parts: Vec<String> = triggers
        .iter()
        .map(|trigger| match trigger {
            TriggerInput::CodexFeedback { body } => format!("codex:{}", fingerprint(body)),
            TriggerInput::CiFailure { name } => format!("ci:{}", name),
            TriggerInput::Rebase {
                base_branch,
                conflicting_files,
            } => {
                let mut files: Vec<&str> =
                    conflicting_files.iter().map(String::as_str).collect();
                files.sort_unstable();
                format!("rebase:{}|{}", base_branch, files.join(","))
            }
        })
        .collect();
    parts.sort_unstable();
    parts.dedup();
    parts.join(";")
}

/// Short, human-readable label describing the trigger set for the dashboard.
/// Matches the rough shape `Codex feedback`, `CI failure: build`, `Conflict`.
pub(crate) fn trigger_summary(triggers: &[TriggerInput<'_>]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut codex_count = 0usize;
    let mut ci_names: Vec<&str> = Vec::new();
    let mut had_rebase = false;
    for trigger in triggers {
        match trigger {
            TriggerInput::CodexFeedback { .. } => codex_count += 1,
            TriggerInput::CiFailure { name } => ci_names.push(name),
            TriggerInput::Rebase { .. } => had_rebase = true,
        }
    }
    if codex_count > 0 {
        parts.push(if codex_count == 1 {
            "Codex feedback".to_string()
        } else {
            format!("Codex feedback ({})", codex_count)
        });
    }
    if !ci_names.is_empty() {
        ci_names.sort_unstable();
        ci_names.dedup();
        parts.push(format!("CI failure: {}", ci_names.join(", ")));
    }
    if had_rebase {
        parts.push("Conflict".to_string());
    }
    if parts.is_empty() {
        "Unknown trigger".to_string()
    } else {
        parts.join(" + ")
    }
}

/// Reasons a Review run can escape into `AwaitingApproval` instead of
/// spawning yet another fix-run. Surfaced verbatim in the run log and
/// notification body so operators see *why* the loop halted.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum EscapeReason {
    /// `review_iteration` is at or above the configured cap.
    IterationCap { iterations: u32, cap: u32 },
    /// Cumulative cost across every run for this issue exceeds the cap.
    CostCap {
        cost_usd: f64,
        cap_usd: f64,
    },
    /// The next would-be trigger signature exactly matches the prior one —
    /// the agent is being asked to fix the same thing twice.
    SignatureDedup { signature: String, summary: String },
}

impl EscapeReason {
    pub(crate) fn human_message(&self) -> String {
        match self {
            EscapeReason::IterationCap { iterations, cap } => format!(
                "review iteration cap reached ({}/{})",
                iterations, cap
            ),
            EscapeReason::CostCap { cost_usd, cap_usd } => format!(
                "cost cap exceeded for this issue (${:.2} > ${:.2})",
                cost_usd, cap_usd
            ),
            EscapeReason::SignatureDedup { summary, .. } => format!(
                "two consecutive fix-runs with identical triggers ({}) — agent is looping",
                summary
            ),
        }
    }
}

/// Decide whether the next fix-run should escape to AwaitingApproval before
/// spawning. `iterations_so_far` is the run's current `review_iteration`
/// counter (i.e. the number of fix-runs that already ran), `next_signature`
/// is the signature of the about-to-fire trigger set.
///
/// Returns the first matching reason in priority order: iteration → cost →
/// signature dedup. Iteration and cost are deterministic environment limits
/// and outrank the signature heuristic.
pub(crate) fn decide_escape(
    iterations_so_far: u32,
    iteration_cap: u32,
    cumulative_cost_usd: f64,
    cost_cap_usd: f64,
    last_signature: Option<&str>,
    next_signature: &str,
    next_summary: &str,
) -> Option<EscapeReason> {
    if iteration_cap > 0 && iterations_so_far >= iteration_cap {
        return Some(EscapeReason::IterationCap {
            iterations: iterations_so_far,
            cap: iteration_cap,
        });
    }
    if cost_cap_usd > 0.0 && cumulative_cost_usd > cost_cap_usd {
        return Some(EscapeReason::CostCap {
            cost_usd: cumulative_cost_usd,
            cap_usd: cost_cap_usd,
        });
    }
    if let Some(prior) = last_signature {
        if !prior.is_empty() && prior == next_signature {
            return Some(EscapeReason::SignatureDedup {
                signature: next_signature.to_string(),
                summary: next_summary.to_string(),
            });
        }
    }
    None
}

/// Returns true when the PR's current HEAD matches the SHA we recorded at the
/// time we asked Codex to review. If we don't have a recorded SHA (e.g. a legacy
/// run from before this field existed) or we don't know the PR HEAD, we accept —
/// the review-request timestamp is still gating the comment.
fn head_sha_matches(reviewed_sha: Option<&str>, current_head: Option<&str>) -> bool {
    match (reviewed_sha, current_head) {
        (Some(reviewed), Some(current)) => reviewed == current,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comment(author: &str, created_at: &str, body: &str) -> PrComment {
        PrComment {
            author: author.to_string(),
            body: body.to_string(),
            created_at: created_at.to_string(),
        }
    }

    fn approve_patterns() -> Vec<String> {
        vec![
            "Didn't find any major issues".to_string(),
            "did not find major issues".to_string(),
        ]
    }

    fn marker() -> &'static str {
        "Useful? React with 👍 / 👎."
    }

    #[test]
    fn newer_than_returns_true_when_no_baseline() {
        assert!(newer_than("2026-01-01T00:00:00Z", None));
    }

    #[test]
    fn newer_than_compares_rfc3339_strings() {
        assert!(newer_than(
            "2026-05-09T10:00:00Z",
            Some("2026-05-09T09:00:00Z"),
        ));
        assert!(!newer_than(
            "2026-05-09T08:00:00Z",
            Some("2026-05-09T09:00:00Z"),
        ));
    }

    #[test]
    fn is_codex_author_only_matches_official_bot_logins_exactly() {
        assert!(is_codex_author("chatgpt-codex-connector[bot]"));
        assert!(is_codex_author("chatgpt-codex-connector"));

        assert!(!is_codex_author("Codex"));
        assert!(!is_codex_author("my-codex-account"));
        assert!(!is_codex_author("codex-user"));
        assert!(!is_codex_author("CHATGPT-CODEX-CONNECTOR[BOT]"));
        assert!(!is_codex_author("kulichevskiy"));
    }

    fn pr_check(
        name: &str,
        state: github::CheckState,
        is_required: bool,
        run_id: Option<&str>,
    ) -> github::PrCheck {
        github::PrCheck {
            name: name.to_string(),
            state,
            is_required,
            run_id: run_id.map(|s| s.to_string()),
            details_url: None,
        }
    }

    #[test]
    fn decide_approval_outcome_advances_when_ci_is_green() {
        let status = github::PrCiStatus::default();
        assert_eq!(decide_approval_outcome(Some(&status)), ApprovalOutcome::Advance);
    }

    #[test]
    fn decide_approval_outcome_holds_when_ci_status_is_unknown() {
        // We can't tell if it's safe to advance without a CI verdict — refuse
        // to advance rather than ship without one.
        assert_eq!(decide_approval_outcome(None), ApprovalOutcome::WaitForCiUnknown);
    }

    #[test]
    fn decide_approval_outcome_holds_when_required_check_pending() {
        let status = github::PrCiStatus {
            checks: vec![
                pr_check("build", github::CheckState::Success, true, None),
                pr_check("test", github::CheckState::Pending, true, None),
            ],
        };
        assert_eq!(decide_approval_outcome(Some(&status)), ApprovalOutcome::WaitForCi);
    }

    #[test]
    fn decide_approval_outcome_falls_through_to_fix_run_when_ci_is_failing() {
        let status = github::PrCiStatus {
            checks: vec![
                pr_check("build", github::CheckState::Failure, true, None),
            ],
        };
        assert_eq!(decide_approval_outcome(Some(&status)), ApprovalOutcome::CiFailing);
    }

    #[test]
    fn ci_fetch_failure_should_terminate_only_at_or_above_threshold() {
        // Below threshold: keep retrying. The orchestrator stays in
        // WaitForCiUnknown and logs "failure N of THRESHOLD".
        for count in 0..CI_STATUS_FETCH_FAILURE_THRESHOLD {
            assert!(
                !ci_fetch_failure_should_terminate(count),
                "count {count} should NOT terminate (threshold is {CI_STATUS_FETCH_FAILURE_THRESHOLD})",
            );
        }
        // At threshold: terminate. This is the boundary Codex's P1 was about —
        // we must not keep silently retrying when `gh` is permanently broken.
        assert!(ci_fetch_failure_should_terminate(
            CI_STATUS_FETCH_FAILURE_THRESHOLD
        ));
        // Far above threshold: still terminate (defensive against counter math
        // overshoot).
        assert!(ci_fetch_failure_should_terminate(
            CI_STATUS_FETCH_FAILURE_THRESHOLD + 100
        ));
    }

    #[test]
    fn should_trigger_ci_fix_run_only_when_failures_present_and_sha_is_new() {
        let failing_with_run =
            pr_check("build", github::CheckState::Failure, true, Some("9999"));
        let failing_refs = vec![&failing_with_run];

        // No failures: never trigger.
        assert!(!should_trigger_ci_fix_run(&[], Some("sha-1"), None));
        // Failures + new SHA + no prior CI fix-run: trigger.
        assert!(should_trigger_ci_fix_run(
            &failing_refs,
            Some("sha-1"),
            None
        ));
        // Failures + same SHA we already kicked a fix-run for: dedupe (don't
        // re-spawn while the fix-run is in flight or hasn't pushed yet).
        assert!(!should_trigger_ci_fix_run(
            &failing_refs,
            Some("sha-1"),
            Some("sha-1")
        ));
        // Failures + different SHA than the one we last spawned for: trigger
        // again — fix-run pushed a new commit but CI is still red.
        assert!(should_trigger_ci_fix_run(
            &failing_refs,
            Some("sha-2"),
            Some("sha-1")
        ));
        // Failures but unknown current HEAD: can't safely dedupe, skip this tick.
        assert!(!should_trigger_ci_fix_run(&failing_refs, None, None));
    }

    #[test]
    fn head_sha_matches_accepts_when_reviewed_and_current_agree() {
        assert!(head_sha_matches(Some("abc123"), Some("abc123")));
    }

    #[test]
    fn head_sha_matches_rejects_when_pr_head_changed_since_review_request() {
        assert!(!head_sha_matches(Some("abc123"), Some("def456")));
    }

    #[test]
    fn head_sha_matches_is_lenient_when_either_side_is_unknown() {
        assert!(head_sha_matches(None, Some("abc123")));
        assert!(head_sha_matches(Some("abc123"), None));
        assert!(head_sha_matches(None, None));
    }

    fn make_review_run(
        id: &str,
        issue_number: u64,
        status: AgentStatus,
        started_at: &str,
    ) -> crate::orchestrator::AgentRun {
        crate::orchestrator::AgentRun {
            id: id.to_string(),
            repo: "kulichevskiy/SymphonyMac".to_string(),
            issue_number,
            issue_title: "test".to_string(),
            status,
            stage: PipelineStage::Review,
            started_at: started_at.to_string(),
            finished_at: None,
            logs: vec![],
            workspace_path: "/tmp/x".to_string(),
            error: None,
            attempt: 1,
            max_retries: 0,
            lines_added: 0,
            lines_removed: 0,
            files_modified_list: vec![],
            report: None,
            command_display: None,
            agent_type: "claude".to_string(),
            last_log_line: None,
            log_count: 0,
            activity: None,
            last_log_timestamp: None,
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: 0.0,
            issue_labels: vec![],
            skipped_stages: vec![],
            stage_context: None,
            pending_next_stage: None,
            last_pushed_sha: None,
            last_review_request_at: None,
            review_iteration: 0,
            last_ci_failure_sha: None,
            ci_status_fetch_failure_count: 0,
            last_trigger_signature: None,
            last_trigger_summary: None,
        }
    }

    #[test]
    fn latest_running_review_per_issue_dedupes_to_newest_started_at() {
        let older = make_review_run("older", 42, AgentStatus::Running, "2026-05-09T10:00:00Z");
        let newer = make_review_run("newer", 42, AgentStatus::Running, "2026-05-09T11:00:00Z");
        let other_issue =
            make_review_run("other-issue", 43, AgentStatus::Running, "2026-05-09T09:00:00Z");
        let preparing =
            make_review_run("preparing", 42, AgentStatus::Preparing, "2026-05-09T11:30:00Z");
        let stopped =
            make_review_run("stopped", 42, AgentStatus::Stopped, "2026-05-09T12:00:00Z");

        let runs = vec![older, newer, other_issue, preparing, stopped];
        let result = latest_running_review_per_issue(runs.iter());
        let mut ids: Vec<&str> = result.iter().map(|r| r.id.as_str()).collect();
        ids.sort();

        // Issue 42: only the newest Running run (`newer`) — `older` is dropped,
        // and Preparing/Stopped runs are filtered out entirely.
        // Issue 43: the lone Running sentinel survives.
        assert_eq!(ids, vec!["newer", "other-issue"]);
    }

    #[test]
    fn another_review_active_flags_sibling_fix_run_in_preparing() {
        // Same repo+issue, Review stage, Preparing — that's a sibling fix-run
        // already dispatched. We must not double-spawn against it.
        assert!(another_review_active(
            "kulichevskiy/SymphonyMac",
            42,
            "sibling-run-id",
            &PipelineStage::Review,
            &AgentStatus::Preparing,
            "kulichevskiy/SymphonyMac",
            42,
            "self-run-id",
        ));
    }

    #[test]
    fn another_review_active_does_not_flag_sibling_polling_sentinels() {
        // A second polling sentinel in `Running` for the same issue is not an
        // in-flight fix-run. Treating it as active would deadlock when two
        // sentinels see each other and refuse to dispatch.
        assert!(!another_review_active(
            "kulichevskiy/SymphonyMac",
            42,
            "sibling-run-id",
            &PipelineStage::Review,
            &AgentStatus::Running,
            "kulichevskiy/SymphonyMac",
            42,
            "self-run-id",
        ));
    }

    #[test]
    fn another_review_active_ignores_self_and_other_issues_or_stages() {
        // Self — that's the polling sentinel itself, never counts as a duplicate.
        assert!(!another_review_active(
            "kulichevskiy/SymphonyMac",
            42,
            "self-run-id",
            &PipelineStage::Review,
            &AgentStatus::Running,
            "kulichevskiy/SymphonyMac",
            42,
            "self-run-id",
        ));
        // Different issue — fix-run for #43 doesn't block fix-run for #42.
        assert!(!another_review_active(
            "kulichevskiy/SymphonyMac",
            43,
            "other-run-id",
            &PipelineStage::Review,
            &AgentStatus::Running,
            "kulichevskiy/SymphonyMac",
            42,
            "self-run-id",
        ));
        // Different stage — Implement run for the same issue is unrelated.
        assert!(!another_review_active(
            "kulichevskiy/SymphonyMac",
            42,
            "implement-run-id",
            &PipelineStage::Implement,
            &AgentStatus::Running,
            "kulichevskiy/SymphonyMac",
            42,
            "self-run-id",
        ));
        // Different repo — coincidence, not a conflict.
        assert!(!another_review_active(
            "other/repo",
            42,
            "other-run-id",
            &PipelineStage::Review,
            &AgentStatus::Running,
            "kulichevskiy/SymphonyMac",
            42,
            "self-run-id",
        ));
    }

    #[test]
    fn another_review_active_skips_non_preparing_statuses() {
        // Only `Preparing` indicates an in-flight fix-run. Every other status —
        // including a sibling polling sentinel in `Running` — must NOT block,
        // otherwise we'd deadlock when two Review records exist for the same
        // issue. Terminal/idle statuses obviously aren't doing work.
        for status in [
            AgentStatus::Running,
            AgentStatus::Completed,
            AgentStatus::Failed,
            AgentStatus::Stopped,
            AgentStatus::Interrupted,
            AgentStatus::AwaitingApproval,
        ] {
            assert!(
                !another_review_active(
                    "kulichevskiy/SymphonyMac",
                    42,
                    "sibling-run-id",
                    &PipelineStage::Review,
                    &status,
                    "kulichevskiy/SymphonyMac",
                    42,
                    "self-run-id",
                ),
                "{status:?} should NOT block fix-run spawn"
            );
        }
    }

    #[test]
    fn comment_is_feedback_requires_marker_and_no_approval_phrase() {
        // Feedback: marker present, no approval phrase.
        let body = "Found a missing test. Please add coverage.\n\nUseful? React with 👍 / 👎.";
        assert!(comment_is_feedback(body, &approve_patterns(), marker()));

        // Approval: marker present but text matches approval pattern -> NOT feedback.
        let approving = "Didn't find any major issues.\n\nUseful? React with 👍 / 👎.";
        assert!(!comment_is_feedback(approving, &approve_patterns(), marker()));

        // No marker — chatter we don't act on.
        let chatter = "agree, going to look later";
        assert!(!comment_is_feedback(chatter, &approve_patterns(), marker()));

        // Empty marker — we never trigger fix-runs (defensive).
        assert!(!comment_is_feedback(body, &approve_patterns(), ""));
    }

    #[test]
    fn collect_codex_feedback_with_no_baseline_treats_all_history_as_new() {
        // Documents the hazard the missing-baseline guard in `check_codex_activity`
        // protects against: without a baseline timestamp, every historical Codex
        // feedback comment is "new". The orchestrator-level guard refuses to
        // spawn a fix-run when `last_review_request_at` is None precisely so
        // this set isn't acted on against a stale review cycle.
        let comments = vec![
            comment(
                "chatgpt-codex-connector[bot]",
                "2025-01-01T00:00:00Z",
                "Old feedback from prior cycle.\n\nUseful? React with 👍 / 👎.",
            ),
            comment(
                "chatgpt-codex-connector[bot]",
                "2025-06-01T00:00:00Z",
                "Other old feedback.\n\nUseful? React with 👍 / 👎.",
            ),
        ];

        let with_no_baseline =
            collect_codex_feedback(&comments, None, &approve_patterns(), marker());
        assert_eq!(
            with_no_baseline.len(),
            2,
            "without a baseline, every historical Codex feedback comment looks new"
        );
    }

    #[test]
    fn collect_codex_feedback_skips_old_and_non_codex_comments() {
        let comments = vec![
            comment(
                "kulichevskiy",
                "2026-05-09T10:00:00Z",
                "Looks fine to me. Useful? React with 👍 / 👎.",
            ),
            comment(
                "chatgpt-codex-connector[bot]",
                "2026-05-09T09:00:00Z", // before baseline; ignored
                "Old feedback. Useful? React with 👍 / 👎.",
            ),
            comment(
                "chatgpt-codex-connector[bot]",
                "2026-05-09T11:00:00Z",
                "Missing test for edge case.\n\nUseful? React with 👍 / 👎.",
            ),
            comment(
                "chatgpt-codex-connector",
                "2026-05-09T12:00:00Z",
                "Didn't find any major issues.\n\nUseful? React with 👍 / 👎.",
            ),
        ];
        let baseline = Some("2026-05-09T09:30:00Z");
        let feedback = collect_codex_feedback(&comments, baseline, &approve_patterns(), marker());

        assert_eq!(feedback.len(), 1);
        assert!(feedback[0].body.contains("Missing test for edge case"));
    }

    #[test]
    fn find_codex_approval_picks_first_matching_codex_comment() {
        let comments = vec![
            comment(
                "kulichevskiy",
                "2026-05-09T11:00:00Z",
                "Didn't find any major issues. Useful? React with 👍 / 👎.",
            ),
            comment(
                "chatgpt-codex-connector[bot]",
                "2026-05-09T12:00:00Z",
                "Didn't find any major issues.\n\nUseful? React with 👍 / 👎.",
            ),
        ];
        let approval =
            find_codex_approval(&comments, None, &approve_patterns(), marker()).expect("approval");
        assert_eq!(approval.author, "chatgpt-codex-connector[bot]");
    }

    #[test]
    fn format_feedback_for_prompt_preserves_chronological_order_and_separators() {
        let one = comment(
            "chatgpt-codex-connector[bot]",
            "2026-05-09T11:00:00Z",
            "first concern\n\nUseful? React with 👍 / 👎.",
        );
        let two = comment(
            "chatgpt-codex-connector[bot]",
            "2026-05-09T11:30:00Z",
            "second concern\n\nUseful? React with 👍 / 👎.",
        );
        let combined = format_feedback_for_prompt(&[&one, &two]);

        let first_pos = combined.find("first concern").expect("first present");
        let separator_pos = combined.find("---").expect("separator present");
        let second_pos = combined.find("second concern").expect("second present");
        assert!(first_pos < separator_pos);
        assert!(separator_pos < second_pos);
        assert!(combined.contains("Comment 1"));
        assert!(combined.contains("Comment 2"));
    }

    // ─── Stuck-loop escape tests (issue #6) ───────────────────────────

    #[test]
    fn compute_trigger_signature_is_deterministic_and_order_independent() {
        // Same trigger set in different order produces the same signature so
        // the dedup is robust to the order Codex/CI report in.
        let body_a = "first concern";
        let body_b = "second concern";
        let files = vec!["a.rs".to_string(), "b.rs".to_string()];
        let triggers_a = vec![
            TriggerInput::CodexFeedback { body: body_a },
            TriggerInput::CiFailure { name: "build" },
            TriggerInput::Rebase {
                base_branch: "main",
                conflicting_files: &files,
            },
            TriggerInput::CodexFeedback { body: body_b },
        ];
        let triggers_b = vec![
            TriggerInput::CiFailure { name: "build" },
            TriggerInput::CodexFeedback { body: body_b },
            TriggerInput::CodexFeedback { body: body_a },
            TriggerInput::Rebase {
                base_branch: "main",
                conflicting_files: &files,
            },
        ];

        assert_eq!(
            compute_trigger_signature(&triggers_a),
            compute_trigger_signature(&triggers_b)
        );
    }

    #[test]
    fn compute_trigger_signature_distinguishes_different_comment_bodies() {
        // Different feedback text must produce different signatures —
        // otherwise we'd incorrectly dedupe genuine new feedback.
        let triggers_a = vec![TriggerInput::CodexFeedback {
            body: "missing test for edge case",
        }];
        let triggers_b = vec![TriggerInput::CodexFeedback {
            body: "rename function for clarity",
        }];
        assert_ne!(
            compute_trigger_signature(&triggers_a),
            compute_trigger_signature(&triggers_b)
        );
    }

    #[test]
    fn decide_escape_returns_iteration_cap_when_counter_at_or_above_limit() {
        let next_signature = "codex:abc";
        let result = decide_escape(10, 10, 0.0, 0.0, None, next_signature, "Codex feedback");
        assert!(matches!(
            result,
            Some(EscapeReason::IterationCap { iterations: 10, cap: 10 })
        ));

        // Same with counter strictly above.
        let result = decide_escape(15, 10, 0.0, 0.0, None, next_signature, "Codex feedback");
        assert!(matches!(
            result,
            Some(EscapeReason::IterationCap { iterations: 15, cap: 10 })
        ));
    }

    #[test]
    fn decide_escape_iteration_cap_zero_disables_check() {
        // cap=0 means "disabled" — even a high iteration count must not
        // escape via this path.
        let result = decide_escape(99, 0, 0.0, 0.0, None, "sig", "Codex feedback");
        assert!(result.is_none());
    }

    #[test]
    fn decide_escape_returns_cost_cap_when_cumulative_cost_exceeds_cap() {
        let result = decide_escape(0, 100, 5.01, 5.0, None, "sig", "Codex feedback");
        match result {
            Some(EscapeReason::CostCap { cost_usd, cap_usd }) => {
                assert!((cost_usd - 5.01).abs() < f64::EPSILON);
                assert!((cap_usd - 5.0).abs() < f64::EPSILON);
            }
            other => panic!("expected CostCap, got {:?}", other),
        }
    }

    #[test]
    fn decide_escape_cost_cap_zero_or_equal_does_not_fire() {
        // Cap=0.0 → disabled, even with non-zero cost.
        assert!(decide_escape(0, 100, 99.0, 0.0, None, "sig", "x").is_none());
        // Cost equal to cap → not strictly greater, so does not fire.
        assert!(decide_escape(0, 100, 5.0, 5.0, None, "sig", "x").is_none());
    }

    #[test]
    fn decide_escape_returns_signature_dedup_when_signatures_match() {
        let signature = "codex:abc;ci:build";
        let result = decide_escape(
            1,
            100,
            0.0,
            0.0,
            Some(signature),
            signature,
            "Codex feedback",
        );
        assert!(matches!(result, Some(EscapeReason::SignatureDedup { .. })));
    }

    #[test]
    fn decide_escape_signature_dedup_does_not_fire_when_signatures_differ() {
        let result = decide_escape(
            1,
            100,
            0.0,
            0.0,
            Some("codex:aaa"),
            "codex:bbb",
            "Codex feedback",
        );
        assert!(result.is_none());
    }

    #[test]
    fn decide_escape_signature_dedup_does_not_fire_on_first_iteration() {
        // No prior signature → never dedup.
        let result = decide_escape(
            0,
            100,
            0.0,
            0.0,
            None,
            "codex:abc",
            "Codex feedback",
        );
        assert!(result.is_none());
    }

    #[test]
    fn decide_escape_priority_iteration_then_cost_then_signature() {
        // All three triggers fire; iteration cap should win because it's the
        // most deterministic.
        let signature = "codex:abc";
        let result = decide_escape(
            10,
            10,
            10.0,
            5.0,
            Some(signature),
            signature,
            "Codex feedback",
        );
        assert!(matches!(result, Some(EscapeReason::IterationCap { .. })));

        // Without the iteration cap (set to 0), cost should win over signature.
        let result = decide_escape(
            10,
            0,
            10.0,
            5.0,
            Some(signature),
            signature,
            "Codex feedback",
        );
        assert!(matches!(result, Some(EscapeReason::CostCap { .. })));
    }

    #[test]
    fn trigger_summary_renders_all_kinds() {
        let files = vec!["a.rs".to_string()];
        let triggers = vec![
            TriggerInput::CodexFeedback { body: "x" },
            TriggerInput::CodexFeedback { body: "y" },
            TriggerInput::CiFailure { name: "build" },
            TriggerInput::Rebase {
                base_branch: "main",
                conflicting_files: &files,
            },
        ];
        let summary = trigger_summary(&triggers);
        assert!(summary.contains("Codex feedback (2)"));
        assert!(summary.contains("CI failure: build"));
        assert!(summary.contains("Conflict"));
    }

    fn make_run_with_cost(
        repo: &str,
        issue_number: u64,
        cost: f64,
    ) -> crate::orchestrator::AgentRun {
        let mut run = make_review_run("run", issue_number, AgentStatus::Completed, "2026-05-09T10:00:00Z");
        run.repo = repo.to_string();
        run.cost_usd = cost;
        run
    }

    #[test]
    fn cumulative_cost_for_issue_sums_all_runs_for_same_issue_only() {
        use std::collections::HashMap;

        let mut state = crate::orchestrator::OrchestratorState {
            is_running: false,
            repos: vec![],
            runs: HashMap::new(),
            config: crate::orchestrator::RunConfig::default(),
            agent_pids: HashMap::new(),
            stop_flag: false,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cost_usd: 0.0,
            total_runtime_secs: 0.0,
        };

        let mut a1 = make_run_with_cost("kulichevskiy/SymphonyMac", 42, 1.25);
        a1.id = "a1".to_string();
        let mut a2 = make_run_with_cost("kulichevskiy/SymphonyMac", 42, 2.5);
        a2.id = "a2".to_string();
        // Different issue — must NOT be summed.
        let mut b1 = make_run_with_cost("kulichevskiy/SymphonyMac", 43, 7.0);
        b1.id = "b1".to_string();
        // Different repo — must NOT be summed.
        let mut c1 = make_run_with_cost("other/repo", 42, 9.0);
        c1.id = "c1".to_string();

        state.runs.insert(a1.id.clone(), a1);
        state.runs.insert(a2.id.clone(), a2);
        state.runs.insert(b1.id.clone(), b1);
        state.runs.insert(c1.id.clone(), c1);

        let total = crate::orchestrator::cumulative_cost_for_issue(
            &state,
            "kulichevskiy/SymphonyMac",
            42,
        );
        assert!((total - 3.75).abs() < f64::EPSILON);
    }
}
