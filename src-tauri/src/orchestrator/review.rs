use crate::github::{self, PrComment};
use crate::SharedState;
use std::collections::HashMap;
use std::path::PathBuf;
use tauri::AppHandle;

use super::{AgentStatus, PipelineStage};

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
}

pub async fn poll_review_runs(app: &AppHandle, state: &SharedState) {
    let (snapshots, approve_patterns, feedback_marker) = {
        let s = state.lock().await;
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
                .map(|run| ReviewRunSnapshot {
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

    // 1. Approval takes priority — if Codex says LGTM, advance to Merge regardless of
    // whether earlier comments were feedback.
    if let Some(comment) = find_codex_approval(
        &pr_state.comments,
        snapshot.last_review_request_at.as_deref(),
        approve_patterns,
        feedback_marker,
    ) {
        // Defense against changes pushed *after* `@codex review`: if HEAD moved,
        // the approval is stale and we wait for a fresh review.
        if !head_sha_matches(snapshot.last_pushed_sha.as_deref(), pr_state.head_ref_oid.as_deref())
        {
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

        crate::agent::runtime_helpers::append_log(
            state,
            &snapshot.run_id,
            format!(
                "[review] Codex approval detected at {} — advancing to Merge.",
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

    // 2. No approval — look for actionable feedback comments and spawn a fix-run.
    //
    // Refuse to fix-run when we don't have a `last_review_request_at` baseline
    // (legacy persisted Review runs, or a run observed before its first
    // `@codex review` post wrote the field). Without a baseline,
    // `collect_codex_feedback` would treat *every* historical Codex comment as
    // new and could force-push fixes for stale feedback that's already been
    // addressed in a prior cycle. Approval doesn't have this hazard (an
    // outdated approval is gated by the `head_sha_matches` check), so we only
    // bail out of the feedback path here.
    let Some(baseline_ts) = snapshot.last_review_request_at.as_deref() else {
        return;
    };

    let feedback_comments = collect_codex_feedback(
        &pr_state.comments,
        Some(baseline_ts),
        approve_patterns,
        feedback_marker,
    );

    if feedback_comments.is_empty() {
        return;
    }

    // External-push guard: if HEAD moved since the last review request, a human
    // (or some other process) pushed to this branch. Don't fire a fix-run on
    // top — we don't know what that push contained, and force-pushing over it
    // could destroy work. Wait until the human investigates.
    if !head_sha_matches(snapshot.last_pushed_sha.as_deref(), pr_state.head_ref_oid.as_deref()) {
        crate::agent::runtime_helpers::append_log(
            state,
            &snapshot.run_id,
            format!(
                "[review] Detected Codex feedback but PR HEAD ({}) differs from last reviewed SHA ({}). Skipping fix-run — external push detected.",
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
            "[review] Detected Codex feedback but another Review run is already active for this issue — skipping fix-run spawn this tick."
                .to_string(),
        )
        .await;
        return;
    }

    let feedback_text = format_feedback_for_prompt(&feedback_comments);
    let pr_number = pr_state.number;
    let branch_name = pr_state.head_ref_name.clone();
    let next_iteration = snapshot.review_iteration.saturating_add(1);

    // Increment review_iteration AND drop the run out of `Running` BEFORE we
    // tokio::spawn the fix-run agent. The poll loop filters by `status ==
    // Running`, so transitioning to `Preparing` synchronously here closes the
    // race Codex flagged: even if the next poll tick fires before the spawned
    // task registers a PID (e.g. while the `before_run` hook is still running),
    // it will see the run is no longer in `Running` and skip it. The fix-run
    // path inside `run_agent_process` keeps the run in `Preparing` until
    // `finalize_fix_run_success` flips it back.
    let log_message = format!(
        "[review] Codex feedback detected ({} comment{}). Spawning fix-run iteration {}.",
        feedback_comments.len(),
        if feedback_comments.len() == 1 { "" } else { "s" },
        next_iteration,
    );
    let _ = crate::agent::runtime_helpers::append_log(state, &snapshot.run_id, log_message).await;
    let _ = crate::agent::pipeline_helpers::set_review_iteration(
        state,
        &snapshot.run_id,
        next_iteration,
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
            run_id: snapshot.run_id,
            repo: snapshot.repo,
            issue_number: snapshot.issue_number,
            issue_title: snapshot.issue_title,
            issue_labels: snapshot.issue_labels,
            workspace_path: PathBuf::from(snapshot.workspace_path),
            previous_context: snapshot.stage_context,
            pr_number,
            branch_name,
            feedback: feedback_text,
        },
    );
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
}
