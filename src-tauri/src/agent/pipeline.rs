use super::process::run_agent_process;
use super::prompt::{build_command_args, build_prompt, format_command_display};
use super::runtime;
use crate::orchestrator::{AgentRun, AgentStatus, PipelineStage, RunConfig, StageContext};
use crate::{report, SharedState};
use chrono::Utc;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

/// Distinguishes the two flavours of Review-stage fix-runs. Both reuse the
/// polling Review run's id and run inside the existing PR worktree, but they
/// differ in prompt, success treatment, and failure escape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FixRunKind {
    /// Codex left actionable review feedback — apply it, push, re-request review.
    Feedback,
    /// `mergeStateStatus == DIRTY` — rebase onto base, push, re-request review.
    /// On non-zero exit (or unmerged paths still present after the agent gives
    /// up), the run escapes to AwaitingApproval with `pending_next_stage = "merge"`
    /// instead of being marked Failed.
    Rebase,
}

#[derive(Debug, Clone)]
pub(crate) struct StageLaunchSpec {
    pub repo: String,
    pub issue_number: u64,
    pub issue_title: String,
    pub issue_body: String,
    pub stage: PipelineStage,
    pub issue_labels: Vec<String>,
    pub workspace_path: PathBuf,
    pub attempt: u32,
    pub max_retries: u32,
    pub previous_error: String,
    pub previous_context: Option<StageContext>,
    /// True when this spec drives a Review-stage fix-run launched in response to
    /// Codex feedback OR a DIRTY mergeStateStatus. Fix-runs reuse the Review
    /// polling run's id, skip the red-gate, and on success re-post `@codex review`
    /// instead of advancing.
    pub is_fix_run: bool,
    /// Set when `is_fix_run == true` to indicate which fix-run variant is
    /// running. None for non-fix-run specs.
    pub fix_run_kind: Option<FixRunKind>,
}

#[derive(Debug, Clone)]
pub(crate) struct AgentProcessRequest {
    pub run_id: String,
    pub command: String,
    pub args: Vec<String>,
    pub spec: StageLaunchSpec,
}

struct PreparedStageRun {
    run: AgentRun,
    request: AgentProcessRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MergeVerification {
    NotRequired,
    Verified,
    NotMerged,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SuccessfulStageAction {
    Advance {
        next_stage: PipelineStage,
        skipped_logs: Vec<String>,
    },
    AwaitingApproval {
        next_stage: PipelineStage,
        skipped_logs: Vec<String>,
    },
    FinishPipeline,
    MergeBlocked,
    Noop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FailureAction {
    Retry {
        next_attempt: u32,
        backoff_secs: u64,
    },
    Exhausted,
}

#[derive(Debug, Clone)]
pub(crate) struct PipelineCompletionSpec {
    pub repo: String,
    pub issue_number: u64,
    pub issue_title: String,
    pub workspace_path: PathBuf,
    pub issue_labels: Vec<String>,
    pub skipped_stages: Vec<PipelineStage>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct UsageTotals {
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
}

pub(crate) fn compute_retry_backoff(config: &RunConfig, attempt: u32) -> u64 {
    let base_delay = if config.retry_base_delay_secs > 0 {
        config.retry_base_delay_secs
    } else {
        config.retry_backoff_secs
    };
    let exponent = attempt.saturating_sub(1);
    let exp_backoff = base_delay.saturating_mul(2u64.saturating_pow(exponent));
    exp_backoff.min(config.retry_max_backoff_secs)
}

pub(crate) fn decide_failure_action(
    current_attempt: u32,
    max_retries: u32,
    config: &RunConfig,
) -> FailureAction {
    if current_attempt <= max_retries {
        FailureAction::Retry {
            next_attempt: current_attempt + 1,
            backoff_secs: compute_retry_backoff(config, current_attempt),
        }
    } else {
        FailureAction::Exhausted
    }
}

pub(crate) fn decide_successful_stage_action(
    stage: &PipelineStage,
    skipped_stages: &[PipelineStage],
    gate_enabled: bool,
    merge_verification: MergeVerification,
) -> SuccessfulStageAction {
    match stage {
        PipelineStage::Merge => match merge_verification {
            MergeVerification::NotMerged => SuccessfulStageAction::MergeBlocked,
            MergeVerification::Verified
            | MergeVerification::Unknown
            | MergeVerification::NotRequired => SuccessfulStageAction::FinishPipeline,
        },
        PipelineStage::Done => SuccessfulStageAction::Noop,
        _ => match crate::orchestrator::next_pipeline_stage(stage, skipped_stages) {
            Some(next_stage) => {
                let skipped_logs = skipped_stage_logs(stage, &next_stage, skipped_stages);
                if gate_enabled {
                    SuccessfulStageAction::AwaitingApproval {
                        next_stage,
                        skipped_logs,
                    }
                } else {
                    SuccessfulStageAction::Advance {
                        next_stage,
                        skipped_logs,
                    }
                }
            }
            None => SuccessfulStageAction::Noop,
        },
    }
}

pub(crate) async fn prepare_and_register_stage_run(
    app: &AppHandle,
    state: &SharedState,
    config: &RunConfig,
    spec: StageLaunchSpec,
    emit_extra: Map<String, Value>,
) -> AgentProcessRequest {
    let PreparedStageRun { run, request } = prepare_stage_run(config, spec);
    runtime::register_preparing_run(app, state, run, emit_extra).await;
    request
}

pub(crate) fn spawn_next_stage(app: AppHandle, state: SharedState, spec: StageLaunchSpec) {
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

        if !wait_for_stage_slot(&state, &spec.stage, spec.issue_number).await {
            return;
        }

        if should_skip_next_stage_launch(&spec).await {
            return;
        }

        let config = {
            let s = state.lock().await;
            s.config.clone()
        };

        let request = prepare_and_register_stage_run(&app, &state, &config, spec, Map::new()).await;
        run_agent_process(app, state, request).await;
    });
}

/// Spawn the Review stage: registers a Running run, posts `@codex review`,
/// and records the request timestamp + HEAD SHA. The orchestrator's poll loop
/// then watches for an approving Codex comment and advances to Merge.
pub(crate) fn spawn_review_stage(app: AppHandle, state: SharedState, spec: StageLaunchSpec) {
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

        if !wait_for_stage_slot(&state, &spec.stage, spec.issue_number).await {
            return;
        }

        if should_skip_next_stage_launch(&spec).await {
            return;
        }

        start_review_stage(&app, &state, spec).await;
    });
}

/// Synchronous variant: prepares the Review run, marks it Running, posts the
/// `@codex review` comment, and persists `last_review_request_at` / `last_pushed_sha`.
pub(crate) async fn start_review_stage(
    app: &AppHandle,
    state: &SharedState,
    spec: StageLaunchSpec,
) -> Option<String> {
    let run_id = register_review_run(app, state, &spec).await;

    super::runtime::transition_run(
        app,
        state,
        &run_id,
        super::runtime::StatusTransition {
            status: AgentStatus::Running,
            stage_label: PipelineStage::Review.to_string(),
            error: None,
            finished: false,
            log_message: Some("[review] Pinging @codex review on the PR".to_string()),
            pending_next_stage: super::runtime::PendingNextStageUpdate::Keep,
            emit_extra: Map::new(),
            persist_meta: true,
        },
    )
    .await;

    let (pr_number, head_sha) = match resolve_pr_for_review(state, &run_id, &spec).await {
        Some(pr) => (pr.number, pr.head_ref_oid),
        None => return Some(run_id),
    };

    let request_timestamp = Utc::now().to_rfc3339();

    if let Err(error) = crate::github::post_codex_review(&spec.repo, pr_number).await {
        let log_message = format!("[review] Failed to post @codex review: {}", error);
        super::runtime::append_run_log(state, &run_id, log_message.clone(), true, true).await;
        let mut emit_extra = Map::new();
        emit_extra.insert("error".to_string(), json!(error.clone()));
        super::runtime::transition_run(
            app,
            state,
            &run_id,
            super::runtime::StatusTransition {
                status: AgentStatus::Failed,
                stage_label: PipelineStage::Review.to_string(),
                error: Some(error),
                finished: true,
                log_message: None,
                pending_next_stage: super::runtime::PendingNextStageUpdate::Keep,
                emit_extra,
                persist_meta: true,
            },
        )
        .await;
        return Some(run_id);
    }

    let request_ts_for_run = request_timestamp.clone();
    let sha_for_run = head_sha.clone();
    super::runtime::mutate_run(state, &run_id, true, move |run| {
        run.last_review_request_at = Some(request_ts_for_run);
        run.last_pushed_sha = sha_for_run;
    })
    .await;

    let posted_log = format!(
        "[review] Posted @codex review (PR #{}{}). Polling for approval.",
        pr_number,
        head_sha
            .as_ref()
            .map(|sha| format!(", PR HEAD {}", sha))
            .unwrap_or_default(),
    );
    super::runtime::append_run_log(state, &run_id, posted_log, true, true).await;

    Some(run_id)
}

async fn register_review_run(
    app: &AppHandle,
    state: &SharedState,
    spec: &StageLaunchSpec,
) -> String {
    let config = {
        let s = state.lock().await;
        s.config.clone()
    };
    let run_id = Uuid::new_v4().to_string();
    let run = AgentRun {
        id: run_id.clone(),
        repo: spec.repo.clone(),
        issue_number: spec.issue_number,
        issue_title: spec.issue_title.clone(),
        status: AgentStatus::Preparing,
        stage: spec.stage.clone(),
        started_at: Utc::now().to_rfc3339(),
        finished_at: None,
        logs: Vec::new(),
        workspace_path: spec.workspace_path.to_string_lossy().to_string(),
        error: None,
        attempt: spec.attempt,
        max_retries: spec.max_retries,
        lines_added: 0,
        lines_removed: 0,
        files_modified_list: Vec::new(),
        report: None,
        command_display: Some("gh pr comment <PR> --body \"@codex review\"".to_string()),
        agent_type: config.agent_type.clone(),
        last_log_line: None,
        log_count: 0,
        activity: Some("Awaiting Codex".to_string()),
        input_tokens: 0,
        output_tokens: 0,
        cost_usd: 0.0,
        last_log_timestamp: None,
        issue_labels: spec.issue_labels.clone(),
        skipped_stages: Vec::new(),
        stage_context: spec.previous_context.clone(),
        pending_next_stage: None,
        last_pushed_sha: None,
        last_review_request_at: None,
        review_iteration: 0,
        last_ci_failure_sha: None,
        ci_status_fetch_failure_count: 0,
        last_trigger_signature: None,
        last_trigger_summary: None,
    };
    super::runtime::register_preparing_run(app, state, run, Map::new()).await;
    run_id
}

async fn resolve_pr_for_review(
    state: &SharedState,
    run_id: &str,
    spec: &StageLaunchSpec,
) -> Option<crate::github::PullRequestFullState> {
    match crate::github::pr_full_state(&spec.repo, spec.issue_number).await {
        Ok(Some(pr)) => Some(pr),
        Ok(None) => {
            let error = format!(
                "No PR found for issue #{} — cannot start Review stage.",
                spec.issue_number
            );
            fail_review_run(state, run_id, error).await;
            None
        }
        Err(error) => {
            fail_review_run(state, run_id, error).await;
            None
        }
    }
}

/// Per-kind payload for a fix-run spawn. `Feedback` carries the actionable
/// signals coming back from the PR — verbatim Codex comments and/or failing
/// CI checks. `Rebase` carries the conflict context the agent needs to know
/// how to rebase.
///
/// Both `feedback` and `ci_failures` on the `Feedback` variant are optional
/// independently — the orchestrator's review poll reaches the spawn path
/// when at least one is non-empty. The prompt builder omits the section it
/// has nothing to render.
#[derive(Debug, Clone)]
pub(crate) enum FixRunPayload {
    Feedback {
        feedback: String,
        ci_failures: Vec<super::prompt::CiFailureContext>,
    },
    Rebase {
        base_branch: String,
        conflicting_files: Vec<String>,
    },
}

impl FixRunPayload {
    pub(crate) fn kind(&self) -> FixRunKind {
        match self {
            FixRunPayload::Feedback { .. } => FixRunKind::Feedback,
            FixRunPayload::Rebase { .. } => FixRunKind::Rebase,
        }
    }
}

/// Snapshot of the Review run + its open PR captured under-lock for the
/// fix-run dispatcher. Carries everything the spawn function needs without
/// re-acquiring the state lock or re-querying GitHub.
///
/// At least one of `feedback` (Codex review feedback) and `ci_failures` (failed
/// CI checks) must be non-empty — the orchestrator's review-poll loop only
/// reaches the spawn path when there's something to fix. Both can coexist:
/// e.g. Codex left feedback AND a workflow failed on the same SHA.
#[derive(Debug, Clone)]
pub(crate) struct FixRunSnapshot {
    pub run_id: String,
    pub repo: String,
    pub issue_number: u64,
    pub issue_title: String,
    pub issue_labels: Vec<String>,
    pub workspace_path: PathBuf,
    pub previous_context: Option<StageContext>,
    pub pr_number: u64,
    pub branch_name: String,
    pub payload: FixRunPayload,
}

/// Spawn a Review-stage fix-run agent. The fix-run reuses the polling Review
/// run's id (so its logs accrue on the same record) and runs the agent in the
/// existing worktree. On success the fix-run completion path inside
/// `run_agent_process` re-posts `@codex review` and refreshes
/// `last_pushed_sha` / `last_review_request_at`.
///
/// `review_iteration` must already have been incremented by the caller —
/// `orchestrator::review` does that under-lock during its pre-spawn guards.
pub(crate) fn spawn_fix_run(app: AppHandle, state: SharedState, snapshot: FixRunSnapshot) {
    tokio::spawn(async move {
        let config = {
            let s = state.lock().await;
            s.config.clone()
        };

        let kind = snapshot.payload.kind();
        let (prompt, activity) = match &snapshot.payload {
            FixRunPayload::Feedback {
                feedback,
                ci_failures,
            } => (
                super::prompt::build_fix_run_prompt(
                    snapshot.issue_number,
                    &snapshot.repo,
                    &snapshot.issue_title,
                    snapshot.pr_number,
                    &snapshot.branch_name,
                    feedback,
                    ci_failures,
                ),
                // Activity label distinguishes "Codex told me to" from "CI is
                // red" so the kanban view shows operators what the agent is
                // actually responding to.
                match (!feedback.trim().is_empty(), !ci_failures.is_empty()) {
                    (true, true) => "Applying Codex feedback + CI fixes",
                    (false, true) => "Fixing CI failures",
                    _ => "Applying Codex feedback",
                },
            ),
            FixRunPayload::Rebase {
                base_branch,
                conflicting_files,
            } => (
                super::prompt::build_rebase_fix_run_prompt(
                    snapshot.issue_number,
                    &snapshot.repo,
                    &snapshot.issue_title,
                    snapshot.pr_number,
                    &snapshot.branch_name,
                    base_branch,
                    conflicting_files,
                ),
                "Rebasing onto base branch",
            ),
        };
        let (command, args) = build_command_args(&config, &prompt);
        let command_display = format_command_display(&command, &args);

        // Reflect the fix-run command + activity on the Review run so the UI
        // shows what's happening while the subprocess executes. The activity
        // label distinguishes "Codex told me to" from "CI is red" so the kanban
        // view shows operators what the agent is actually responding to.
        let new_command_display = command_display.clone();
        let activity_label = activity.to_string();
        let _ = super::runtime::mutate_run(&state, &snapshot.run_id, true, move |run| {
            run.command_display = Some(new_command_display);
            run.activity = Some(activity_label);
        })
        .await;

        let spec = StageLaunchSpec {
            repo: snapshot.repo.clone(),
            issue_number: snapshot.issue_number,
            issue_title: snapshot.issue_title.clone(),
            issue_body: String::new(),
            stage: PipelineStage::Review,
            issue_labels: snapshot.issue_labels.clone(),
            workspace_path: snapshot.workspace_path.clone(),
            attempt: 1,
            max_retries: 0,
            previous_error: String::new(),
            previous_context: snapshot.previous_context.clone(),
            is_fix_run: true,
            fix_run_kind: Some(kind),
        };

        let request = AgentProcessRequest {
            run_id: snapshot.run_id.clone(),
            command,
            args,
            spec,
        };

        run_agent_process(app, state, request).await;
    });
}

/// Called after a fix-run subprocess exits successfully. Re-posts
/// `@codex review` on the PR, refreshes `last_pushed_sha` from the new HEAD,
/// updates `last_review_request_at`, and transitions the Review run back to
/// Running so the orchestrator's poll loop resumes watching for approval.
///
/// Returns Ok on success or an error string on failure (caller marks the run
/// Failed).
pub(crate) async fn finalize_fix_run_success(
    app: &AppHandle,
    state: &SharedState,
    run_id: &str,
    repo: &str,
    issue_number: u64,
) -> Result<(), String> {
    let pr = match crate::github::pr_full_state(repo, issue_number).await {
        Ok(Some(pr)) => pr,
        Ok(None) => {
            return Err(format!(
                "fix-run finished but no open PR found for issue #{}",
                issue_number
            ));
        }
        Err(error) => return Err(format!("fix-run finished but PR lookup failed: {}", error)),
    };

    crate::github::post_codex_review(repo, pr.number)
        .await
        .map_err(|error| format!("fix-run finished but @codex review re-post failed: {}", error))?;

    let request_timestamp = Utc::now().to_rfc3339();
    let new_sha = pr.head_ref_oid.clone();
    let request_ts_for_run = request_timestamp.clone();
    let _ = super::runtime::mutate_run(state, run_id, true, move |run| {
        run.last_pushed_sha = new_sha;
        run.last_review_request_at = Some(request_ts_for_run);
        run.activity = Some("Awaiting Codex".to_string());
        run.command_display = Some(
            "gh pr comment <PR> --body \"@codex review\"".to_string(),
        );
    })
    .await;

    let posted_log = format!(
        "[review] Fix-run pushed; re-posted @codex review (PR #{}{}). Polling for approval.",
        pr.number,
        pr.head_ref_oid
            .as_ref()
            .map(|sha| format!(", PR HEAD {}", sha))
            .unwrap_or_default(),
    );

    super::runtime::transition_run(
        app,
        state,
        run_id,
        super::runtime::StatusTransition {
            status: AgentStatus::Running,
            stage_label: PipelineStage::Review.to_string(),
            error: None,
            finished: false,
            log_message: Some(posted_log),
            pending_next_stage: super::runtime::PendingNextStageUpdate::Keep,
            emit_extra: Map::new(),
            persist_meta: true,
        },
    )
    .await;

    Ok(())
}

/// Classification of the worktree state after a rebase fix-run subprocess
/// exits. Used to decide whether the rebase actually completed cleanly or
/// the agent gave up partway through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RebaseOutcome {
    /// `git status --porcelain` shows no unmerged paths and no rebase is in
    /// progress. Safe to treat as a successful rebase.
    Clean,
    /// `git status --porcelain` reports lines starting with conflict markers
    /// (`UU`, `AA`, `DD`, `AU`, `UA`, `DU`, `UD`). The rebase did not finish.
    UnmergedPaths { paths: Vec<String> },
    /// A rebase is still in progress (`.git/rebase-apply/` or
    /// `.git/rebase-merge/` directory exists), or `git status` reports a
    /// rebase-state header. The agent left the worktree mid-rebase.
    RebaseInProgress,
    /// `git status` itself failed — couldn't classify. Treated like a failed
    /// rebase by the caller (escape to AwaitingApproval) so we don't push a
    /// half-resolved branch by accident.
    StatusFailed { error: String },
}

impl RebaseOutcome {
    pub(crate) fn is_clean(&self) -> bool {
        matches!(self, RebaseOutcome::Clean)
    }

    pub(crate) fn describe(&self) -> String {
        match self {
            RebaseOutcome::Clean => "worktree clean, rebase complete".to_string(),
            RebaseOutcome::UnmergedPaths { paths } => format!(
                "unmerged paths remain after rebase: {}",
                paths.join(", "),
            ),
            RebaseOutcome::RebaseInProgress => {
                "rebase still in progress in worktree (.git/rebase-* present)".to_string()
            }
            RebaseOutcome::StatusFailed { error } => {
                format!("could not run git status to verify rebase: {}", error)
            }
        }
    }
}

/// Inspect the worktree post-rebase. Returns `Clean` only when there are no
/// unmerged paths AND no in-progress rebase state. The orchestrator uses this
/// to detect rebase failures the agent itself didn't surface (e.g. exited zero
/// after `git rebase --abort` without saying so).
///
/// The rebase-state directory check goes through `git rev-parse --git-dir`
/// rather than hardcoding `<workspace>/.git`. SymphonyMac runs the agent in a
/// *linked git worktree* whenever `local_repos[repo]` is configured (see
/// `workspace::ensure_workspace_worktree`), and in that mode the worktree's
/// `.git` is a *file* pointing at the real gitdir — `<workspace>/.git/rebase-merge/`
/// would never exist even with a rebase actively paused. Without `git rev-parse`
/// we'd also miss the case where the agent ran `git add` on the conflict
/// resolutions but never `git rebase --continue`: status would show only `M/A/D`
/// codes (which `classify_porcelain_status` treats as clean), so the rebase-dir
/// existence is the only remaining signal.
pub(crate) async fn verify_rebase_outcome(workspace: &std::path::Path) -> RebaseOutcome {
    match resolve_git_dir(workspace).await {
        Ok(git_dir) => {
            if git_dir.join("rebase-apply").exists() || git_dir.join("rebase-merge").exists() {
                return RebaseOutcome::RebaseInProgress;
            }
        }
        Err(error) => {
            // We could not resolve the gitdir at all — treat the same as a
            // failed `git status`. The caller (`finalize_rebase_fix_run_failure`
            // path) escapes to AwaitingApproval, which is the conservative
            // choice rather than racing into `finalize_fix_run_success` and
            // re-requesting review on a worktree we don't understand.
            return RebaseOutcome::StatusFailed {
                error: format!("git rev-parse --git-dir failed: {}", error),
            };
        }
    }

    let output = tokio::process::Command::new("git")
        .args(["status", "--porcelain=v1"])
        .current_dir(workspace)
        .env("PATH", crate::paths::build_path_env())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .output()
        .await;

    let output = match output {
        Ok(output) => output,
        Err(error) => {
            return RebaseOutcome::StatusFailed {
                error: error.to_string(),
            };
        }
    };

    if !output.status.success() {
        return RebaseOutcome::StatusFailed {
            error: format!(
                "git status exited with code {}: {}",
                output.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&output.stderr).trim(),
            ),
        };
    }

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    classify_porcelain_status(&stdout)
}

/// Run `git rev-parse --git-dir` in `workspace` and return the resolved path
/// to the gitdir (absolute when git returns one, otherwise joined under the
/// workspace). For linked worktrees this resolves to
/// `<main-repo>/.git/worktrees/<name>`, which is where `rebase-apply` /
/// `rebase-merge` actually live during a paused rebase.
async fn resolve_git_dir(
    workspace: &std::path::Path,
) -> Result<std::path::PathBuf, String> {
    let output = tokio::process::Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .current_dir(workspace)
        .env("PATH", crate::paths::build_path_env())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .output()
        .await
        .map_err(|error| error.to_string())?;

    if !output.status.success() {
        return Err(format!(
            "git rev-parse exited with code {}: {}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }

    let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if raw.is_empty() {
        return Err("git rev-parse --git-dir returned empty output".to_string());
    }

    let path = std::path::PathBuf::from(&raw);
    Ok(if path.is_absolute() {
        path
    } else {
        workspace.join(path)
    })
}

/// Pure parser over `git status --porcelain=v1` output. Lines beginning with
/// any unmerged code (`UU`, `AA`, `DD`, `AU`, `UA`, `DU`, `UD`) signal an
/// unfinished merge/rebase. Extracted so the classification is unit-testable
/// without a real git workspace.
pub(crate) fn classify_porcelain_status(stdout: &str) -> RebaseOutcome {
    const UNMERGED: &[&str] = &["UU", "AA", "DD", "AU", "UA", "DU", "UD"];

    let mut unmerged_paths = Vec::new();
    for line in stdout.lines() {
        if line.len() < 3 {
            continue;
        }
        let code = &line[..2];
        if UNMERGED.contains(&code) {
            let path = line[3..].trim().to_string();
            if !path.is_empty() {
                unmerged_paths.push(path);
            }
        }
    }

    if unmerged_paths.is_empty() {
        RebaseOutcome::Clean
    } else {
        RebaseOutcome::UnmergedPaths {
            paths: unmerged_paths,
        }
    }
}

/// Called when a rebase fix-run cannot land — either the agent exited
/// non-zero, or it exited zero but the worktree still has unmerged paths /
/// an active rebase. Transitions the Review run to `AwaitingApproval` with
/// `pending_next_stage = "merge"` so the operator can advance manually past
/// the conflict (acceptance criterion: "Manual approve_pending_stage from UI
/// advances to Merge despite known conflicts").
pub(crate) async fn finalize_rebase_fix_run_failure(
    app: &AppHandle,
    state: &SharedState,
    run_id: &str,
    issue_number: u64,
    reason: String,
) {
    let mut emit_extra = Map::new();
    emit_extra.insert(
        "pending_next_stage".to_string(),
        json!(PipelineStage::Merge.to_string()),
    );
    emit_extra.insert("error".to_string(), json!(reason.clone()));

    let log_message = format!(
        "[review] Rebase fix-run could not resolve conflicts ({}) — pausing as AwaitingApproval so an operator can resolve manually before Merge.",
        reason,
    );

    let _ = super::runtime::transition_run(
        app,
        state,
        run_id,
        super::runtime::StatusTransition {
            status: AgentStatus::AwaitingApproval,
            stage_label: PipelineStage::Review.to_string(),
            error: Some(reason.clone()),
            finished: false,
            log_message: Some(log_message),
            pending_next_stage: super::runtime::PendingNextStageUpdate::Set(
                PipelineStage::Merge.to_string(),
            ),
            emit_extra,
            persist_meta: true,
        },
    )
    .await;

    let (notifications_enabled, notification_sound) = {
        let s = state.lock().await;
        (s.config.notifications_enabled, s.config.notification_sound)
    };
    if notifications_enabled {
        crate::notification::notify_awaiting_approval(
            app,
            issue_number,
            &PipelineStage::Review.to_string(),
            notification_sound,
        );
    }
    super::runtime::update_dock_badge(state).await;
}

async fn fail_review_run(state: &SharedState, run_id: &str, error: String) {
    super::runtime::append_run_log(
        state,
        run_id,
        format!("[review] {}", error),
        true,
        true,
    )
    .await;
    let _ = super::runtime::mutate_run(state, run_id, true, move |run| {
        run.status = AgentStatus::Failed;
        run.error = Some(error.clone());
        run.finished_at = Some(Utc::now().to_rfc3339());
    })
    .await;
}

pub(crate) fn spawn_retry(
    app: AppHandle,
    state: SharedState,
    spec: StageLaunchSpec,
    backoff_secs: u64,
) {
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(backoff_secs)).await;

        if !wait_for_stage_slot(&state, &spec.stage, spec.issue_number).await {
            return;
        }

        let config = {
            let s = state.lock().await;
            s.config.clone()
        };

        let mut emit_extra = Map::new();
        emit_extra.insert("attempt".to_string(), json!(spec.attempt));
        emit_extra.insert("max_retries".to_string(), json!(spec.max_retries + 1));

        let request = prepare_and_register_stage_run(&app, &state, &config, spec, emit_extra).await;
        run_agent_process(app, state, request).await;
    });
}

pub(crate) async fn finish_pipeline(
    app: &AppHandle,
    state: &SharedState,
    completion: PipelineCompletionSpec,
) {
    let stage_runs =
        collect_latest_stage_runs(state, &completion.repo, completion.issue_number).await;
    let usage_totals = collect_usage_totals(state, &completion.repo, completion.issue_number).await;
    let (done_run, pipeline_report) = build_done_run(&completion, stage_runs, usage_totals);
    let done_run_id = done_run.id.clone();

    {
        let mut s = state.lock().await;
        s.runs.insert(done_run_id.clone(), done_run);
        s.persist();
    }

    runtime::emit_status(app, &done_run_id, "completed", "done", Map::new());
    let _ = app.emit("pipeline-report", &pipeline_report);

    {
        let s = state.lock().await;
        if s.config.notifications_enabled {
            crate::notification::notify_pipeline_done(
                app,
                completion.issue_number,
                &completion.issue_title,
                s.config.notification_sound,
            );
        }
    }

    runtime::update_dock_badge(state).await;

    let cleanup_hooks = {
        let s = state.lock().await;
        s.config.hooks.clone()
    };
    let _ = crate::workspace::cleanup_workspace(
        &completion.repo,
        completion.issue_number,
        &cleanup_hooks,
    );
}

pub(crate) fn extract_stage_context(run: &AgentRun, repo: &str) -> StageContext {
    let from_stage = run.stage.to_string();
    let files_changed = run.files_modified_list.clone();
    let lines_added = run.lines_added;
    let lines_removed = run.lines_removed;
    let pr_number = detect_pr_number_from_logs(&run.logs, repo);
    let branch_name = detect_branch_from_logs(&run.logs);
    let summary = build_stage_summary(&run.stage, &run.logs);

    StageContext {
        from_stage,
        files_changed,
        lines_added,
        lines_removed,
        pr_number,
        branch_name,
        summary,
    }
}

fn prepare_stage_run(config: &RunConfig, spec: StageLaunchSpec) -> PreparedStageRun {
    let run_id = Uuid::new_v4().to_string();
    let prompt = build_prompt(
        &spec.stage,
        spec.issue_number,
        &spec.repo,
        &spec.issue_title,
        &spec.issue_body,
        &config.stage_prompts,
        spec.attempt,
        &spec.previous_error,
        spec.previous_context.as_ref(),
    );
    let (command, args) = build_command_args(config, &prompt);
    let command_display = format_command_display(&command, &args);
    let skipped_stage_names: Vec<String> = Vec::new();

    let mut logs = Vec::new();
    if spec.attempt > 1 {
        logs.push(format!(
            "Retry attempt {}/{} (previous error: {})",
            spec.attempt,
            spec.max_retries + 1,
            spec.previous_error
        ));
    }

    let run = AgentRun {
        id: run_id.clone(),
        repo: spec.repo.clone(),
        issue_number: spec.issue_number,
        issue_title: spec.issue_title.clone(),
        status: AgentStatus::Preparing,
        stage: spec.stage.clone(),
        started_at: Utc::now().to_rfc3339(),
        finished_at: None,
        logs,
        workspace_path: spec.workspace_path.to_string_lossy().to_string(),
        error: None,
        attempt: spec.attempt,
        max_retries: spec.max_retries,
        lines_added: 0,
        lines_removed: 0,
        files_modified_list: Vec::new(),
        report: None,
        command_display: Some(command_display),
        agent_type: config.agent_type.clone(),
        last_log_line: None,
        log_count: 0,
        activity: None,
        input_tokens: 0,
        output_tokens: 0,
        cost_usd: 0.0,
        last_log_timestamp: None,
        issue_labels: spec.issue_labels.clone(),
        skipped_stages: skipped_stage_names,
        stage_context: None,
        pending_next_stage: None,
        last_pushed_sha: None,
        last_review_request_at: None,
        review_iteration: 0,
        last_ci_failure_sha: None,
        ci_status_fetch_failure_count: 0,
        last_trigger_signature: None,
        last_trigger_summary: None,
    };

    let request = AgentProcessRequest {
        run_id,
        command,
        args,
        spec,
    };

    PreparedStageRun { run, request }
}

fn skipped_stage_logs(
    _current_stage: &PipelineStage,
    _next_stage: &PipelineStage,
    _skipped_stages: &[PipelineStage],
) -> Vec<String> {
    Vec::new()
}

async fn wait_for_stage_slot(
    state: &SharedState,
    stage: &PipelineStage,
    issue_number: u64,
) -> bool {
    let stage_label = stage.to_string();

    loop {
        let (can_launch, stopped) = {
            let s = state.lock().await;
            (
                crate::orchestrator::can_launch_stage(&s, stage),
                s.stop_flag,
            )
        };

        if stopped {
            eprintln!(
                "Orchestrator stopped; aborting queued stage {stage_label} for issue #{issue_number}"
            );
            return false;
        }

        if can_launch {
            return true;
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
}

async fn should_skip_next_stage_launch(spec: &StageLaunchSpec) -> bool {
    let repo = &spec.repo;
    let issue_number = spec.issue_number;
    let stage = &spec.stage;
    let stage_label = stage.to_string();

    match crate::github::get_issue_state(repo, issue_number).await {
        Ok(ref issue_state) if issue_state != "OPEN" => {
            eprintln!(
                "Issue #{issue_number} is {issue_state}; skipping stage {stage_label}"
            );
            return true;
        }
        Err(ref error) => {
            eprintln!(
                "Warning: could not re-check issue #{issue_number} state: {error}; proceeding anyway"
            );
        }
        _ => {}
    }

    if matches!(stage, PipelineStage::Merge) {
        match crate::github::is_pr_merged_for_issue(repo, issue_number).await {
            Ok(true) => {
                eprintln!(
                    "PR for issue #{issue_number} is already merged; skipping Merge stage"
                );
                return true;
            }
            Err(ref error) => {
                eprintln!(
                    "Warning: could not check PR merge status for #{issue_number}: {error}; proceeding anyway"
                );
            }
            _ => {}
        }
    }

    false
}

async fn collect_latest_stage_runs(
    state: &SharedState,
    repo: &str,
    issue_number: u64,
) -> Vec<AgentRun> {
    let s = state.lock().await;
    let stage_order = ["implement", "review", "merge"];

    stage_order
        .iter()
        .filter_map(|stage_name| {
            s.runs
                .values()
                .filter(|run| {
                    run.repo == repo
                        && run.issue_number == issue_number
                        && run.stage.to_string() == *stage_name
                })
                .max_by_key(|run| run.started_at.clone())
                .cloned()
        })
        .collect()
}

async fn collect_usage_totals(state: &SharedState, repo: &str, issue_number: u64) -> UsageTotals {
    let s = state.lock().await;

    aggregate_usage_totals(s.runs.values().filter(|run| {
        run.repo == repo && run.issue_number == issue_number && run.stage != PipelineStage::Done
    }))
}

fn aggregate_usage_totals<'a>(runs: impl IntoIterator<Item = &'a AgentRun>) -> UsageTotals {
    runs.into_iter().fold(
        UsageTotals {
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: 0.0,
        },
        |totals, run| UsageTotals {
            input_tokens: totals.input_tokens + run.input_tokens,
            output_tokens: totals.output_tokens + run.output_tokens,
            cost_usd: totals.cost_usd + run.cost_usd,
        },
    )
}

fn build_done_run(
    completion: &PipelineCompletionSpec,
    stage_runs: Vec<AgentRun>,
    usage_totals: UsageTotals,
) -> (AgentRun, crate::report::PipelineReport) {
    let done_id = Uuid::new_v4().to_string();
    let mut aggregated_logs = Vec::new();
    let stage_order = ["implement", "review", "merge"];

    for stage_name in &stage_order {
        if let Some(run) = stage_runs
            .iter()
            .find(|run| run.stage.to_string() == *stage_name)
        {
            aggregated_logs.push(format!(
                "═══ {} ═══",
                stage_name.to_uppercase().replace('_', " ")
            ));
            aggregated_logs.extend(run.logs.iter().cloned());
            aggregated_logs.push(String::new());
        }
    }
    aggregated_logs.push("═══ PIPELINE COMPLETED ═══".to_string());

    let stage_refs: Vec<&AgentRun> = stage_runs.iter().collect();
    let pipeline_report = report::generate_report(
        completion.issue_number,
        &completion.issue_title,
        &completion.repo,
        stage_refs,
    );

    let done_run = AgentRun {
        id: done_id,
        repo: completion.repo.clone(),
        issue_number: completion.issue_number,
        issue_title: completion.issue_title.clone(),
        status: AgentStatus::Completed,
        stage: PipelineStage::Done,
        started_at: Utc::now().to_rfc3339(),
        finished_at: Some(Utc::now().to_rfc3339()),
        logs: aggregated_logs,
        workspace_path: completion.workspace_path.to_string_lossy().to_string(),
        error: None,
        attempt: 1,
        max_retries: 0,
        lines_added: 0,
        lines_removed: 0,
        files_modified_list: Vec::new(),
        report: Some(pipeline_report.clone()),
        command_display: None,
        agent_type: String::new(),
        last_log_line: None,
        log_count: 0,
        activity: Some("Completed".to_string()),
        input_tokens: usage_totals.input_tokens,
        output_tokens: usage_totals.output_tokens,
        cost_usd: usage_totals.cost_usd,
        last_log_timestamp: None,
        issue_labels: completion.issue_labels.clone(),
        skipped_stages: completion
            .skipped_stages
            .iter()
            .map(ToString::to_string)
            .collect(),
        stage_context: None,
        pending_next_stage: None,
        last_pushed_sha: None,
        last_review_request_at: None,
        review_iteration: 0,
        last_ci_failure_sha: None,
        ci_status_fetch_failure_count: 0,
        last_trigger_signature: None,
        last_trigger_summary: None,
    };

    (done_run, pipeline_report)
}

fn detect_pr_number_from_logs(logs: &[String], repo: &str) -> Option<u64> {
    let pr_url_suffix = format!("{}/pull/", repo);
    for line in logs.iter().rev() {
        if let Some(position) = line.find(&pr_url_suffix) {
            let after = &line[position + pr_url_suffix.len()..];
            let number: String = after
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .collect();
            if let Ok(pr_number) = number.parse::<u64>() {
                return Some(pr_number);
            }
        }

        let line_lower = line.to_lowercase();
        for prefix in ["pull request #", "pr #"] {
            if let Some(position) = line_lower.find(prefix) {
                let after = &line[position + prefix.len()..];
                let number: String = after
                    .chars()
                    .take_while(|character| character.is_ascii_digit())
                    .collect();
                if let Ok(pr_number) = number.parse::<u64>() {
                    return Some(pr_number);
                }
            }
        }
    }
    None
}

fn detect_branch_from_logs(logs: &[String]) -> Option<String> {
    for line in logs.iter().rev() {
        if line.contains("headRefName") {
            if let Some(position) = line.find("headRefName") {
                let after = &line[position..];
                if let Some(colon_position) = after.find(':') {
                    let value_part = after[colon_position + 1..]
                        .trim()
                        .trim_matches(|character| {
                            character == '"' || character == ',' || character == ' '
                        });
                    let branch: String = value_part
                        .chars()
                        .take_while(|character| {
                            !character.is_whitespace() && *character != '"' && *character != ','
                        })
                        .collect();
                    if !branch.is_empty() {
                        return Some(branch);
                    }
                }
            }
        }

        if let Some(position) = line.find("checkout -b ") {
            let after = &line[position + "checkout -b ".len()..];
            let branch: String = after
                .chars()
                .take_while(|character| !character.is_whitespace())
                .collect();
            if !branch.is_empty() {
                return Some(branch);
            }
        }
    }
    None
}

fn build_stage_summary(stage: &PipelineStage, logs: &[String]) -> String {
    match stage {
        PipelineStage::Implement => {
            let mut commits = Vec::new();
            for line in logs {
                if line.contains("git commit") || line.contains("Commit") {
                    commits.push(line.chars().take(100).collect::<String>());
                }
            }
            if commits.is_empty() {
                "Implementation completed.".to_string()
            } else {
                commits.into_iter().take(3).collect::<Vec<_>>().join("; ")
            }
        }
        PipelineStage::Review => {
            let mut findings = Vec::new();
            for line in logs {
                let lower = line.to_lowercase();
                if lower.contains("approved")
                    || lower.contains("review completed")
                    || lower.contains("codex")
                    || lower.contains("@codex review")
                {
                    findings.push(line.chars().take(100).collect::<String>());
                }
            }
            if findings.is_empty() {
                "Review completed.".to_string()
            } else {
                findings.into_iter().take(3).collect::<Vec<_>>().join("; ")
            }
        }
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        aggregate_usage_totals, build_done_run, classify_porcelain_status, decide_failure_action,
        decide_successful_stage_action, resolve_git_dir, verify_rebase_outcome, FailureAction,
        MergeVerification, PipelineCompletionSpec, RebaseOutcome, SuccessfulStageAction,
        UsageTotals,
    };
    use crate::orchestrator::{AgentRun, AgentStatus, PipelineStage, RunConfig};
    use std::path::{Path, PathBuf};

    fn sample_run(
        id: &str,
        stage: PipelineStage,
        started_at: &str,
        input_tokens: u64,
        output_tokens: u64,
        cost_usd: f64,
    ) -> AgentRun {
        AgentRun {
            id: id.to_string(),
            repo: "pedrocid/SymphonyMac".to_string(),
            issue_number: 57,
            issue_title: "Split agent.rs".to_string(),
            status: AgentStatus::Completed,
            stage,
            started_at: started_at.to_string(),
            finished_at: Some(started_at.to_string()),
            logs: vec![format!("log {}", id)],
            workspace_path: "/tmp/workspace".to_string(),
            error: None,
            attempt: 1,
            max_retries: 1,
            lines_added: 0,
            lines_removed: 0,
            files_modified_list: Vec::new(),
            report: None,
            command_display: None,
            agent_type: "claude".to_string(),
            last_log_line: None,
            log_count: 0,
            activity: None,
            input_tokens,
            output_tokens,
            cost_usd,
            last_log_timestamp: None,
            issue_labels: Vec::new(),
            skipped_stages: Vec::new(),
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
    fn implement_advances_to_review_in_three_stage_pipeline() {
        let action = decide_successful_stage_action(
            &PipelineStage::Implement,
            &[],
            false,
            MergeVerification::NotRequired,
        );

        assert_eq!(
            action,
            SuccessfulStageAction::Advance {
                next_stage: PipelineStage::Review,
                skipped_logs: Vec::new(),
            }
        );
    }

    #[test]
    fn review_advances_to_merge_in_three_stage_pipeline() {
        let action = decide_successful_stage_action(
            &PipelineStage::Review,
            &[],
            false,
            MergeVerification::NotRequired,
        );

        assert_eq!(
            action,
            SuccessfulStageAction::Advance {
                next_stage: PipelineStage::Merge,
                skipped_logs: Vec::new(),
            }
        );
    }

    #[test]
    fn approval_gate_pauses_after_successful_stage() {
        let action = decide_successful_stage_action(
            &PipelineStage::Review,
            &[],
            true,
            MergeVerification::NotRequired,
        );

        assert_eq!(
            action,
            SuccessfulStageAction::AwaitingApproval {
                next_stage: PipelineStage::Merge,
                skipped_logs: Vec::new(),
            }
        );
    }

    #[test]
    fn merge_requires_verified_pr() {
        let action = decide_successful_stage_action(
            &PipelineStage::Merge,
            &[],
            false,
            MergeVerification::NotMerged,
        );

        assert_eq!(action, SuccessfulStageAction::MergeBlocked);
    }

    #[test]
    fn verified_merge_finishes_pipeline() {
        let action = decide_successful_stage_action(
            &PipelineStage::Merge,
            &[],
            true,
            MergeVerification::Verified,
        );

        assert_eq!(action, SuccessfulStageAction::FinishPipeline);
    }

    #[test]
    fn done_stage_does_not_advance() {
        let action = decide_successful_stage_action(
            &PipelineStage::Done,
            &[],
            false,
            MergeVerification::NotRequired,
        );

        assert_eq!(action, SuccessfulStageAction::Noop);
    }

    #[test]
    fn retry_action_uses_exponential_backoff() {
        let config = RunConfig {
            retry_base_delay_secs: 5,
            retry_max_backoff_secs: 30,
            ..RunConfig::default()
        };

        assert_eq!(
            decide_failure_action(2, 3, &config),
            FailureAction::Retry {
                next_attempt: 3,
                backoff_secs: 10,
            }
        );
    }

    #[test]
    fn retry_action_falls_back_to_legacy_retry_backoff_setting() {
        let config = RunConfig {
            retry_base_delay_secs: 0,
            retry_backoff_secs: 7,
            retry_max_backoff_secs: 30,
            ..RunConfig::default()
        };

        assert_eq!(
            decide_failure_action(1, 3, &config),
            FailureAction::Retry {
                next_attempt: 2,
                backoff_secs: 7,
            }
        );
    }

    #[test]
    fn retry_action_stops_after_max_attempts() {
        let config = RunConfig::default();
        assert_eq!(
            decide_failure_action(3, 2, &config),
            FailureAction::Exhausted
        );
    }

    #[test]
    fn aggregate_usage_totals_counts_all_attempts_and_ignores_done_stage() {
        let runs = vec![
            sample_run(
                "attempt-1",
                PipelineStage::Implement,
                "2026-03-08T10:00:00Z",
                100,
                40,
                1.5,
            ),
            sample_run(
                "attempt-2",
                PipelineStage::Implement,
                "2026-03-08T10:05:00Z",
                250,
                90,
                2.25,
            ),
            sample_run(
                "review",
                PipelineStage::Review,
                "2026-03-08T10:10:00Z",
                30,
                15,
                0.5,
            ),
            sample_run(
                "done",
                PipelineStage::Done,
                "2026-03-08T10:15:00Z",
                999,
                999,
                9.9,
            ),
        ];

        let totals =
            aggregate_usage_totals(runs.iter().filter(|run| run.stage != PipelineStage::Done));

        assert_eq!(
            totals,
            UsageTotals {
                input_tokens: 380,
                output_tokens: 145,
                cost_usd: 4.25,
            }
        );
    }

    #[test]
    fn build_done_run_uses_aggregate_usage_totals() {
        let completion = PipelineCompletionSpec {
            repo: "pedrocid/SymphonyMac".to_string(),
            issue_number: 57,
            issue_title: "Split agent.rs".to_string(),
            workspace_path: PathBuf::from("/tmp/workspace"),
            issue_labels: vec!["refactor".to_string()],
            skipped_stages: Vec::new(),
        };
        let latest_stage_runs = vec![
            sample_run(
                "implement-latest",
                PipelineStage::Implement,
                "2026-03-08T10:05:00Z",
                250,
                90,
                2.25,
            ),
            sample_run(
                "review",
                PipelineStage::Review,
                "2026-03-08T10:10:00Z",
                30,
                15,
                0.5,
            ),
        ];

        let (done_run, _) = build_done_run(
            &completion,
            latest_stage_runs,
            UsageTotals {
                input_tokens: 380,
                output_tokens: 145,
                cost_usd: 4.25,
            },
        );

        assert_eq!(done_run.input_tokens, 380);
        assert_eq!(done_run.output_tokens, 145);
        assert_eq!(done_run.cost_usd, 4.25);
        assert!(done_run.skipped_stages.is_empty());
    }

    #[test]
    fn test_extract_stage_context_detects_pr_branch_and_summary() {
        use super::extract_stage_context;

        let run = AgentRun {
            id: "run-1".to_string(),
            repo: "pedrocid/SymphonyMac".to_string(),
            issue_number: 62,
            issue_title: "Add automated coverage".to_string(),
            status: AgentStatus::Completed,
            stage: PipelineStage::Implement,
            started_at: "2026-03-08T10:00:00Z".to_string(),
            finished_at: Some("2026-03-08T10:30:00Z".to_string()),
            logs: vec![
                "$ git checkout -b symphony/issue-62".to_string(),
                "Created pull request https://github.com/pedrocid/SymphonyMac/pull/91".to_string(),
                "git commit -m \"Add automated test coverage\"".to_string(),
            ],
            workspace_path: "/tmp/symphony".to_string(),
            error: None,
            attempt: 1,
            max_retries: 2,
            lines_added: 24,
            lines_removed: 6,
            files_modified_list: vec![
                "src/App.tsx".to_string(),
                "src-tauri/src/agent.rs".to_string(),
            ],
            report: None,
            command_display: Some("claude --print".to_string()),
            agent_type: "claude".to_string(),
            last_log_line: None,
            log_count: 0,
            activity: None,
            last_log_timestamp: None,
            input_tokens: 100,
            output_tokens: 150,
            cost_usd: 0.025,
            issue_labels: vec!["feature".to_string()],
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
        };

        let context = extract_stage_context(&run, "pedrocid/SymphonyMac");

        assert_eq!(context.from_stage, "implement");
        assert_eq!(context.pr_number, Some(91));
        assert_eq!(context.branch_name.as_deref(), Some("symphony/issue-62"));
        assert!(context.summary.contains("git commit"));
    }

    #[test]
    fn classify_porcelain_status_treats_empty_output_as_clean() {
        // Clean rebase: nothing to commit, no unmerged paths.
        assert_eq!(classify_porcelain_status(""), RebaseOutcome::Clean);
    }

    #[test]
    fn classify_porcelain_status_ignores_normal_change_codes() {
        // Modified/added/deleted entries are not unmerged. After a successful
        // rebase the worktree may have these (e.g. a file moved across stages),
        // but they don't represent conflict markers.
        let stdout = "\
 M src/lib.rs\n\
A  src/new.rs\n\
?? untracked.txt\n\
 D removed.rs\n\
";
        assert_eq!(classify_porcelain_status(stdout), RebaseOutcome::Clean);
    }

    #[test]
    fn classify_porcelain_status_flags_every_unmerged_code() {
        // Every git unmerged-state code must trigger the failure path. Missing
        // any of these would let a half-resolved rebase look "clean" and we'd
        // force-push conflict markers to the PR branch.
        for code in ["UU", "AA", "DD", "AU", "UA", "DU", "UD"] {
            let stdout = format!("{} src/conflict.rs\n", code);
            match classify_porcelain_status(&stdout) {
                RebaseOutcome::UnmergedPaths { paths } => {
                    assert_eq!(paths, vec!["src/conflict.rs".to_string()], "code {}", code);
                }
                other => panic!("code {} should flag unmerged, got {:?}", code, other),
            }
        }
    }

    #[test]
    fn classify_porcelain_status_collects_all_unmerged_paths() {
        // Multi-file conflicts: all unmerged paths appear in the failure
        // describe() output so the operator sees the full conflict set.
        let stdout = "\
UU src/a.rs\n\
 M src/clean.rs\n\
AA src/b.rs\n\
UD src/c.rs\n\
";
        match classify_porcelain_status(stdout) {
            RebaseOutcome::UnmergedPaths { paths } => {
                assert_eq!(
                    paths,
                    vec![
                        "src/a.rs".to_string(),
                        "src/b.rs".to_string(),
                        "src/c.rs".to_string(),
                    ]
                );
            }
            other => panic!("expected UnmergedPaths, got {:?}", other),
        }
    }

    #[test]
    fn fix_run_payload_kind_round_trips_to_correct_variant() {
        // The dispatch in `spawn_fix_run` and the rebase-escape branches in
        // `process.rs` both rely on `FixRunPayload::kind()` correctly mapping
        // each payload to its FixRunKind. Locking this mapping in a test
        // prevents accidental regressions where a new payload variant would
        // silently fall through to the feedback path.
        use super::{FixRunKind, FixRunPayload};
        assert_eq!(
            FixRunPayload::Feedback {
                feedback: "x".into(),
                ci_failures: Vec::new(),
            }
            .kind(),
            FixRunKind::Feedback,
        );
        assert_eq!(
            FixRunPayload::Rebase {
                base_branch: "main".into(),
                conflicting_files: vec![],
            }
            .kind(),
            FixRunKind::Rebase,
        );
    }

    /// Run `git` with the given args inside `cwd`. Panics on non-zero exit.
    /// Helper for the rebase-state integration tests below.
    fn run_git(cwd: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(cwd)
            .env("GIT_AUTHOR_NAME", "Symphony Test")
            .env("GIT_AUTHOR_EMAIL", "symphony@example.com")
            .env("GIT_COMMITTER_NAME", "Symphony Test")
            .env("GIT_COMMITTER_EMAIL", "symphony@example.com")
            .output()
            .expect("git available on PATH for tests");
        assert!(
            output.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr),
        );
    }

    #[tokio::test]
    async fn resolve_git_dir_resolves_linked_worktree_gitdir_not_workspace_dotgit() {
        // Regression test for the P1 Codex flagged on PR #10: SymphonyMac runs
        // the agent in a *linked* git worktree when `local_repos[repo]` is set,
        // and in that mode `<workspace>/.git` is a file pointing at the real
        // gitdir under `<main-repo>/.git/worktrees/<name>/`. If
        // `verify_rebase_outcome` checks `<workspace>/.git/rebase-*` directly
        // (the bug), it will *never* fire on a paused rebase in a linked
        // worktree, and we'd silently force-push a half-rebased branch.

        let tmp = tempfile::tempdir().expect("tempdir");
        let main_repo = tmp.path().join("main");
        let linked = tmp.path().join("linked");
        std::fs::create_dir(&main_repo).unwrap();

        run_git(&main_repo, &["init", "--initial-branch=main"]);
        // Need at least one commit before `git worktree add` will accept the
        // current branch as a starting point.
        std::fs::write(main_repo.join("seed.txt"), "seed\n").unwrap();
        run_git(&main_repo, &["add", "seed.txt"]);
        run_git(&main_repo, &["commit", "-m", "seed"]);
        run_git(
            &main_repo,
            &[
                "worktree",
                "add",
                "-b",
                "feature/issue-5",
                linked.to_str().unwrap(),
            ],
        );

        // Sanity: in a linked worktree, `.git` is a file (not a directory).
        // If this changes the test premise is wrong.
        let dot_git = linked.join(".git");
        assert!(dot_git.exists());
        assert!(
            dot_git.is_file(),
            ".git in a linked worktree should be a file pointing at the gitdir",
        );
        assert!(
            !linked.join(".git").join("rebase-merge").exists(),
            "the buggy path must not resolve to anything",
        );

        let git_dir = resolve_git_dir(&linked).await.expect("resolve_git_dir");
        // The resolved gitdir for a linked worktree lives under the main
        // repo's `.git/worktrees/<name>/`. This is where rebase-merge /
        // rebase-apply actually appear during a paused rebase, and the
        // verify_rebase_outcome check must look here.
        let expected_suffix = std::path::Path::new(".git").join("worktrees").join("linked");
        assert!(
            git_dir.ends_with(&expected_suffix),
            "git_dir was {:?}, expected to end with {:?}",
            git_dir,
            expected_suffix,
        );
    }

    #[tokio::test]
    async fn verify_rebase_outcome_detects_paused_rebase_in_linked_worktree() {
        // End-to-end: in a linked worktree, a `rebase-merge/` directory under
        // the resolved gitdir (NOT under <workspace>/.git) must classify as
        // RebaseInProgress. This is the case Codex flagged: the agent might
        // exit zero with a clean `git status` while a rebase is still paused.
        let tmp = tempfile::tempdir().expect("tempdir");
        let main_repo = tmp.path().join("main");
        let linked = tmp.path().join("linked");
        std::fs::create_dir(&main_repo).unwrap();

        run_git(&main_repo, &["init", "--initial-branch=main"]);
        std::fs::write(main_repo.join("seed.txt"), "seed\n").unwrap();
        run_git(&main_repo, &["add", "seed.txt"]);
        run_git(&main_repo, &["commit", "-m", "seed"]);
        run_git(
            &main_repo,
            &[
                "worktree",
                "add",
                "-b",
                "feature/issue-5",
                linked.to_str().unwrap(),
            ],
        );

        // Plant a rebase-merge marker under the linked worktree's resolved
        // gitdir. We don't want to actually trigger a rebase conflict (that's
        // brittle to git versions) — the directory's existence is the signal
        // we read. Git itself uses this same convention to detect a paused
        // rebase.
        let resolved = resolve_git_dir(&linked).await.expect("gitdir");
        std::fs::create_dir_all(resolved.join("rebase-merge")).unwrap();

        let outcome = verify_rebase_outcome(&linked).await;
        assert_eq!(outcome, RebaseOutcome::RebaseInProgress);
    }

    #[tokio::test]
    async fn verify_rebase_outcome_returns_clean_for_quiescent_linked_worktree() {
        // Counterpart to the previous test: when no rebase markers exist and
        // status is clean, we should return Clean (so `finalize_fix_run_success`
        // proceeds to re-request review). Asserts the gitdir-resolution path
        // doesn't accidentally treat *every* linked worktree as in-progress.
        let tmp = tempfile::tempdir().expect("tempdir");
        let main_repo = tmp.path().join("main");
        let linked = tmp.path().join("linked");
        std::fs::create_dir(&main_repo).unwrap();

        run_git(&main_repo, &["init", "--initial-branch=main"]);
        std::fs::write(main_repo.join("seed.txt"), "seed\n").unwrap();
        run_git(&main_repo, &["add", "seed.txt"]);
        run_git(&main_repo, &["commit", "-m", "seed"]);
        run_git(
            &main_repo,
            &[
                "worktree",
                "add",
                "-b",
                "feature/issue-5",
                linked.to_str().unwrap(),
            ],
        );

        let outcome = verify_rebase_outcome(&linked).await;
        assert_eq!(outcome, RebaseOutcome::Clean);
    }

    #[test]
    fn rebase_outcome_describe_includes_paths_for_operator() {
        // The describe() string is what `finalize_rebase_fix_run_failure`
        // surfaces as the run's error message and AwaitingApproval log line.
        // It must contain the conflicting file list — that's what the
        // operator needs to act.
        let outcome = RebaseOutcome::UnmergedPaths {
            paths: vec!["src/a.rs".into(), "src/b.rs".into()],
        };
        let described = outcome.describe();
        assert!(described.contains("src/a.rs"));
        assert!(described.contains("src/b.rs"));
        assert!(!outcome.is_clean());

        let in_progress = RebaseOutcome::RebaseInProgress;
        assert!(in_progress.describe().contains("rebase"));
        assert!(!in_progress.is_clean());

        let status_failed = RebaseOutcome::StatusFailed {
            error: "git: command not found".into(),
        };
        assert!(status_failed.describe().contains("git: command not found"));
        assert!(!status_failed.is_clean());

        assert!(RebaseOutcome::Clean.is_clean());
    }
}
