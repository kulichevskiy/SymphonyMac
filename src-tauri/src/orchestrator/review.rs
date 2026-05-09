use crate::github::{self, PrComment};
use crate::SharedState;
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
    has_active_subprocess: bool,
}

pub async fn poll_review_runs(app: &AppHandle, state: &SharedState) {
    let (snapshots, approve_patterns, feedback_marker) = {
        let s = state.lock().await;
        let snapshots: Vec<ReviewRunSnapshot> = s
            .runs
            .values()
            .filter(|run| {
                run.stage == PipelineStage::Review && run.status == AgentStatus::Running
            })
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
                has_active_subprocess: s.agent_pids.contains_key(&run.id),
            })
            .collect();
        (
            snapshots,
            s.config.codex_approve_patterns.clone(),
            s.config.codex_feedback_marker.clone(),
        )
    };

    for snapshot in snapshots {
        if snapshot.has_active_subprocess {
            // A fix-run subprocess is currently rebasing, fixing, or pushing for
            // this Review run. Don't poll for approval — we'd race the fix-run
            // and could either advance to Merge prematurely or spawn a second
            // fix-run on top of the first.
            continue;
        }
        check_codex_activity(app, state, snapshot, &approve_patterns, &feedback_marker).await;
    }
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
    let feedback_comments = collect_codex_feedback(
        &pr_state.comments,
        snapshot.last_review_request_at.as_deref(),
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

    // Increment review_iteration BEFORE spawning so the UI immediately reflects
    // the new iteration count and a re-entrant poll wouldn't double-increment.
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
/// orchestrator state) represents an *active* Review run for `target_repo` /
/// `target_issue` that isn't the polling sentinel itself (`target_run_id`).
///
/// "Active" here means status Running or Preparing. We use this to refuse a
/// fix-run spawn when one is already in flight — preventing the case where two
/// agents would race to push to the same branch.
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
        && matches!(
            candidate_status,
            AgentStatus::Running | AgentStatus::Preparing
        )
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

    #[test]
    fn another_review_active_flags_sibling_review_run_for_same_issue() {
        // Same repo+issue, Review stage, Preparing — that's a sibling fix-run
        // about to start; we must not double-spawn.
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
        assert!(another_review_active(
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
    fn another_review_active_skips_terminal_statuses() {
        // Failed/Completed/Stopped Review runs aren't doing work — don't block.
        for status in [
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
