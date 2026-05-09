use crate::github;
use crate::SharedState;
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
            })
            .collect();
        (
            snapshots,
            s.config.codex_approve_patterns.clone(),
            s.config.codex_feedback_marker.clone(),
        )
    };

    for snapshot in snapshots {
        check_codex_approval(app, state, snapshot, &approve_patterns, &feedback_marker).await;
    }
}

async fn check_codex_approval(
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

    let approved_comment = pr_state.comments.iter().find(|comment| {
        is_codex_author(&comment.author)
            && newer_than(&comment.created_at, snapshot.last_review_request_at.as_deref())
            && github::parse_codex_approval(&comment.body, approve_patterns, feedback_marker)
    });

    let Some(comment) = approved_comment else {
        return;
    };

    // Reject approvals that target an older commit than the one we asked Codex to review.
    // This prevents merging unreviewed changes pushed after the `@codex review` request.
    if !head_sha_matches(snapshot.last_pushed_sha.as_deref(), pr_state.head_ref_oid.as_deref()) {
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
        // exact official login
        assert!(is_codex_author("chatgpt-codex-connector[bot]"));
        // bare bot login (some GitHub API responses)
        assert!(is_codex_author("chatgpt-codex-connector"));

        // substrings or look-alikes must NOT pass — that was the spoofing risk.
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
        // Legacy runs without `last_pushed_sha`, or PR API responses missing
        // `head_ref_oid`, fall back to timestamp gating only.
        assert!(head_sha_matches(None, Some("abc123")));
        assert!(head_sha_matches(Some("abc123"), None));
        assert!(head_sha_matches(None, None));
    }
}
