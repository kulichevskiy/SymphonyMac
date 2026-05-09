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
}

pub async fn poll_review_runs(app: &AppHandle, state: &SharedState) {
    let (snapshots, approve_patterns) = {
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
            })
            .collect();
        (snapshots, s.config.codex_approve_patterns.clone())
    };

    for snapshot in snapshots {
        check_codex_approval(app, state, snapshot, &approve_patterns).await;
    }
}

async fn check_codex_approval(
    app: &AppHandle,
    state: &SharedState,
    snapshot: ReviewRunSnapshot,
    approve_patterns: &[String],
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
            && github::parse_codex_approval(&comment.body, approve_patterns)
    });

    let Some(comment) = approved_comment else {
        return;
    };

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

fn is_codex_author(author: &str) -> bool {
    let lower = author.to_lowercase();
    lower.contains("codex") || lower.contains("chatgpt-codex-connector")
}

fn newer_than(comment_at: &str, baseline: Option<&str>) -> bool {
    let Some(baseline) = baseline else {
        return true;
    };
    comment_at > baseline
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
    fn is_codex_author_matches_known_logins() {
        assert!(is_codex_author("chatgpt-codex-connector[bot]"));
        assert!(is_codex_author("Codex"));
        assert!(!is_codex_author("kulichevskiy"));
    }
}
