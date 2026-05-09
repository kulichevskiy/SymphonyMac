use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use tokio::process::Command;
use ts_rs::TS;

const ISSUE_STATE_BATCH_SIZE: usize = 25;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "contracts.ts")]
pub struct Repo {
    pub full_name: String,
    pub name: String,
    pub owner: String,
    pub description: Option<String>,
    pub url: String,
    pub default_branch: String,
    pub is_private: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "contracts.ts")]
pub struct Issue {
    pub number: u64,
    pub title: String,
    pub body: Option<String>,
    pub state: String,
    pub labels: Vec<String>,
    pub assignee: Option<String>,
    pub url: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    pub number: u64,
    pub title: String,
    pub body: Option<String>,
    pub state: String,
    pub head_branch: String,
    pub url: String,
    pub created_at: String,
    pub updated_at: String,
    pub author: Option<String>,
    /// The issue number this PR closes, extracted from body "Closes #N"
    pub closes_issue: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrComment {
    pub author: String,
    pub body: String,
    pub created_at: String,
}

/// Full state of a PR, used by the Review-stage poll loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequestFullState {
    pub number: u64,
    pub state: String,
    pub head_ref_name: String,
    pub head_ref_oid: Option<String>,
    pub base_ref_name: Option<String>,
    /// GraphQL `mergeStateStatus`. Common values include `CLEAN`, `DIRTY`,
    /// `BLOCKED`, `BEHIND`, `UNKNOWN`, `UNSTABLE`, `HAS_HOOKS`. We only act on
    /// `DIRTY` (PR has conflicts with base) — everything else is a soft signal
    /// the Review loop ignores.
    pub merge_state_status: Option<String>,
    /// PR file paths (from `gh pr list --json files`). Used to seed the
    /// rebase fix-run prompt with a "likely conflicting files" hint.
    pub files: Vec<String>,
    pub comments: Vec<PrComment>,
}

/// Returns true when GitHub reports `mergeStateStatus == DIRTY` for this PR —
/// the only state that indicates a hard merge conflict the rebase fix-run can
/// act on. Other unmergeable states (`BLOCKED`, `BEHIND`, `UNKNOWN`, etc.) are
/// not conflicts and should not trigger a rebase agent.
pub fn pr_is_dirty(merge_state: Option<&str>) -> bool {
    matches!(merge_state, Some(state) if state.eq_ignore_ascii_case("DIRTY"))
}

#[async_trait]
pub trait GitHubGateway: Send + Sync {
    async fn list_repos(&self, filter: Option<String>) -> Result<Vec<Repo>, String>;
    async fn list_issues(
        &self,
        repo: &str,
        state: Option<&str>,
        label: Option<&str>,
    ) -> Result<Vec<Issue>, String>;
    async fn list_open_prs(&self, repo: &str) -> Result<Vec<PullRequest>, String>;
    async fn get_issue_states(
        &self,
        repo: &str,
        issue_numbers: &[u64],
    ) -> Result<HashMap<u64, String>, String>;
    async fn get_issue_state(&self, repo: &str, issue_number: u64) -> Result<String, String>;
    async fn get_issue_detail(&self, repo: &str, number: u64) -> Result<Issue, String>;
    async fn is_pr_merged_for_issue(&self, repo: &str, issue_number: u64) -> Result<bool, String>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GhCliGateway;

#[async_trait]
impl GitHubGateway for GhCliGateway {
    async fn list_repos(&self, filter: Option<String>) -> Result<Vec<Repo>, String> {
        let json_fields = "nameWithOwner,name,owner,description,url,defaultBranchRef,isPrivate";
        let output = run_gh(&["repo", "list", "--limit", "100", "--json", json_fields]).await?;
        let raw: Vec<serde_json::Value> = serde_json::from_str(&output)
            .map_err(|e| format!("Failed to parse repos JSON: {}", e))?;

        let mut repos: Vec<Repo> = raw.iter().map(parse_repo).collect();
        if let Some(filter) = filter {
            let filter = filter.to_lowercase();
            repos.retain(|repo| repo.full_name.to_lowercase().contains(&filter));
        }

        Ok(repos)
    }

    async fn list_issues(
        &self,
        repo: &str,
        state: Option<&str>,
        label: Option<&str>,
    ) -> Result<Vec<Issue>, String> {
        let json_fields = "number,title,body,state,labels,assignees,url,createdAt,updatedAt";
        let state_filter = state.unwrap_or("open");
        let mut args = vec![
            "issue",
            "list",
            "-R",
            repo,
            "--state",
            state_filter,
            "--limit",
            "100",
            "--json",
            json_fields,
        ];

        if let Some(label) = label {
            args.push("--label");
            args.push(label);
        }

        let output = run_gh(&args).await?;
        let raw: Vec<serde_json::Value> = serde_json::from_str(&output)
            .map_err(|e| format!("Failed to parse issues JSON: {}", e))?;

        Ok(raw.iter().map(parse_issue).collect())
    }

    async fn list_open_prs(&self, repo: &str) -> Result<Vec<PullRequest>, String> {
        let json_fields = "number,title,body,state,headRefName,url,createdAt,updatedAt,author";
        let output = run_gh(&[
            "pr",
            "list",
            "-R",
            repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            json_fields,
        ])
        .await?;

        let raw: Vec<serde_json::Value> = serde_json::from_str(&output)
            .map_err(|e| format!("Failed to parse PRs JSON: {}", e))?;

        Ok(raw.iter().map(parse_pull_request).collect())
    }

    async fn get_issue_states(
        &self,
        repo: &str,
        issue_numbers: &[u64],
    ) -> Result<HashMap<u64, String>, String> {
        let issue_numbers = unique_issue_numbers(issue_numbers);
        if issue_numbers.is_empty() {
            return Ok(HashMap::new());
        }

        let (owner, name) = split_repo_full_name(repo)?;
        let mut states = HashMap::new();

        for chunk in issue_numbers.chunks(ISSUE_STATE_BATCH_SIZE) {
            let query = build_issue_states_query(owner, name, chunk)?;
            let query_arg = format!("query={}", query);
            let output = run_gh(&["api", "graphql", "-f", &query_arg]).await?;
            let response: serde_json::Value = serde_json::from_str(&output)
                .map_err(|e| format!("Failed to parse issue states JSON: {}", e))?;

            if let Some(errors) = response["errors"].as_array() {
                if let Some(message) = errors
                    .iter()
                    .filter_map(|error| error["message"].as_str())
                    .next()
                {
                    return Err(format!("GitHub GraphQL failed: {}", message));
                }
            }

            let repository = response["data"]["repository"]
                .as_object()
                .ok_or_else(|| "GitHub GraphQL response missing repository data".to_string())?;

            for issue_number in chunk {
                let field_name = format!("issue_{}", issue_number);
                if let Some(state) = repository
                    .get(&field_name)
                    .and_then(|issue| issue["state"].as_str())
                {
                    states.insert(*issue_number, state.to_string());
                }
            }
        }

        Ok(states)
    }

    async fn get_issue_state(&self, repo: &str, issue_number: u64) -> Result<String, String> {
        let issue_number = issue_number.to_string();
        let output = run_gh(&[
            "issue",
            "view",
            &issue_number,
            "-R",
            repo,
            "--json",
            "state",
        ])
        .await?;
        let value: serde_json::Value = serde_json::from_str(&output)
            .map_err(|e| format!("Failed to parse issue JSON: {}", e))?;
        Ok(value["state"].as_str().unwrap_or("OPEN").to_string())
    }

    async fn get_issue_detail(&self, repo: &str, number: u64) -> Result<Issue, String> {
        let issue_number = number.to_string();
        let json_fields = "number,title,body,state,labels,assignees,url,createdAt,updatedAt";
        let output = run_gh(&[
            "issue",
            "view",
            &issue_number,
            "-R",
            repo,
            "--json",
            json_fields,
        ])
        .await?;

        let value: serde_json::Value = serde_json::from_str(&output)
            .map_err(|e| format!("Failed to parse issue JSON: {}", e))?;
        Ok(parse_issue(&value))
    }

    async fn is_pr_merged_for_issue(&self, repo: &str, issue_number: u64) -> Result<bool, String> {
        let output = run_gh(&[
            "pr",
            "list",
            "-R",
            repo,
            "--state",
            "all",
            "--limit",
            "50",
            "--json",
            "number,title,body,state",
        ])
        .await?;

        let prs: Vec<serde_json::Value> =
            serde_json::from_str(&output).map_err(|e| format!("Failed to parse PRs: {}", e))?;

        for pr in &prs {
            let body = pr["body"].as_str().unwrap_or("");
            let title = pr["title"].as_str().unwrap_or("");
            let state = pr["state"].as_str().unwrap_or("");

            let references_issue = parse_closes_issue(body) == Some(issue_number)
                || parse_issue_from_title(title) == Some(issue_number);

            if references_issue {
                return Ok(state == "MERGED");
            }
        }

        Err(format!("No PR found referencing issue #{}", issue_number))
    }
}

fn cli_gateway() -> GhCliGateway {
    GhCliGateway
}

async fn run_gh(args: &[&str]) -> Result<String, String> {
    let output = Command::new(crate::paths::resolve("gh"))
        .env("PATH", crate::paths::build_path_env())
        .args(args)
        .output()
        .await
        .map_err(|e| {
            format!(
                "Failed to run gh CLI: {}. Make sure gh is installed and authenticated.",
                e
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("gh command failed: {}", stderr.trim()));
    }

    String::from_utf8(output.stdout).map_err(|e| format!("Invalid UTF-8 output: {}", e))
}

fn parse_repo(value: &serde_json::Value) -> Repo {
    let owner = value["owner"]["login"].as_str().unwrap_or("").to_string();
    Repo {
        full_name: value["nameWithOwner"].as_str().unwrap_or("").to_string(),
        name: value["name"].as_str().unwrap_or("").to_string(),
        owner,
        description: value["description"].as_str().map(|s| s.to_string()),
        url: value["url"].as_str().unwrap_or("").to_string(),
        default_branch: value["defaultBranchRef"]
            .as_object()
            .and_then(|branch| branch["name"].as_str())
            .unwrap_or("main")
            .to_string(),
        is_private: value["isPrivate"].as_bool().unwrap_or(false),
    }
}

fn parse_issue(value: &serde_json::Value) -> Issue {
    let labels = value["labels"]
        .as_array()
        .map(|labels| {
            labels
                .iter()
                .filter_map(|label| label["name"].as_str().map(|name| name.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let assignee = value["assignees"]
        .as_array()
        .and_then(|assignees| assignees.first())
        .and_then(|assignee| assignee["login"].as_str())
        .map(|login| login.to_string());

    Issue {
        number: value["number"].as_u64().unwrap_or(0),
        title: value["title"].as_str().unwrap_or("").to_string(),
        body: value["body"].as_str().map(|s| s.to_string()),
        state: value["state"].as_str().unwrap_or("OPEN").to_string(),
        labels,
        assignee,
        url: value["url"].as_str().unwrap_or("").to_string(),
        created_at: value["createdAt"].as_str().unwrap_or("").to_string(),
        updated_at: value["updatedAt"].as_str().unwrap_or("").to_string(),
    }
}

fn parse_pull_request(value: &serde_json::Value) -> PullRequest {
    let body = value["body"].as_str().map(|s| s.to_string());
    let closes_issue = body.as_ref().and_then(|body| parse_closes_issue(body));

    PullRequest {
        number: value["number"].as_u64().unwrap_or(0),
        title: value["title"].as_str().unwrap_or("").to_string(),
        body,
        state: value["state"].as_str().unwrap_or("OPEN").to_string(),
        head_branch: value["headRefName"].as_str().unwrap_or("").to_string(),
        url: value["url"].as_str().unwrap_or("").to_string(),
        created_at: value["createdAt"].as_str().unwrap_or("").to_string(),
        updated_at: value["updatedAt"].as_str().unwrap_or("").to_string(),
        author: value["author"]
            .as_object()
            .and_then(|author| author["login"].as_str())
            .map(|login| login.to_string()),
        closes_issue,
    }
}

fn split_repo_full_name(repo: &str) -> Result<(&str, &str), String> {
    repo.split_once('/')
        .ok_or_else(|| format!("Invalid repo full name: {}", repo))
}

fn build_issue_states_query(
    owner: &str,
    name: &str,
    issue_numbers: &[u64],
) -> Result<String, String> {
    let owner = serde_json::to_string(owner)
        .map_err(|e| format!("Failed to encode GitHub owner for GraphQL: {}", e))?;
    let name = serde_json::to_string(name)
        .map_err(|e| format!("Failed to encode GitHub repo name for GraphQL: {}", e))?;
    let fields = issue_numbers
        .iter()
        .map(|issue_number| {
            format!(
                "issue_{0}: issue(number: {0}) {{ number state }}",
                issue_number
            )
        })
        .collect::<Vec<_>>()
        .join(" ");

    Ok(format!(
        "query {{ repository(owner: {owner}, name: {name}) {{ {fields} }} }}",
    ))
}

fn unique_issue_numbers(issue_numbers: &[u64]) -> Vec<u64> {
    let mut seen = BTreeSet::new();
    issue_numbers
        .iter()
        .copied()
        .filter(|issue_number| *issue_number > 0 && seen.insert(*issue_number))
        .collect()
}

#[tauri::command]
pub async fn list_repos(filter: Option<String>) -> Result<Vec<Repo>, String> {
    cli_gateway().list_repos(filter).await
}

#[tauri::command]
pub async fn list_issues(
    repo: String,
    state: Option<String>,
    label: Option<String>,
) -> Result<Vec<Issue>, String> {
    cli_gateway()
        .list_issues(&repo, state.as_deref(), label.as_deref())
        .await
}

pub async fn list_open_prs(repo: String) -> Result<Vec<PullRequest>, String> {
    cli_gateway().list_open_prs(&repo).await
}

pub async fn get_issue_states(
    repo: &str,
    issue_numbers: &[u64],
) -> Result<HashMap<u64, String>, String> {
    cli_gateway().get_issue_states(repo, issue_numbers).await
}

/// Parse blocker references from issue body text.
/// Looks for patterns like "blocked by #X", "depends on #X", "requires #X".
pub fn parse_blockers(text: &str) -> Vec<u64> {
    let text_lower = text.to_lowercase();
    let mut blockers = Vec::new();
    let patterns = [
        "blocked by #",
        "depends on #",
        "requires #",
        "waiting on #",
        "waiting for #",
        "after #",
    ];

    for pattern in &patterns {
        let mut search_from = 0;
        while let Some(pos) = text_lower[search_from..].find(pattern) {
            let abs_pos = search_from + pos + pattern.len();
            let after = &text_lower[abs_pos..];
            let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = num_str.parse::<u64>() {
                if n > 0 && !blockers.contains(&n) {
                    blockers.push(n);
                }
            }
            search_from = abs_pos;
        }
    }

    blockers
}

/// Parse "Closes #123" or "Fixes #123" from PR body
fn parse_closes_issue(body: &str) -> Option<u64> {
    let body_lower = body.to_lowercase();
    for keyword in &[
        "closes #",
        "fixes #",
        "resolves #",
        "close #",
        "fix #",
        "resolve #",
    ] {
        if let Some(pos) = body_lower.find(keyword) {
            let after = &body_lower[pos + keyword.len()..];
            let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = num_str.parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

/// Parse issue number from PR title like "Fix #14: ..."
pub fn parse_issue_from_title(title: &str) -> Option<u64> {
    let title_lower = title.to_lowercase();
    for keyword in &[
        "fix #",
        "fixes #",
        "closes #",
        "resolve #",
        "resolves #",
        "close #",
        "feat #",
        "issue #",
    ] {
        if let Some(pos) = title_lower.find(keyword) {
            let after = &title_lower[pos + keyword.len()..];
            let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = num_str.parse::<u64>() {
                return Some(n);
            }
        }
    }
    // Try pattern "#123" anywhere
    for (index, ch) in title.char_indices() {
        if ch == '#' {
            let after = &title[index + 1..];
            let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = num_str.parse::<u64>() {
                if n > 0 {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Fetch the current state of an issue (e.g. "OPEN", "CLOSED").
/// Returns the state string or an error if the check fails.
pub async fn get_issue_state(repo: &str, issue_number: u64) -> Result<String, String> {
    cli_gateway().get_issue_state(repo, issue_number).await
}

/// Check if a PR associated with a given issue number is actually merged.
/// Returns Ok(true) if merged, Ok(false) if still open/closed-not-merged, Err on failure.
pub async fn is_pr_merged_for_issue(repo: &str, issue_number: u64) -> Result<bool, String> {
    cli_gateway()
        .is_pr_merged_for_issue(repo, issue_number)
        .await
}

/// Find the open PR associated with `issue_number` in `repo`. Returns its full state
/// (number, state, branch, HEAD oid, and comments) — used by the Review stage poll loop.
///
/// Restricted to `--state open` on purpose: an issue may have historical merged or
/// closed PRs, but the Review stage targets the *active* PR. Picking up an old
/// merged PR here would let `@codex review` go to the wrong thread and accept stale
/// approvals.
pub async fn pr_full_state(
    repo: &str,
    issue_number: u64,
) -> Result<Option<PullRequestFullState>, String> {
    let json_fields = "number,title,body,state,headRefName,headRefOid,baseRefName,mergeStateStatus,files,comments";
    let output = run_gh(&[
        "pr",
        "list",
        "-R",
        repo,
        "--state",
        "open",
        "--limit",
        "50",
        "--json",
        json_fields,
    ])
    .await?;

    let prs: Vec<serde_json::Value> =
        serde_json::from_str(&output).map_err(|e| format!("Failed to parse PRs: {}", e))?;

    Ok(select_pr_full_state(&prs, issue_number))
}

fn select_pr_full_state(
    prs: &[serde_json::Value],
    issue_number: u64,
) -> Option<PullRequestFullState> {
    for pr in prs {
        let body = pr["body"].as_str().unwrap_or("");
        let title = pr["title"].as_str().unwrap_or("");
        let state = pr["state"].as_str().unwrap_or("");
        let references_issue = parse_closes_issue(body) == Some(issue_number)
            || parse_issue_from_title(title) == Some(issue_number);

        // Defense in depth against API responses that include non-open PRs:
        // even though we ask for `--state open`, ignore anything that came back
        // tagged otherwise so a stale merged PR can never win this lookup.
        if !references_issue || (!state.is_empty() && state != "OPEN") {
            continue;
        }

        let comments = pr["comments"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .map(|item| PrComment {
                        author: item["author"]["login"].as_str().unwrap_or("").to_string(),
                        body: item["body"].as_str().unwrap_or("").to_string(),
                        created_at: item["createdAt"].as_str().unwrap_or("").to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let files = pr["files"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item["path"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        return Some(PullRequestFullState {
            number: pr["number"].as_u64().unwrap_or(0),
            state: state.to_string(),
            head_ref_name: pr["headRefName"].as_str().unwrap_or("").to_string(),
            head_ref_oid: pr["headRefOid"].as_str().map(|s| s.to_string()),
            base_ref_name: pr["baseRefName"].as_str().map(|s| s.to_string()),
            merge_state_status: pr["mergeStateStatus"].as_str().map(|s| s.to_string()),
            files,
            comments,
        });
    }

    None
}

/// Normalized state of a single CI check on a PR.
///
/// `gh pr view --json statusCheckRollup` returns two distinct shapes — `CheckRun`
/// (Actions) carries `status` + `conclusion`, while `StatusContext` (external CI)
/// carries `state`. This enum collapses both into the four buckets the orchestrator
/// actually cares about, so callers don't have to re-do that normalization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CheckState {
    /// Completed successfully, or treated-as-success (skipped, neutral, stale).
    Success,
    /// Completed and failed (FAILURE / TIMED_OUT / CANCELLED / ACTION_REQUIRED / STARTUP_FAILURE).
    Failure,
    /// Errored (typically a `StatusContext` with state == ERROR).
    Error,
    /// Still running, queued, or otherwise not yet a terminal verdict.
    Pending,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrCheck {
    pub name: String,
    pub state: CheckState,
    pub is_required: bool,
    /// GitHub Actions workflow run id, if this check is a CheckRun whose
    /// detailsUrl looks like `.../actions/runs/<id>/...`. Used for
    /// `gh run view <id> --log-failed`.
    pub run_id: Option<String>,
    pub details_url: Option<String>,
}

/// Aggregated CI status for a PR: just a flat list of normalized checks.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct PrCiStatus {
    pub checks: Vec<PrCheck>,
}

impl PrCiStatus {
    /// "CI green" definition (per issue #4):
    /// - All required checks are SUCCESS.
    /// - No check (required or not) is FAILURE / ERROR.
    /// - Non-required PENDING is ignored.
    /// - Required PENDING blocks (we wait, not green yet).
    /// - Zero checks → green.
    pub fn is_green(&self) -> bool {
        if self.checks.iter().any(|check| {
            matches!(check.state, CheckState::Failure | CheckState::Error)
        }) {
            return false;
        }
        self.checks
            .iter()
            .filter(|check| check.is_required)
            .all(|check| check.state == CheckState::Success)
    }

    /// Checks currently in FAILURE or ERROR — the set we'd surface to a fix-run.
    pub fn failing_checks(&self) -> Vec<&PrCheck> {
        self.checks
            .iter()
            .filter(|check| matches!(check.state, CheckState::Failure | CheckState::Error))
            .collect()
    }

    /// True when at least one *required* check is still PENDING. Useful for
    /// distinguishing the "approved but CI not yet finished" case from the
    /// "approved and CI failed" case in callers' log lines and approval
    /// dispositions. Not used in `is_green()` itself — kept as a separate
    /// predicate so consumers can act on partial CI state.
    #[allow(dead_code)]
    pub fn has_pending_required(&self) -> bool {
        self.checks
            .iter()
            .any(|check| check.is_required && check.state == CheckState::Pending)
    }
}

/// Fetch the structured CI status for `pr_number` on `repo` via
/// `gh pr view --json statusCheckRollup`.
pub async fn pr_ci_status(repo: &str, pr_number: u64) -> Result<PrCiStatus, String> {
    let pr_number_str = pr_number.to_string();
    let output = run_gh(&[
        "pr",
        "view",
        &pr_number_str,
        "-R",
        repo,
        "--json",
        "statusCheckRollup",
    ])
    .await?;
    let value: serde_json::Value = serde_json::from_str(&output)
        .map_err(|e| format!("Failed to parse PR status JSON: {}", e))?;
    Ok(parse_status_check_rollup(&value["statusCheckRollup"]))
}

/// Best-effort fetch of the failure log for a GitHub Actions run, capped at
/// `max_chars` so we don't blow up the agent prompt. Returns `None` when the
/// `gh run view` command fails for any reason — we surface the check name and
/// state regardless, so a missing log shouldn't block the fix-run.
///
/// IMPORTANT: callers must treat the returned text as **untrusted user input**
/// (anyone who can edit a workflow can write arbitrary text into a failing
/// log, including prompt-injection text). This function strips ANSI escape
/// sequences and most control characters before truncation so the raw bytes
/// can't manipulate terminal display, but the *content* is still adversarial
/// — the fix-run prompt must fence it explicitly.
pub async fn run_failure_log_excerpt(
    repo: &str,
    run_id: &str,
    max_chars: usize,
) -> Option<String> {
    let output = Command::new(crate::paths::resolve("gh"))
        .env("PATH", crate::paths::build_path_env())
        .args(["run", "view", run_id, "-R", repo, "--log-failed"])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8(output.stdout).ok()?;
    let sanitized = sanitize_untrusted_log(&raw);
    let trimmed = sanitized.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(truncate_chars(trimmed, max_chars))
}

/// Strip control characters and ANSI escape sequences from a raw log blob so
/// it can be safely embedded as fenced text in a downstream agent prompt.
///
/// We keep `\n` and `\t` (legitimate in compiler output) but drop:
/// - ESC sequences (`\x1b[...m`, etc.) used to color/move terminal output
/// - Other C0 controls (`\x00..\x1F` minus `\n`/`\t`) and `\x7f`
///
/// Sanitization is defense-in-depth: even with the prompt's fence markers,
/// stripping ANSI keeps the fenced data textually clean and avoids confusing
/// the agent with terminal-escape garbage.
pub(crate) fn sanitize_untrusted_log(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' => {
                // Drop the entire escape sequence. The most common forms are
                // CSI (`ESC [ ... <final>`) and OSC (`ESC ] ... BEL/ST`); we
                // approximate by consuming up to the next ASCII letter or
                // bell, which covers both without a full ANSI parser.
                if chars.peek() == Some(&'[') {
                    chars.next();
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        if next.is_ascii_alphabetic() {
                            break;
                        }
                    }
                } else if chars.peek() == Some(&']') {
                    chars.next();
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        if next == '\x07' {
                            break;
                        }
                    }
                } else {
                    // Two-byte escape (e.g. ESC =) — skip the next char.
                    chars.next();
                }
            }
            '\n' | '\t' => out.push(c),
            c if (c as u32) < 0x20 || c == '\x7f' => {
                // Other control chars: drop silently rather than render.
            }
            _ => out.push(c),
        }
    }
    out
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars).collect();
    out.push_str("\n…[truncated]");
    out
}

fn parse_status_check_rollup(rollup: &serde_json::Value) -> PrCiStatus {
    let Some(items) = rollup.as_array() else {
        return PrCiStatus::default();
    };
    let checks = items.iter().filter_map(parse_pr_check).collect();
    PrCiStatus { checks }
}

fn parse_pr_check(item: &serde_json::Value) -> Option<PrCheck> {
    let typename = item["__typename"].as_str().unwrap_or("");
    let is_required = item["isRequired"].as_bool().unwrap_or(false);

    match typename {
        "CheckRun" => {
            let name = item["name"].as_str().unwrap_or("").to_string();
            let status = item["status"].as_str().unwrap_or("");
            let conclusion = item["conclusion"].as_str().unwrap_or("");
            let state = normalize_check_run_state(status, conclusion);
            let details_url = item["detailsUrl"].as_str().map(|s| s.to_string());
            let run_id = details_url.as_deref().and_then(parse_actions_run_id);
            Some(PrCheck {
                name,
                state,
                is_required,
                run_id,
                details_url,
            })
        }
        "StatusContext" => {
            let name = item["context"].as_str().unwrap_or("").to_string();
            let state_str = item["state"].as_str().unwrap_or("");
            let state = normalize_status_context_state(state_str);
            let details_url = item["targetUrl"].as_str().map(|s| s.to_string());
            Some(PrCheck {
                name,
                state,
                is_required,
                run_id: None,
                details_url,
            })
        }
        // Defensive: GitHub adds new __typename values periodically. Skip
        // unknown shapes rather than misclassify them.
        _ => None,
    }
}

/// Map a GitHub Actions `CheckRun` (status + conclusion) onto our 4-state enum.
///
/// Status drives the verdict only when the run hasn't completed; once it's
/// COMPLETED we read the conclusion. We treat skipped/neutral/stale as success
/// (they don't block — that matches GitHub's own "checks passed" UI).
fn normalize_check_run_state(status: &str, conclusion: &str) -> CheckState {
    let status_upper = status.to_ascii_uppercase();
    if status_upper != "COMPLETED" {
        return CheckState::Pending;
    }
    match conclusion.to_ascii_uppercase().as_str() {
        "SUCCESS" => CheckState::Success,
        "NEUTRAL" | "SKIPPED" | "STALE" => CheckState::Success,
        "FAILURE" | "TIMED_OUT" | "CANCELLED" | "ACTION_REQUIRED" | "STARTUP_FAILURE" => {
            CheckState::Failure
        }
        // Unknown / empty conclusion on a COMPLETED run — treat as pending so
        // we don't silently advance on a check we can't interpret.
        _ => CheckState::Pending,
    }
}

fn normalize_status_context_state(state: &str) -> CheckState {
    match state.to_ascii_uppercase().as_str() {
        "SUCCESS" => CheckState::Success,
        "FAILURE" => CheckState::Failure,
        "ERROR" => CheckState::Error,
        "PENDING" | "EXPECTED" => CheckState::Pending,
        // Unknown — bias toward Pending; we'd rather wait than misreport green.
        _ => CheckState::Pending,
    }
}

/// Pull the workflow-run id out of an Actions detailsUrl, e.g.
/// `https://github.com/owner/repo/actions/runs/123456789/job/987` → `123456789`.
/// Returns `None` for any URL that doesn't follow that pattern.
fn parse_actions_run_id(url: &str) -> Option<String> {
    let marker = "/actions/runs/";
    let position = url.find(marker)?;
    let after = &url[position + marker.len()..];
    let id: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    if id.is_empty() {
        None
    } else {
        Some(id)
    }
}

/// Post `@codex review` as an issue-level comment on the given PR.
pub async fn post_codex_review(repo: &str, pr_number: u64) -> Result<(), String> {
    let pr_number_str = pr_number.to_string();
    let _ = run_gh(&[
        "pr",
        "comment",
        &pr_number_str,
        "-R",
        repo,
        "--body",
        "@codex review",
    ])
    .await?;
    Ok(())
}

/// Returns true if `text` matches any of the configured Codex approval `patterns`
/// (case-insensitive substring match).
///
/// `feedback_marker` is the trailing footer Codex appends to its review comments
/// (e.g. "Useful? React with 👍 / 👎."). It is stripped from `text` before matching,
/// so emoji or phrases inside the footer cannot cause false-positive approvals.
pub fn parse_codex_approval(text: &str, patterns: &[String], feedback_marker: &str) -> bool {
    if text.is_empty() || patterns.is_empty() {
        return false;
    }
    let stripped = strip_feedback_marker(text, feedback_marker);
    let lower = stripped.to_lowercase();
    patterns
        .iter()
        .any(|pattern| !pattern.is_empty() && lower.contains(&pattern.to_lowercase()))
}

fn strip_feedback_marker<'a>(text: &'a str, feedback_marker: &str) -> std::borrow::Cow<'a, str> {
    if feedback_marker.is_empty() || !text.contains(feedback_marker) {
        return std::borrow::Cow::Borrowed(text);
    }
    std::borrow::Cow::Owned(text.replace(feedback_marker, ""))
}

#[tauri::command]
pub async fn get_issue_detail(repo: String, number: u64) -> Result<Issue, String> {
    cli_gateway().get_issue_detail(&repo, number).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_blockers_blocked_by() {
        let text = "This issue is blocked by #10 and blocked by #20";
        let blockers = parse_blockers(text);
        assert_eq!(blockers, vec![10, 20]);
    }

    #[test]
    fn test_parse_blockers_depends_on() {
        let text = "Depends on #5";
        let blockers = parse_blockers(text);
        assert_eq!(blockers, vec![5]);
    }

    #[test]
    fn test_parse_blockers_requires() {
        let text = "This requires #42 to be done first";
        let blockers = parse_blockers(text);
        assert_eq!(blockers, vec![42]);
    }

    #[test]
    fn test_parse_blockers_waiting_on() {
        let text = "Waiting on #3";
        let blockers = parse_blockers(text);
        assert_eq!(blockers, vec![3]);
    }

    #[test]
    fn test_parse_blockers_waiting_for() {
        let text = "Waiting for #7";
        let blockers = parse_blockers(text);
        assert_eq!(blockers, vec![7]);
    }

    #[test]
    fn test_parse_blockers_after() {
        let text = "Should be done after #15";
        let blockers = parse_blockers(text);
        assert_eq!(blockers, vec![15]);
    }

    #[test]
    fn test_parse_blockers_case_insensitive() {
        let text = "BLOCKED BY #99 and Depends On #88";
        let blockers = parse_blockers(text);
        assert_eq!(blockers, vec![99, 88]);
    }

    #[test]
    fn test_parse_blockers_no_duplicates() {
        let text = "Blocked by #10, also depends on #10";
        let blockers = parse_blockers(text);
        assert_eq!(blockers, vec![10]);
    }

    #[test]
    fn test_parse_blockers_empty_text() {
        let blockers = parse_blockers("");
        assert!(blockers.is_empty());
    }

    #[test]
    fn test_parse_blockers_no_matches() {
        let text = "This is a regular issue with no blockers";
        let blockers = parse_blockers(text);
        assert!(blockers.is_empty());
    }

    #[test]
    fn test_parse_blockers_multiple_patterns() {
        let text =
            "Blocked by #1, depends on #2, requires #3, waiting on #4, waiting for #5, after #6";
        let blockers = parse_blockers(text);
        assert_eq!(blockers, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_parse_blockers_unicode_text() {
        let text = "🚫 Blocked by #42 — needs résumé feature first";
        let blockers = parse_blockers(text);
        assert_eq!(blockers, vec![42]);
    }

    #[test]
    fn test_parse_blockers_invalid_number() {
        let text = "Blocked by #abc";
        let blockers = parse_blockers(text);
        assert!(blockers.is_empty());
    }

    #[test]
    fn test_parse_blockers_zero_ignored() {
        let text = "Blocked by #0";
        let blockers = parse_blockers(text);
        assert!(blockers.is_empty());
    }

    #[test]
    fn test_parse_closes_issue_unicode_prefix() {
        let body = "Résumé update complete. Closes #42";
        assert_eq!(parse_closes_issue(body), Some(42));
    }

    #[test]
    fn test_parse_issue_from_title_unicode_prefix() {
        let title = "🚀 Fix #91: Improve pipeline coverage";
        assert_eq!(parse_issue_from_title(title), Some(91));
    }

    #[test]
    fn test_parse_issue_from_title_unicode_before_hash() {
        let title = "Résumé polish before landing #108";
        assert_eq!(parse_issue_from_title(title), Some(108));
    }

    #[test]
    fn test_unique_issue_numbers_dedupes_and_filters_zero() {
        let issue_numbers = unique_issue_numbers(&[3, 0, 2, 3, 1, 2]);
        assert_eq!(issue_numbers, vec![3, 2, 1]);
    }

    #[test]
    fn test_build_issue_states_query_escapes_repo_names() {
        let query = build_issue_states_query("octo\"cat", "repo-name", &[12, 27]).unwrap();
        assert!(query.contains("issue_12: issue(number: 12)"));
        assert!(query.contains("issue_27: issue(number: 27)"));
        assert!(query.contains("owner: \"octo\\\"cat\""));
        assert!(query.contains("name: \"repo-name\""));
    }

    #[test]
    fn test_parse_codex_approval_matches_default_patterns_case_insensitively() {
        let patterns = vec![
            "Didn't find any major issues".to_string(),
            "did not find major issues".to_string(),
        ];
        let marker = "Useful? React with 👍 / 👎.";

        // exact match (case-insensitive)
        assert!(parse_codex_approval(
            "DIDN'T FIND ANY MAJOR ISSUES — looks good to me.",
            &patterns,
            marker,
        ));

        // alt phrasing
        assert!(parse_codex_approval(
            "After review, did NOT find major issues.",
            &patterns,
            marker,
        ));

        // no match
        assert!(!parse_codex_approval(
            "Found a couple of issues that need fixing.",
            &patterns,
            marker,
        ));

        // empty inputs
        assert!(!parse_codex_approval("", &patterns, marker));
        assert!(!parse_codex_approval("anything", &[], marker));
    }

    #[test]
    fn test_select_pr_full_state_skips_merged_pr_and_picks_open_one() {
        // Simulates the case where an issue has both a historical merged PR and a
        // current open PR. The selector must return the open one even if the
        // merged PR appears first in the list.
        let prs = serde_json::json!([
            {
                "number": 100,
                "title": "Fix #42: old attempt",
                "body": "Closes #42",
                "state": "MERGED",
                "headRefName": "old-branch",
                "headRefOid": "old111",
                "baseRefName": "main",
                "mergeStateStatus": "CLEAN",
                "files": [],
                "comments": []
            },
            {
                "number": 200,
                "title": "Fix #42: current attempt",
                "body": "Closes #42",
                "state": "OPEN",
                "headRefName": "new-branch",
                "headRefOid": "new222",
                "baseRefName": "main",
                "mergeStateStatus": "CLEAN",
                "files": [{"path": "src/foo.rs"}, {"path": "src/bar.rs"}],
                "comments": []
            }
        ]);
        let prs_array = prs.as_array().unwrap();

        let selected = select_pr_full_state(prs_array, 42).expect("an open PR should be selected");
        assert_eq!(selected.number, 200);
        assert_eq!(selected.state, "OPEN");
        assert_eq!(selected.head_ref_oid.as_deref(), Some("new222"));
        assert_eq!(selected.base_ref_name.as_deref(), Some("main"));
        assert_eq!(selected.merge_state_status.as_deref(), Some("CLEAN"));
        assert_eq!(selected.files, vec!["src/foo.rs".to_string(), "src/bar.rs".to_string()]);
    }

    #[test]
    fn test_select_pr_full_state_extracts_dirty_merge_state() {
        // The Review-loop DIRTY trigger relies on this field surfacing through
        // pr_full_state. Without it, conflicting PRs would be invisible to the
        // rebase fix-run dispatcher.
        let prs = serde_json::json!([
            {
                "number": 200,
                "title": "Fix #42",
                "body": "Closes #42",
                "state": "OPEN",
                "headRefName": "feature",
                "headRefOid": "abc",
                "baseRefName": "main",
                "mergeStateStatus": "DIRTY",
                "files": [{"path": "src/lib.rs"}],
                "comments": []
            }
        ]);
        let prs_array = prs.as_array().unwrap();
        let selected = select_pr_full_state(prs_array, 42).expect("PR present");
        assert_eq!(selected.merge_state_status.as_deref(), Some("DIRTY"));
        assert!(pr_is_dirty(selected.merge_state_status.as_deref()));
    }

    #[test]
    fn test_pr_is_dirty_only_matches_dirty_case_insensitively() {
        // We only act on DIRTY — the only mergeable status that means hard
        // conflicts. BLOCKED/BEHIND/etc. are not conflicts and must NOT trigger
        // a rebase fix-run, which would otherwise destroy work or churn forever.
        assert!(pr_is_dirty(Some("DIRTY")));
        assert!(pr_is_dirty(Some("dirty")));
        assert!(pr_is_dirty(Some("Dirty")));

        assert!(!pr_is_dirty(Some("CLEAN")));
        assert!(!pr_is_dirty(Some("BLOCKED")));
        assert!(!pr_is_dirty(Some("BEHIND")));
        assert!(!pr_is_dirty(Some("UNKNOWN")));
        assert!(!pr_is_dirty(Some("UNSTABLE")));
        assert!(!pr_is_dirty(Some("HAS_HOOKS")));
        assert!(!pr_is_dirty(Some("")));
        assert!(!pr_is_dirty(None));
    }

    #[test]
    fn test_select_pr_full_state_returns_none_when_no_open_pr_references_issue() {
        let prs = serde_json::json!([
            {
                "number": 100,
                "title": "Fix #42: old attempt",
                "body": "Closes #42",
                "state": "MERGED",
                "headRefName": "old-branch",
                "headRefOid": "old111",
                "baseRefName": "main",
                "mergeStateStatus": "CLEAN",
                "files": [],
                "comments": []
            }
        ]);
        let prs_array = prs.as_array().unwrap();
        assert!(select_pr_full_state(prs_array, 42).is_none());
    }

    fn check(name: &str, state: CheckState, is_required: bool) -> PrCheck {
        PrCheck {
            name: name.to_string(),
            state,
            is_required,
            run_id: None,
            details_url: None,
        }
    }

    #[test]
    fn pr_ci_status_is_green_when_no_checks_exist() {
        let status = PrCiStatus::default();
        assert!(status.is_green());
        assert!(status.failing_checks().is_empty());
        assert!(!status.has_pending_required());
    }

    #[test]
    fn pr_ci_status_is_green_when_all_required_succeed_and_nothing_is_failing() {
        // Mixed required/non-required checks where every required one is green
        // and no check failed. A non-required PENDING is allowed by spec.
        let status = PrCiStatus {
            checks: vec![
                check("build", CheckState::Success, true),
                check("test", CheckState::Success, true),
                check("lint", CheckState::Pending, false),
            ],
        };
        assert!(status.is_green());
    }

    #[test]
    fn pr_ci_status_is_not_green_when_required_check_is_pending() {
        // Required PENDING blocks merge — we wait for it to settle before
        // declaring CI green.
        let status = PrCiStatus {
            checks: vec![
                check("build", CheckState::Success, true),
                check("test", CheckState::Pending, true),
            ],
        };
        assert!(!status.is_green());
        assert!(status.has_pending_required());
        assert!(status.failing_checks().is_empty());
    }

    #[test]
    fn pr_ci_status_is_not_green_when_any_check_failed_even_if_not_required() {
        // Spec: "No checks in state FAILURE or ERROR (even non-required)."
        let status = PrCiStatus {
            checks: vec![
                check("build", CheckState::Success, true),
                check("optional-smoke", CheckState::Failure, false),
            ],
        };
        assert!(!status.is_green());
        assert_eq!(status.failing_checks().len(), 1);
        assert_eq!(status.failing_checks()[0].name, "optional-smoke");
    }

    #[test]
    fn pr_ci_status_failing_checks_includes_both_failure_and_error_states() {
        let status = PrCiStatus {
            checks: vec![
                check("ci/jenkins", CheckState::Error, false),
                check("test", CheckState::Failure, true),
                check("lint", CheckState::Success, true),
                check("docs", CheckState::Pending, false),
            ],
        };
        assert!(!status.is_green());
        let failing: Vec<&str> = status
            .failing_checks()
            .into_iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(failing, vec!["ci/jenkins", "test"]);
    }

    #[test]
    fn parse_status_check_rollup_handles_check_run_and_status_context_shapes() {
        let rollup = serde_json::json!([
            {
                "__typename": "CheckRun",
                "name": "build",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
                "isRequired": true,
                "detailsUrl": "https://github.com/owner/repo/actions/runs/9876543/job/111"
            },
            {
                "__typename": "CheckRun",
                "name": "flaky-test",
                "status": "COMPLETED",
                "conclusion": "FAILURE",
                "isRequired": false,
                "detailsUrl": "https://github.com/owner/repo/actions/runs/9876544/job/222"
            },
            {
                "__typename": "CheckRun",
                "name": "long-running",
                "status": "IN_PROGRESS",
                "conclusion": null,
                "isRequired": true,
                "detailsUrl": null
            },
            {
                "__typename": "StatusContext",
                "context": "ci/jenkins",
                "state": "ERROR",
                "isRequired": true,
                "targetUrl": "https://jenkins.example.com/job/ci"
            },
            {
                "__typename": "Unknown",
                "name": "future-shape"
            }
        ]);
        let status = parse_status_check_rollup(&rollup);
        assert_eq!(status.checks.len(), 4, "unknown __typename must be skipped");

        let by_name: std::collections::HashMap<&str, &PrCheck> = status
            .checks
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();

        assert_eq!(by_name["build"].state, CheckState::Success);
        assert_eq!(by_name["build"].run_id.as_deref(), Some("9876543"));
        assert!(by_name["build"].is_required);

        assert_eq!(by_name["flaky-test"].state, CheckState::Failure);
        assert!(!by_name["flaky-test"].is_required);
        assert_eq!(by_name["flaky-test"].run_id.as_deref(), Some("9876544"));

        assert_eq!(by_name["long-running"].state, CheckState::Pending);
        assert!(by_name["long-running"].is_required);
        assert!(by_name["long-running"].run_id.is_none());

        assert_eq!(by_name["ci/jenkins"].state, CheckState::Error);
        assert!(by_name["ci/jenkins"].run_id.is_none());
    }

    #[test]
    fn parse_status_check_rollup_returns_empty_for_pr_with_no_checks() {
        // Spec: "Repo with zero checks → green." Verify the parse step gives us
        // an empty list, which is_green() then accepts.
        let empty = serde_json::json!([]);
        let status = parse_status_check_rollup(&empty);
        assert!(status.checks.is_empty());
        assert!(status.is_green());
    }

    #[test]
    fn normalize_check_run_state_treats_neutral_and_skipped_as_success() {
        // GitHub's "checks passed" UI considers these non-blocking; mirror that
        // so a Skipped non-required check doesn't keep us out of Merge.
        assert_eq!(
            normalize_check_run_state("COMPLETED", "NEUTRAL"),
            CheckState::Success
        );
        assert_eq!(
            normalize_check_run_state("COMPLETED", "SKIPPED"),
            CheckState::Success
        );
        assert_eq!(
            normalize_check_run_state("COMPLETED", "STALE"),
            CheckState::Success
        );
    }

    #[test]
    fn normalize_check_run_state_groups_terminal_failure_modes() {
        for conclusion in [
            "FAILURE",
            "TIMED_OUT",
            "CANCELLED",
            "ACTION_REQUIRED",
            "STARTUP_FAILURE",
        ] {
            assert_eq!(
                normalize_check_run_state("COMPLETED", conclusion),
                CheckState::Failure,
                "{conclusion} should map to Failure"
            );
        }
    }

    #[test]
    fn normalize_check_run_state_in_progress_is_pending_regardless_of_conclusion() {
        // GitHub sometimes leaves a stale conclusion field on an in-progress run.
        // Status drives the verdict until the run is COMPLETED.
        assert_eq!(
            normalize_check_run_state("IN_PROGRESS", "SUCCESS"),
            CheckState::Pending
        );
        assert_eq!(
            normalize_check_run_state("QUEUED", ""),
            CheckState::Pending
        );
    }

    #[test]
    fn normalize_status_context_state_distinguishes_failure_from_error() {
        assert_eq!(normalize_status_context_state("SUCCESS"), CheckState::Success);
        assert_eq!(normalize_status_context_state("FAILURE"), CheckState::Failure);
        assert_eq!(normalize_status_context_state("ERROR"), CheckState::Error);
        assert_eq!(normalize_status_context_state("PENDING"), CheckState::Pending);
        assert_eq!(normalize_status_context_state("EXPECTED"), CheckState::Pending);
        // Unknown -> Pending (don't pretend success).
        assert_eq!(normalize_status_context_state("WAT"), CheckState::Pending);
    }

    #[test]
    fn parse_actions_run_id_extracts_id_from_canonical_actions_url() {
        assert_eq!(
            parse_actions_run_id("https://github.com/owner/repo/actions/runs/123456789/job/987"),
            Some("123456789".to_string())
        );
        assert_eq!(
            parse_actions_run_id("https://github.com/owner/repo/actions/runs/42"),
            Some("42".to_string())
        );
    }

    #[test]
    fn parse_actions_run_id_returns_none_for_non_actions_urls() {
        assert!(parse_actions_run_id("https://jenkins.example.com/job/ci").is_none());
        assert!(parse_actions_run_id("").is_none());
        assert!(parse_actions_run_id("https://github.com/owner/repo/actions/runs/abc").is_none());
    }

    #[test]
    fn sanitize_untrusted_log_strips_ansi_csi_color_sequences() {
        // Common CI output: `gcc` / `cargo` / `pytest` color codes.
        let raw = "\x1b[31mFAILED\x1b[0m tests/foo.rs\n\x1b[1;33mwarning\x1b[0m: dead";
        let cleaned = sanitize_untrusted_log(raw);
        assert!(!cleaned.contains('\x1b'));
        assert!(cleaned.contains("FAILED tests/foo.rs"));
        assert!(cleaned.contains("warning: dead"));
    }

    #[test]
    fn sanitize_untrusted_log_strips_osc_terminal_title_sequences() {
        // OSC sequences (used to set terminal title) end with BEL or ST.
        let raw = "before\x1b]0;malicious title\x07after";
        let cleaned = sanitize_untrusted_log(raw);
        assert_eq!(cleaned, "beforeafter");
    }

    #[test]
    fn sanitize_untrusted_log_keeps_newlines_and_tabs_drops_other_controls() {
        // \n and \t are legitimate in compiler output — keep them. \r and
        // other C0 controls (e.g. \x07 BEL outside an OSC) are dropped so
        // they can't manipulate terminal state.
        let raw = "ok\nline2\twith tab\rcarriage\x07bell";
        let cleaned = sanitize_untrusted_log(raw);
        assert_eq!(cleaned, "ok\nline2\twith tabcarriagebell");
    }

    #[test]
    fn sanitize_untrusted_log_preserves_plain_text_unchanged() {
        // No escapes, no controls — output equals input.
        let raw = "error[E0382]: borrow of moved value\n  --> src/lib.rs:42:5";
        let cleaned = sanitize_untrusted_log(raw);
        assert_eq!(cleaned, raw);
    }

    #[test]
    fn truncate_chars_keeps_short_inputs_intact_and_truncates_long_ones() {
        assert_eq!(truncate_chars("hello", 10), "hello");
        // Boundary case: exact length should not be truncated.
        assert_eq!(truncate_chars("hello", 5), "hello");
        let truncated = truncate_chars("abcdefghij", 4);
        assert!(truncated.starts_with("abcd"));
        assert!(truncated.contains("[truncated]"));
    }

    #[test]
    fn test_parse_codex_approval_ignores_feedback_marker_footer() {
        // A non-approving Codex comment with the footer (which contains 👍) should NOT match
        // even if the patterns list contains a bare 👍 — the footer is stripped first.
        let patterns = vec![
            "Didn't find any major issues".to_string(),
            "👍".to_string(),
        ];
        let marker = "Useful? React with 👍 / 👎.";

        let non_approving_with_footer = "Found a P1 bug; please fix.\n\nUseful? React with 👍 / 👎.";
        assert!(
            !parse_codex_approval(non_approving_with_footer, &patterns, marker),
            "footer-only 👍 must not be treated as approval"
        );

        // Real approval phrase still matches even with the footer.
        let approving_with_footer =
            "Didn't find any major issues.\n\nUseful? React with 👍 / 👎.";
        assert!(parse_codex_approval(approving_with_footer, &patterns, marker));
    }
}
