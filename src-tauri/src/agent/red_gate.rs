//! TDD red-gate: verifies the Implement stage produced a failing test before
//! production code. Walks the PR-branch git history from the base branch and
//! requires that the earliest non-merge commit modifies only test files and
//! that running the project's test command at that commit yields a failing
//! suite.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

/// Issue label that disables the gate entirely.
pub const RED_GATE_BYPASS_LABEL: &str = "no-tdd";

/// Languages whose source files participate in the gate.
const PRODUCTION_EXTENSIONS: &[&str] = &["ts", "tsx", "rs", "py", "swift", "go"];

/// Maximum wall time for the test invocation at the red commit.
const TEST_RUN_TIMEOUT_SECS: u64 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileClass {
    Production,
    Test,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    NoProductionFiles,
    BypassLabel,
    NoSupportedTestCommand,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureReason {
    NoTestOnlyCommit,
    TestPassedAtRedCommit { sha: String, command: String },
    GitError(String),
    TestRunError(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedGateOutcome {
    Skipped(SkipReason),
    Passed { red_sha: String, command: String },
    Failed(FailureReason),
}

impl RedGateOutcome {
    pub fn human_summary(&self) -> String {
        match self {
            RedGateOutcome::Skipped(SkipReason::NoProductionFiles) => {
                "[red-gate] Skipped: diff touches no production source files.".to_string()
            }
            RedGateOutcome::Skipped(SkipReason::BypassLabel) => format!(
                "[red-gate] Skipped: issue carries the `{}` label.",
                RED_GATE_BYPASS_LABEL
            ),
            RedGateOutcome::Skipped(SkipReason::NoSupportedTestCommand) => {
                "[red-gate] Skipped: no supported test runner detected (Node/Rust/Python/Go/Swift)."
                    .to_string()
            }
            RedGateOutcome::Passed { red_sha, command } => format!(
                "[red-gate] Passed: commit {} contains failing tests via `{}`.",
                short_sha(red_sha),
                command
            ),
            RedGateOutcome::Failed(FailureReason::NoTestOnlyCommit) => {
                "[red-gate] Failed: no test-only commit precedes the production changes. \
The Implement stage must commit a failing test before any production code."
                    .to_string()
            }
            RedGateOutcome::Failed(FailureReason::TestPassedAtRedCommit { sha, command }) => {
                format!(
                    "[red-gate] Failed: tests passed at the test-only commit {} (`{}`). \
The first commit must contain a *failing* test (red), not a passing one.",
                    short_sha(sha),
                    command
                )
            }
            RedGateOutcome::Failed(FailureReason::GitError(message)) => {
                format!("[red-gate] Failed: git error: {}", message)
            }
            RedGateOutcome::Failed(FailureReason::TestRunError(message)) => {
                format!("[red-gate] Failed: could not run test command: {}", message)
            }
        }
    }

}

fn short_sha(sha: &str) -> String {
    sha.chars().take(8).collect()
}

/// Classify a path as production, test, or other.
pub fn classify_path(path: &str) -> FileClass {
    let normalized = path.replace('\\', "/");
    let lower = normalized.to_lowercase();

    let extension = Path::new(&lower)
        .extension()
        .and_then(|ext| ext.to_str());

    let Some(extension) = extension else {
        return FileClass::Other;
    };

    if !PRODUCTION_EXTENSIONS.contains(&extension) {
        return FileClass::Other;
    }

    let file_name = Path::new(&normalized)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");

    let stem = Path::new(file_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("");

    let in_test_dir = normalized.split('/').any(|segment| {
        matches!(segment, "tests" | "__tests__" | "test")
    });

    let dotted_test_suffix = stem.ends_with(".test") || stem.ends_with(".spec");

    let go_rust_py_test_suffix = matches!(extension, "go" | "rs" | "py")
        && (file_name.ends_with(&format!("_test.{}", extension))
            || file_name.starts_with("test_"));

    let swift_test_suffix = extension == "swift" && file_name.ends_with("Tests.swift");

    if in_test_dir || dotted_test_suffix || go_rust_py_test_suffix || swift_test_suffix {
        FileClass::Test
    } else {
        FileClass::Production
    }
}

/// Determine whether a list of changed paths includes at least one production
/// source file.
pub fn diff_has_production_files(paths: &[String]) -> bool {
    paths
        .iter()
        .any(|path| classify_path(path) == FileClass::Production)
}

/// Determine whether all changed paths are test files (and at least one is).
pub fn diff_is_test_only(paths: &[String]) -> bool {
    let mut saw_test = false;
    for path in paths {
        match classify_path(path) {
            FileClass::Test => saw_test = true,
            FileClass::Production => return false,
            FileClass::Other => return false,
        }
    }
    saw_test
}

/// Auto-detect the test command to run at the workspace root. Returns the
/// command tokens (binary first, args after) or `None` if no supported runner
/// is found.
///
/// For Tauri-style hybrid layouts (a `package.json` at the root and a
/// `src-tauri/Cargo.toml` underneath), Cargo wins — the gate is meant to verify
/// the production source diff, and Rust changes under `src-tauri/` are the
/// dominant case for a Tauri app.
pub fn detect_test_command(workspace: &Path) -> Option<Vec<String>> {
    if workspace.join("Cargo.toml").exists() {
        return Some(vec!["cargo".into(), "test".into()]);
    }
    if workspace.join("src-tauri").join("Cargo.toml").exists() {
        return Some(vec![
            "cargo".into(),
            "test".into(),
            "--manifest-path".into(),
            "src-tauri/Cargo.toml".into(),
        ]);
    }
    if workspace.join("package.json").exists() {
        return Some(vec!["npm".into(), "test".into(), "--silent".into()]);
    }
    if workspace.join("pytest.ini").exists() || workspace.join("pyproject.toml").exists() {
        return Some(vec!["pytest".into(), "-q".into()]);
    }
    if workspace.join("go.mod").exists() {
        return Some(vec!["go".into(), "test".into(), "./...".into()]);
    }
    if workspace.join("Package.swift").exists() {
        return Some(vec!["swift".into(), "test".into()]);
    }
    None
}

pub fn render_test_command(tokens: &[String]) -> String {
    tokens.join(" ")
}

/// Determine whether the issue's labels disable the gate.
pub fn bypass_via_labels(labels: &[String]) -> bool {
    labels
        .iter()
        .any(|label| label.eq_ignore_ascii_case(RED_GATE_BYPASS_LABEL))
}

/// Public entry point for running the gate after a successful Implement run.
pub async fn run_red_gate(
    workspace: &Path,
    base_ref: &str,
    issue_labels: &[String],
) -> RedGateOutcome {
    if bypass_via_labels(issue_labels) {
        return RedGateOutcome::Skipped(SkipReason::BypassLabel);
    }

    let diff_files = match list_diff_files(workspace, base_ref).await {
        Ok(files) => files,
        Err(error) => {
            return RedGateOutcome::Failed(FailureReason::GitError(error));
        }
    };

    if !diff_has_production_files(&diff_files) {
        return RedGateOutcome::Skipped(SkipReason::NoProductionFiles);
    }

    let test_command = match detect_test_command(workspace) {
        Some(tokens) => tokens,
        None => return RedGateOutcome::Skipped(SkipReason::NoSupportedTestCommand),
    };

    let commit_log = match list_commits_since_base(workspace, base_ref).await {
        Ok(log) => log,
        Err(error) => {
            return RedGateOutcome::Failed(FailureReason::GitError(error));
        }
    };

    let red_sha = match find_first_test_only_commit(workspace, &commit_log).await {
        Ok(Some(sha)) => sha,
        Ok(None) => {
            return RedGateOutcome::Failed(FailureReason::NoTestOnlyCommit);
        }
        Err(error) => {
            return RedGateOutcome::Failed(FailureReason::GitError(error));
        }
    };

    let original_head = match capture_head(workspace).await {
        Ok(head) => head,
        Err(error) => {
            return RedGateOutcome::Failed(FailureReason::GitError(error));
        }
    };

    let outcome = run_tests_at_commit(workspace, &red_sha, &test_command).await;

    if let Err(error) = restore_head(workspace, &original_head).await {
        // We could not put the workspace back; surface as a failure even if the
        // gate itself would have passed, because subsequent stages depend on a
        // clean workspace.
        return RedGateOutcome::Failed(FailureReason::GitError(format!(
            "test invocation finished but failed to restore HEAD ({}): {}",
            short_sha(&original_head),
            error
        )));
    }

    match outcome {
        TestExecution::Failed => RedGateOutcome::Passed {
            red_sha,
            command: render_test_command(&test_command),
        },
        TestExecution::Passed => RedGateOutcome::Failed(FailureReason::TestPassedAtRedCommit {
            sha: red_sha,
            command: render_test_command(&test_command),
        }),
        TestExecution::Errored(message) => RedGateOutcome::Failed(FailureReason::TestRunError(message)),
    }
}

#[derive(Debug)]
enum TestExecution {
    Failed,
    Passed,
    Errored(String),
}

async fn run_git(workspace: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new(resolve_git())
        .args(args)
        .current_dir(workspace)
        .env("PATH", crate::paths::build_path_env())
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("git {} failed to spawn: {}", args.join(" "), error))?;

    if !output.status.success() {
        return Err(format!(
            "git {} exited with status {}: {}",
            args.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn resolve_git() -> String {
    crate::paths::resolve("git")
}

async fn list_diff_files(workspace: &Path, base_ref: &str) -> Result<Vec<String>, String> {
    let output = run_git(
        workspace,
        &["diff", "--name-only", &format!("{}...HEAD", base_ref)],
    )
    .await?;
    Ok(parse_name_only(&output))
}

fn parse_name_only(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect()
}

async fn list_commits_since_base(
    workspace: &Path,
    base_ref: &str,
) -> Result<Vec<String>, String> {
    let output = run_git(
        workspace,
        &[
            "rev-list",
            "--no-merges",
            "--reverse",
            &format!("{}..HEAD", base_ref),
        ],
    )
    .await?;
    Ok(parse_name_only(&output))
}

async fn list_commit_files(workspace: &Path, sha: &str) -> Result<Vec<String>, String> {
    let output = run_git(
        workspace,
        &[
            "show",
            "--no-renames",
            "--name-only",
            "--pretty=format:",
            sha,
        ],
    )
    .await?;
    Ok(parse_name_only(&output))
}

async fn find_first_test_only_commit(
    workspace: &Path,
    commits: &[String],
) -> Result<Option<String>, String> {
    for sha in commits {
        let files = list_commit_files(workspace, sha).await?;
        if diff_is_test_only(&files) {
            return Ok(Some(sha.clone()));
        }
    }
    Ok(None)
}

async fn capture_head(workspace: &Path) -> Result<String, String> {
    let stdout = run_git(workspace, &["rev-parse", "HEAD"]).await?;
    Ok(stdout.trim().to_string())
}

async fn restore_head(workspace: &Path, sha: &str) -> Result<(), String> {
    let _ = run_git(
        workspace,
        &["-c", "advice.detachedHead=false", "checkout", sha],
    )
    .await?;
    Ok(())
}

async fn run_tests_at_commit(
    workspace: &Path,
    sha: &str,
    test_command: &[String],
) -> TestExecution {
    if let Err(error) = run_git(
        workspace,
        &["-c", "advice.detachedHead=false", "checkout", sha],
    )
    .await
    {
        return TestExecution::Errored(error);
    }

    let mut command = Command::new(&test_command[0]);
    if test_command.len() > 1 {
        command.args(&test_command[1..]);
    }
    command
        .current_dir(workspace)
        .env("PATH", crate::paths::build_path_env())
        .env("CI", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let child = command.spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            return TestExecution::Errored(format!(
                "failed to spawn `{}`: {}",
                render_test_command(test_command),
                error
            ));
        }
    };

    let wait = tokio::time::timeout(
        Duration::from_secs(TEST_RUN_TIMEOUT_SECS),
        child.wait(),
    )
    .await;

    match wait {
        Ok(Ok(status)) => {
            if status.success() {
                TestExecution::Passed
            } else {
                TestExecution::Failed
            }
        }
        Ok(Err(error)) => TestExecution::Errored(format!("test process error: {}", error)),
        Err(_) => {
            let _ = child.kill().await;
            TestExecution::Errored(format!(
                "test command `{}` timed out after {}s",
                render_test_command(test_command),
                TEST_RUN_TIMEOUT_SECS
            ))
        }
    }
}

/// Resolve the diff base for the workspace. Tries `origin/main` first, then
/// falls back to `origin/master`. Returns the ref name.
pub async fn resolve_base_ref(workspace: &Path) -> Result<String, String> {
    for candidate in ["origin/main", "origin/master"] {
        let exists = Command::new(resolve_git())
            .args(["rev-parse", "--verify", candidate])
            .current_dir(workspace)
            .env("PATH", crate::paths::build_path_env())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map_err(|error| format!("git rev-parse: {}", error))?;
        if exists.success() {
            return Ok(candidate.to_string());
        }
    }
    Err("no origin/main or origin/master ref found".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command as StdCommand;
    use tempfile::TempDir;

    #[test]
    fn classify_production_typescript() {
        assert_eq!(classify_path("src/components/App.tsx"), FileClass::Production);
        assert_eq!(classify_path("src/utils/parse.ts"), FileClass::Production);
    }

    #[test]
    fn classify_rust_in_tests_dir() {
        assert_eq!(classify_path("tests/integration.rs"), FileClass::Test);
    }

    #[test]
    fn classify_rust_top_level_source_is_production() {
        assert_eq!(classify_path("src-tauri/src/agent/red_gate.rs"), FileClass::Production);
    }

    #[test]
    fn classify_jest_dot_test_suffix() {
        assert_eq!(classify_path("src/Foo.test.ts"), FileClass::Test);
        assert_eq!(classify_path("src/Foo.spec.tsx"), FileClass::Test);
    }

    #[test]
    fn classify_python_test_prefix_and_directory() {
        assert_eq!(classify_path("tests/test_thing.py"), FileClass::Test);
        assert_eq!(classify_path("app/test_helpers.py"), FileClass::Test);
        assert_eq!(classify_path("app/helpers.py"), FileClass::Production);
    }

    #[test]
    fn classify_go_underscore_test_suffix() {
        assert_eq!(classify_path("internal/api/handler_test.go"), FileClass::Test);
        assert_eq!(classify_path("internal/api/handler.go"), FileClass::Production);
    }

    #[test]
    fn classify_swift_tests_directory() {
        assert_eq!(classify_path("Tests/MyAppTests/FooTests.swift"), FileClass::Test);
        assert_eq!(classify_path("Sources/MyApp/Foo.swift"), FileClass::Production);
    }

    #[test]
    fn classify_double_underscore_tests_directory() {
        assert_eq!(classify_path("src/__tests__/Foo.tsx"), FileClass::Test);
    }

    #[test]
    fn classify_unknown_extension_is_other() {
        assert_eq!(classify_path("README.md"), FileClass::Other);
        assert_eq!(classify_path("package.json"), FileClass::Other);
        assert_eq!(classify_path("docs/architecture.png"), FileClass::Other);
    }

    #[test]
    fn diff_with_only_docs_has_no_production_files() {
        let diff = vec![
            "README.md".to_string(),
            "docs/spec.md".to_string(),
            "package.json".to_string(),
        ];
        assert!(!diff_has_production_files(&diff));
    }

    #[test]
    fn diff_with_production_typescript_is_detected() {
        let diff = vec![
            "README.md".to_string(),
            "src/App.tsx".to_string(),
        ];
        assert!(diff_has_production_files(&diff));
    }

    #[test]
    fn diff_test_only_requires_at_least_one_test() {
        assert!(!diff_is_test_only(&[]));
        assert!(!diff_is_test_only(&["README.md".to_string()]));
        assert!(diff_is_test_only(&["tests/foo.rs".to_string()]));
        assert!(!diff_is_test_only(&[
            "tests/foo.rs".to_string(),
            "src/lib.rs".to_string(),
        ]));
        assert!(!diff_is_test_only(&[
            "tests/foo.rs".to_string(),
            "README.md".to_string(),
        ]));
    }

    #[test]
    fn detect_test_command_picks_cargo_for_rust_workspace() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname=\"x\"\n").unwrap();
        let cmd = detect_test_command(dir.path()).expect("expected cargo command");
        assert_eq!(cmd[0], "cargo");
        assert_eq!(cmd[1], "test");
    }

    #[test]
    fn detect_test_command_picks_npm_for_node_workspace() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        let cmd = detect_test_command(dir.path()).expect("expected npm command");
        assert_eq!(cmd[0], "npm");
        assert_eq!(cmd[1], "test");
    }

    #[test]
    fn detect_test_command_picks_pytest_for_python_workspace() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("pyproject.toml"), "[project]\nname=\"x\"\n").unwrap();
        let cmd = detect_test_command(dir.path()).expect("expected pytest command");
        assert_eq!(cmd[0], "pytest");
    }

    #[test]
    fn detect_test_command_picks_go_test() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("go.mod"), "module example\n").unwrap();
        let cmd = detect_test_command(dir.path()).expect("expected go test command");
        assert_eq!(cmd, vec!["go", "test", "./..."]);
    }

    #[test]
    fn detect_test_command_picks_swift_test() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("Package.swift"), "// swift-tools-version:5.0\n").unwrap();
        let cmd = detect_test_command(dir.path()).expect("expected swift test command");
        assert_eq!(cmd, vec!["swift", "test"]);
    }

    #[test]
    fn detect_test_command_prefers_cargo_when_tauri_layout_has_package_json_at_root() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        std::fs::create_dir_all(dir.path().join("src-tauri")).unwrap();
        std::fs::write(
            dir.path().join("src-tauri").join("Cargo.toml"),
            "[package]\nname=\"x\"\n",
        )
        .unwrap();
        let cmd = detect_test_command(dir.path()).expect("expected cargo for tauri layout");
        assert_eq!(
            cmd,
            vec![
                "cargo".to_string(),
                "test".to_string(),
                "--manifest-path".to_string(),
                "src-tauri/Cargo.toml".to_string(),
            ]
        );
    }

    #[test]
    fn detect_test_command_returns_none_when_no_marker_present() {
        let dir = TempDir::new().unwrap();
        assert!(detect_test_command(dir.path()).is_none());
    }

    #[test]
    fn bypass_label_is_case_insensitive() {
        assert!(bypass_via_labels(&["no-tdd".to_string()]));
        assert!(bypass_via_labels(&["NO-TDD".to_string()]));
        assert!(bypass_via_labels(&["bug".to_string(), "No-TDD".to_string()]));
        assert!(!bypass_via_labels(&["bug".to_string()]));
    }

    #[test]
    fn human_summary_describes_skip_and_failure_cases() {
        assert!(RedGateOutcome::Skipped(SkipReason::NoProductionFiles)
            .human_summary()
            .contains("no production"));
        assert!(RedGateOutcome::Skipped(SkipReason::BypassLabel)
            .human_summary()
            .contains("no-tdd"));
        assert!(RedGateOutcome::Failed(FailureReason::NoTestOnlyCommit)
            .human_summary()
            .contains("test-only commit"));
        assert!(RedGateOutcome::Passed {
            red_sha: "abcdef1234".to_string(),
            command: "cargo test".to_string(),
        }
        .human_summary()
        .contains("abcdef12"));
    }

    fn run(repo: &PathBuf, args: &[&str]) {
        let status = StdCommand::new("git")
            .args(args)
            .current_dir(repo)
            .env("GIT_AUTHOR_NAME", "Test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "Test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .status()
            .expect("git command must run");
        assert!(status.success(), "git {:?} failed", args);
    }

    fn write(repo: &PathBuf, rel: &str, content: &str) {
        let full = repo.join(rel);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(full, content).unwrap();
    }

    fn init_repo(dir: &TempDir) -> PathBuf {
        let repo = dir.path().to_path_buf();
        run(&repo, &["init", "-q", "-b", "main"]);
        run(&repo, &["config", "commit.gpgsign", "false"]);
        write(&repo, "README.md", "init\n");
        run(&repo, &["add", "README.md"]);
        run(&repo, &["commit", "-q", "-m", "initial"]);
        // Establish a base ref the gate can resolve.
        run(&repo, &["update-ref", "refs/remotes/origin/main", "HEAD"]);
        repo
    }

    #[test]
    fn find_first_test_only_commit_locates_red_commit() {
        let dir = TempDir::new().unwrap();
        let repo = init_repo(&dir);

        write(&repo, "tests/integration.rs", "#[test]\nfn red() { assert!(false); }\n");
        run(&repo, &["add", "tests/integration.rs"]);
        run(&repo, &["commit", "-q", "-m", "red: failing test"]);

        write(&repo, "src/lib.rs", "pub fn answer() -> i32 { 42 }\n");
        run(&repo, &["add", "src/lib.rs"]);
        run(&repo, &["commit", "-q", "-m", "green: implementation"]);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let commits = runtime
            .block_on(list_commits_since_base(&repo, "origin/main"))
            .unwrap();
        let red = runtime
            .block_on(find_first_test_only_commit(&repo, &commits))
            .unwrap();
        assert!(red.is_some(), "expected to find a test-only commit");
    }

    #[test]
    fn find_first_test_only_commit_returns_none_when_first_commit_mixes_files() {
        let dir = TempDir::new().unwrap();
        let repo = init_repo(&dir);

        write(&repo, "tests/integration.rs", "#[test]\nfn red() { assert!(false); }\n");
        write(&repo, "src/lib.rs", "pub fn answer() -> i32 { 42 }\n");
        run(&repo, &["add", "."]);
        run(&repo, &["commit", "-q", "-m", "feature with tests"]);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let commits = runtime
            .block_on(list_commits_since_base(&repo, "origin/main"))
            .unwrap();
        let red = runtime
            .block_on(find_first_test_only_commit(&repo, &commits))
            .unwrap();
        assert!(red.is_none(), "first commit mixed prod+test, gate must miss");
    }

    #[test]
    fn run_red_gate_skips_when_diff_is_docs_only() {
        let dir = TempDir::new().unwrap();
        let repo = init_repo(&dir);

        write(&repo, "docs/architecture.md", "# spec\n");
        run(&repo, &["add", "docs/architecture.md"]);
        run(&repo, &["commit", "-q", "-m", "docs only"]);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let outcome = runtime.block_on(run_red_gate(&repo, "origin/main", &[]));
        assert_eq!(outcome, RedGateOutcome::Skipped(SkipReason::NoProductionFiles));
    }

    #[test]
    fn run_red_gate_skips_when_no_tdd_label_present() {
        let dir = TempDir::new().unwrap();
        let repo = init_repo(&dir);

        // even with production files, the bypass label short-circuits the gate
        write(&repo, "src/lib.rs", "pub fn answer() -> i32 { 42 }\n");
        run(&repo, &["add", "src/lib.rs"]);
        run(&repo, &["commit", "-q", "-m", "ship without tests"]);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let outcome = runtime.block_on(run_red_gate(
            &repo,
            "origin/main",
            &["no-tdd".to_string()],
        ));
        assert_eq!(outcome, RedGateOutcome::Skipped(SkipReason::BypassLabel));
    }

    #[test]
    fn run_red_gate_fails_when_no_test_only_commit_precedes_production() {
        let dir = TempDir::new().unwrap();
        let repo = init_repo(&dir);

        // single commit mixing production and test changes — no red commit
        write(&repo, "src/lib.rs", "pub fn answer() -> i32 { 42 }\n");
        write(&repo, "tests/integration.rs", "#[test] fn ok() {}\n");
        run(&repo, &["add", "."]);
        run(&repo, &["commit", "-q", "-m", "implement and test together"]);

        std::fs::write(repo.join("Cargo.toml"), "[package]\nname=\"x\"\n").unwrap();
        run(&repo, &["add", "Cargo.toml"]);
        run(&repo, &["commit", "-q", "-m", "add cargo manifest"]);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let outcome = runtime.block_on(run_red_gate(&repo, "origin/main", &[]));
        match outcome {
            RedGateOutcome::Failed(FailureReason::NoTestOnlyCommit) => {}
            other => panic!("expected NoTestOnlyCommit failure, got {:?}", other),
        }
    }
}
