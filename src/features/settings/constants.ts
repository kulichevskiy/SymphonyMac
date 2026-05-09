import type { RunConfig } from "../../lib/types";

export const STAGE_KEYS = ["implement", "review", "merge"] as const;

export const STAGE_LABELS: Record<(typeof STAGE_KEYS)[number], string> = {
  implement: "Implement",
  review: "Review",
  merge: "Merge",
};

export const DEFAULT_RUN_CONFIG: RunConfig = {
  agent_type: "claude",
  auto_approve: true,
  max_concurrent: 3,
  poll_interval_secs: 60,
  issue_label: null,
  max_turns: 1,
  notifications_enabled: true,
  notification_sound: true,
  max_retries: 1,
  retry_backoff_secs: 10,
  retry_base_delay_secs: 10,
  retry_max_backoff_secs: 300,
  cleanup_on_failure: false,
  cleanup_on_stop: false,
  workspace_ttl_days: 7,
  max_concurrent_by_stage: {},
  stage_prompts: {},
  hooks: {
    after_create: null,
    before_run: null,
    after_run: null,
    before_remove: null,
    timeout_secs: 60,
  },
  priority_labels: ["priority:critical", "priority:high", "priority:medium", "priority:low"],
  stall_timeout_secs: 300,
  approval_gates: {},
  codex_approve_patterns: [
    "Didn't find any major issues",
    "did not find major issues",
  ],
  codex_feedback_marker: "Useful? React with 👍 / 👎.",
  local_repos: {},
  custom_agent_command: "",
};
