export interface KanbanCard {
  id: string;
  issueKey: string;
  repo?: string;
  number: number;
  title: string;
  labels: string[];
  assignee: string | null;
  updated: string;
  runId?: string;
  runStatus?: string;
  runStage?: string;
  error?: string | null;
  elapsed?: string;
  attempt?: number;
  maxRetries?: number;
  blockedBy?: number[];
  skippedStages?: string[];
  pendingNextStage?: string | null;
  reviewIteration?: number;
  /// Configured iteration cap, mirrored from RunConfig so the Review card can
  /// render `Iteration: N/M`. 0 means the cap is disabled.
  maxReviewIterations?: number;
  /// Cumulative cost (USD) summed across every run for this (repo, issue) —
  /// mirrors the backend value used for the cost-cap check.
  issueCostUsd?: number;
  /// Configured cost cap (USD), mirrored from RunConfig. 0.0 means disabled.
  costCapPerIssueUsd?: number;
  /// Short label describing the trigger that fired the most recent fix-run.
  lastTriggerSummary?: string | null;
}

export interface DashboardColumn {
  id: string;
  title: string;
  color: string;
  items: KanbanCard[];
}
