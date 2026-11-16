/**
 * Plain-language texts of the Studio, kept free of the DOM so the Node test runner can import them.
 */

// why a run stopped, as the person reads it in the status pill and the feed
export const STOP_REASON_LABELS = {
  completed: "Shipped to prod",
  clarification_needed: "The agent has a question",
  refused: "Refused: nothing changed",
  quiz_failed: "Blocked: quiz failed twice",
  human_rejected: "You rejected the release",
  release_failed: "Push failed",
  max_steps: "Stopped: step budget used up",
  api_error: "Model API error",
  harness_error: "Internal error",
};

// what each tool call did, in plain words, for the Simple view's one-line summary of an agent's steps
export const TOOL_PLAIN_LABELS = {
  read_notion_rule: "read the rules",
  search_notion_rules: "searched the rules",
  read_code_file: "read code",
  list_workspace_files: "listed files",
  write_staging_file: "edited code",
  run_staging_tests: "ran the tests",
  get_staging_diff: "checked the diff",
  upsert_ship_log_doc: "wrote the Ship Log",
  create_comprehension_quiz: "wrote the quiz",
  deploy_to_production: "deployed",
  read_audit_credentials: "read credentials",
};

/** "edited code ×2, ran the tests": the tools of a step group in call order, repeats counted. */
export function summarizeToolActions(tools) {
  const counts = new Map();
  for (const tool of tools.filter(Boolean)) {
    const label = TOOL_PLAIN_LABELS[tool] || tool;
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  return [...counts].map(([label, n]) => (n > 1 ? `${label} ×${n}` : label)).join(", ");
}

/** The line under the deploy gate that says what is still missing before the change can ship. */
export function gateMissingText({ treesIdentical, blocked, missing }) {
  if (treesIdentical) {
    return "Nothing to ship yet: ask the agent for a change on the left.";
  }
  if (blocked) {
    return "This change is blocked and cannot ship. Start a new thread to try again.";
  }
  return missing.length ? `Still needed: ${missing.join(", ")}.` : "All checks done.";
}
