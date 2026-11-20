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
export function gateMissingText({ treesIdentical, blocked, missing, busy = false }) {
  if (treesIdentical) {
    return busy
      ? "Working on staging changes — locks unlock automatically once checks pass."
      : "Nothing to ship yet: ask the agent for a change on the left.";
  }
  if (blocked) {
    return "This change is blocked and cannot ship. Start a new thread to try again.";
  }
  return missing.length ? `Still needed: ${missing.join(", ")}.` : "All checks done.";
}

/**
 * Derive a clear, live stage indicator for active async operations so the user always sees
 * what is generating and when it has finished.
 */
export function deriveLiveActivity({
  isStreaming = false,
  isSubmittingQuiz = false,
  isSubmittingDeploy = false,
  isResetting = false,
  serverBusy = null,
  stepCount = 0,
  lastAgent = null,
  testsPassed = false,
  hasShipLog = false,
  hasQuiz = false,
  quizPassed = false,
  isBlocked = false,
} = {}) {
  if (isResetting) {
    return {
      badge: "Resetting…",
      title: "Resetting workspace seed…",
      detail: "Restoring clean baseline staging/ and prod/ trees.",
      tone: "system",
    };
  }
  if (isSubmittingDeploy) {
    return {
      badge: "Deploying…",
      title: "Deploying verified change to prod/…",
      detail: "Promoting staging diff, committing release & syncing Ship Log.",
      tone: "examiner",
    };
  }
  if (isSubmittingQuiz) {
    return {
      badge: "Grading quiz…",
      title: "Verifying your answer server-side…",
      detail: "Checking comprehension (generating Remediation v2 if missed).",
      tone: "examiner",
    };
  }
  if (serverBusy) {
    return {
      badge: `Busy: ${serverBusy}`,
      title: `Running ${serverBusy} in another session…`,
      detail: "Waiting for the active workspace operation to finish.",
      tone: "coder",
    };
  }
  if (!isStreaming) {
    return null;
  }
  if (testsPassed && hasShipLog && !hasQuiz && !quizPassed && !isBlocked) {
    return {
      badge: "Generating quiz…",
      title: "Examiner is writing your comprehension quiz…",
      detail: "Tests passed & Ship Log ready. Crafting targeted question for this diff.",
      tone: "examiner",
    };
  }
  if (testsPassed && !hasShipLog) {
    return {
      badge: "Writing Ship Log…",
      title: "Tests passed — Examiner generating Ship Log & Quiz…",
      detail: "Summarizing changed invariants, applied rules, and preparing quiz.",
      tone: "examiner",
    };
  }
  if (lastAgent === "examiner") {
    return {
      badge: "Examiner working…",
      title: "Examiner is preparing Ship Log & Quiz…",
      detail: "Drafting review documentation and comprehension check.",
      tone: "examiner",
    };
  }
  if (!stepCount) {
    return {
      badge: "Coder planning…",
      title: "Coder is reading the workspace…",
      detail: "Inspecting files and engineering rules before staging changes.",
      tone: "coder",
    };
  }
  return {
    badge: `Coder · step ${stepCount}`,
    title: `Coder is working (step ${stepCount})…`,
    detail: "Editing staging files and running pytest verification.",
    tone: "coder",
  };
}
