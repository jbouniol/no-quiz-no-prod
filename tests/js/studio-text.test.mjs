import assert from "node:assert/strict";
import { test } from "node:test";

import { staticModuleUrl } from "./helpers.mjs";

const { STOP_REASON_LABELS, TOOL_PLAIN_LABELS, gateMissingText, summarizeToolActions } = await import(
  staticModuleUrl("studio-text.js")
);

test("every stop reason of the workflow has a plain label", () => {
  const reasons = [
    "completed",
    "clarification_needed",
    "refused",
    "quiz_failed",
    "human_rejected",
    "release_failed",
    "max_steps",
    "api_error",
    "harness_error",
  ];
  for (const reason of reasons) {
    assert.ok(STOP_REASON_LABELS[reason], `no label for ${reason}`);
    assert.doesNotMatch(STOP_REASON_LABELS[reason], /_/, `${reason} shows a raw identifier`);
  }
});

test("a step group summary counts repeated tools in call order", () => {
  const tools = ["read_code_file", "write_staging_file", "run_staging_tests", "write_staging_file", ""];
  assert.equal(summarizeToolActions(tools), "read code, edited code ×2, ran the tests");
});

test("a tool without a plain label keeps its name", () => {
  assert.equal(summarizeToolActions(["sync_ship_log"]), "sync_ship_log");
  assert.equal(summarizeToolActions([]), "");
  assert.equal(TOOL_PLAIN_LABELS.create_comprehension_quiz, "wrote the quiz");
});

test("the gate says what is missing, in priority order", () => {
  const missing = ["the quiz", "your approval"];
  assert.match(gateMissingText({ treesIdentical: true, blocked: true, missing }), /^Nothing to ship yet/);
  assert.match(gateMissingText({ treesIdentical: false, blocked: true, missing }), /blocked and cannot ship/);
  assert.equal(
    gateMissingText({ treesIdentical: false, blocked: false, missing }),
    "Still needed: the quiz, your approval."
  );
  assert.equal(gateMissingText({ treesIdentical: false, blocked: false, missing: [] }), "All checks done.");
});
