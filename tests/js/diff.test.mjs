import assert from "node:assert/strict";
import { test } from "node:test";

import { staticModuleUrl } from "./helpers.mjs";

const { computeInlineWordDiff, diffFileDomId, escapeHtml, parseUnifiedDiff, renderDiffFiles } = await import(
  staticModuleUrl("diff.js")
);

const rowsOf = (hunk) => hunk.rows.map((row) => [row.kind, row.text]);
// The rendered text without its markup (word-diff <mark> tags split a line into several nodes).
const textOf = (html) => html.replace(/<[^>]+>/g, "");

test("--- and +++ are file headers only before the first hunk (UI-03)", () => {
  const diff = [
    "--- a/prod/query.sql",
    "+++ b/staging/query.sql",
    "@@ -1,3 +1,4 @@",
    " keep",
    "--- old sql comment",
    "+++ plus-prefixed line",
    "+-- new sql comment",
    " end",
    "",
  ].join("\n");
  const [hunk] = parseUnifiedDiff(diff);
  assert.deepEqual(rowsOf(hunk), [
    ["ctx", "keep"],
    ["del", "-- old sql comment"],
    ["add", "++ plus-prefixed line"],
    ["add", "-- new sql comment"],
    ["ctx", "end"],
  ]);
});

test("every hunk line of a deleted SQL comment and an added ++ statement is rendered", () => {
  const file = {
    path: "auth.py",
    additions: 1,
    deletions: 1,
    unified_diff: "--- a/prod/auth.py\n+++ b/staging/auth.py\n@@ -1,2 +1,2 @@\n--- drop the audit trail\n+++ disable_hmac()\n same\n",
  };
  for (const mode of ["unified", "split"]) {
    const text = textOf(renderDiffFiles([file], mode));
    assert.ok(text.includes("-- drop the audit trail"), mode);
    assert.ok(text.includes("++ disable_hmac()"), mode);
  }
});

test("line numbers follow the hunk header and survive a second hunk", () => {
  const diff = "--- a/x\n+++ b/x\n@@ -3,2 +3,3 @@\n a\n+b\n c\n@@ -10,2 +11,1 @@\n-d\n e\n";
  const [first, second] = parseUnifiedDiff(diff);
  assert.deepEqual(
    first.rows.map((row) => [row.kind, row.oldNo, row.newNo]),
    [["ctx", 3, 3], ["add", null, 4], ["ctx", 4, 5]]
  );
  assert.deepEqual(
    second.rows.map((row) => [row.kind, row.oldNo, row.newNo]),
    [["del", 10, null], ["ctx", 11, 11]]
  );
});

test("the no-newline marker and text before the first hunk are ignored", () => {
  const diff = "diff --git a/x b/x\nindex 1..2\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n\\ No newline at end of file\n+new\n";
  const [hunk] = parseUnifiedDiff(diff);
  assert.deepEqual(rowsOf(hunk), [["del", "old"], ["add", "new"]]);
});

test("diff content is HTML-escaped in both modes", () => {
  const file = {
    path: '<b onload="x">.py',
    additions: 1,
    deletions: 1,
    unified_diff: '@@ -1 +1 @@\n-<script>alert(1)</script>\n+<img src=x onerror="y">\n',
  };
  for (const mode of ["unified", "split"]) {
    const html = renderDiffFiles([file], mode);
    assert.ok(!html.includes("<script>") && !html.includes("<img") && !html.includes("<b "), mode);
    assert.ok(textOf(html).includes("&lt;script&gt;alert(1)&lt;/script&gt;"), mode);
    assert.ok(html.includes("&lt;b onload="), mode);
  }
});

test("word-level diff marks only the changed tokens and escapes them", () => {
  const { delHtml, addHtml } = computeInlineWordDiff("return round(x, 2)", "return quantize(x, 2)");
  assert.equal(delHtml, 'return <mark class="diff-word-del">round</mark>(x, 2)');
  assert.equal(addHtml, 'return <mark class="diff-word-add">quantize</mark>(x, 2)');
  assert.ok(computeInlineWordDiff("a<b", "a>b").addHtml.includes("&gt;"));
});

test("a file with no hunk renders the binary/whitespace note; no files render nothing", () => {
  assert.ok(renderDiffFiles([{ path: "logo.png", unified_diff: "" }]).includes("Binary or whitespace-only change"));
  assert.equal(renderDiffFiles([]), "");
  assert.equal(renderDiffFiles(null), "");
});

test("escapeHtml and diffFileDomId are safe on odd input", () => {
  assert.equal(escapeHtml(`&<>"'`), "&amp;&lt;&gt;&quot;&#39;");
  assert.equal(escapeHtml(null), "");
  assert.equal(diffFileDomId('a/b "c".py', 2), "diff-file-2-a-b--c--py");
});
