import assert from "node:assert/strict";
import { test } from "node:test";

import { callInChild, staticModuleUrl } from "./helpers.mjs";

const { extractRemediationCallout, renderInlineMarkdown, renderNotionMarkdown, sanitizeHref } = await import(
  staticModuleUrl("markdown.js")
);

// Lines that start like a block (heading, table) without being one: each used to freeze the renderer (UI-01).
const NON_BLOCK_LINES = {
  "hashtag without a space": "#tag",
  "five hashes": "##### Title",
  "shebang after a paragraph": "ok\n#!/bin/sh",
  "pipe line with no separator row": "| a | b |\ntext",
  "lone pipe as the last line": "intro\n|",
  "hash only": "#",
};

for (const [name, markdown] of Object.entries(NON_BLOCK_LINES)) {
  test(`renderNotionMarkdown terminates on a ${name}`, () => {
    const { finished, value } = callInChild("markdown.js", "renderNotionMarkdown", [markdown]);
    assert.equal(finished, true, "the renderer did not return");
    for (const line of markdown.split("\n")) {
      const visible = line.replace(/[|#]/g, "").trim();
      if (visible) {
        assert.ok(value.includes(visible.split(" ")[0]), `"${line}" is missing from the output`);
      }
    }
  });
}

test("a non-heading # line is kept as literal text, one paragraph per line", () => {
  const { value } = callInChild("markdown.js", "renderNotionMarkdown", ["#tag\n#!/bin/sh"]);
  assert.equal(value, '<p class="md-p">#tag</p>\n<p class="md-p">#!/bin/sh</p>');
});

test("a pipe line without a separator row is a paragraph, not a table", () => {
  const { value } = callInChild("markdown.js", "renderNotionMarkdown", ["| a | b |\ntext"]);
  assert.ok(value !== null && !value.includes("<table"));
  assert.ok(value.includes("| a | b |") && value.includes("text"));
});

test("real blocks still render after the paragraph fix", () => {
  const html = renderNotionMarkdown(
    ["# Title", "", "para one", "para two", "", "- a", "- b", "", "1. x", "", "| h |", "| - |", "| c |", "", "```py", "x = 1", "```", "", "> quoted"].join("\n")
  );
  assert.ok(html.includes('<h2 class="md-h2">Title</h2>'));
  assert.ok(html.includes('<p class="md-p">para one para two</p>'));
  assert.ok(html.includes('<ul class="md-list"><li>a</li><li>b</li></ul>'));
  assert.ok(html.includes('<ol class="md-list md-list--ordered"><li>x</li></ol>'));
  assert.ok(html.includes("<th>h</th>") && html.includes("<td>c</td>"));
  assert.ok(html.includes("<pre><code>x = 1</code></pre>"));
  assert.ok(html.includes('<blockquote class="md-quote">quoted</blockquote>'));
});

test("an empty document renders the empty-state paragraph", () => {
  assert.ok(renderNotionMarkdown("   ").includes("md-empty"));
});

test("raw HTML is escaped in every block type", () => {
  const payload = '<img src=x onerror="alert(1)">';
  for (const markdown of [payload, `# ${payload}`, `- ${payload}`, `> ${payload}`, `| ${payload} |\n| - |\n| ${payload} |`, "```\n" + payload + "\n```"]) {
    const html = renderNotionMarkdown(markdown);
    assert.ok(!html.includes("<img"), `unescaped tag in: ${markdown}`);
    assert.ok(html.includes("&lt;img"));
  }
});

test("inline markdown escapes text and only links http(s) URLs", () => {
  const html = renderInlineMarkdown('[x](javascript:alert(1)) **<b>** `<i>` [ok](https://ok.example/a?b=1&c=2)');
  assert.ok(!html.includes("javascript:"));
  assert.ok(html.includes("<strong>&lt;b&gt;</strong>"));
  assert.ok(html.includes('<code class="md-inline-code">&lt;i&gt;</code>'));
  assert.ok(html.includes('href="https://ok.example/a?b=1&amp;c=2"'));
  assert.ok(html.includes('rel="noopener noreferrer"'));
});

test("sanitizeHref accepts only quote-free http(s) URLs", () => {
  assert.equal(sanitizeHref(" https://github.com/acme/repo/commit/abc "), "https://github.com/acme/repo/commit/abc");
  for (const bad of ["javascript:alert(1)", "data:text/html,x", "//evil.example", 'https://a.example/"onmouseover=1', "", null, "ftp://a.example"]) {
    assert.equal(sanitizeHref(bad), null, String(bad));
  }
});

test("extractRemediationCallout lifts the v2 blockquote out of the document", () => {
  const { calloutBody, cleanedMarkdown } = extractRemediationCallout(
    "> Pedagogical Remediation (v2): floats drift\n\n## Part I\nBody",
    null,
    true
  );
  assert.equal(calloutBody, "Pedagogical Remediation (v2): floats drift");
  assert.equal(cleanedMarkdown, "## Part I\nBody");
});
