/**
 * Pure XSS-safe unified & split diff parser + inline word-level token highlighter.
 * Zero external dependencies. Strictly escapes raw source text before wrapping tokens.
 */

const HTML_ESCAPE_MAP = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

/**
 * Escape untrusted text for safe HTML insertion.
 * @param {unknown} raw
 * @returns {string}
 */
export function escapeHtml(raw) {
  if (raw === null || raw === undefined) {
    return "";
  }
  return String(raw).replace(/[&<>"']/g, (ch) => HTML_ESCAPE_MAP[ch]);
}

/**
 * Split a code line into fine-grained word/operator/whitespace tokens for inline diffing.
 * @param {string} line
 * @returns {string[]}
 */
function tokenizeLine(line) {
  return line.match(/[A-Za-z0-9_]+|\s+|[^\w\s]/g) || [];
}

/**
 * Compute inline word-level diff markup for a paired deletion and addition line.
 * @param {string} delText
 * @param {string} addText
 * @returns {{ delHtml: string, addHtml: string }}
 */
export function computeInlineWordDiff(delText, addText) {
  const a = tokenizeLine(delText);
  const b = tokenizeLine(addText);

  // Guard pathological long lines to keep rendering O(1) snappy
  if (a.length * b.length > 16000) {
    return {
      delHtml: `<mark class="diff-word-del">${escapeHtml(delText)}</mark>`,
      addHtml: `<mark class="diff-word-add">${escapeHtml(addText)}</mark>`,
    };
  }

  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));

  for (let i = m - 1; i >= 0; i -= 1) {
    for (let j = n - 1; j >= 0; j -= 1) {
      if (a[i] === b[j]) {
        dp[i][j] = dp[i + 1][j + 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }

  const delParts = [];
  const addParts = [];
  let i = 0;
  let j = 0;

  while (i < m && j < n) {
    if (a[i] === b[j]) {
      const esc = escapeHtml(a[i]);
      delParts.push({ changed: false, html: esc });
      addParts.push({ changed: false, html: esc });
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      delParts.push({ changed: true, html: escapeHtml(a[i]) });
      i += 1;
    } else {
      addParts.push({ changed: true, html: escapeHtml(b[j]) });
      j += 1;
    }
  }
  while (i < m) {
    delParts.push({ changed: true, html: escapeHtml(a[i]) });
    i += 1;
  }
  while (j < n) {
    addParts.push({ changed: true, html: escapeHtml(b[j]) });
    j += 1;
  }

  const collapseSegments = (parts, cls) => {
    let out = "";
    let buf = "";
    let inMark = false;
    for (const part of parts) {
      if (part.changed) {
        if (!inMark) {
          inMark = true;
          buf = part.html;
        } else {
          buf += part.html;
        }
      } else {
        if (inMark) {
          out += `<mark class="${cls}">${buf}</mark>`;
          buf = "";
          inMark = false;
        }
        out += part.html;
      }
    }
    if (inMark) {
      out += `<mark class="${cls}">${buf}</mark>`;
    }
    return out;
  };

  return {
    delHtml: collapseSegments(delParts, "diff-word-del"),
    addHtml: collapseSegments(addParts, "diff-word-add"),
  };
}

const HUNK_HEADER_RE = /^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$/;

/**
 * Parse unified diff text into structured hunks with line numbers and word-diff HTML.
 * @param {string} unifiedText
 */
export function parseUnifiedDiff(unifiedText) {
  const lines = String(unifiedText || "").split(/\r?\n/);
  const hunks = [];
  let currentHunk = null;
  let oldLine = 1;
  let newLine = 1;

  for (const rawLine of lines) {
    // "--- a/x" and "+++ b/x" are file headers only before the first hunk. Inside a hunk the same prefixes are
    // a deleted "-- ..." line and an added "++ ..." line, and they must reach the reviewer.
    if (!currentHunk && (rawLine.startsWith("--- ") || rawLine.startsWith("+++ "))) {
      continue;
    }
    const hunkMatch = rawLine.match(HUNK_HEADER_RE);
    if (hunkMatch) {
      oldLine = Number.parseInt(hunkMatch[1], 10);
      newLine = Number.parseInt(hunkMatch[3], 10);
      currentHunk = {
        header: rawLine,
        rows: [],
      };
      hunks.push(currentHunk);
      continue;
    }
    if (!currentHunk) {
      continue;
    }
    if (rawLine.startsWith("\\ No newline")) {
      continue;
    }
    if (rawLine.startsWith("-")) {
      currentHunk.rows.push({
        kind: "del",
        oldNo: oldLine,
        newNo: null,
        text: rawLine.slice(1),
        html: escapeHtml(rawLine.slice(1)),
      });
      oldLine += 1;
    } else if (rawLine.startsWith("+")) {
      currentHunk.rows.push({
        kind: "add",
        oldNo: null,
        newNo: newLine,
        text: rawLine.slice(1),
        html: escapeHtml(rawLine.slice(1)),
      });
      newLine += 1;
    } else if (rawLine.startsWith(" ") || rawLine === "") {
      const content = rawLine.startsWith(" ") ? rawLine.slice(1) : "";
      // Skip trailing empty split artifact at EOF
      if (rawLine === "" && currentHunk.rows.length > 0 && rawLine === lines[lines.length - 1]) {
        continue;
      }
      currentHunk.rows.push({
        kind: "ctx",
        oldNo: oldLine,
        newNo: newLine,
        text: content,
        html: escapeHtml(content),
      });
      oldLine += 1;
      newLine += 1;
    }
  }

  // Enrich adjacent del/add blocks with inline word-level highlights
  for (const hunk of hunks) {
    let idx = 0;
    while (idx < hunk.rows.length) {
      if (hunk.rows[idx].kind !== "del") {
        idx += 1;
        continue;
      }
      const delStart = idx;
      while (idx < hunk.rows.length && hunk.rows[idx].kind === "del") {
        idx += 1;
      }
      const delEnd = idx;
      const addStart = idx;
      while (idx < hunk.rows.length && hunk.rows[idx].kind === "add") {
        idx += 1;
      }
      const addEnd = idx;
      const pairCount = Math.min(delEnd - delStart, addEnd - addStart);
      for (let k = 0; k < pairCount; k += 1) {
        const delRow = hunk.rows[delStart + k];
        const addRow = hunk.rows[addStart + k];
        const { delHtml, addHtml } = computeInlineWordDiff(delRow.text, addRow.text);
        delRow.html = delHtml;
        addRow.html = addHtml;
      }
    }
  }

  return hunks;
}

function renderUnifiedTable(hunks) {
  const trs = [];
  for (const hunk of hunks) {
    trs.push(
      `<tr class="diff-row diff-row--hunk">` +
        `<td class="diff-gutter diff-gutter--hunk" colspan="2">@@</td>` +
        `<td class="diff-sign diff-sign--hunk"></td>` +
        `<td class="diff-code diff-code--hunk">${escapeHtml(hunk.header)}</td>` +
      `</tr>`
    );
    for (const row of hunk.rows) {
      const sign = row.kind === "add" ? "+" : row.kind === "del" ? "-" : " ";
      const oldLabel = row.oldNo !== null ? String(row.oldNo) : "";
      const newLabel = row.newNo !== null ? String(row.newNo) : "";
      trs.push(
        `<tr class="diff-row diff-row--${row.kind}">` +
          `<td class="diff-gutter tabular-nums">${oldLabel}</td>` +
          `<td class="diff-gutter tabular-nums">${newLabel}</td>` +
          `<td class="diff-sign">${sign}</td>` +
          `<td class="diff-code"><code>${row.html || "&nbsp;"}</code></td>` +
        `</tr>`
      );
    }
  }
  return `<table class="diff-table diff-table--unified" aria-label="Unified code diff"><tbody>${trs.join("")}</tbody></table>`;
}

function renderSplitTable(hunks) {
  const trs = [];
  for (const hunk of hunks) {
    trs.push(
      `<tr class="diff-row diff-row--hunk">` +
        `<td class="diff-gutter diff-gutter--hunk">@@</td>` +
        `<td class="diff-code diff-code--hunk" colspan="3">${escapeHtml(hunk.header)}</td>` +
      `</tr>`
    );
    let idx = 0;
    while (idx < hunk.rows.length) {
      const cur = hunk.rows[idx];
      if (cur.kind === "ctx") {
        trs.push(
          `<tr class="diff-row diff-row--split">` +
            `<td class="diff-gutter tabular-nums">${cur.oldNo}</td>` +
            `<td class="diff-code diff-split-cell diff-split-cell--ctx"><code>${cur.html || "&nbsp;"}</code></td>` +
            `<td class="diff-gutter tabular-nums diff-split-divider">${cur.newNo}</td>` +
            `<td class="diff-code diff-split-cell diff-split-cell--ctx"><code>${cur.html || "&nbsp;"}</code></td>` +
          `</tr>`
        );
        idx += 1;
        continue;
      }

      const dels = [];
      while (idx < hunk.rows.length && hunk.rows[idx].kind === "del") {
        dels.push(hunk.rows[idx]);
        idx += 1;
      }
      const adds = [];
      while (idx < hunk.rows.length && hunk.rows[idx].kind === "add") {
        adds.push(hunk.rows[idx]);
        idx += 1;
      }
      const maxLen = Math.max(dels.length, adds.length);
      for (let k = 0; k < maxLen; k += 1) {
        const left = dels[k] || null;
        const right = adds[k] || null;
        const leftCls = left ? "diff-split-cell--del" : "diff-split-cell--empty";
        const rightCls = right ? "diff-split-cell--add" : "diff-split-cell--empty";
        trs.push(
          `<tr class="diff-row diff-row--split">` +
            `<td class="diff-gutter ${left ? "diff-gutter--del" : ""} tabular-nums">${left ? left.oldNo : ""}</td>` +
            `<td class="diff-code diff-split-cell ${leftCls}"><code>${left ? `- ${left.html}` : "&nbsp;"}</code></td>` +
            `<td class="diff-gutter ${right ? "diff-gutter--add" : ""} tabular-nums diff-split-divider">${right ? right.newNo : ""}</td>` +
            `<td class="diff-code diff-split-cell ${rightCls}"><code>${right ? `+ ${right.html}` : "&nbsp;"}</code></td>` +
          `</tr>`
        );
      }
    }
  }
  return `<table class="diff-table diff-table--split" aria-label="Side-by-side prod vs staging code diff"><tbody>${trs.join("")}</tbody></table>`;
}

const FILE_CODE_SVG =
  `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">` +
  `<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>` +
  `<polyline points="14 2 14 8 20 8"></polyline>` +
  `<path d="m10 13-2 2 2 2"></path>` +
  `<path d="m14 17 2-2-2-2"></path>` +
  `</svg>`;

/**
 * Canonical DOM element ID for a FileDiff card, shared across diff.js and app.js.
 * @param {string} path
 * @param {number} index
 * @returns {string}
 */
export function diffFileDomId(path, index) {
  return `diff-file-${index}-${String(path || "file").replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

/**
 * Render all changed FileDiff cards in either "unified" or "split" mode.
 * @param {Array<{path: string, additions: number, deletions: number, unified_diff: string}>} files
 * @param {"unified" | "split"} mode
 * @returns {string}
 */
export function renderDiffFiles(files, mode = "unified") {
  if (!Array.isArray(files) || files.length === 0) {
    return "";
  }
  return files
    .map((file, index) => {
      const safeId = diffFileDomId(file.path, index);
      const hunks = parseUnifiedDiff(file.unified_diff || "");
      const bodyHtml =
        hunks.length === 0
          ? `<div class="diff-file__empty">Binary or whitespace-only change in <code>${escapeHtml(file.path)}</code></div>`
          : mode === "split"
            ? renderSplitTable(hunks)
            : renderUnifiedTable(hunks);
      return (
        `<section class="diff-file-card" id="${safeId}" data-file-path="${escapeHtml(file.path)}">` +
          `<header class="diff-file-card__header">` +
            `<div class="diff-file-card__title">` +
              `<span class="diff-file-card__icon" aria-hidden="true">${FILE_CODE_SVG}</span>` +
              `<span class="mono-text">${escapeHtml(file.path)}</span>` +
              `<span class="diff-file-card__scope">prod..staging</span>` +
            `</div>` +
            `<div class="diff-file-card__stats tabular-nums">` +
              `<span class="diff-stat-add">+${Number(file.additions || 0)}</span>` +
              `<span class="diff-stat-del">−${Number(file.deletions || 0)}</span>` +
            `</div>` +
          `</header>` +
          `<div class="diff-file-card__scroll">${bodyHtml}</div>` +
        `</section>`
      );
    })
    .join("");
}
