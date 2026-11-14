/**
 * Lightweight XSS-safe Notion-flavored Markdown + Remediation v2 Callout renderer.
 * Every raw text segment is escaped via escapeHtml() before structural HTML tags are emitted.
 */

import { escapeHtml } from "./diff.js";

/**
 * Validate external URL so javascript:/data: URIs can never execute.
 * @param {string} rawUrl
 * @returns {string | null}
 */
export function sanitizeHref(rawUrl) {
  const trimmed = String(rawUrl || "").trim();
  if (/^https?:\/\/[^\s"'<>]+$/i.test(trimmed)) {
    return trimmed;
  }
  return null;
}

/**
 * Render safe inline Markdown (`code`, **bold**, *italic*, [link](https://...)).
 * @param {string} rawText
 * @returns {string}
 */
export function renderInlineMarkdown(rawText) {
  if (!rawText) {
    return "";
  }
  // Extract inline code spans first so inner asterisks/links stay literal
  const codePlaceholders = [];
  const withPlaceholders = String(rawText).replace(/`([^`\n]+)`/g, (_m, codeBody) => {
    const token = `\u0000CODE_${codePlaceholders.length}\u0000`;
    codePlaceholders.push(`<code class="md-inline-code">${escapeHtml(codeBody)}</code>`);
    return token;
  });

  // Extract safe [label](url) links before escaping remaining text
  const linkPlaceholders = [];
  const withLinks = withPlaceholders.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (_m, label, url) => {
    const safeUrl = sanitizeHref(url);
    const token = `\u0000LINK_${linkPlaceholders.length}\u0000`;
    if (safeUrl) {
      linkPlaceholders.push(
        `<a class="md-link" href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`
      );
    } else {
      linkPlaceholders.push(escapeHtml(label));
    }
    return token;
  });

  let html = escapeHtml(withLinks);

  // Bold & italic transforms on already-escaped text
  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?:;]|$)/g, "$1<em>$2</em>");

  // Restore links and inline code spans
  html = html.replace(/\u0000LINK_(\d+)\u0000/g, (_m, idx) => linkPlaceholders[Number(idx)] || "");
  html = html.replace(/\u0000CODE_(\d+)\u0000/g, (_m, idx) => codePlaceholders[Number(idx)] || "");

  return html;
}

/**
 * Extract an explicit v2 Remediation Callout from ShipLog doc_markdown if present,
 * returning { calloutBody, cleanedMarkdown }.
 * @param {string} rawMarkdown
 * @param {string | null} fallbackMisconception
 * @param {boolean} isV2
 */
export function extractRemediationCallout(rawMarkdown, fallbackMisconception = null, isV2 = false) {
  let text = String(rawMarkdown || "");
  let extracted = "";

  // 1. Check Notion-style <callout icon="⚠️" color="orange_bg">...Remediation...</callout>
  const notionCalloutRe = /<callout([^>]*)>([\s\S]*?)<\/callout>/gi;
  text = text.replace(notionCalloutRe, (fullMatch, attrs, inner) => {
    const isRemediation =
      /remediation|trap|previous answer|attempt\s*1/i.test(inner) || /orange/i.test(attrs);
    if (isRemediation && !extracted) {
      extracted = inner.trim().replace(/^\t+/gm, "");
      return "";
    }
    // Convert any other Notion <callout> to standard blockquote so it renders cleanly
    const iconMatch = String(attrs).match(/icon="([^"]+)"/);
    const prefix = iconMatch ? `${iconMatch[1]} ` : "💡 ";
    const lines = inner
      .trim()
      .split(/\r?\n/)
      .map((l) => `> ${l.replace(/^\t+/, "")}`);
    if (lines.length > 0) {
      lines[0] = `> ${prefix}${lines[0].slice(2)}`;
    }
    return `\n${lines.join("\n")}\n`;
  });

  // 2. Check blockquote starting with > ⚠️ Pedagogical Remediation or ## Remediation v2 section
  if (!extracted && isV2) {
    const bqRe = /(?:^|\n)((?:>\s*[^\n]*(?:Remediation|trap|misconception)[^\n]*\n)(?:>\s*[^\n]*\n?)*)/i;
    const bqMatch = text.match(bqRe);
    if (bqMatch) {
      extracted = bqMatch[1]
        .split(/\r?\n/)
        .map((line) => line.replace(/^>\s?/, ""))
        .join("\n")
        .trim();
      text = text.replace(bqMatch[1], "\n");
    }
  }

  if (!extracted && isV2 && fallbackMisconception) {
    extracted = fallbackMisconception;
  }

  return {
    calloutBody: extracted,
    cleanedMarkdown: text.trim(),
  };
}

/**
 * Render Notion-flavored Markdown into clean, XSS-safe semantic HTML.
 * @param {string} rawMarkdown
 * @returns {string}
 */
export function renderNotionMarkdown(rawMarkdown) {
  if (!rawMarkdown || !String(rawMarkdown).trim()) {
    return `<p class="md-empty">No documentation body recorded yet.</p>`;
  }

  // Normalize any remaining Notion <details><summary>...</summary>...</details> safely
  const lines = String(rawMarkdown)
    .replace(/<\/?(?:details|summary)[^>]*>/gi, "\n")
    .split(/\r?\n/);

  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    // Fenced code block ```lang
    if (trimmed.startsWith("```")) {
      const lang = trimmed.slice(3).trim();
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) {
        i += 1; // skip closing ```
      }
      const langBadge = lang ? `<span class="md-code-block__lang">${escapeHtml(lang)}</span>` : "";
      blocks.push(
        `<div class="md-code-block">` +
          (langBadge ? `<div class="md-code-block__bar">${langBadge}</div>` : "") +
          `<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>` +
        `</div>`
      );
      continue;
    }

    // Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      blocks.push(`<hr class="md-hr" />`);
      i += 1;
      continue;
    }

    // Headings
    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      const level = Math.min(headingMatch[1].length + 1, 4); // map # -> h2 inside Notion page
      blocks.push(`<h${level} class="md-h${level}">${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      i += 1;
      continue;
    }

    // Blockquote / Notion Callout (> ...)
    if (trimmed.startsWith(">")) {
      const quoteLines = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ""));
        i += 1;
      }
      const joined = quoteLines.join(" ");
      const iconMatch = joined.match(/^(💡|🚫|⚠️|🛡️|🎯|✅|📌)\s*(.*)$/);
      if (iconMatch) {
        const icon = iconMatch[1];
        const tone = icon === "🚫" ? "danger" : icon === "⚠️" ? "warn" : "info";
        blocks.push(
          `<aside class="md-callout md-callout--${tone}">` +
            `<span class="md-callout__icon" aria-hidden="true">${icon}</span>` +
            `<div class="md-callout__body">${renderInlineMarkdown(iconMatch[2])}</div>` +
          `</aside>`
        );
      } else {
        blocks.push(`<blockquote class="md-quote">${renderInlineMarkdown(joined)}</blockquote>`);
      }
      continue;
    }

    // Markdown pipe table
    if (trimmed.startsWith("|") && i + 1 < lines.length && /^\s*\|?[\s:-]+\|/.test(lines[i + 1])) {
      const parseCells = (rowStr) =>
        rowStr
          .trim()
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
          .map((c) => c.trim());
      const headers = parseCells(lines[i]);
      i += 2; // skip header + separator
      const bodyRows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        bodyRows.push(parseCells(lines[i]));
        i += 1;
      }
      const thHtml = headers.map((h) => `<th>${renderInlineMarkdown(h)}</th>`).join("");
      const trHtml = bodyRows
        .map((cells) => `<tr>${cells.map((c) => `<td>${renderInlineMarkdown(c)}</td>`).join("")}</tr>`)
        .join("");
      blocks.push(
        `<div class="md-table-wrap"><table class="md-table"><thead><tr>${thHtml}</tr></thead><tbody>${trHtml}</tbody></table></div>`
      );
      continue;
    }

    // Unordered list
    if (/^[-*+]\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^[-*+]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*+]\s+/, ""));
        i += 1;
      }
      blocks.push(
        `<ul class="md-list">${items.map((it) => `<li>${renderInlineMarkdown(it)}</li>`).join("")}</ul>`
      );
      continue;
    }

    // Ordered list
    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i += 1;
      }
      blocks.push(
        `<ol class="md-list md-list--ordered">${items.map((it) => `<li>${renderInlineMarkdown(it)}</li>`).join("")}</ol>`
      );
      continue;
    }

    // Standard paragraph
    const paraLines = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith("#") &&
      !lines[i].trim().startsWith(">") &&
      !lines[i].trim().startsWith("|") &&
      !/^([-*+]|\d+\.)\s+/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i].trim());
      i += 1;
    }
    blocks.push(`<p class="md-p">${renderInlineMarkdown(paraLines.join(" "))}</p>`);
  }

  return blocks.join("\n");
}
