/**
 * Shared helpers for the static-module unit tests (Node's built-in runner, no npm dependency).
 * Run from the repository root: node --test "tests/js/*.test.mjs"
 */

import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const STATIC_DIR = new URL("../../src/no_quiz_no_prod/ui/static/", import.meta.url);
const CHILD_TIMEOUT_MS = 10_000;
const CHILD_HEAP_MB = 64;

/** file:// URL of a browser module of the Studio. */
export function staticModuleUrl(name) {
  return new URL(name, STATIC_DIR).href;
}

/**
 * Call one exported function of a static module in a child process and return its JSON result.
 * A synchronous infinite loop cannot be interrupted from inside its own thread, so anything that must be
 * proven to terminate runs here: a hang ends as `{ finished: false }` instead of blocking the whole run.
 */
export function callInChild(moduleName, exportName, args) {
  const script =
    `const mod = await import(${JSON.stringify(staticModuleUrl(moduleName))});` +
    `const out = mod[${JSON.stringify(exportName)}](...${JSON.stringify(args)});` +
    `process.stdout.write(JSON.stringify(out));`;
  const child = spawnSync(
    process.execPath,
    [`--max-old-space-size=${CHILD_HEAP_MB}`, "--input-type=module", "-e", script],
    { cwd: fileURLToPath(new URL(".", pathToFileURL(import.meta.filename))), encoding: "utf8", timeout: CHILD_TIMEOUT_MS }
  );
  if (child.status !== 0) {
    return { finished: false, value: null };
  }
  return { finished: true, value: JSON.parse(child.stdout) };
}
