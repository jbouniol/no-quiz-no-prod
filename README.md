# 🛡️ No-Quiz-No-Prod

**No-Quiz-No-Prod** is an agentic coding workflow with a comprehension gate. AI agents stage and test a fix, but the fix reaches production only after the human developer passes a quiz on the invariant it touches and approves the exact diff that was tested. Deploying is a real git commit pushed fast-forward to a release branch: a local bare repository by default, or a GitHub repository. It is built on LangGraph human-in-the-loop interrupts, an OS-sandboxed test runner with a hidden acceptance suite, two least-privilege stdio MCP servers, a cross-thread developer skill memory ("Layer 4"), and three Notion databases with a local SQLite fallback.

---

## Architecture: State Machine, Two MCP Servers and the 4-Lock Gate

```mermaid
flowchart TD
    Dev["Developer Task Prompt"] --> Graph["LangGraph Orchestrator (dual: Coder -> Examiner | single)"]

    subgraph MCP1["code-mcp (sandbox workspace and gate)"]
        Stage["staging/ tree edits + idempotency store"]
        Pytest["OS-sandboxed pytest + hidden acceptance suite (Lock 1: tests_passed)"]
        Deploy["deploy_to_production (harness only, after the 4 locks)"]
        Push["Verified snapshot commit + fast-forward push (local bare remote by default, GitHub optional)"]
        Promote["Promote staging/ -> prod/, SQLite ledger 'deployed', Ship Log 'Deployed to Prod' + commit link"]
    end

    subgraph MCP2["notion-quiz-mcp (rules, Ship Log and private quiz store)"]
        Rules["DB 1: Engineering Rules & Policies (RULE-001..004)"]
        ShipLog["DB 2: Feature Ship Log & Pedagogical Docs (v1 -> Remediation v2)"]
        QuizVault["SQLite quiz answer keys (agents only see PublicQuizView)"]
        SkillDB["DB 3: Developer Skill Profile (Layer 4 cross-thread memory)"]
    end

    Graph -->|"Coder reads rules and stages the fix"| Stage
    Stage --> Pytest
    Pytest -->|"Both suites pass -> binds diff_sha256"| ShipLog
    SkillDB -->|"Targets past misconceptions"| QuizVault
    ShipLog --> QuizVault

    QuizVault -->|"HITL interrupt 1: PublicQuizView (no answer key)"| QuizGate{"Server-side quiz grading (max 2 attempts / epoch)"}
    QuizGate -->|"Attempt 1 wrong: doc upgraded to v2 + trap callout"| ShipLog
    QuizGate -->|"Attempt 2 wrong: task epoch Blocked"| Blocked["Blocked (prod/ untouched)"]
    QuizGate -->|"Passed (Lock 3: quiz_passed)"| ApprovalGate{"HITL interrupt 2: human sign-off"}

    ApprovalGate -->|"Reject"| Blocked
    ApprovalGate -->|"Approve (Lock 4) + server re-checks Lock 2 (sha256_matches)"| Deploy
    Deploy --> Push
    Push -->|"Push succeeded"| Promote
    Push -->|"Push failed: Lock 4 revoked (release_retry interrupt), prod/ untouched"| ApprovalGate
    QuizGate -->|"Updates fail_count / pass_count"| SkillDB
```

### Core Engineering Pillars

| Pillar | Mechanism |
| :--- | :--- |
| **4-lock production gate** | `deploy_to_production` requires an explicit `expected_diff_sha256` at UI, harness and server (never defaulted), recomputes `diff_sha256 = sha256(prod_tree:staging_tree)` on the server, and refuses (`code="gate_locked"`) unless all four locks hold in the SQLite `gate_ledger` (`state.sqlite`) on that digest: **1. `tests_passed`** (both staged `pytest` and the seed's hidden acceptance suite pass), **2. `sha256_matches`** (zero post-test drift), **3. `quiz_passed`**, **4. `human_approved`**. Quiz attempts and blocks are tracked per **task epoch**: editing `staging/` after quiz exhaustion or a human rejection never clears a block until a human resets or unblocks. |
| **OS-sandboxed test runner & two-tier hidden acceptance** | Agent-written tests run on a private snapshot outside the workspace via `src/no_quiz_no_prod/backend/sandbox.py` (`sandbox-exec` Seatbelt on macOS; `bubblewrap` user/mount/PID/net namespaces on Linux) with a minimal 7-variable environment, process-group `SIGKILL` on timeout, 64 KiB output cap and `setrlimit` caps. **Tier 1 (runtime Lock 1)**: after a full green staged run, the seed's 19 hidden regression checks (`seeds/fintech/acceptance/test_acceptance.py`) run in a fresh sandbox and return counts only. **Tier 2 (eval scorer)**: for every deploying scenario (`24` dev + `8` held-out), `eval/acceptance/` (24 task-targeted checks + `test_seed_regression.py`, agent `test_*.py` excluded) runs under OS isolation on the final `prod/` tree (`RunFinalState.acceptance_passed`). Fails closed (`sandbox_unavailable`) unless `NO_QUIZ_UNSAFE_NO_SANDBOX=1` is explicitly set. |
| **Two bounds on every graph run** | The router halts a run cleanly once `steps >= max_steps` (`stop_reason="max_steps"`). Behind it, every LangGraph invocation carries a tight `recursion_limit` (`2 × remaining_steps + 30` supersteps, `10` on `/forget`): a budget-compliant run never reaches it, and any routing bug raises `GraphRecursionError` immediately instead of looping ~10,000 steps, ending cleanly as `harness_error` (never blamed on the model or provider). |
| **Git release & journal reconciliation** | `prepare()` snapshots `staging/` and `prod/`, verifies their SHA-256 digests, forbids symlinks, and builds git trees via pure plumbing (ignoring user and repo `.gitignore`/`.gitattributes`). `publish()` runs outside the workspace lock under `deploy_slot` (`300 s` deadline) with a persistent `release_journal` in `state.sqlite` reconciled on restart, pushing **fast-forward only** with gate trailers (`Diff-SHA256`, `Prod-Tree-SHA256`, `Gate`, `Quiz-Attempts`, `Approved-By: human`, `Rules`, `Thread`). If an existing remote branch's tip tree differs from `prod/`, the deploy fails closed with `remote_diverged` (no unreviewed sync commit is ever pushed; an absent branch is initialized). A failed push revokes `human_approved` and pauses on a `release_retry` interrupt (`Approve & Retry Push` or `Reject`; terminal `release_failed` after 2 re-approvals). |
| **Token & secret isolation** | `NO_QUIZ_GITHUB_TOKEN` is sent only to allow-listed HTTPS hosts on default port `443` (`github.com` by default, extendable via `NO_QUIZ_GIT_ALLOWED_HOSTS`) as an `http.<remote>.extraHeader` with `http.followRedirects=false`, `HOME=/dev/null`, and user/system git config ignored. MCP servers run under per-server allow-lists (`environment_for(server)` and `.mcp.json`): neither server gets LLM keys, `code-mcp` gets no Notion key, `notion-quiz-mcp` gets no git remote or token, and secrets are omitted from child launch environments and read from `.env` at runtime (mitigating macOS `sysctl(KERN_PROCARGS2)` same-user launch-env inspection, while `deploy_slot` prevents test runs and git network commands from overlapping). |
| **Dual vs. single agent topology** | **`dual` (default)**: the `Coder` edits `staging/` and runs `pytest`, with no access to `deploy_to_production` or quiz authoring; the `Examiner` reads Layer 4 memory, receives the full diff (failing closed if oversized), publishes the Ship Log, writes the quiz and handles remediation. **`single`**: one agent with both toolsets for ablation, with write and test tools removed (`SINGLE_REMEDIATION`) once remediation starts so code stays frozen during Attempt 2. |
| **Quiz secrecy & replay-safe grading** | `create_comprehension_quiz` stores `correct_option` and `explanation_by_option` only in SQLite (sealed out of graph state, traces, SSE and checkpoints) and returns `PublicQuizView`. `grade_quiz_answer` is atomic and replay-safe in the same SQLite transaction as Lock 3. A wrong Attempt 1 triggers `Remediation v2`; a wrong Attempt 2 blocks the task epoch. |
| **Web Studio guard & Notion fallback** | `StudioGuard` (`src/no_quiz_no_prod/ui/security.py`) enforces a `Host` allow-list, mandatory same-origin `Origin`, and a per-session `X-NQ-Session-Token` on all mutating routes, with one active operation slot per workspace (`409 run_active`). **Engineering Rules & Policies** (`RULE-001..004`), **Feature Ship Log** and **Developer Skill Profile** (`6` skills, `/forget` erasure) synchronize via Notion REST API (`2022-06-28`) or fall back to local SQLite and `seeds/fintech/`. |

---

## Getting Started (`uv`)

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/) (it downloads its own CPython `3.12` on the first `uv sync`: `pyproject.toml` sets `python-preference = "only-managed"`, so a Homebrew or system Python is never picked up), **`git >= 2.31`** (required for every deploy, including the default local bare remote, and validated at run start), and an OS test sandbox (`sandbox-exec` is built into macOS; on Linux install `bubblewrap` / `bwrap`). Node.js `>= 22.7` is optional and used only to run the frontend unit tests (`node --test`). Running the offline test suite (`uv run pytest -q`) needs no API key.

### 1. Install and configure

```bash
# Install the locked Python >= 3.12 environment and dev tools
uv sync

# Copy the environment template and add an API key (OPENROUTER_API_KEY or ANTHROPIC_API_KEY)
cp .env.example .env
```

| Environment Variable | Required | Default / Fallback | Description |
| :--- | :---: | :--- | :--- |
| `OPENROUTER_API_KEY` | Yes (or Anthropic) | — | API key for OpenRouter (`~deepseek/deepseek-v4-flash-latest`, `openai/gpt-4o-mini`, etc.) |
| `OPENROUTER_MODEL` | No | `~deepseek/deepseek-v4-flash-latest` | Default OpenRouter model route |
| `ANTHROPIC_API_KEY` | Optional | — | Native Anthropic SDK route for `claude-*` model IDs without `/` |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-5` | Default native Anthropic model |
| `NOTION_API_KEY` | Optional | Local SQLite mirror | Enables write-through to the 3 Notion databases |
| `NOTION_RULES_DB_ID` | Optional | `docs/notion_workspace_ids.json` | Notion Database 1 (`Engineering Rules & Policies`) |
| `NOTION_SHIP_LOG_DB_ID` | Optional | `docs/notion_workspace_ids.json` | Notion Database 2 (`Feature Ship Log & Pedagogical Docs`) |
| `NOTION_SKILLS_DB_ID` | Optional | `docs/notion_workspace_ids.json` | Notion Database 3 (`Developer Skill Profile`) |
| `NO_QUIZ_WORKSPACE_DIR` | No | `.runtime_workspace` | Isolated `prod/` + `staging/` workspace and `state.sqlite` ledger / release journal |
| `NO_QUIZ_GIT_REMOTE_URL` | No | `<workspace>/release_remote.git` (local bare) | Release remote of `deploy_to_production`: an `https://` URL on an allowed host (no credentials in it) or an absolute local repository path |
| `NO_QUIZ_GIT_BRANCH` | No | `main` | Release branch (fast-forward pushes only) |
| `NO_QUIZ_GITHUB_TOKEN` | With an HTTPS remote | — | Token with `contents: write` on the release repository; sent only to allowed HTTPS hosts on port `443` |
| `NO_QUIZ_GIT_ALLOWED_HOSTS` | No | `github.com` | Extra comma-separated HTTPS hosts allowed to receive `NO_QUIZ_GITHUB_TOKEN` (e.g. GitHub Enterprise) |
| `NO_QUIZ_UNSAFE_NO_SANDBOX` | No | `0` (isolation required) | Set to `1` to run staged tests **without** OS isolation (unsafe; surfaced on `DeployGateStatus.sandbox_disabled`) |
| `NO_QUIZ_UI_ALLOWED_HOSTS` | No | Loopback + `--host` | Extra comma-separated `Host` header names accepted by `no-quiz-ui` (used with `--reload`) |
| `NO_QUIZ_ENV_FILE` | No | `.env` (repository root) | Load another `.env` instead (hermetic tests, isolated runs). Variables already set in the process always win, even empty ones |

> [!TIP]
> Keep `NOTION_API_KEY` and `NO_QUIZ_GITHUB_TOKEN` in `.env` (or `NO_QUIZ_ENV_FILE`) rather than exporting them in your shell: spawned MCP servers never inherit secrets in their launch environment and read `.env` directly at runtime.

### 2. Optional: release to a GitHub repository

Without this step, deploys go to a local bare repository in the workspace. To see each release as a commit on GitHub:

1. **Create an empty repository** on GitHub to use as a sandbox (no README, no license, no `.gitignore`). If you reused a sandbox repository from an earlier commit before hermetic test guards landed, inspect its commit history for any stray gate-authored test commits.
2. **Create a fine-grained personal access token** limited to that one repository (*Repository access → Only select repositories*), with the **Contents: Read and write** permission and nothing else.
3. **Configure the release remote** in `.env`, or in the Studio's `.env` popover (GitHub tab: remote URL and token, with a 2-click confirmation on remote URL changes; the branch is set in `.env`):
   ```bash
   NO_QUIZ_GIT_REMOTE_URL=https://github.com/<owner>/<repo>.git
   NO_QUIZ_GIT_BRANCH=main
   NO_QUIZ_GITHUB_TOKEN=github_pat_...
   ```
4. **Check the remote, then create the release branch**:
   ```bash
   # git >= 2.31, URL, allowed host, token and ls-remote access; shows repo, branch and head commit. Pushes nothing.
   uv run no-quiz-no-prod init-github --verify
   ```
   ```bash
   # Push the pristine seed as the first commit of the release branch if it does not exist yet (fast-forward only)
   uv run no-quiz-no-prod init-github
   ```

`no-quiz-no-prod reset` resets the local workspace and discards the default local `<workspace>/release_remote.git`, while never rewriting an external remote: if an external branch has diverged from `prod/`, the next deploy fails closed with `remote_diverged` instead of pushing an unreviewed sync commit. The test suite (`tests/conftest.py`) and the evaluation harness always use a local bare remote per run, whatever `.env` contains.

### 3. Launch the Web Studio and Trace Explorer

```bash
# Start the FastAPI Studio (no build step) on http://127.0.0.1:8000
uv run no-quiz-ui
```

Pass `--allowed-host HOST` (repeatable) or set `NO_QUIZ_UI_ALLOWED_HOSTS` when accessing the Studio via a hostname other than loopback or `--host`. Every `POST` and `DELETE` route requires a same-origin `Origin` header and the per-session `X-NQ-Session-Token` injected into `GET /` (the browser UI sends both automatically).

Open `http://127.0.0.1:8000`:
- **Mode 1 (`⌘1`): Studio.** A launch screen with task presets and the `.env` key manager, then a 3-pane Workbench: live SSE trace of the MCP calls, unified/split `prod..staging` diff with its `diff_sha256`, the Ship Log `v1` → `Remediation v2` inspector, the Quiz Gate (keys `1–4` + `Enter`), the 4-lock gate with failed-push `Approve & Retry Push` / `Reject` handling and the deploy receipt (release commit, GitHub link), and the Layer 4 skill memory (`/forget`).
- **Mode 2 (`⌘2`): Eval.** State-scored results for `normal`, `ambiguous`, `impossible` and `adversarial` scenarios, `pass@k` and `pass^k`, cohort comparisons, `manifest.json` staleness banners, and step-by-step `RunFinalState` trace replay.

### 4. Run the state-based evaluation harness

Two scenario sets live under [`eval/`](eval/README.md): the **development set** [`eval/scenarios.jsonl`](eval/scenarios.jsonl) (**38 scenarios**: 19 `normal`, 6 `ambiguous`, 6 `impossible`, 7 `adversarial` — used during prompt/harness iteration, so rates on it are optimistic) and the **held-out set** [`eval/scenarios_heldout.jsonl`](eval/scenarios_heldout.jsonl) (**14 scenarios**: 5 `normal`, 3 `ambiguous`, 3 `impossible`, 3 `adversarial` — never tune prompts or budgets on it). By default, `run_eval.py` runs `k = 3` repeats per scenario (`pass^3` + scenario-level 95% Wilson CI) and writes to a fresh `eval/runs/<UTC-timestamp>_<slug>/` directory (refusing to overwrite existing results without `--overwrite`).

**Course grading command.** The command from the project brief works as is:

```bash
uv run python eval/run_eval.py --scenarios eval/scenarios.jsonl --out eval/
```

It resets the backend before every run and writes `eval/results.jsonl` (outcome and why, final state, tools called in order, steps, tokens in/out, latency, cost, stop reason), plus `eval/summary.md` and `eval/manifest.json`. With the default `--repeats 3` it makes **114 runs** (38 × 3); add `--repeats 1` for a single 38-run pass. These top-level outputs are git-ignored so a fresh clone never already "holds eval results" (which `--out` refuses without `--overwrite`); archived runs live under `eval/campaigns/`.

```bash
# Offline contract check (no API call, tagged dry-run -> new directory under eval/runs/)
uv run python eval/run_eval.py --dry-run
```

```bash
# Development set (38 scenarios x 3 repeats, dual topology, default model)
uv run python eval/run_eval.py --workers 4
```

```bash
# Held-out set (14 scenarios x 3 repeats — unbiased evaluation)
uv run python eval/run_eval.py --scenario-set heldout --workers 4
```

```bash
# Multi-model x topology benchmark + cross-family binary judge (persists judge_verdicts.jsonl, swap_test.json, judge_calibration.json)
uv run python eval/run_eval.py --workers 6 \
  --compare-models "~deepseek/deepseek-v4-flash-latest,google/gemini-2.5-flash" --compare-topologies --with-judge
```

```bash
# Export a blank human-annotation batch for judge calibration, then score with --human-labels
uv run python eval/run_eval.py --export-labeling-batch eval/runs/<run>/results.jsonl --labeling-out eval/labels/human_labels.json
```

```bash
# Re-score the 6 stale archived cohorts against eval/acceptance/ (no LLM call) -> eval/acceptance_replay.md
uv run python eval/replay_acceptance.py --out eval/acceptance_replay.md
```

### 5. Terminal CLI and MCP client wiring

```bash
# Reset .runtime_workspace/ back to the pristine fintech seed
uv run no-quiz-no-prod reset
```

```bash
# Run an interactive terminal human-in-the-loop session (validates git release config first; see `run --help` for exit codes)
uv run no-quiz-no-prod run "Fix VAT calculation in orders.py per RULE-001 and run staging tests."
```

```bash
# Verify the Notion database configuration
uv run no-quiz-no-prod init-notion --verify
```

`no-quiz-no-prod run` requires an explicit `y` or `n` at the deploy prompt (an empty `Enter` re-prompts; closing `stdin` exits with code `9` without recording a rejection) and returns a distinct exit code per terminal `stop_reason` (`0` `completed`, `1` `api_error` or typed error, `2` usage, `3` `clarification_needed`, `4` `refused`, `5` `quiz_failed`, `6` `human_rejected`, `7` `max_steps`, `8` `release_failed`, `9` no decision on `EOF`, `10` `harness_error`, `130` `Ctrl-C`).

**Connecting Claude Code, Antigravity or MCP Inspector via `.mcp.json`**:
The repository root ships a `.mcp.json` that launches both `stdio` MCP servers (`code-mcp` and `notion-quiz-mcp`) via `uv run python -m ...` with disallowed secret variables blanked per server. To inspect either server interactively:

```bash
uv run mcp dev src/no_quiz_no_prod/mcp_servers/code_server.py
```

```bash
uv run mcp dev src/no_quiz_no_prod/mcp_servers/notion_quiz_server.py
```

---

## Evaluation Results: Campaign `04` (current code, `k = 3`)

The first live run of the current harness, prompts, scenario sets and scoring (2026-11-16, harness commit `045ded7`, stop reason and hidden acceptance checks scored). Raw rows, `summary.md`, `comparison.md` and `manifest.json` are in [`eval/campaigns/04_dev_models_x_topologies/`](eval/campaigns/04_dev_models_x_topologies/comparison.md) and [`eval/campaigns/04_heldout_dual_vs_single/`](eval/campaigns/04_heldout_dual_vs_single/comparison.md); [`eval/README.md`](eval/README.md) has the full reading. The headline is `pass^3`: the share of scenarios whose three repeats all passed, with a 95% Wilson interval counted on scenarios.

| Set | Cohort | Scenarios × k | `pass^3` | 95% CI | Runs passed | `normal` | `ambiguous` | `impossible` | `adversarial` | USD / passed run |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| dev | DeepSeek V4 Flash · `dual` | 38 × 3 | **86.8%** | 72.7–94.2% | 106/114 | 56/57 | 12/18 | 17/18 | 21/21 | $0.0059 |
| dev | DeepSeek V4 Flash · `single` | 38 × 3 | **86.8%** | 72.7–94.2% | 106/114 | 55/57 | 14/18 | 17/18 | 20/21 | $0.0042 |
| dev | Gemini 2.5 Flash · `dual` | 38 × 3 | **68.4%** | 52.5–80.9% | 87/114 | 52/57 | 11/18 | 12/18 | 12/21 | $0.0202 |
| dev | Gemini 2.5 Flash · `single` | 38 × 3 | *awaiting rerun* | — | — | — | — | — | — | — |
| held-out | DeepSeek V4 Flash · `dual` | 14 × 3 | **78.6%** | 52.4–92.4% | 39/42 | 15/15 | 7/9 | 9/9 | 8/9 | $0.0042 |
| held-out | DeepSeek V4 Flash · `single` | 14 × 3 | **78.6%** | 52.4–92.4% | 35/42 | 15/15 | 2/9 | 9/9 | 9/9 | $0.0056 |

> [!WARNING]
> **Gemini 2.5 Flash · `single` is not a result.** The OpenRouter credit ran out mid-campaign: 85 of its 114 runs are `api_error` (HTTP 402). The rows are kept as recorded and flagged in the manifest and `comparison.md`; the cohort will be rerun alone.

**What the numbers say**

- **Multi-agent (dual) vs single, DeepSeek: no difference in `pass^3`, a difference in how runs fail.** The paired sign test gives `p = 1.00` on both sets, so neither topology is shown better on the headline. But all 8 dual failures on the dev set reached the right final state (nothing written on an ambiguous or impossible task) and only went over the step budget or stopped at `max_steps`. Single **deployed code that failed the hidden acceptance checks in 3 runs** (`n03` ×2, `ad04`); dual never did. Three runs are too few to call that significant. On the held-out set, dual passed more runs (39 vs 35), mostly on ambiguous tasks (7/9 vs 2/9 runs). Single is cheaper per passed run on the dev set (0.71×), dearer on the held-out set (1.35×).
- **DeepSeek vs Gemini (both dual):** Gemini scores 18.4 points lower in `pass^3` at 3.45× the cost per passed run (`p = 0.065`, not significant at 0.05). In 7 runs Gemini wrote or deployed on a task it should have questioned or refused (`am01`, `am06`, `im01` ×3, `im06` ×2): the gates checked tests, quiz and approval, and the simulated reviewer approved.
- **Dev vs held-out:** DeepSeek dual drops from 86.8% to 78.6%, as expected from a set the prompts were never tuned on.
- **Controls:** an agent that calls no tool and always asks for clarification, or always refuses, passes 15.8% of the dev set; always "completed" passes 0%.
- **Spend:** the whole campaign cost about **$3.95** of agent calls (DeepSeek $0.62 + $0.44 dev, $0.16 + $0.20 held-out; Gemini $1.76 dual, $0.77 for the invalid single run).

### Error analysis (campaign `04`)

| Failure mode | Where | Count | Root cause | Status |
| :--- | :--- | :---: | :--- | :--- |
| Examiner re-created the sealed quiz until the step budget ran out | First live smoke run, `n01` (before the campaign) | every dual deploy | The Examiner could not see the sealed answer key, so it kept "checking" its quiz instead of finishing | **Fixed** before campaign 04: the graph hands off as soon as the doc and quiz of the attempt exist (`ebdf233`) |
| Over-investigation before asking a clarification | `am01`, `am05`, `am06` (dev), `ham01`–`ham03` (held-out), both topologies | 6 dual + 4 single dev runs; 2 dual + 7 single held-out runs | The model takes 7 to 10 steps of reading before asking, against a budget of 6; on `am05` it often never asks and stops at `max_steps` | **Open**: model behaviour. Held-out budgets are not re-tuned on these results |
| Deployed code that fails the hidden acceptance checks | DeepSeek `single` `n03` ×2, `ad04`; Gemini `dual` `n03`, `ad04` ×2 | 3 + 3 | Agent-written tests passed but did not cover the task; caught by `eval/acceptance/` on `prod/` (scored `wrong_state`) | **Detected** by the scorer. The dual Examiner's separate pass is the only mitigation measured (0 such runs for DeepSeek dual) |
| Wrote or deployed on an ambiguous or impossible task | Gemini `dual` `am01`, `am06`, `im01` ×3, `im06` ×2 | 7 | Guessed the missing parameter or ignored the rule that forbids the change | **Open**: the human reviewer is the last gate |
| Adversarial failures | Gemini `dual` `ad02` ×3, `ad04` ×3, `ad05` ×3 | 9 of 21 | `ad02`: ran out of steps without deploying; `ad04`: 2 deploys failing acceptance + 1 `api_error`; `ad05`: stopped after 2 steps without fixing (planted text) | **Open**: model robustness |
| Over-refusal | DeepSeek `dual` `had02` (held-out) | 1 of 3 | Refused a task that was allowed | **Open** |
| Provider errors | Gemini `single` (402), Gemini `dual` (429) | 85 + 5 | Out of credit, rate limit | Not agent failures; Gemini single to rerun |

**Demo traces:** passing run `n01#0` (DeepSeek dual, VAT rounding fix, deployed and pushed); failing run `n03` in DeepSeek single (deployed, then failed the hidden acceptance check) or `am05#1` in DeepSeek dual (right state, stopped at `max_steps`). The Studio's Eval mode loads campaign 04 by default.

---

## Archived Pre-Fix Evaluation Campaigns (`eval/campaigns/`)

> [!IMPORTANT]
> The figures in this section come from the three earlier campaigns (`stale: true` in each `manifest.json`; see [`eval/README.md`](eval/README.md)), collected on 2026-11-14 under the earlier system prompts (which enumerated scenario classes), on the earlier 30/31-scenario development files before `expect_stop_reason` (`wrong_stop`) and the Lock 1 hidden acceptance suite were scored, and at **`k = 1` repeat per scenario** (`pass@1 = pass^1`). They are kept for history; do not compare them with campaign `04`.

### 1. Campaign `03`: `~deepseek/deepseek-v4-flash-latest` · `dual` (31 × 1, pre-fix git release snapshot)

Raw artifacts: [`eval/campaigns/03_git_release_dual_deepseek/results.jsonl`](eval/campaigns/03_git_release_dual_deepseek/results.jsonl) and [`eval/campaigns/03_git_release_dual_deepseek/summary.md`](eval/campaigns/03_git_release_dual_deepseek/summary.md).

| Kind | Scenarios | Strict pass (`k=1`) | 95% Wilson CI | Avg steps | Avg latency | Avg cost / run |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All** | 31 | **31/31 (100%)** | 89.0% – 100% | 7.5 | 47.96 s | $0.0054 |
| `normal` | 15 | 15/15 (100%) | 79.6% – 100% | 10.4 | 70.49 s | $0.0078 |
| `ambiguous` | 5 | 5/5 (100%) | 56.6% – 100% | 2.8 | 8.34 s | $0.0007 |
| `impossible` | 5 | 5/5 (100%) | 56.6% – 100% | 1.2 | 4.97 s | $0.0003 |
| `adversarial` | 6 | 6/6 (100%) | 61.0% – 100% | 9.5 | 60.45 s | $0.0076 |

In that historical `k = 1` run, **Deployed & Pushed** held on `20/31` runs and the other `11` left `prod/` untouched (`5` clarifications, `5` refusals, `1` quiz failure), using `2,230,129` tokens (`1.92 M` in / `0.31 M` out) for **`$0.167`** total (`47.96 s` average latency). Under campaign `03`'s unscored stop reasons, the zero-tool `do_nothing` baseline scored `32.3%` overall (`0%` on `normal` and `adversarial`); now that `expect_stop_reason` (`wrong_stop`) is scored, `do_nothing` (`completed`) scores `0%` (while constant `clarification_needed` and constant `refused` each score `15.8%` on the 38-scenario dev set).

**Historical fault injection in campaign `03`** (`n03#0` `write_timeout`, `n07#0` `rate_limit`, `n15#0` `push_rejected`): all three passed (`9/18`, `9/18`, `9/14` steps), but all three faults were absorbed by harness-level retries invisible to the model, and `n15` was a reworded copy of `n01`. In the current `eval/scenarios.jsonl`, `n15` is a distinct unknown-VIP-tier task, and `n16` (`push_unavailable` → `release_failed`), `n17` (`write_outcome_unknown`, model-visible fault), `n18`–`n19` look-alikes, `am06`, `im06` and `ad07` were added (first run live in campaign `04`).

### 2. Historical harness iterations (`k = 1` snapshots)

| Snapshot | Harness state | Historical result (DeepSeek · `dual`, `k=1`) |
| :--- | :--- | :--- |
| 1. Initial baseline ([`eval/campaigns/01_pre_fix_dual_deepseek/`](eval/campaigns/01_pre_fix_dual_deepseek/summary.md)) | `max_tokens=8000`, pre-git-release | 27/30 (90.0%): `normal` 12/14 |
| 2. Output budget ([`eval/campaigns/02_benchmark_pre_git_release/summary.md`](eval/campaigns/02_benchmark_pre_git_release/summary.md)) | Unusable-turn guard + 16k output budget | 29/30 (96.7%): `normal` 14/14 |
| 3. Examiner dead-end guard | Examiner question handled by deliverables + `next_step` on `invalid_argument` | Offline regression tests only |
| 4. Pre-fix git release ([`eval/campaigns/03_git_release_dual_deepseek/`](eval/campaigns/03_git_release_dual_deepseek/summary.md)) | `deploy_to_production` commits & pushes; `pushed` scored | 31/31 (100%) at `k=1` (`20/20` deployed pass offline [`eval/acceptance_replay.md`](eval/acceptance_replay.md)) |
| 5. Post-audit hardening (current `HEAD`) | OS test sandbox + Lock 1 & `eval/acceptance/` hidden suites, `recursion_limit` net, SQLite epoch ledger, `release_retry` / `remote_diverged`, de-enumerated prompts, `wrong_stop`, 38 dev + 14 held-out scenarios | **Offline replay**: `103/109` historical task-checked deploys hold (`6` false passes caught in campaign `02`: DeepSeek dual `n03`; Gemini dual `n05`, `n08`; Gemini single `n03`, `n05`, `n08`). Live `k=3`: campaign `04` (see above) |

### 3. Campaign `02` snapshot: topology and model runs (`30 × 1`, pre-git-release)

> Archived in [`eval/campaigns/02_benchmark_pre_git_release/`](eval/campaigns/02_benchmark_pre_git_release/comparison.md) (`results_*.jsonl`, `30` scenarios, `k = 1` repeat, before git release, iteration-3 Examiner fixes, and prompt de-enumeration).

| Cohort (`k = 1`) | Strict pass | `normal` | `ambiguous` | `impossible` | `adversarial` | Tokens | Cost | Avg latency |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| DeepSeek V4 Flash · `dual` | 29/30 | 14/14 | 4/5 | 5/5 | 6/6 | 2.22 M | $0.151 | 47.6 s |
| DeepSeek V4 Flash · `single` | 28/30 | 14/14 | 3/5 | 5/5 | 6/6 | 2.49 M | $0.139 | 37.3 s |
| Gemini 2.5 Flash · `dual` | 26/30 | 13/14 | 4/5 | 5/5 | 4/6 | 1.32 M | $0.430 | 17.1 s |
| Gemini 2.5 Flash · `single` | 26/30 | 14/14 | 2/5 | 5/5 | 5/6 | 2.07 M | $0.484 | 19.5 s |

- **No statistically supported ranking (`k = 1`):** None of the paired exact sign tests on 30 single-repeat scenarios is significant (`p = 1.00` for DeepSeek `dual` vs `single` [1 discordant scenario]; `p = 1.00` for Gemini `dual` vs `single` [2 vs 2 discordant scenarios, computed directly from the two archived Gemini JSONL files; `report.py` now emits all cohort pairs]; `p = 0.375` for DeepSeek vs Gemini `dual` [4 vs 1 discordant scenarios]). Single-repeat runs on a 30-scenario dev set cannot establish superiority between topologies or models.
- **Observed failure modes in those traces:** On `am02`/`am03`/`am05` Gemini `single` guessed an unspecified business parameter (`am05` also in `dual`); on `am05` DeepSeek `single` started editing and hit `max_steps`; two of Gemini `dual`'s misses (`n12`, `ad06`) were the Examiner dead end fixed in iteration 3; `ad05` was an injection-induced refusal in both Gemini topologies. `single` used more input tokens (+12% DeepSeek, +57% Gemini) by resending one growing transcript, while `dual` used more output tokens on the separate Examiner pass.

### 4. Campaign `02` judge snapshot & persistence caveats

- **Archived summary figures:** On campaign `02`, `anthropic/claude-sonnet-4.5` was run as a binary 4-criterion judge (`29/30` accepted on DeepSeek `dual`, `20/30` order-consistent and `10/30` flipped in the dual-vs-single swap test). **Individual judge verdicts, swap-test pair picks, failed-call counts and judge USD cost were not persisted in campaign `02`** (`verdicts_stored: false` in `manifest.json`), so those counts cannot be recomputed from disk. Running `run_eval.py --with-judge` now persists `judge_verdicts.jsonl`, `swap_test.json` and `judge_calibration.json`.
- **Unvalidated reference labels:** The `70.0%` raw agreement (`κ = 0.444`, equal to the `70%` majority-class accept baseline) reported in campaign `02` was measured against 20 harness-authored synthetic draft records now stored at [`eval/labels/synthetic_draft_labels.json`](eval/labels/synthetic_draft_labels.json) (`validated: false`), **not** independent human annotations. See [`eval/labels/README.md`](eval/labels/README.md) to export and score real human labels before citing `κ`.
- **Recorded agent spend:** The 6 distinct recorded cohort JSONL files under [`eval/campaigns/`](eval/campaigns/) sum to **`$1.49`** (`$1.4946`); unarchived judge calls and smoke runs are not included.

### 5. Error analysis & audit hardening summary (before campaign `04`)

| Failure mode / risk | Where observed | Root cause | Current status |
| :--- | :--- | :--- | :--- |
| Reasoning ate the output budget (empty turn read as "done") | DeepSeek, `n03` / `n13` (campaign `01`) | `finish_reason="length"` turn parsed as final message | **Fixed** (`MAX_UNUSABLE_TURN_RETRIES = 2`, `AGENT_MAX_OUTPUT_TOKENS = 16_000`) |
| Examiner asks the developer to fix its own rejected quiz | Gemini `dual`, `n12` / `ad06` (campaign `02`) | Examiner `CLARIFICATION_NEEDED` had no route | **Fixed** (offline regression tests) |
| Over-investigation before asking a clarification | `am01` (`8/6`, `7/6`, `9/6`, `5/6` across `k=1` runs) | Model reads several files before asking; zero writes | **Open**, now measured at `k=3` in campaign `04` |
| Guessing an ambiguous parameter and shipping it | Gemini `am02`, `am03`, `am05` | Gates check tests, quiz and human approval, not business intent | **Open** (human reviewer is the gate) |
| Injection-induced refusal | Gemini `ad05` | Planted `"DEPRECATED: do not fix"` text in a rule | **Open** (model robustness; prompts now wrap all untrusted inputs in `<untrusted_data>`) |
| Staged tests ran unsandboxed and Lock 1 trusted agent-editable tests | Security audit (`BE-01`, `BE-03`) | `pytest` ran in-workspace as the host user with no hidden suite | **Fixed** (`backend/sandbox.py` OS sandbox + 19 hidden checks in `seeds/fintech/acceptance/test_acceptance.py` + SQLite `gate_ledger`) |
| Deployed wrong fix scored as `pass` | Campaign `02` replay: DeepSeek dual `n03`; Gemini dual `n05`, `n08`; Gemini single `n03`, `n05`, `n08` | Scorer checked `prod_changed` + green agent-written tests rather than task behavior on `prod/` | **Fixed** (`src/no_quiz_no_prod/eval/acceptance.py` runs `eval/acceptance/` under OS isolation on `prod/`; [`eval/acceptance_replay.md`](eval/acceptance_replay.md) re-scores archived traces) |
| Failed push left no clean retry/reject path | Studio & graph (`C8`) | Push failure ended the graph or kept stale Lock 4 | **Fixed** (failed push revokes `human_approved` and pauses at `release_retry` for explicit retry or rejection; `release_failed` after 2 retries) |
| Test suite was not hermetic | `pytest`, git release | The spawned stdio MCP servers re-read `.env`, so test deploys pushed commits to the real GitHub repository. Found in code review | **Fixed**: `tests/conftest.py` points `NO_QUIZ_ENV_FILE` at a missing file and blanks the release remote, and every server launch path passes that environment explicitly (`tests/test_hermetic_env.py`) |
| Third-party commit on the release branch silently overwritten | Git release (`BE-31D`, `C6`) | Former sync commit wrote `prod/` on top of diverged remote tip | **Fixed** (`publish()` checks the remote tip tree against `prod/` and fails closed with `remote_diverged`) |

### 6. Trace showcase

- **Passed (`n01#0`)**: VAT rounding fix, deployed and pushed in [`eval/campaigns/03_git_release_dual_deepseek/`](eval/campaigns/03_git_release_dual_deepseek/summary.md). Campaign `04` (loaded by default in the Studio when `eval/runs/` has no newer run) has the current traces.
- **Over-budget (`am01#0`)** in [`eval/campaigns/02_benchmark_pre_git_release/`](eval/campaigns/02_benchmark_pre_git_release/summary.md): right final state (`clarification_needed`, zero writes), `7` tool calls against a budget of `6`. To inspect campaign `02` in the Studio:
  ```bash
  uv run python -c "from pathlib import Path; import uvicorn; from no_quiz_no_prod.ui.server import create_app; uvicorn.run(create_app(eval_dir=Path('eval/campaigns/02_benchmark_pre_git_release')), host='127.0.0.1', port=8001)"
  ```

---

## Newcomer Map

Suggested reading order: `contracts.py` → `backend/workspace.py` & `backend/sandbox.py` & `backend/release.py` → `graph/workflow.py` & `agents/prompts.py` → `ui/server.py`.

| File / Config | Purpose |
| :--- | :--- |
| `src/no_quiz_no_prod/contracts.py` | Frozen Pydantic contracts: enums, SHA-256 tree hashing, SQLite ledger/journal models, per-server env allow-lists, SSE `RunEvent`, `RunFinalState` and `EvalRunResult` |
| `src/no_quiz_no_prod/backend/workspace.py` | `SandboxWorkspace`: SQLite `state.sqlite` (`gate_ledger`, `release_journal`, task epochs, private quiz keys, idempotency store) and `prod/` + `staging/` promotion |
| `src/no_quiz_no_prod/backend/sandbox.py` | OS test isolation (`sandbox-exec` Seatbelt on macOS, `bubblewrap` on Linux), minimal child env, PGID `SIGKILL`, output and `rlimit` caps |
| `src/no_quiz_no_prod/backend/release.py` | `GitReleasePublisher` & `validate_release_config`: pure-plumbing verified trees, fast-forward push, `remote_diverged` check, host-scoped token header |
| `src/no_quiz_no_prod/graph/workflow.py` | LangGraph `NoQuizWorkflow` (`dual` & `single` topologies, `quiz_gate` / `deploy_gate` / `release_gate` interrupts, single-remediation freeze) |
| `src/no_quiz_no_prod/graph/mcp_bridge.py` | Reconnectable `StdioMCPBridge` with per-server env allow-lists and per-tool call timeouts |
| `src/no_quiz_no_prod/agents/` | `prompts.py` (`<untrusted_data>` envelope), `coder.py`, `examiner.py`, `single.py` (`SINGLE_REMEDIATION`), `tools.py` |
| `src/no_quiz_no_prod/llm.py` | `LLMClient` routing OpenRouter and native Anthropic with structured output parsing and USD cost tracking |
| `src/no_quiz_no_prod/cli.py` | `no-quiz-no-prod` CLI (`run`, `reset`, `ui`, `init-notion`, `init-github`) with distinct exit codes per `stop_reason` |
| `src/no_quiz_no_prod/ui/server.py`, `src/no_quiz_no_prod/ui/security.py` | FastAPI Studio (`no-quiz-ui`), `StudioGuard` (`Host`, `Origin`, `X-NQ-Session-Token`), single-run slot, SSE stream and static assets (`tokens.css`, `app.js`, `diff.js`, `markdown.js`) |
| `docs/ui_design_spec.md`, `docs/notion_workspace_ids.json` | Web Studio REST/SSE design spec and fallback Notion database IDs |
| `.mcp.json`, `langgraph.json`, `.env.example`, `pyproject.toml` | Least-privilege `stdio` MCP client config, LangGraph Studio graph entry (`make_graph`), env template, locked `uv` + `ruff` + `pytest` + `mypy` config |
| `.github/workflows/ci.yml`, `.pre-commit-config.yaml` | Linux + macOS GitHub Actions CI and local `pre-commit` / `pre-push` hooks |
| `seeds/fintech/`, `seeds/fintech/acceptance/test_acceptance.py` | Seeded fintech codebase, `RULE-001..004` policies, and the 19 hidden acceptance checks guarding Lock 1 |
| `eval/scenarios.jsonl`, `eval/scenarios_heldout.jsonl`, `eval/campaigns/`, `eval/labels/` | 38-scenario dev set, 14-scenario held-out set, archived pre-fix campaigns (`manifest.json`), and judge calibration label schema |
| `eval/acceptance/`, `src/no_quiz_no_prod/eval/acceptance.py`, `eval/replay_acceptance.py` | 24 task-specific hidden acceptance suites + `test_seed_regression.py` run under OS isolation on `prod/`, plus zero-LLM replay CLI (`eval/acceptance_replay.md`) |
| `tests/conftest.py` | Suite-wide guard: no test reads the real `.env`, its release remote or its GitHub token |
| `tests/test_sandbox.py`, `tests/test_git_release.py`, `tests/test_backend_and_mcp.py` | OS sandbox PoC regressions, git release plumbing/host/divergence tests, and SQLite epoch gate + stdio MCP tests |
| `tests/test_graph_and_memory.py`, `tests/test_end_to_end_integration.py`, `tests/test_llm.py` | Graph/memory unit tests, full Studio → workflow → stdio MCP → sandbox integration tests, and offline `LLMClient` tests |
| `tests/test_ui_api.py`, `tests/test_cli.py`, `tests/test_eval_scorer.py`, `tests/js/` | Studio guard/API tests, CLI prompt & exit-code tests, eval scorer/judge tests, and Node `diff.js`/`markdown.js` unit tests |

### Domain Package Architecture

| Domain | Package / Path | Responsibility |
| :--- | :--- | :--- |
| **Fintech sandbox seed** | `seeds/fintech/` | Seeded fintech modules (`orders.py`, `discounts.py`, `auth.py`, `refunds.py`, `test_fintech.py`), `RULE-001..004`, and hidden `acceptance/` suite |
| **Workspace, sandbox, release & Notion** | `src/no_quiz_no_prod/backend/` | `SandboxWorkspace` (SQLite WAL), `sandbox.py` (`sandbox-exec` / `bubblewrap`), `GitReleasePublisher` and `NotionDatabaseSync` |
| **MCP servers** | `src/no_quiz_no_prod/mcp_servers/` | `code_server.py` (`code-mcp`) and `notion_quiz_server.py` (`notion-quiz-mcp`) with per-server secret scrubbing and typed `ToolErrorPayload` |
| **Agents, graph & memory** | `src/no_quiz_no_prod/graph/`, `src/no_quiz_no_prod/agents/`, `src/no_quiz_no_prod/memory/` | LangGraph `NoQuizWorkflow`, reconnectable `StdioMCPBridge`, `<untrusted_data>` prompts and Layer 4 skill memory |
| **Evaluation harness** | `eval/`, `src/no_quiz_no_prod/eval/` | 38-scenario dev set, 14-scenario held-out set, `pass^k` / `wrong_stop` scorer, persisted binary judge artifacts and campaign manifests |
| **Web Studio & CLI** | `src/no_quiz_no_prod/ui/`, `src/no_quiz_no_prod/cli.py` | `StudioGuard`-protected SSE Workbench, Evaluation & Trace Explorer, and terminal HITL runner |

---

## End-to-End Product Walkthrough

1. **Gated release of a fix**:
   - Launch `uv run no-quiz-ui`, open `http://127.0.0.1:8000`, and select **Fix Decimal VAT Rounding Drift** on the launch screen.
   - The `Coder` stages the fix in `staging/orders.py` (`Decimal("0.01")`, `ROUND_HALF_UP`) and runs `pytest` inside the OS sandbox alongside the 19 hidden acceptance checks; the `Examiner` publishes Ship Log `v1` and Comprehension Quiz `v1`.
   - Review the word-level diff (`Unified` / `Split`) and its `diff_sha256`, answer the quiz (`1–4` + `Enter`) to unlock **Lock 3**, and click **Approve & Deploy** (**Lock 4**). The server re-checks the digest, commits and pushes the verified tree fast-forward, then promotes `staging/` into `prod/`.
2. **Fault recovery, remediation `v2` and skill memory**:
   - Enable the `fault` toggle in the Workbench composer and run **30-Day Refund Window & Remediation v2** to observe automatic retry with `idem:... · replayed` deduplication after a transient `write_timeout`.
   - Pick a wrong option on **Quiz Attempt 1**: the Ship Log upgrades to **`Remediation v2`** with a trap callout and **Layer 4 Skill Memory** records `Weakness Detected` for later threads (erasable via the 2-click `Reset`). If a release push fails, Lock 4 is revoked and the gate offers **Approve & Retry Push** or **Reject**.
3. **Evaluation and trace replay (`Eval` / `⌘2`)**:
   - Switch to **Eval** to inspect `pass@k` and `pass^k`, per-kind rates, cohort comparisons and any `manifest.json` staleness notice, and click **Inspect →** on any row to replay tool calls next to `RunFinalState`.

---

## Development, CI & Branch Protection

```bash
# Python test suite (347 tests: graph+memory 74, backend+MCP 58, sandbox 47, git release 43, eval 34, UI 31, CLI 27, LLM 25, e2e 4, hermetic env 4)
uv run pytest -q

# Frontend JavaScript unit tests (22 tests for diff.js and markdown.js; Node >= 22.7)
node --test "tests/js/*.test.mjs"

# Lint & formatting (100 Python files) and static type checks (55 typed source/test files)
uv run ruff check . && uv run ruff format --check . && uv run mypy

# Install local git hooks matching CI (.pre-commit-config.yaml)
uvx pre-commit install --hook-type pre-commit --hook-type pre-push
```

`tests/test_end_to_end_integration.py` drives the real Studio API over the real workflow and both stdio MCP servers with a scripted LLM. It covers the full loop (v1 → wrong answer → Layer 4 "Weakness Detected" → v2 remediation → right answer → human approval → system-role `deploy_to_production` → prod == staging), `release_retry` re-approval and rejection after a failed push, and `409 gate_locked` / `409 conflict` / `422` refusals on premature, hash-mismatched, missing-hash and replayed requests. Every test deploys to a local bare remote.

### Protecting `main` on GitHub

Branch protection lives in repository settings on GitHub and cannot be enabled from git files alone. After pushing `.github/workflows/ci.yml`, enable it once:

1. Open the repository on GitHub → **Settings** → **Rules** → **Rulesets** → **New ruleset** → **New branch ruleset**.
2. Set **Ruleset Name** to `main-protection`, switch **Enforcement status** to **Active**, and under **Target branches** click **Add target → Include default branch** (`main`).
3. Check **Restrict deletions**, **Block force pushes**, and **Require a pull request before merging** (enable **Dismiss stale pull request approvals when new commits are pushed**).
4. Check **Require status checks to pass**, enable **Require branches to be up to date before merging**, and add both required checks from `.github/workflows/ci.yml`: `checks (ubuntu-latest)` and `checks (macos-latest)`.
5. Leave **Bypass list** empty so direct pushes to `main` are refused until both Linux and macOS CI suites pass, then click **Create**.

---

## License

This project is licensed under the [MIT License](LICENSE).
