"""Command-line interface (`no-quiz-no-prod`): interactive HITL runs, workspace reset, Studio UI, Notion & GitHub setup."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import httpx

from no_quiz_no_prod.backend.errors import WorkspaceError
from no_quiz_no_prod.backend.release import GitReleasePublisher, describe_release_remote, github_repo
from no_quiz_no_prod.backend.seeds import populate_area, seed_path
from no_quiz_no_prod.backend.store import workspace_lock
from no_quiz_no_prod.contracts import (
    ALL_QUIZ_OPTION_KEYS,
    ALL_TOPOLOGIES,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    DEFAULT_USER_ID,
    NOTION_IDS_FILE,
    DeployDecision,
    QuizAnswer,
    QuizOptionKey,
    RunEvent,
    RuntimeSettings,
    Topology,
)
from no_quiz_no_prod.ui.server import _load_default_workspace
from no_quiz_no_prod.ui.server import main as run_ui_server

NOTION_API_VERSION = "2022-06-28"
KEY_SHORTCUT_MAP: dict[str, QuizOptionKey] = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
}


def _load_workflow(
    settings: RuntimeSettings,
    *,
    topology: Topology,
    model: str | None,
    transport: Literal["stdio", "inprocess"],
) -> Any:
    try:
        from no_quiz_no_prod.graph.workflow import NoQuizWorkflow  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph workflow (`no_quiz_no_prod.graph.workflow`) is not available in this environment."
        ) from exc
    return NoQuizWorkflow(
        settings=settings,
        topology=topology,
        model=model,
        mcp_transport=transport,
    )


def _print_run_event(ev: RunEvent, *, out: Any = sys.stdout) -> None:
    if ev.type == "run_started":
        print(
            f"\n▶ Run started [thread={ev.thread_id} topology={ev.topology} model={ev.model}]",
            file=out,
        )
        return
    if ev.type == "step" and ev.step is not None:
        st = ev.step
        srv = f" · {st.server}" if st.server else ""
        ms = round(st.latency_s * 1000)
        mark = "✓" if st.ok and st.attempts <= 1 else ("↻" if st.ok else "✗")
        extras: list[str] = []
        if st.attempts > 1:
            extras.append(f"attempts={st.attempts}")
        if st.idempotency_key:
            extras.append(f"idem={st.idempotency_key[:8]}")
        if st.replayed:
            extras.append("replayed=True")
        if st.contains_injection:
            extras.append("injection_contained=True")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        print(f"  #{st.step:02d} [{st.agent}{srv}] {st.tool} {ms}ms {mark}{extra_str}", file=out)


def _prompt_quiz_choice(input_fn: Callable[[str], str], *, out: Any = sys.stdout) -> QuizOptionKey:
    while True:
        raw = input_fn("Select option [A/B/C/D or 1-4]: ").strip().upper()
        mapped = KEY_SHORTCUT_MAP.get(raw)
        if mapped in ALL_QUIZ_OPTION_KEYS:
            return mapped
        print("  Invalid choice — enter A, B, C, D (or 1, 2, 3, 4).", file=out)


def _cmd_run(
    args: argparse.Namespace,
    *,
    settings: RuntimeSettings | None = None,
    workflow: Any | None = None,
    input_fn: Callable[[str], str] = input,
    out: Any = sys.stdout,
) -> int:
    cfg = settings or RuntimeSettings.from_env()
    task = (args.task or "").strip()
    if not task:
        try:
            task = input_fn("Task prompt: ").strip()
        except EOFError:
            task = ""
    if not task:
        print("Error: a non-empty task prompt is required.", file=sys.stderr)
        return 2

    wf = workflow or _load_workflow(
        cfg,
        topology=args.topology,
        model=args.model,
        transport=args.transport,
    )
    thread_id = args.thread_id or f"cli_{uuid.uuid4().hex[:8]}"

    outcome = wf.start_run(
        task,
        thread_id=thread_id,
        user_id=args.user_id,
        max_steps=args.max_steps,
        on_event=lambda ev: _print_run_event(ev, out=out),
    )

    while True:
        if outcome.final_answer:
            print(f"\nAssistant:\n{outcome.final_answer}", file=out)

        if outcome.interrupt is not None:
            intr = outcome.interrupt
            if intr.kind == "quiz" and intr.quiz is not None:
                q = intr.quiz
                print(
                    f"\n=== Comprehension Quiz Gate (Attempt {q.attempt}/{q.max_attempts} · "
                    f"skill={q.skill_tag} · doc={q.doc_version}) ===",
                    file=out,
                )
                print(f"\n{q.stem}\n", file=out)
                for key in ALL_QUIZ_OPTION_KEYS:
                    print(f"  [{key}] {q.options[key]}", file=out)
                print("", file=out)
                try:
                    choice = _prompt_quiz_choice(input_fn, out=out)
                except EOFError:
                    print("Aborted (EOF during quiz prompt).", file=out)
                    return 1

                grade, outcome = wf.submit_quiz(
                    thread_id,
                    QuizAnswer(selected_option=choice),
                    user_id=args.user_id,
                    on_event=lambda ev: _print_run_event(ev, out=out),
                )
                verdict = "PASSED ✓" if grade.passed else "INCORRECT ✗"
                print(f"\nQuiz Result: {verdict} — {grade.explanation}", file=out)
                if grade.remediation_required:
                    print(
                        "→ Examiner upgraded Ship Log to Remediation v2 and prepared Attempt 2.",
                        file=out,
                    )
                continue

            if intr.kind == "deploy_approval" and intr.gate is not None:
                gate = intr.gate
                print(f"\n=== 4-Lock Production Gate [diff={gate.diff_sha256[:12]}] ===", file=out)
                print(f"  Lock 1 (tests_passed)   : {gate.tests_passed}", file=out)
                print(f"  Lock 2 (sha256_matches) : {gate.sha256_matches}", file=out)
                print(f"  Lock 3 (quiz_passed)    : {gate.quiz_passed}", file=out)
                print(f"  Lock 4 (human_approved) : {gate.human_approved} (awaiting sign-off)", file=out)
                try:
                    ans = input_fn("\nApprove production deployment? [y/N]: ").strip().lower()
                except EOFError:
                    ans = "n"
                action = "approve" if ans in ("y", "yes", "a", "approve") else "reject"
                decision = DeployDecision(
                    action=action,
                    expected_diff_sha256=gate.diff_sha256,
                    reason="CLI HITL decision",
                )
                outcome = wf.submit_deploy_decision(
                    thread_id,
                    decision,
                    user_id=args.user_id,
                    on_event=lambda ev: _print_run_event(ev, out=out),
                )
                continue

        if outcome.stop_reason == "clarification_needed":
            try:
                followup = input_fn("\nClarification reply (Enter to stop): ").strip()
            except EOFError:
                followup = ""
            if followup:
                outcome = wf.start_run(
                    followup,
                    thread_id=thread_id,
                    user_id=args.user_id,
                    max_steps=args.max_steps,
                    on_event=lambda ev: _print_run_event(ev, out=out),
                )
                continue

        break

    fs = outcome.final_state
    print(
        f"\n■ Run finished: stop_reason={outcome.stop_reason} | deployed={fs.deployed} "
        f"| ship_log={fs.ship_log_status} | steps={outcome.budget.steps}/{outcome.budget.max_steps} "
        f"| cost=${outcome.budget.cost_usd:.4f}",
        file=out,
    )
    return 0 if outcome.stop_reason != "api_error" else 1


def _cmd_reset(
    args: argparse.Namespace,
    *,
    settings: RuntimeSettings | None = None,
    workspace: Any | None = None,
    out: Any = sys.stdout,
) -> int:
    cfg = settings or RuntimeSettings.from_env()
    ws = workspace or _load_default_workspace(cfg)
    state = ws.reset(
        args.seed,
        preserve_skill_memory=not args.reset_skill_memory,
    )
    print(
        f"Workspace reset to seed '{args.seed}' "
        f"(prod_sha={state.prod_tree_sha256[:12]} staging_sha={state.staging_tree_sha256[:12]} "
        f"skill_memory_reset={args.reset_skill_memory}).",
        file=out,
    )
    return 0


def _verify_notion_database(api_key: str, db_id: str) -> tuple[bool, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": NOTION_API_VERSION,
    }
    try:
        resp = httpx.get(f"https://api.notion.com/v1/databases/{db_id}", headers=headers, timeout=10.0)
    except httpx.HTTPError as exc:
        return False, f"network error: {exc}"
    if resp.status_code == 200:
        data = resp.json()
        title_parts = data.get("title") or []
        title = "".join(part.get("plain_text", "") for part in title_parts) or db_id
        return True, title
    return False, f"HTTP {resp.status_code}"


def _cmd_init_notion(
    args: argparse.Namespace,
    *,
    settings: RuntimeSettings | None = None,
    out: Any = sys.stdout,
) -> int:
    ids_path = Path(args.ids_file)
    cfg = settings or RuntimeSettings.from_env(notion_ids_file=ids_path)

    if args.parent_page_id:
        if cfg.notion_api_key is None:
            print(
                "Error: NOTION_API_KEY is not set in .env; cannot provision live Notion databases.",
                file=sys.stderr,
            )
            return 1
        try:
            from no_quiz_no_prod.backend.notion_sync import (
                provision_notion_databases,  # type: ignore[import-not-found]
            )
        except ImportError as exc:
            print(f"Error: Notion sync module unavailable ({exc}).", file=sys.stderr)
            return 1

        provisioned = provision_notion_databases(
            args.parent_page_id,
            cfg.notion_api_key.get_secret_value(),
        )
        existing: dict[str, Any] = {}
        if ids_path.is_file():
            try:
                loaded = json.loads(ids_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing.update(loaded)
            except (OSError, ValueError):
                existing = {}
        existing.update(provisioned)
        existing.setdefault(
            "hub_url",
            f"https://www.notion.so/{args.parent_page_id.replace('-', '')}",
        )
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Provisioned Notion databases and saved IDs to {ids_path}:", file=out)
        for key, val in sorted(provisioned.items()):
            print(f"  {key}: {val}", file=out)
        return 0

    # Verification / inspection mode
    print("Notion Relational Hub Configuration Status:", file=out)
    print(f"  IDs file          : {ids_path} ({'present' if ids_path.is_file() else 'missing'})", file=out)
    print(
        f"  NOTION_API_KEY    : {'configured' if cfg.notion_api_key else 'not set (local SQLite fallback)'}", file=out
    )
    print(f"  rules_db_id       : {cfg.notion_rules_db_id or 'unset'}", file=out)
    print(f"  ship_log_db_id    : {cfg.notion_ship_log_db_id or 'unset'}", file=out)
    print(f"  skill_profile_db  : {cfg.notion_skills_db_id or 'unset'}", file=out)
    print(f"  notion_enabled    : {cfg.notion_enabled}", file=out)

    if args.verify and cfg.notion_enabled and cfg.notion_api_key is not None:
        key_val = cfg.notion_api_key.get_secret_value()
        all_ok = True
        for label, db_id in (
            ("Engineering Rules DB", cfg.notion_rules_db_id),
            ("Feature Ship Log DB", cfg.notion_ship_log_db_id),
            ("Developer Skill Profile DB", cfg.notion_skills_db_id),
        ):
            if db_id:
                ok, detail = _verify_notion_database(key_val, db_id)
                mark = "✓" if ok else "✗"
                print(f"  {mark} {label}: {detail}", file=out)
                all_ok = all_ok and ok
        return 0 if all_ok else 1

    return 0


def _print_git_failure(label: str, error: WorkspaceError, out: Any) -> None:
    """One failed check: the typed message, plus git's (already redacted) output when there is some."""
    print(f"  ✗ {label}: {error.payload.message}", file=out)
    if output := str(error.payload.details.get("git_output", "")).strip():
        print(f"      {output.splitlines()[-1]}", file=out)


def _cmd_init_github(
    args: argparse.Namespace,
    *,
    settings: RuntimeSettings | None = None,
    out: Any = sys.stdout,
) -> int:
    """Inspect or verify the release remote (``--verify``), or push the seed as the first commit of its branch.

    Never prints the token (only whether one is set) nor a remote URL that failed validation.
    """
    cfg = settings or RuntimeSettings.from_env()
    publisher = GitReleasePublisher(cfg)
    summary = describe_release_remote(cfg)
    print("GitHub Release Remote Configuration Status:", file=out)
    print(f"  remote_kind       : {summary['remote_kind']}", file=out)
    print(f"  repo              : {summary['repo'] or '-'}", file=out)
    print(f"  branch            : {summary['branch']}", file=out)
    print(f"  NO_QUIZ_GITHUB_TOKEN : {'configured' if summary['token_set'] else 'not set'}", file=out)

    try:
        version = publisher.git_version()
    except WorkspaceError as error:
        _print_git_failure("git >= 2.31", error, out)
        return 1
    print(f"  ✓ git {version}", file=out)
    try:
        remote = publisher.resolve_remote()
    except WorkspaceError as error:
        _print_git_failure("remote URL", error, out)
        return 1
    repo = github_repo(cfg.git_remote_url) if remote.kind == "https" else None
    print(f"  ✓ remote: {repo or remote.display} ({remote.kind})", file=out)
    if remote.kind == "https" and not summary["token_set"]:
        print("  ✗ token: NO_QUIZ_GITHUB_TOKEN is not set (an HTTPS push needs it)", file=out)
        return 1
    print(f"  ✓ token: {'configured' if remote.kind == 'https' else 'not needed for a local remote'}", file=out)

    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    if args.verify:
        try:
            with workspace_lock(cfg.workspace_dir):
                head = publisher.inspect_remote()
        except WorkspaceError as error:
            _print_git_failure("access (ls-remote)", error, out)
            return 1
        print("  ✓ access: ls-remote answered" if head.remote_exists else "  ✓ access: created on first use", file=out)
        print(f"  repo   : {repo or remote.display}", file=out)
        print(f"  branch : {head.branch}", file=out)
        print(f"  head   : {head.tip or 'absent (run `no-quiz-no-prod init-github` to create it)'}", file=out)
        return 0

    message = (
        f"chore(release): initialize the release branch with the {args.seed} seed\n\n"
        "Pristine production tree: every later commit on this branch is a gated deploy.\n"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="nqnp_seed_") as tmp, workspace_lock(cfg.workspace_dir):
            populate_area(Path(tmp), seed_path(args.seed), injection_target=None, injection_text="")
            head = publisher.initialize_branch(Path(tmp), message=message)
    except WorkspaceError as error:
        _print_git_failure("initialize branch", error, out)
        return 1
    if head.created:
        print(f"  ✓ pushed the {args.seed} seed as {head.tip} to {head.branch} (fast-forward)", file=out)
    else:
        print(f"  ✓ {head.branch} already exists at {head.tip}: left untouched (never forced)", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``no-quiz-no-prod`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="no-quiz-no-prod",
        description=(
            "No-Quiz-No-Prod: Pedagogical coding agent gating production behind staging pytest, "
            "adaptive comprehension quizzes, and a 4-lock SHA-256 gate."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. run
    run_p = sub.add_parser("run", help="Run an interactive terminal HITL coding & quiz session.")
    run_p.add_argument("task", nargs="?", default="", help="Developer task prompt.")
    run_p.add_argument(
        "--topology",
        choices=ALL_TOPOLOGIES,
        default="dual",
        help="Agent graph topology: 'dual' (Coder -> Examiner) or 'single' (default: dual).",
    )
    run_p.add_argument("--model", default=None, help="Override LLM model identifier.")
    run_p.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum LLM-initiated MCP tool calls (default: {DEFAULT_MAX_STEPS}).",
    )
    run_p.add_argument("--thread-id", default=None, help="Explicit thread_id for multi-turn continuation.")
    run_p.add_argument("--user-id", default=DEFAULT_USER_ID, help=f"Developer user_id (default: {DEFAULT_USER_ID}).")
    run_p.add_argument(
        "--transport",
        choices=("stdio", "inprocess"),
        default="stdio",
        help="MCP transport mode (default: stdio).",
    )

    # 2. reset
    reset_p = sub.add_parser("reset", help="Reset .runtime_workspace/ back to the pristine Fintech seed.")
    reset_p.add_argument("--seed", default=DEFAULT_SEED, help=f"Seed name under seeds/ (default: {DEFAULT_SEED}).")
    reset_p.add_argument(
        "--reset-skill-memory",
        action="store_true",
        help="Also erase Layer 4 Developer Skill Profile records.",
    )
    reset_p.add_argument("--user-id", default=DEFAULT_USER_ID, help=f"Developer user_id (default: {DEFAULT_USER_ID}).")

    # 3. ui
    ui_p = sub.add_parser("ui", help="Launch the FastAPI Conversational Web Studio & Trace Explorer.")
    ui_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    ui_p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    ui_p.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")

    # 4. init-notion
    notion_p = sub.add_parser(
        "init-notion",
        help="Verify Notion Relational Hub DB IDs or provision the 3 databases under a parent page.",
    )
    notion_p.add_argument(
        "--parent-page-id",
        default=None,
        help="Notion parent page UUID to provision fresh Rules, Ship Log, and Skill Profile databases.",
    )
    notion_p.add_argument(
        "--ids-file",
        default=str(NOTION_IDS_FILE),
        help="Path to docs/notion_workspace_ids.json.",
    )
    notion_p.add_argument(
        "--verify",
        action="store_true",
        help="Ping the Notion REST API to verify reachability of configured database IDs.",
    )

    # 5. init-github
    github_p = sub.add_parser(
        "init-github",
        help="Verify the release remote (--verify) or push the seed as the first commit of the release branch.",
    )
    github_p.add_argument(
        "--verify",
        action="store_true",
        help="Check git >= 2.31, the remote URL, the token and access (ls-remote); show repo, branch and head.",
    )
    github_p.add_argument(
        "--seed", default=DEFAULT_SEED, help=f"Seed pushed to a new branch (default: {DEFAULT_SEED})."
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``no-quiz-no-prod``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "reset":
        return _cmd_reset(args)
    if args.command == "ui":
        ui_args = ["--host", args.host, "--port", str(args.port)]
        if args.reload:
            ui_args.append("--reload")
        return run_ui_server(ui_args)
    if args.command == "init-notion":
        return _cmd_init_notion(args)
    if args.command == "init-github":
        return _cmd_init_github(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
