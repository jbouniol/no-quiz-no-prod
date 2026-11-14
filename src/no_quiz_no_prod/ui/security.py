"""Request guard of the Studio server: Host allow-list, same-origin check and per-session token.

The Studio has no user accounts: it is a local tool that can write ``.env``, start LLM runs and approve a deploy.
Three checks keep a web page, a rebinding DNS name or another machine from driving it:

1. **Host allow-list** (every request): the ``Host`` header must name an allowed host, loopback names by default.
   A page served from ``attacker.example`` that resolves to 127.0.0.1 still sends ``Host: attacker.example``.
2. **Same origin** (every state-changing request): ``Origin`` must be present and equal to the scheme and host the
   request was sent to. Browsers always send ``Origin`` on POST and DELETE, and a page cannot forge it.
3. **Session token** (every state-changing request): a random token generated when the server starts, delivered
   only inside the Studio page and sent back in ``X-NQ-Session-Token``.

A missing header is a refusal, never a pass. Safe methods (GET, HEAD, OPTIONS) need only the Host check: they
change nothing, and without CORS headers another origin cannot read their answers.
"""

from __future__ import annotations

import hmac
import os
import secrets
from collections.abc import Iterable, Sequence
from typing import Any, Final
from urllib.parse import urlsplit

from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

SESSION_TOKEN_HEADER: Final = "X-NQ-Session-Token"
SESSION_TOKEN_PLACEHOLDER: Final = "__NQ_SESSION_TOKEN__"  # in index.html, replaced when the page is served
SESSION_TOKEN_BYTES: Final = 32
SAFE_METHODS: Final = frozenset({"GET", "HEAD", "OPTIONS"})
LOOPBACK_HOSTS: Final[tuple[str, ...]] = ("127.0.0.1", "localhost", "::1")
WILDCARD_BIND_HOSTS: Final = frozenset({"", "0.0.0.0", "::"})  # "listen everywhere": not a name a client can send
ALLOWED_HOSTS_ENV: Final = "NO_QUIZ_UI_ALLOWED_HOSTS"  # comma-separated extra hosts (set by ``no-quiz-ui --reload``)
DEFAULT_TEST_ORIGIN: Final = "http://127.0.0.1:8000"

GuardRefusal = tuple[int, str, str]  # (HTTP status, machine code, human message)


def new_session_token() -> str:
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def hostname_of(authority: str) -> str:
    """The lower-case host of a ``Host`` header or URL authority, without port or IPv6 brackets."""
    authority = authority.strip().lower()
    if authority.startswith("["):
        return authority[1:].partition("]")[0]
    if authority.count(":") > 1:  # a bare IPv6 literal such as "::1" (an allow-list entry, never a Host header)
        return authority
    return authority.partition(":")[0]


def allowed_hosts_from(bind_host: str | None = None, extra: Iterable[str] = ()) -> tuple[str, ...]:
    """Loopback names, plus the address the server is bound to and any explicitly allowed host."""
    hosts = list(LOOPBACK_HOSTS)
    if bind_host is not None and bind_host not in WILDCARD_BIND_HOSTS:
        hosts.append(bind_host)
    hosts.extend(extra)
    return tuple(dict.fromkeys(hostname_of(host) for host in hosts if host.strip()))


def allowed_hosts_from_env() -> tuple[str, ...]:
    return allowed_hosts_from(extra=os.environ.get(ALLOWED_HOSTS_ENV, "").split(","))


def _same_origin(origin: str, host_header: str, scheme: str) -> bool:
    parts = urlsplit(origin)
    return parts.scheme == scheme and parts.netloc.lower() == host_header.strip().lower()


class StudioGuard:
    """Pure ASGI middleware applying the three checks before any route runs (streaming responses pass untouched)."""

    def __init__(self, app: ASGIApp, *, allowed_hosts: Sequence[str], session_token: str) -> None:
        if not session_token:
            raise ValueError("the Studio session token must not be empty")
        self.app = app
        self.allowed_hosts = frozenset(allowed_hosts)
        self.session_token = session_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        refusal = self.refusal(scope["method"], Headers(scope=scope), scope.get("scheme", "http"))
        if refusal is None:
            await self.app(scope, receive, send)
            return
        status, code, message = refusal
        await JSONResponse({"ok": False, "code": code, "detail": message}, status_code=status)(scope, receive, send)

    def refusal(self, method: str, headers: Headers, scheme: str) -> GuardRefusal | None:
        """Why this request is refused, or None when it may proceed."""
        host_header = headers.get("host", "")
        if hostname_of(host_header) not in self.allowed_hosts:
            return 400, "invalid_host", "This Host is not allowed. Open the Studio on its loopback address."
        if method in SAFE_METHODS:
            return None
        origin = headers.get("origin")
        if origin is None or not _same_origin(origin, host_header, scheme):
            return 403, "forbidden_origin", "State-changing requests must come from the Studio page itself."
        token = headers.get(SESSION_TOKEN_HEADER, "")
        if not hmac.compare_digest(token.encode(), self.session_token.encode()):
            return 403, "invalid_session", "Missing or stale session token: reload the Studio page."
        return None


def same_origin_client_kwargs(app: FastAPI, origin: str = DEFAULT_TEST_ORIGIN) -> dict[str, Any]:
    """Keyword arguments making an in-process HTTP client behave like the Studio page of ``app``.

    ``TestClient(app, **same_origin_client_kwargs(app))`` sends an allowed Host, a matching Origin and the
    session token of this app instance, which is exactly what the page's own requests carry.
    """
    return {"base_url": origin, "headers": {"Origin": origin, SESSION_TOKEN_HEADER: app.state.session_token}}


__all__ = [
    "ALLOWED_HOSTS_ENV",
    "LOOPBACK_HOSTS",
    "SESSION_TOKEN_HEADER",
    "SESSION_TOKEN_PLACEHOLDER",
    "StudioGuard",
    "allowed_hosts_from",
    "allowed_hosts_from_env",
    "new_session_token",
    "same_origin_client_kwargs",
]
