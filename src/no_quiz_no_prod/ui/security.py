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
import ipaddress
import os
import re
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
_ALLOWED_SCHEMES: Final = frozenset({"http", "https"})
_HOST_LABEL_RE: Final = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

GuardRefusal = tuple[int, str, str]  # (HTTP status, machine code, human message)


def new_session_token() -> str:
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def _parse_port(port_str: str) -> int | None:
    if not port_str or not port_str.isascii() or not port_str.isdigit() or port_str.startswith("0"):
        return None
    port = int(port_str)
    return port if 1 <= port <= 65535 else None


def _parse_authority(authority: str) -> tuple[str, int | None] | None:
    """Strictly parse an HTTP ``Host`` or URL authority ``host[:port]`` / ``[ipv6][:port]``; fail closed on invalid."""
    if not authority or any(ord(ch) <= 0x20 or ord(ch) >= 0x7F for ch in authority):
        return None
    if any(ch in authority for ch in ("@", "/", "\\", "?", "#")):
        return None

    if authority.startswith("["):
        close = authority.find("]")
        if close == -1:
            return None
        ipv6_literal = authority[1:close]
        rest = authority[close + 1 :]
        if "%" in ipv6_literal or "[" in ipv6_literal:
            return None
        try:
            host = str(ipaddress.IPv6Address(ipv6_literal))
        except ValueError:
            return None
        if not rest:
            return host, None
        if not rest.startswith(":"):
            return None
        port = _parse_port(rest[1:])
        return (host, port) if port is not None else None

    if "[" in authority or "]" in authority:
        return None
    host_part, sep, port_str = authority.partition(":")
    port = None
    if sep:
        if ":" in port_str:
            return None
        port = _parse_port(port_str)
        if port is None:
            return None

    lowered = host_part.lower()
    if not lowered or len(lowered) > 253:
        return None
    labels = lowered.split(".")
    if any(not _HOST_LABEL_RE.match(label) for label in labels):
        return None
    if labels[-1].isdigit():
        try:
            return str(ipaddress.IPv4Address(lowered)), port
        except ValueError:
            return None
    return lowered, port


def hostname_of(authority: str) -> str | None:
    """The canonical lower-case host of a strict ``Host`` header or URL authority, or ``None`` if malformed."""
    parsed = _parse_authority(authority)
    return parsed[0] if parsed is not None else None


def normalize_allowed_hosts(hosts: Iterable[str]) -> tuple[str, ...]:
    """Normalize allow-list entries (`host[:port]`, `[ipv6][:port]`, or bare IPv6); discard empty/wildcard/invalid."""
    normalized: list[str] = []
    for raw in hosts:
        candidate = raw.strip()
        if not candidate or candidate.lower() in WILDCARD_BIND_HOSTS:
            continue
        host = hostname_of(candidate)
        if (
            host is None
            and not candidate.startswith("[")
            and candidate.count(":") >= 2
            and "%" not in candidate
            and all(0x21 <= ord(ch) <= 0x7E for ch in candidate)
        ):
            try:
                host = str(ipaddress.IPv6Address(candidate))
            except ValueError:
                host = None
        if host is not None and host not in WILDCARD_BIND_HOSTS:
            normalized.append(host)
    return tuple(dict.fromkeys(normalized))


def allowed_hosts_from(bind_host: str | None = None, extra: Iterable[str] = ()) -> tuple[str, ...]:
    """Loopback names, plus the address the server is bound to and any explicitly allowed host."""
    hosts = list(LOOPBACK_HOSTS)
    if bind_host is not None:
        hosts.append(bind_host)
    hosts.extend(extra)
    return normalize_allowed_hosts(hosts)


def allowed_hosts_from_env() -> tuple[str, ...]:
    return allowed_hosts_from(extra=os.environ.get(ALLOWED_HOSTS_ENV, "").split(","))


def _same_origin(origin: str, host_header: str, scheme: str) -> bool:
    if not origin or any(ord(ch) <= 0x20 or ord(ch) >= 0x7F for ch in origin):
        return False
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    expected_scheme = scheme.lower()
    if parts.scheme not in _ALLOWED_SCHEMES or parts.scheme != expected_scheme:
        return False
    if parts.path or parts.query or parts.fragment or parts.username is not None or parts.password is not None:
        return False
    if origin.lower() != f"{parts.scheme}://{parts.netloc.lower()}":
        return False
    origin_authority = _parse_authority(parts.netloc)
    request_authority = _parse_authority(host_header)
    return origin_authority is not None and origin_authority == request_authority


class StudioGuard:
    """Pure ASGI middleware applying the three checks before any route runs (streaming responses pass untouched)."""

    def __init__(self, app: ASGIApp, *, allowed_hosts: Sequence[str], session_token: str) -> None:
        if not session_token:
            raise ValueError("the Studio session token must not be empty")
        self.app = app
        self.allowed_hosts = frozenset(normalize_allowed_hosts(allowed_hosts))
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
        host_headers = headers.getlist("host")
        if len(host_headers) != 1 or not host_headers[0]:
            return 400, "invalid_host", "This Host is not allowed. Open the Studio on its loopback address."
        host_header = host_headers[0]
        parsed_host = hostname_of(host_header)
        if parsed_host is None or parsed_host not in self.allowed_hosts:
            return 400, "invalid_host", "This Host is not allowed. Open the Studio on its loopback address."
        if method in SAFE_METHODS:
            return None
        origins = headers.getlist("origin")
        if len(origins) != 1 or not _same_origin(origins[0], host_header, scheme):
            return 403, "forbidden_origin", "State-changing requests must come from the Studio page itself."
        tokens = headers.getlist(SESSION_TOKEN_HEADER)
        if len(tokens) != 1 or not hmac.compare_digest(tokens[0].encode(), self.session_token.encode()):
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
    "hostname_of",
    "new_session_token",
    "normalize_allowed_hosts",
    "same_origin_client_kwargs",
]
