"""Production safety: environment gating, CORS allowlist, rate limiting,
request-ID + structured JSON logging, and security headers (incl. a
Leaflet-compatible CSP).

All behaviour is environment-driven so the same image runs locally (open,
convenient) and in production (fail-closed). ``APP_ENV=production`` is the
single switch that flips the strict defaults on.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_log = logging.getLogger("fire-date-api.access")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV in ("production", "prod")


# ── CORS allowlist ───────────────────────────────────────────────────────────

def cors_allow_origins() -> List[str]:
    """Explicit origin allowlist. In production an empty CORS_ALLOW_ORIGINS
    fails closed (no cross-origin browser access); dev defaults to localhost."""
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    if IS_PRODUCTION:
        return []
    return [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ]


# ── Content-Security-Policy (Leaflet + configured tile host) ─────────────────

_DEFAULT_TILE_HOSTS = "https://*.basemaps.cartocdn.com"
# The map popup falls back to Open-Meteo when the local ERA5 cache is absent
# or too old.  The SPA is served with this CSP in production, so omitting
# these hosts makes the browser reject that fallback before it reaches the
# network (and the UI misleadingly reports "Weather load failed").
_DEFAULT_CONNECT_HOSTS = (
    "https://api.open-meteo.com https://archive-api.open-meteo.com"
)


def content_security_policy() -> str:
    tiles = os.getenv("CSP_TILE_HOSTS", _DEFAULT_TILE_HOSTS).strip()
    extra_connect = os.getenv("CSP_CONNECT_SRC", "").strip()
    connect_hosts = " ".join(
        part for part in (_DEFAULT_CONNECT_HOSTS, extra_connect) if part
    )
    connect = f"'self' {connect_hosts}".strip()
    return "; ".join([
        "default-src 'self'",
        f"img-src 'self' data: blob: {tiles}",
        "style-src 'self' 'unsafe-inline'",          # Leaflet injects inline styles
        "script-src 'self'",
        f"connect-src {connect}",
        "font-src 'self' data:",
        "worker-src 'self' blob:",
        "object-src 'none'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
    ])


def documentation_content_security_policy() -> str:
    """Narrow CSP for FastAPI's generated Swagger/ReDoc pages.

    FastAPI's default documentation HTML loads versioned bundles from
    jsDelivr and contains a small inline bootstrap script. Applying the SPA's
    stricter ``script-src 'self'`` made /docs and /redoc return HTTP 200 while
    rendering as blank pages. Keep these allowances scoped to the exact docs
    routes; the dashboard and API responses retain the strict default policy.
    """
    return "; ".join([
        "default-src 'self'",
        (
            "img-src 'self' data: "
            "https://fastapi.tiangolo.com"
        ),
        (
            "style-src 'self' 'unsafe-inline' "
            "https://cdn.jsdelivr.net https://fonts.googleapis.com"
        ),
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        "connect-src 'self'",
        "font-src 'self' data: https://fonts.gstatic.com",
        "object-src 'none'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
    ])


# ── In-process rate limiter (best-effort; per replica) ───────────────────────

# Cap the number of distinct client keys held in memory so a flood of spoofed
# / rotating source IPs cannot grow the limiter unboundedly. When exceeded, the
# oldest-touched keys are evicted (they simply get a fresh window next time).
_MAX_LIMITER_KEYS = int(os.getenv("RATE_LIMIT_MAX_KEYS", "50000"))


class RateLimiter:
    """Sliding-window limiter keyed by (client, route-class). In-process, so
    limits are PER REPLICA — best-effort only; use a shared limiter (Redis) for
    strict multi-replica enforcement. The key store is bounded and self-expiring
    so spoofed source IPs cannot grow memory without limit."""

    def __init__(self, max_keys: int = _MAX_LIMITER_KEYS) -> None:
        # OrderedDict as an LRU: most-recently-touched key moves to the end.
        self._hits: "OrderedDict[str, Deque[float]]" = OrderedDict()
        self._max_keys = max(10, max_keys)

    def allow(self, key: str, limit: int, window_s: float) -> bool:
        now = time.monotonic()
        dq = self._hits.get(key)
        if dq is None:
            dq = deque()
            self._hits[key] = dq
        else:
            self._hits.move_to_end(key)
        cutoff = now - window_s
        while dq and dq[0] <= cutoff:
            dq.popleft()
        # Bound the store: evict least-recently-used keys when over capacity
        # (spoofed/rotating source IPs otherwise grow memory without limit). An
        # evicted key simply starts a fresh window next time it is seen — an
        # acceptable trade for a hard memory ceiling. Never evict the key we are
        # currently servicing (it was just moved to the end).
        while len(self._hits) > self._max_keys and next(iter(self._hits)) != key:
            self._hits.popitem(last=False)
        if len(dq) >= limit:
            return False
        dq.append(now)
        return True


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def route_rate_rule(path: str) -> Optional[Tuple[str, int, int]]:
    """(route-class, limit, window_s) for a path, or None to skip limiting
    (health checks + static assets are never limited)."""
    if path.startswith("/api/admin") or path.startswith("/api/notify"):
        return ("admin", _int_env("RATE_LIMIT_ADMIN", 20), 60)
    if path.startswith("/api/fires/stream"):
        return ("stream", _int_env("RATE_LIMIT_STREAM", 10), 60)
    if (path.startswith("/api") or path.startswith("/predict")
            or path.startswith("/live-hotspots")
            or path in ("/geojson", "/metrics", "/metadata")):
        return ("public", _int_env("RATE_LIMIT_PUBLIC", 120), 60)
    return None


def _trusted_proxies() -> set:
    """Peer IPs whose X-Forwarded-For header we trust (the platform's load
    balancer). Env RATE_LIMIT_TRUSTED_PROXIES (comma-separated). Empty ⇒ trust
    none — never believe a client-supplied XFF from an untrusted peer."""
    raw = os.getenv("RATE_LIMIT_TRUSTED_PROXIES", "").strip()
    return {p.strip() for p in raw.split(",") if p.strip()}


def _client_ip(request: Request) -> str:
    """Real client IP for rate limiting. X-Forwarded-For is honoured ONLY when
    the direct peer is a configured trusted proxy — otherwise an attacker could
    spoof the header to dodge or poison limits. Falls back to the socket peer."""
    peer = request.client.host if request.client else "unknown"
    if peer in _trusted_proxies():
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # left-most entry is the original client per the platform LB
            return xff.split(",")[0].strip() or peer
    return peer


class SecurityMiddleware(BaseHTTPMiddleware):
    """One pass per request: assign a request ID, enforce the rate limit, time
    the handler, emit a structured JSON access log, and stamp security headers
    (including the CSP) onto the response."""

    def __init__(self, app):
        super().__init__(app)
        self._limiter = RateLimiter()
        self._csp = content_security_policy()
        self._docs_csp = documentation_content_security_policy()

    async def dispatch(self, request: Request, call_next):
        supplied_rid = (request.headers.get("x-request-id") or "").strip()
        # Request IDs are reflected in response headers and structured logs.
        # Accept a small safe character set only; otherwise generate our own
        # so an untrusted header cannot inject controls or unbounded log data.
        rid = (
            supplied_rid
            if _REQUEST_ID_RE.fullmatch(supplied_rid)
            else uuid.uuid4().hex[:16]
        )
        request.state.request_id = rid
        started = time.monotonic()
        path = request.url.path

        rule = route_rate_rule(path)
        if rule is not None:
            cls, limit, window = rule
            key = f"{_client_ip(request)}|{cls}"
            if not self._limiter.allow(key, limit, window):
                _log.warning(json.dumps({
                    "event": "rate_limited", "request_id": rid,
                    "path": path, "class": cls, "limit": limit,
                }))
                resp = JSONResponse(
                    status_code=429,
                    content={"error": "rate_limited",
                             "detail": f"Too many requests ({cls}). Retry shortly.",
                             "request_id": rid},
                )
                self._apply_headers(resp, rid, path)
                resp.headers["Retry-After"] = str(window)
                return resp

        try:
            response = await call_next(request)
        except Exception:
            _log.error(json.dumps({
                "event": "request_error", "request_id": rid,
                "method": request.method, "path": path,
            }))
            raise
        dur_ms = round((time.monotonic() - started) * 1000, 1)
        _log.info(json.dumps({
            "event": "request", "request_id": rid,
            "method": request.method, "path": path,
            "status": response.status_code, "duration_ms": dur_ms,
        }))
        self._apply_headers(response, rid, path)
        return response

    def _apply_headers(self, response, rid: str, path: str = "") -> None:
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            self._docs_csp
            if path in {"/docs", "/redoc", "/docs/oauth2-redirect"}
            else self._csp
        )
        if IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = \
                "max-age=31536000; includeSubDomains"


def configure_security(app: FastAPI) -> None:
    """Attach CORS (allowlist) + the security/rate-limit/logging middleware.
    Call once, replacing any wildcard CORS."""
    origins = cors_allow_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=bool(origins),   # credentials require explicit origins
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityMiddleware)
    if IS_PRODUCTION and not origins:
        logging.getLogger("fire-date-api").warning(
            "APP_ENV=production but CORS_ALLOW_ORIGINS is empty — cross-origin "
            "browser access is disabled (fail-closed).")
