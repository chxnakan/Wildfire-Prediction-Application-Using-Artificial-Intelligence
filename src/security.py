"""HTTP security headers for the dashboard, the API documentation and the JSON API."""

import base64
import hashlib
import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
INLINE_SCRIPT = re.compile(r"<script>(.*?)</script>", re.S)

COMMON = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
LOCKDOWN = "base-uri 'self'; object-src 'none'; frame-ancestors 'none'"
CDNS = "https://unpkg.com https://cdn.jsdelivr.net"
# Every third-party origin the dashboard uses: CDN libraries, Google Fonts,
# OpenFreeMap vector tiles (with an OpenStreetMap raster fallback) and Open-Meteo.
DASHBOARD = {
    "default-src": "'self'",
    "style-src": f"'self' 'unsafe-inline' https://fonts.googleapis.com {CDNS}",
    "font-src": "'self' https://fonts.gstatic.com",
    "img-src": "'self' data: blob: https://unpkg.com https://tile.openstreetmap.org",
    "connect-src": "'self' https://tiles.openfreemap.org https://api.open-meteo.com https://archive-api.open-meteo.com",
    "worker-src": "'self' blob:",
    "child-src": "blob:",
    "form-action": "'self'",
}
# FastAPI's /docs and /redoc pages are generated with inline bootstrap scripts.
DOCS = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data: https://fastapi.tiangolo.com "
        "https://cdn.redoc.ly; worker-src 'self' blob:; " + LOCKDOWN)
API = "default-src 'none'; " + LOCKDOWN
DOCS_PATHS = ("/docs", "/redoc")
_script_hashes = {}


def inline_script_hashes(index=FRONTEND / "index.html"):
    """Allow exactly the inline scripts shipped in index.html, recomputed when it changes."""
    try:
        stamp = index.stat().st_mtime_ns
    except FileNotFoundError:
        return ""
    if _script_hashes.get("stamp") != stamp:
        bodies = INLINE_SCRIPT.findall(index.read_text(encoding="utf-8"))
        digests = (base64.b64encode(hashlib.sha256(body.encode()).digest()).decode() for body in bodies)
        _script_hashes.update(stamp=stamp, value=" ".join(f"'sha256-{d}'" for d in digests))
    return _script_hashes["value"]


def dashboard_policy():
    scripts = " ".join(filter(None, ["'self'", inline_script_hashes(), CDNS]))
    directives = {**DASHBOARD, "script-src": scripts}
    return "; ".join(f"{name} {value}" for name, value in directives.items()) + "; " + LOCKDOWN


def headers_for(path):
    if path.startswith("/dashboard"):
        policy = dashboard_policy()
    elif path.startswith(DOCS_PATHS):
        policy = DOCS
    else:
        policy = API
    return {**COMMON, "Content-Security-Policy": policy}
