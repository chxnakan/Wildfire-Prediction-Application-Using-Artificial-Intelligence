import asyncio
import base64
import hashlib
import re
import unittest
from urllib.parse import urlparse

import src.api as api
from src.security import FRONTEND, headers_for


def get(path):
    """Call the ASGI app directly; returns the status and lower-cased response headers."""
    messages, requested = [], []

    async def receive():
        if requested:  # Later reads wait for a disconnect, like a client that stays connected.
            await asyncio.Event().wait()
        requested.append(True)
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": [(b"host", b"testserver")],
             "client": ("127.0.0.1", 1234), "server": ("testserver", 80)}
    asyncio.run(api.app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    return start["status"], {k.decode().lower(): v.decode() for k, v in start["headers"]}


def directives(policy):
    return {part.split()[0]: part.split()[1:] for part in policy.split("; ")}


INDEX = (FRONTEND / "index.html").read_text(encoding="utf-8")
EXTERNAL_TAGS = re.findall(r'<(?:script|link rel="stylesheet")[^>]+(?:src|href)="https://[^"]+"[^>]*>', INDEX)


class SecurityHeaderTests(unittest.TestCase):
    def test_every_response_carries_baseline_headers(self):
        for path in ("/", "/dashboard/", "/docs", "/no-such-route"):
            status, headers = get(path)
            self.assertIn(status, (200, 404), path)
            self.assertEqual(headers["x-content-type-options"], "nosniff", path)
            self.assertEqual(headers["x-frame-options"], "DENY", path)
            self.assertIn("frame-ancestors 'none'", headers["content-security-policy"], path)

    def test_dashboard_scripts_are_limited_to_known_sources(self):
        status, headers = get("/dashboard/")
        self.assertEqual(status, 200)
        self.assertEqual(headers["cache-control"], "no-cache")
        scripts = directives(headers["content-security-policy"])["script-src"]
        self.assertNotIn("'unsafe-inline'", scripts)
        self.assertNotIn("'unsafe-eval'", scripts)
        for body in re.findall(r"<script>(.*?)</script>", INDEX, re.S):
            digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
            self.assertIn(f"'sha256-{digest}'", scripts)

    def test_json_api_allows_nothing_to_load(self):
        self.assertTrue(headers_for("/metadata")["Content-Security-Policy"].startswith("default-src 'none'"))
        self.assertTrue(headers_for("/forecasts/x/geojson")["Content-Security-Policy"].startswith("default-src 'none'"))

    def test_cdn_files_are_pinned_by_integrity_hash(self):
        self.assertTrue(EXTERNAL_TAGS)
        for tag in EXTERNAL_TAGS:
            if "fonts.googleapis.com" in tag:
                continue  # Google Fonts CSS varies by browser, so it cannot carry a fixed hash.
            self.assertRegex(tag, r'integrity="sha(256|384|512)-[A-Za-z0-9+/=]+"', tag)
            self.assertIn('crossorigin="anonymous"', tag)

    def test_cdn_origins_are_allowed_by_the_policy(self):
        policy = directives(headers_for("/dashboard/")["Content-Security-Policy"])
        for tag in EXTERNAL_TAGS:
            url = re.search(r'(?:src|href)="(https://[^"]+)"', tag).group(1)
            origin = f"https://{urlparse(url).netloc}"
            directive = "script-src" if tag.startswith("<script") else "style-src"
            self.assertIn(origin, policy[directive], url)

    def test_forecast_names_cannot_escape_the_forecast_folder(self):
        for name in ("../metadata/dataset_info.json", "..%2Fsecret.json", "/etc/passwd"):
            with self.assertRaises(api.HTTPException) as caught:
                api.snapshot_path(name)
            self.assertEqual(caught.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
