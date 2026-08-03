"""Phase 1 production-safety: auth fail-closed, notify gating + PII, CORS
allowlist, rate limiting, health split, security headers, Thailand bounds.

Hermetic — no real model load, no network. Run with the suite:
    cd src && ../.venv/bin/python -m pytest tests -q
"""

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import api
    with mock.patch.object(api, "_build_model_state", return_value=None):
        with TestClient(api.app) as c:
            yield c


# ── CORS allowlist (unit) ────────────────────────────────────────────────────

def test_cors_fails_closed_in_production(monkeypatch):
    import security
    monkeypatch.setattr(security, "IS_PRODUCTION", True)
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    assert security.cors_allow_origins() == []          # no wildcard, no origins


def test_cors_reads_env_allowlist(monkeypatch):
    import security
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://a.example, https://b.example")
    assert security.cors_allow_origins() == ["https://a.example", "https://b.example"]


def test_dev_cors_defaults_to_localhost(monkeypatch):
    import security
    monkeypatch.setattr(security, "IS_PRODUCTION", False)
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    assert any("localhost" in o for o in security.cors_allow_origins())


# ── security headers + request id ────────────────────────────────────────────

def test_security_headers_and_request_id(client):
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert "basemaps.cartocdn.com" in r.headers["Content-Security-Policy"]
    assert r.headers.get("X-Request-ID")


def test_api_docs_have_scoped_csp_for_fastapi_bundles(client):
    docs = client.get("/docs")
    assert docs.status_code == 200
    docs_csp = docs.headers["Content-Security-Policy"]
    assert "https://cdn.jsdelivr.net" in docs_csp
    assert "script-src 'self' 'unsafe-inline'" in docs_csp

    # The external-script exception must not leak onto ordinary app/API routes.
    health_csp = client.get("/health/live").headers["Content-Security-Policy"]
    assert "https://cdn.jsdelivr.net" not in health_csp
    assert "script-src 'self' 'unsafe-inline'" not in health_csp


# ── health split ─────────────────────────────────────────────────────────────

def test_health_live_always_ok(client):
    assert client.get("/health/live").json() == {"status": "alive"}


def test_health_ready_503_when_degraded(client):
    import api
    api._apply_model_state(client.app, dict(api._DEGRADED_STATE))
    r = client.get("/health/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "not_ready"
    assert body["checks"]["model_loaded"] is False


# ── admin auth: fail-closed in prod ──────────────────────────────────────────

def test_admin_fails_closed_in_prod_without_token(client, monkeypatch):
    import api, security
    monkeypatch.setattr(security, "IS_PRODUCTION", True)
    monkeypatch.setattr(api, "ADMIN_TOKEN", "")
    r = client.post("/api/admin/reload-model")
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "admin_disabled"


def test_admin_rejects_wrong_token(client, monkeypatch):
    import api, security
    monkeypatch.setattr(security, "IS_PRODUCTION", True)
    monkeypatch.setattr(api, "ADMIN_TOKEN", "s3cret")
    r = client.post("/api/admin/reload-model", headers={"X-Admin-Token": "nope"})
    assert r.status_code == 403


# ── notify: gated + no PII previews ──────────────────────────────────────────

def test_notify_gated_in_prod(client, monkeypatch):
    import api, security
    monkeypatch.setattr(security, "IS_PRODUCTION", True)
    monkeypatch.setattr(api, "ADMIN_TOKEN", "")
    r = client.post("/api/notify", json={
        "channel": "sms", "recipients": ["+66123"], "message": "hi"})
    assert r.status_code == 503


def test_notify_log_never_exposes_recipients_or_message(client, monkeypatch):
    import api, security
    monkeypatch.setattr(security, "IS_PRODUCTION", False)   # dev: open
    monkeypatch.setattr(api, "ADMIN_TOKEN", "")
    client.post("/api/notify", json={
        "channel": "email", "recipients": ["a@b.com", "c@d.com"],
        "message": "secret body"})
    rec = client.get("/api/notify/log").json()["records"][0]
    assert rec["recipients_count"] == 2
    assert "recipients_preview" not in rec
    assert "message_preview" not in rec
    assert "secret body" not in str(rec)


# ── Thailand-bounds validation ───────────────────────────────────────────────

def test_predict_location_rejects_out_of_country(client):
    r = client.get("/predict/location", params={"lat": 48.85, "lon": 2.35})  # Paris
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "coordinates_out_of_bounds"


# ── rate limiting (isolated bucket via unique X-Forwarded-For) ───────────────

def test_public_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PUBLIC", "3")
    monkeypatch.setenv("RATE_LIMIT_TRUSTED_PROXIES", "testclient")  # trust the test peer
    hdr = {"X-Forwarded-For": "203.0.113.77"}   # unique IP → fresh bucket
    codes = [client.get("/metrics", headers=hdr).status_code for _ in range(5)]
    assert 429 in codes
    assert codes.count(429) >= 1 and codes[0] != 429


# ── X-Forwarded-For trust + bounded store (finding 14) ───────────────────────

class _FakeReq:
    def __init__(self, peer, xff=None):
        self.client = type("C", (), {"host": peer})()
        self.headers = {"x-forwarded-for": xff} if xff else {}


def test_xff_ignored_from_untrusted_peer(monkeypatch):
    import security
    monkeypatch.delenv("RATE_LIMIT_TRUSTED_PROXIES", raising=False)
    # peer is not a trusted proxy → spoofed XFF must be ignored
    assert security._client_ip(_FakeReq("198.51.100.9", xff="1.2.3.4")) == "198.51.100.9"


def test_xff_honoured_only_from_trusted_proxy(monkeypatch):
    import security
    monkeypatch.setenv("RATE_LIMIT_TRUSTED_PROXIES", "10.0.0.1")
    assert security._client_ip(_FakeReq("10.0.0.1", xff="1.2.3.4")) == "1.2.3.4"
    assert security._client_ip(_FakeReq("10.0.0.2", xff="1.2.3.4")) == "10.0.0.2"


def test_rate_limiter_key_store_is_bounded():
    import security
    rl = security.RateLimiter(max_keys=100)
    for i in range(1000):                       # 1000 distinct spoofed keys
        rl.allow(f"ip-{i}|public", limit=100, window_s=60)
    assert len(rl._hits) <= 100                 # bounded, no unbounded growth
