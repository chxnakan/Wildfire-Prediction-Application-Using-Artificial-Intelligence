"""GISTDA live-hotspot pipeline tests (client + store + poller + API helpers).

Run the complete suite:  cd src && ../.venv/bin/python -m pytest tests -q
All GISTDA responses are mocked — no network access required.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import db
import gistda_client as gc
import gistda_poller as gp


def _npp_feature(lat=18.5, lon=98.9, date_ms=1770854400000, time="07:35:00",
                 sat="N", conf="high", **extra):
    attrs = {
        "FID": 101, "latitude": lat, "longitude": lon,
        "date": date_ms, "time": time, "satellite": sat, "confident": conf,
        "lu_name": "ป่าอนุรักษ์", "pv_tn": "เชียงใหม่",
        "ap_tn": "แม่แจ่ม", "tb_tn": "ช่างเคิ่ง",
    }
    attrs.update(extra)
    return {"attributes": attrs}


def _modis_feature(lat=18.5, lon=98.9, dt_ms=1770881700000, sat="T", conf=75):
    return {"attributes": {
        "FID": 55, "latitude": lat, "longitude": lon,
        "datetime": dt_ms, "satellite": sat, "confident": conf,
        "lu_name": "เกษตร", "pv_tn": "เชียงใหม่", "ap_tn": "แม่แจ่ม",
        "tb_tb": "ช่างเคิ่ง",
    }}


def _recent_ms(hours_ago=1) -> int:
    return int((datetime.now(timezone.utc) - timedelta(hours=hours_ago)).timestamp() * 1000)


# ─── normalization + timezone handling ───────────────────────────────────────

def test_normalize_valid_npp_feature():
    rec = gc.normalize_feature(_npp_feature(), "hotspot_npp_daily")
    assert rec["source"] == "GISTDA"
    assert rec["latitude"] == 18.5 and rec["longitude"] == 98.9
    assert rec["confidence"] == 80.0                      # "high" → 80
    assert rec["province"] == "เชียงใหม่"
    assert rec["subdistrict"] == "ช่างเคิ่ง"
    # NPP `time` is UTC (verified vs SNPP overpass — see module docstring)
    assert rec["detected_at_utc"][11:19] == "07:35:00"
    assert rec["detected_at_thailand"][11:19] == "14:35:00"   # +07:00 display
    assert rec["detected_at_thailand"].endswith("+07:00")
    assert rec["time_precision"] == "second"
    utc = datetime.fromisoformat(rec["detected_at_utc"])
    th = datetime.fromisoformat(rec["detected_at_thailand"])
    assert utc == th                                       # same instant


def test_normalize_modis_datetime_layer():
    rec = gc.normalize_feature(_modis_feature(), "hotspot_daily")
    assert rec["time_precision"] == "second"               # epoch ms = full instant
    assert rec["subdistrict"] == "ช่างเคิ่ง"                # tb_tb typo handled
    assert rec["source_layer"] == "hotspot_daily"


def test_date_only_precision_never_invents_time():
    feat = _npp_feature(time="")                           # no time-of-day given
    rec = gc.normalize_feature(feat, "hotspot_npp_daily")
    assert rec["time_precision"] == "date_only"            # flagged, not invented
    assert rec["detected_at_utc"][11:19] == "00:00:00"     # date midnight UTC

    garbled = gc.normalize_feature(_npp_feature(time="bogus"), "hotspot_npp_daily")
    assert garbled["time_precision"] == "date_only"


def test_normalize_rejects_bad_coordinates():
    assert gc.normalize_feature(_npp_feature(lat=None), "hotspot_npp_daily") is None
    assert gc.normalize_feature(_npp_feature(lat="abc"), "hotspot_npp_daily") is None
    assert gc.normalize_feature(_npp_feature(lat=55.0), "hotspot_npp_daily") is None
    assert gc.normalize_feature(_npp_feature(lon=150.0), "hotspot_npp_daily") is None


def test_normalize_rejects_missing_timestamp():
    feat = _npp_feature()
    del feat["attributes"]["date"]
    assert gc.normalize_feature(feat, "hotspot_npp_daily") is None


def test_normalize_missing_optional_fields_stay_missing():
    feat = _npp_feature(confident=None, lu_name=None, pv_tn=None,
                        ap_tn=None, tb_tn=None)
    rec = gc.normalize_feature(feat, "hotspot_npp_daily")
    assert rec["confidence"] is None
    assert rec["province"] is None and rec["land_use"] is None


def test_confidence_out_of_range_rejected():
    rec = gc.normalize_feature(_npp_feature(confident=250), "hotspot_npp_daily")
    assert rec["confidence"] is None


# ─── identity: source_detection_id vs canonical event id ─────────────────────

def test_source_detection_id_layer_scoped():
    t = datetime(2026, 2, 12, 7, 35, tzinfo=timezone.utc)
    a = gc.source_detection_id("hotspot_npp_daily", "N", 18.5, 98.9, t)
    assert a == gc.source_detection_id("hotspot_npp_daily", "N", 18.5, 98.9, t)
    assert a != gc.source_detection_id("hotspot_daily", "N", 18.5, 98.9, t)
    assert a != gc.source_detection_id("hotspot_npp_daily", "T", 18.5, 98.9, t)


def test_event_id_independent_of_layer_and_satellite():
    """Canonical identity must NOT depend on which source reported it."""
    # Anchor at a cell CENTRE and a time-bucket start so the within-cell /
    # within-bucket probes below can't accidentally cross a boundary.
    lat = (int(18.5 / gc.CELL_LAT_DEG) + 0.5) * gc.CELL_LAT_DEG
    lon = (int(98.9 / gc.CELL_LON_DEG) + 0.5) * gc.CELL_LON_DEG
    t0 = datetime(2026, 2, 12, 7, 35, tzinfo=timezone.utc)
    bucket_start = datetime.fromtimestamp(
        (int(t0.timestamp()) // gc.EVENT_TIME_TOLERANCE_S) * gc.EVENT_TIME_TOLERANCE_S,
        tz=timezone.utc,
    )
    base = gc.event_id(lat, lon, bucket_start)
    assert base == gc.event_id(lat, lon, bucket_start)              # deterministic
    # nearby point in the same identity cell + same bucket → same event
    assert base == gc.event_id(lat + gc.CELL_LAT_DEG * 0.4, lon,
                               bucket_start + timedelta(seconds=60))
    # far point / different overpass → different event
    assert base != gc.event_id(lat + 1.0, lon, bucket_start)
    assert base != gc.event_id(lat, lon, bucket_start + timedelta(hours=6))


def test_partial_layer_availability_keeps_event_id_stable():
    """Regression for the duplicate-row bug: poll 1 sees only VIIRS, poll 2
    sees VIIRS + MODIS (same physical event, richer metadata). The canonical
    ID must be identical in both polls, whatever the response order."""
    npp = gc.normalize_feature(_npp_feature(), "hotspot_npp_daily")
    # same instant reported by the MODIS layer (epoch ms of 07:35 UTC)
    dt_ms = int(datetime(2026, 2, 12, 7, 36, tzinfo=timezone.utc).timestamp() * 1000)
    modis = gc.normalize_feature(
        _modis_feature(lat=18.5001, lon=98.9001, dt_ms=dt_ms), "hotspot_daily")

    poll1, _, _ = gc.dedupe_events([npp])                       # VIIRS only
    poll2, _, _ = gc.dedupe_events([modis, npp])                # both, reversed order
    assert poll1[0]["hotspot_id"] == [
        r for r in poll2 if r["hotspot_id"] == poll1[0]["hotspot_id"]
    ][0]["hotspot_id"]

    # merged record unions the contributions
    merged = [r for r in poll2 if r["hotspot_id"] == poll1[0]["hotspot_id"]][0]
    assert set(merged["detected_by"]) == {"N", "T"}
    assert set(merged["source_layers"]) == {"hotspot_npp_daily", "hotspot_daily"}
    assert len(merged["source_detection_ids"]) == 2


def test_repeated_poll_same_data_same_ids():
    feats = [_npp_feature(), _npp_feature(lat=19.0)]
    a = [gc.normalize_feature(f, "hotspot_npp_daily") for f in feats]
    b = [gc.normalize_feature(f, "hotspot_npp_daily") for f in feats]
    ids_a = sorted(r["hotspot_id"] for r in gc.dedupe_events(a)[0])
    ids_b = sorted(r["hotspot_id"] for r in gc.dedupe_events(b)[0])
    assert ids_a == ids_b


# ─── dedupe semantics ─────────────────────────────────────────────────────────

def test_dedupe_merges_same_cell_detections():
    a = gc.normalize_feature(_npp_feature(sat="N"), "hotspot_npp_daily")
    b = gc.normalize_feature(
        _npp_feature(lat=18.5001, lon=98.9001, sat="T", time="07:38:00"),
        "hotspot_npp_daily")
    if a["hotspot_id"] != b["hotspot_id"]:
        pytest.skip("fixture straddles an identity-cell edge — covered by "
                    "test_boundary_straddle_is_consistent")
    out, merged, _ = gc.dedupe_events([a, b])
    assert merged == 1 and len(out) == 1
    assert sorted(out[0]["detected_by"]) == ["N", "T"]


def test_distinct_overpasses_stay_separate():
    a = gc.normalize_feature(_npp_feature(time="07:35:00"), "hotspot_npp_daily")
    b = gc.normalize_feature(_npp_feature(time="13:40:00"), "hotspot_npp_daily")
    out, merged, _ = gc.dedupe_events([a, b])
    assert merged == 0 and len(out) == 2      # re-detection hours later = signal


def test_boundary_straddle_is_consistent_and_counted():
    """Two detections within EVENT_DISTANCE_M but in adjacent identity
    cells: the documented trade-off keeps them as TWO events — but always
    the SAME two events on every poll, and the near-split is measured."""
    lat_edge = (int(18.5 / gc.CELL_LAT_DEG) + 1) * gc.CELL_LAT_DEG
    just_below = lat_edge - gc.CELL_LAT_DEG * 0.05
    just_above = lat_edge + gc.CELL_LAT_DEG * 0.05
    a = gc.normalize_feature(_npp_feature(lat=just_below), "hotspot_npp_daily")
    b = gc.normalize_feature(_npp_feature(lat=just_above), "hotspot_npp_daily")
    assert gc.haversine_m(a["latitude"], a["longitude"],
                          b["latitude"], b["longitude"]) <= gc.EVENT_DISTANCE_M

    out1, _, splits1 = gc.dedupe_events([a, b])
    out2, _, splits2 = gc.dedupe_events([b, a])          # order flipped
    assert len(out1) == len(out2) == 2                    # conservative, stable
    assert splits1 == splits2 == 1                        # and observable
    assert sorted(r["hotspot_id"] for r in out1) == \
           sorted(r["hotspot_id"] for r in out2)


def test_haversine_sanity():
    # 1 degree of latitude ≈ 111.32 km
    assert abs(gc.haversine_m(18.0, 98.0, 19.0, 98.0) - 111_320) < 400


# ─── retry behaviour ──────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"features": []}
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_retry_on_5xx_then_success():
    responses = [_FakeResponse(503), _FakeResponse(503),
                 _FakeResponse(200, {"features": [_npp_feature()]})]
    sess = mock.Mock()
    sess.get = mock.Mock(side_effect=responses)
    feats = gc.query_layer("hotspot_npp_daily", session=sess, _sleep=lambda *a: None)
    assert len(feats) == 1 and sess.get.call_count == 3


def test_no_retry_on_permanent_4xx():
    sess = mock.Mock()
    sess.get = mock.Mock(return_value=_FakeResponse(404))
    with pytest.raises(gc.GistdaRequestError, match="permanent"):
        gc.query_layer("hotspot_npp_daily", session=sess, _sleep=lambda *a: None)
    assert sess.get.call_count == 1


def test_retry_on_429_respects_retry_after():
    sleeps = []
    responses = [_FakeResponse(429, headers={"Retry-After": "2"}),
                 _FakeResponse(200, {"features": []})]
    sess = mock.Mock()
    sess.get = mock.Mock(side_effect=responses)
    gc.query_layer("hotspot_npp_daily", session=sess,
                   _sleep=lambda attempt, ra=None: sleeps.append(ra))
    assert sleeps == ["2"]


def test_arcgis_inband_error_raises():
    sess = mock.Mock()
    sess.get = mock.Mock(return_value=_FakeResponse(200, {"error": {"code": 400}}))
    with pytest.raises(gc.GistdaRequestError, match="ArcGIS"):
        gc.query_layer("hotspot_npp_daily", session=sess, _sleep=lambda *a: None)


def test_partial_layer_failure_reported_not_fatal():
    """One layer up + one down → data flows AND the failure stays visible."""
    def fake_query(layer_key, session=None, **kw):
        if layer_key == "hotspot_daily":
            raise gc.GistdaRequestError("hotspot_daily: HTTP 503 (retries exhausted)")
        return [_npp_feature()]

    with mock.patch.object(gc, "query_layer", side_effect=fake_query):
        records, stats = gc.fetch_live_hotspots()
    assert len(records) == 1
    assert stats["layers_ok"] == ["hotspot_npp_daily"]
    assert stats["layers_failed"] == ["hotspot_daily"]
    assert "hotspot_daily" in stats["errors"]


# ─── store: enrichment upsert / retention / filters ──────────────────────────

def test_upsert_enrichment_union_on_repeat_polls():
    """Poll 2 brings a second satellite + fills missing admin fields —
    the stored row must be enriched, never duplicated or nulled."""
    first = gc.normalize_feature(
        _npp_feature(pv_tn=None, conf="nominal"), "hotspot_npp_daily")
    ins, upd, ids = db.upsert_live_hotspots([first])
    assert (ins, upd) == (1, 0)

    richer = dict(first)
    richer = gc.normalize_feature(_npp_feature(conf="high"), "hotspot_npp_daily")
    richer["hotspot_id"] = first["hotspot_id"]           # same physical event
    richer["detected_by"] = ["T"]
    richer["source_layers"] = ["hotspot_daily"]
    richer["source_detection_ids"] = ["gistda-det-xyz"]
    ins2, upd2, _ = db.upsert_live_hotspots([richer])
    assert (ins2, upd2) == (0, 1)

    rows = db.get_live_hotspots()
    assert len(rows) == 1
    row = rows[0]
    assert set(row["detected_by"]) == {"N", "T"}                        # union
    assert set(row["source_layers"]) == {"hotspot_npp_daily", "hotspot_daily"}
    assert row["province"] == "เชียงใหม่"        # null filled, then kept
    assert row["confidence"] == 80.0             # max(nominal 50, high 80)


def test_upsert_never_overwrites_with_null():
    first = gc.normalize_feature(_npp_feature(), "hotspot_npp_daily")
    db.upsert_live_hotspots([first])
    worse = gc.normalize_feature(
        _npp_feature(pv_tn=None, lu_name=None, confident=None), "hotspot_npp_daily")
    worse["hotspot_id"] = first["hotspot_id"]
    db.upsert_live_hotspots([worse])
    row = db.get_live_hotspots()[0]
    assert row["province"] == "เชียงใหม่"
    assert row["land_use"] == "ป่าอนุรักษ์"
    assert row["confidence"] == 80.0


def test_retention_cleanup():
    old = gc.normalize_feature(_npp_feature(), "hotspot_npp_daily")
    old["detected_at_utc"] = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    old["hotspot_id"] = "gistda-ev-old"
    fresh = gc.normalize_feature(
        _npp_feature(lat=19.0, date_ms=_recent_ms(), time=""), "hotspot_npp_daily")
    fresh["detected_at_utc"] = datetime.now(timezone.utc).isoformat()
    db.upsert_live_hotspots([old, fresh])
    assert db.cleanup_live_hotspots(retention_days=7) == 1
    remaining = db.get_live_hotspots()
    assert [r["hotspot_id"] for r in remaining] == [fresh["hotspot_id"]]


def test_query_filters():
    now = datetime.now(timezone.utc)
    a = gc.normalize_feature(_npp_feature(lat=18.5, sat="N"), "hotspot_npp_daily")
    b = gc.normalize_feature(
        _npp_feature(lat=15.0, lon=101.0, sat="T", pv_tn="ลพบุรี"),
        "hotspot_npp_daily")
    for r, ts in ((a, now), (b, now - timedelta(days=2))):
        r["detected_at_utc"] = ts.isoformat()
    db.upsert_live_hotspots([a, b])

    assert len(db.get_live_hotspots(province="ลพบุรี")) == 1
    assert len(db.get_live_hotspots(satellite="N")) == 1
    assert len(db.get_live_hotspots(since=(now - timedelta(days=1)).isoformat())) == 1
    assert len(db.get_live_hotspots(bbox=(100.0, 14.0, 102.0, 16.0))) == 1
    assert db.get_live_hotspots()[0]["hotspot_id"] == a["hotspot_id"]  # newest first


# ─── single-poller lock ───────────────────────────────────────────────────────

def test_advisory_lock_skips_second_holder():
    with db.advisory_lock(db.GISTDA_POLLER_LOCK_KEY) as first:
        assert first is True
        with db.advisory_lock(db.GISTDA_POLLER_LOCK_KEY) as second:
            assert second is False
    with db.advisory_lock(db.GISTDA_POLLER_LOCK_KEY) as third:
        assert third is True


def test_gistda_and_prediction_locks_are_independent():
    with db.advisory_lock(db.GISTDA_POLLER_LOCK_KEY) as a:
        with db.advisory_lock(db.PREDICTION_LOCK_KEY) as b:
            assert a is True and b is True


# ─── poller cycle: ok / partial / error / thailand filter ────────────────────

def _stats(n, layers_failed=None, errors=None):
    return {"raw_count": n, "normalized_count": n, "invalid_count": 0,
            "merged_count": 0, "near_split_count": 0,
            "errors": errors or {}, "layers_ok": ["hotspot_npp_daily"],
            "layers_failed": layers_failed or []}


def test_poll_once_ok_flow():
    gp._FIRST_POLL_DONE = False
    fresh = gc.normalize_feature(
        _npp_feature(date_ms=_recent_ms(), time=""), "hotspot_npp_daily")
    with mock.patch.object(gc, "fetch_live_hotspots",
                           return_value=([fresh], _stats(1))):
        out = gp.poll_once()
    assert out["status"] == "ok" and out["inserted"] == 1
    st = gp.status()
    assert st["hotspot_count"] == 1
    assert st["is_stale"] is False and st["last_error"] is None


def test_poll_once_partial_keeps_layer_error_visible():
    gp._FIRST_POLL_DONE = False
    fresh = gc.normalize_feature(
        _npp_feature(date_ms=_recent_ms(), time=""), "hotspot_npp_daily")
    stats = _stats(1, layers_failed=["hotspot_daily"],
                   errors={"hotspot_daily": "HTTP 503"})
    with mock.patch.object(gc, "fetch_live_hotspots", return_value=([fresh], stats)):
        out = gp.poll_once()
    assert out["status"] == "partial"
    assert out["layers_failed"] == ["hotspot_daily"]
    st = gp.status()
    assert st["is_stale"] is False              # partial still counts as fresh data
    assert "hotspot_daily" in (st["last_error"] or "")   # but the failure shows
    assert st["last_poll"]["layers_failed"] == "hotspot_daily"


def test_poll_failure_keeps_cache_and_reports_stale_error():
    gp._FIRST_POLL_DONE = False
    rec = gc.normalize_feature(_npp_feature(), "hotspot_npp_daily")
    db.upsert_live_hotspots([rec])
    with mock.patch.object(
        gc, "fetch_live_hotspots",
        side_effect=gc.GistdaRequestError("all GISTDA layers failed: HTTP 503"),
    ):
        out = gp.poll_once()
    assert out["status"] == "error"
    assert len(db.get_live_hotspots()) == 1     # cached data survives
    st = gp.status()
    assert st["last_error"] is not None and st["is_stale"] is True


def test_thailand_polygon_filter():
    inside = gc.normalize_feature(
        _npp_feature(lat=18.79, lon=98.98, date_ms=_recent_ms(), time=""),
        "hotspot_npp_daily")                    # Chiang Mai
    outside = gc.normalize_feature(
        _npp_feature(lat=21.5, lon=105.5, date_ms=_recent_ms(), time=""),
        "hotspot_npp_daily")                    # northern Vietnam — in bbox, not in TH
    kept, dropped = gp._filter_thailand([inside, outside])
    assert dropped == 1
    assert [r["hotspot_id"] for r in kept] == [inside["hotspot_id"]]


def test_first_poll_does_not_flood_sse_buffer():
    gp._FIRST_POLL_DONE = False
    with gp._NEW_LOCK:
        gp._NEW_BUFFER.clear()
    recs = [gc.normalize_feature(
                _npp_feature(lat=18.0 + i * 0.1, date_ms=_recent_ms(), time=""),
                "hotspot_npp_daily") for i in range(5)]
    with mock.patch.object(gc, "fetch_live_hotspots", return_value=(recs, _stats(5))):
        gp.poll_once()
    gen0, flood = gp.records_since(0)
    assert flood == []

    # 19.2N 99.5E = Phayao, comfortably inside the Thailand polygon (19.9N
    # 98.9E sits ON the Myanmar border and gets dropped by the country filter)
    newer = gc.normalize_feature(
        _npp_feature(lat=19.2, lon=99.5, date_ms=_recent_ms(), time=""),
        "hotspot_npp_daily")
    with mock.patch.object(gc, "fetch_live_hotspots",
                           return_value=(recs + [newer], _stats(6))):
        gp.poll_once()
    _, fresh = gp.records_since(gen0)
    assert [r["hotspot_id"] for r in fresh] == [newer["hotspot_id"]]


# ─── DB-backed SSE cursor (multi-replica event delivery) ─────────────────────

def test_sse_cursor_sees_events_regardless_of_writer():
    """Replica B's SSE reads events that replica A's poller inserted —
    modelled here as: writer inserts into the shared store, reader walks
    first_seen_at with its own cursor."""
    cursor = datetime.now(timezone.utc).isoformat()
    rec = gc.normalize_feature(
        _npp_feature(date_ms=_recent_ms(), time=""), "hotspot_npp_daily")
    db.upsert_live_hotspots([rec])              # "replica A" writes
    new = db.get_hotspots_first_seen_after(cursor)   # "replica B" reads
    assert [r["hotspot_id"] for r in new] == [rec["hotspot_id"]]

    # Cursor advances → no replay of the same event
    cursor2 = max(str(r["first_seen_at"]) for r in new)
    assert db.get_hotspots_first_seen_after(cursor2) == []


def test_sse_cursor_starts_at_connect_time_no_replay_flood():
    old = gc.normalize_feature(_npp_feature(), "hotspot_npp_daily")
    db.upsert_live_hotspots([old])              # historical event
    connect_cursor = datetime.now(timezone.utc).isoformat()
    assert db.get_hotspots_first_seen_after(connect_cursor) == []


# ─── API validation helpers ───────────────────────────────────────────────────

def _api():
    os.environ.setdefault("SCHEDULER_ENABLED", "false")
    import api
    return api


def test_bbox_validation_rejects_bad_input():
    from fastapi import HTTPException
    api = _api()
    assert api._parse_bbox_param(None) is None
    assert api._parse_bbox_param("96,4,107,22") == (96.0, 4.0, 107.0, 22.0)
    for bad in ("bad", "1,2,3", "1,2,3,nan", "96,4,107", "107,4,96,22",
                "96,22,107,4", "500,4,107,22"):
        with pytest.raises(HTTPException) as e:
            api._parse_bbox_param(bad)
        assert e.value.status_code == 422


def test_since_validation():
    from fastapi import HTTPException
    api = _api()
    assert api._parse_since_param(None) is None
    out = api._parse_since_param("2026-07-13T00:00:00+07:00")
    assert out == "2026-07-12T17:00:00+00:00"      # normalized to UTC
    for bad in ("yesterday", "2026-07-13T00:00:00"):   # naive tz rejected
        with pytest.raises(HTTPException) as e:
            api._parse_since_param(bad)
        assert e.value.status_code == 422


def test_legacy_attributes_conversion():
    api = _api()
    rec = gc.normalize_feature(_npp_feature(), "hotspot_npp_daily")
    attrs = api._live_to_legacy_attributes(rec)
    assert attrs["pv_tn"] == "เชียงใหม่"
    assert attrs["confident"] == 80.0
    assert isinstance(attrs["date"], int) and len(str(attrs["date"])) == 13
    ms = int(datetime.fromisoformat(rec["detected_at_utc"]).timestamp() * 1000)
    assert attrs["date"] == ms
    assert attrs["source"] == "GISTDA"
