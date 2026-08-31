"""GeoJSON provenance and map-selection trust contracts."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

import risk_map


class _TwoCellModel:
    def predict_proba(self, _features):
        return np.array([0.9, 0.2])

    def predict(self, _features):
        return np.array([0.2, 4.2])


class _SignalModel:
    def predict_proba(self, features):
        return features["signal"].to_numpy(dtype=float)

    def predict(self, features):
        # Deliberately put scores into different compatibility buckets. The
        # binary snapshot cap must ignore these pseudo-day values.
        return np.where(features["signal"].to_numpy() >= 0.8, 6.2, 2.2)


class _PlateauSignalModel:
    def _raw_proba(self, features):
        return features["signal"].to_numpy(dtype=float)

    def predict_proba(self, features):
        return np.full(len(features), 0.01)

    def predict(self, features):
        return np.full(len(features), 6.5)


def _predicted_frame(base_date: date) -> pd.DataFrame:
    long_col = f"historical_fire_count_{risk_map.LONG_HISTORY_WINDOW_DAYS}d"
    frame = pd.DataFrame([{
        "lat_grid": 18.5,
        "lon_grid": 98.5,
        "days_until_fire": 0,
        "predicted_fire_date": "2026-08-08",
        "urgency_level": "MEDIUM",
        "ranking_tier": "WATCH",
        "rank": 1,
        "percentile": 1.0,
        "prediction_confidence": 0.45,
        "raw_prediction": 0.2,
        "probability": 0.45,
        "historical_fire_count_30d": 4,
        long_col: 12,
        "fire_days_total": 20,
        "fire_days_per_year": 5.0,
        "nearest_urban_area": "Chiang Mai",
        "nearest_urban_distance_km": 30.0,
        "province": "Chiang Mai",
        "tree_cover_pct_2000": 40.0,
        "tree_loss_pct_recent": 2.0,
    }])
    frame.attrs["thresholds_source"] = (
        "absolute calibrated probability; cutoffs selected on calibration split"
    )
    frame.attrs["background_context_omitted"] = 3
    frame.attrs["n_suppressed_by_cap"] = 3
    frame.attrs["selection_policy"] = {
        "total_scored_cells": 100,
        "retained_map_candidate_cells": 1,
        "selection_changes_probability": False,
    }
    return frame


def test_threshold_copy_labels_calibration_statistics_explicitly():
    copy = risk_map._probability_thresholds_source({
        "calib_precision_at_high": 0.66,
        "calib_recall_at_high": 0.77,
        "calib_recall_target_medium": 0.9,
    })

    assert "calibration precision=0.66" in copy
    assert "calibration recall=0.77" in copy
    assert "calibration recall target≥0.9" in copy
    assert "test" not in copy.lower()


def test_critical_metadata_is_ignored_until_api_and_training_share_contract():
    cells = pd.DataFrame({"probability": [0.95, 0.8, 0.5, 0.1]})
    meta = {
        "probability_tier_thresholds": {
            "CRITICAL": 0.9,
            "HIGH": 0.7,
            "WATCH": 0.4,
        }
    }

    tiered, thresholds, _source = risk_map._assign_urgency_tiers(cells, meta)

    assert tiered["urgency_level"].tolist() == [
        "HIGH", "HIGH", "MEDIUM", "LOW",
    ]
    assert thresholds["CRITICAL"] > 1.0

    without_critical, thresholds, _source = risk_map._assign_urgency_tiers(
        cells,
        {"probability_tier_thresholds": {"HIGH": 0.7, "WATCH": 0.4}},
    )
    assert without_critical["urgency_level"].tolist() == [
        "HIGH", "HIGH", "MEDIUM", "LOW",
    ]
    assert thresholds["CRITICAL"] > 1.0


def test_legacy_artifact_fails_closed_to_low_and_keeps_relative_ranking():
    cells = pd.DataFrame({
        "probability": np.linspace(0.99, 0.01, 10),
        "lat_grid": np.arange(10, dtype=float),
        "lon_grid": np.full(10, 100.0),
    })

    tiered, thresholds, source = risk_map._assign_urgency_tiers(cells, {})

    assert set(tiered["urgency_level"]) == {"LOW"}
    assert thresholds["HIGH"] > 1.0
    assert thresholds["WATCH"] > 1.0
    assert "fails closed to LOW" in source

    import inference
    ranked = inference.assign_operational_ranking(
        tiered,
        high_threshold=thresholds["HIGH"],
    )
    assert "ABSOLUTE_HIGH" not in set(ranked["ranking_tier"])
    assert set(ranked["ranking_tier"]) <= {
        "RELATIVE_TOP_10", "RELATIVE_TOP_20", "UNRANKED",
    }


def test_observed_layer_excludes_foreign_bbox_rows_in_thailand_mode(monkeypatch):
    observed = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-05"] * 3),
        "lat_grid": [18.5, 18.6, 18.7],
        "lon_grid": [98.5, 101.0, 104.0],
        "fire_count": [2, 0, 1],
    })
    monkeypatch.setattr(risk_map, "COUNTRY_FILTER_ENABLED", True)
    # Of the two positive rows, only the first lies inside Thailand.
    monkeypatch.setattr(
        risk_map,
        "is_in_thailand",
        lambda _lat, _lon: np.array([True, False]),
    )

    scoped, provenance = risk_map._scope_observed_rows(observed)

    assert scoped[["lat_grid", "lon_grid"]].to_dict("records") == [
        {"lat_grid": 18.5, "lon_grid": 98.5}
    ]
    assert provenance == {
        "scope": "Thailand",
        "country_filter_enabled": True,
        "positive_source_rows": 2,
        "emitted_rows": 1,
        "excluded_outside_scope": 1,
        "note": (
            "Regional observations remain available through transboundary analytics."
        ),
    }


def test_observed_layer_reports_regional_scope_when_filter_disabled(monkeypatch):
    observed = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-05", "2026-08-05"]),
        "lat_grid": [18.5, 18.7],
        "lon_grid": [98.5, 104.0],
        "fire_count": [2, 1],
    })
    monkeypatch.setattr(risk_map, "COUNTRY_FILTER_ENABLED", False)

    scoped, provenance = risk_map._scope_observed_rows(observed)

    assert len(scoped) == 2
    assert provenance["scope"] == "regional FIRMS bbox"
    assert provenance["excluded_outside_scope"] == 0


def test_build_predicted_records_selection_funnel_without_changing_probability(
    monkeypatch,
):
    base_date = date(2026, 8, 5)
    frame = pd.DataFrame({
        "date": [base_date, base_date],
        "lat_grid": [18.5, 18.6],
        "lon_grid": [98.5, 98.6],
        "fire_count": [2, 0],
        "signal": [1.0, 2.0],
    })
    monkeypatch.setattr(risk_map, "MIN_HISTORICAL_FIRES_FOR_DISPLAY", 1)
    monkeypatch.setattr(risk_map, "MIN_LONG_HISTORICAL_FIRES", 1)
    monkeypatch.setattr(risk_map, "MIN_FIRE_DAYS_PER_YEAR", 0.0)
    monkeypatch.setattr(risk_map, "URBAN_FILTER_ENABLED", False)
    monkeypatch.setattr(risk_map, "COUNTRY_FILTER_ENABLED", False)
    monkeypatch.setattr(risk_map, "BACKGROUND_CONTEXT_LIMIT", 0)
    monkeypatch.setattr(
        risk_map,
        "classify_urban",
        lambda lat, _lon, **_kwargs: (
            np.zeros(len(lat), dtype=bool),
            np.full(len(lat), 30.0),
            np.full(len(lat), "Test City", dtype=object),
        ),
    )
    monkeypatch.setattr(
        risk_map,
        "find_province",
        lambda lat, _lon: np.full(len(lat), "Chiang Mai", dtype=object),
    )
    metadata = {
        "features": ["signal"],
        "imminent_days": 3,
        "probability_tier_thresholds": {
            "HIGH": 0.8,
            "MEDIUM": 0.5,
            "calib_precision_at_high": 0.7,
            "calib_recall_at_high": 0.8,
            "calib_recall_target_medium": 0.9,
        },
    }

    predicted, _, _ = risk_map.build_predicted(
        frame, _TwoCellModel(), base_date, metadata
    )

    assert predicted["probability"].tolist() == [0.9]
    policy = predicted.attrs["selection_policy"]
    assert policy["total_scored_cells"] == 2
    assert policy["retained_map_candidate_cells"] == 1
    assert policy["selection_changes_probability"] is False
    assert policy["filters"]["recent_activity"]["retained_after_filter"] == 1
    assert policy["filters"]["display_cap"]["enabled"] is False


def test_history_evidence_is_combined_with_or(monkeypatch):
    base_date = date(2026, 8, 5)
    frame = pd.DataFrame({
        "date": [base_date] * 4,
        "lat_grid": [10.0, 11.0, 12.0, 13.0],
        "lon_grid": [100.0] * 4,
        "fire_count": [1, 0, 0, 0],
        "signal": [0.9, 0.8, 0.7, 0.1],
    })
    monkeypatch.setattr(risk_map, "MIN_HISTORICAL_FIRES_FOR_DISPLAY", 1)
    monkeypatch.setattr(risk_map, "MIN_LONG_HISTORICAL_FIRES", 3)
    monkeypatch.setattr(risk_map, "MIN_FIRE_DAYS_PER_YEAR", 3.0)
    monkeypatch.setattr(risk_map, "URBAN_FILTER_ENABLED", False)
    monkeypatch.setattr(risk_map, "COUNTRY_FILTER_ENABLED", False)
    monkeypatch.setattr(risk_map, "BACKGROUND_CONTEXT_LIMIT", 0)
    monkeypatch.setattr(
        risk_map,
        "historical_fire_counts",
        lambda _df, _base, window_days, column_name: pd.DataFrame({
            "lat_grid": [10.0, 11.0, 12.0, 13.0],
            "lon_grid": [100.0] * 4,
            column_name: [1, 3 if window_days == risk_map.LONG_HISTORY_WINDOW_DAYS else 0, 0, 0],
        }),
    )
    monkeypatch.setattr(
        risk_map,
        "historical_fire_rate",
        lambda _df: pd.DataFrame({
            "lat_grid": [10.0, 11.0, 12.0, 13.0],
            "lon_grid": [100.0] * 4,
            "fire_days_total": [1, 3, 4, 0],
            "fire_days_per_year": [1.0, 2.0, 3.0, np.nan],
        }),
    )
    monkeypatch.setattr(
        risk_map,
        "classify_urban",
        lambda lat, _lon, **_kwargs: (
            np.zeros(len(lat), dtype=bool),
            np.full(len(lat), 30.0),
            np.full(len(lat), "Test City", dtype=object),
        ),
    )
    monkeypatch.setattr(
        risk_map,
        "find_province",
        lambda lat, _lon: np.full(len(lat), "Test", dtype=object),
    )
    metadata = {
        "features": ["signal"],
        "imminent_days": 3,
        "probability_tier_thresholds": {"HIGH": 0.8, "MEDIUM": 0.5},
    }

    predicted, _, _ = risk_map.build_predicted(
        frame, _SignalModel(), base_date, metadata
    )

    assert predicted["lat_grid"].tolist() == [10.0, 11.0, 12.0]
    policy = predicted.attrs["selection_policy"]
    assert policy["filters"]["history_combination"] == {
        "mode": "any_enabled",
        "retained_after_filter": 3,
    }


def test_display_cap_ranks_complete_binary_snapshot(monkeypatch):
    base_date = date(2026, 8, 5)
    frame = pd.DataFrame({
        "date": [base_date] * 3,
        "lat_grid": [10.0, 11.0, 12.0],
        "lon_grid": [100.0] * 3,
        "fire_count": [0, 0, 0],
        "signal": [0.9, 0.8, 0.7],
    })
    monkeypatch.setattr(risk_map, "MIN_HISTORICAL_FIRES_FOR_DISPLAY", 0)
    monkeypatch.setattr(risk_map, "MIN_LONG_HISTORICAL_FIRES", 0)
    monkeypatch.setattr(risk_map, "MIN_FIRE_DAYS_PER_YEAR", 0.0)
    monkeypatch.setattr(risk_map, "URBAN_FILTER_ENABLED", False)
    monkeypatch.setattr(risk_map, "COUNTRY_FILTER_ENABLED", False)
    monkeypatch.setattr(risk_map, "BACKGROUND_CONTEXT_LIMIT", 2)
    monkeypatch.setattr(
        risk_map,
        "classify_urban",
        lambda lat, _lon, **_kwargs: (
            np.zeros(len(lat), dtype=bool),
            np.full(len(lat), 30.0),
            np.full(len(lat), "Test City", dtype=object),
        ),
    )
    monkeypatch.setattr(
        risk_map,
        "find_province",
        lambda lat, _lon: np.full(len(lat), "Test", dtype=object),
    )
    metadata = {
        "features": ["signal"],
        "imminent_days": 3,
        "probability_tier_thresholds": {"HIGH": 0.95, "WATCH": 0.94},
    }

    predicted, _, _ = risk_map.build_predicted(
        frame, _SignalModel(), base_date, metadata
    )

    assert predicted["probability"].tolist() == [0.9, 0.8]
    display_policy = predicted.attrs["selection_policy"]["filters"]["display_cap"]
    assert display_policy["scope"] == "all_operational_plus_optional_background_context"
    assert display_policy["suppressed"] == 1
    assert display_policy["operational_alerts_suppressed"] == 0


def test_plateau_context_uses_raw_order_not_southern_coordinates(monkeypatch):
    base_date = date(2026, 8, 5)
    frame = pd.DataFrame({
        "date": [base_date] * 3,
        # Lowest latitude deliberately has the weakest raw model signal.
        "lat_grid": [5.0, 15.0, 10.0],
        "lon_grid": [100.0] * 3,
        "fire_count": [0, 0, 0],
        "signal": [0.1, 0.9, 0.8],
    })
    monkeypatch.setattr(risk_map, "MIN_HISTORICAL_FIRES_FOR_DISPLAY", 0)
    monkeypatch.setattr(risk_map, "MIN_LONG_HISTORICAL_FIRES", 0)
    monkeypatch.setattr(risk_map, "MIN_FIRE_DAYS_PER_YEAR", 0.0)
    monkeypatch.setattr(risk_map, "URBAN_FILTER_ENABLED", False)
    monkeypatch.setattr(risk_map, "COUNTRY_FILTER_ENABLED", False)
    monkeypatch.setattr(risk_map, "BACKGROUND_CONTEXT_LIMIT", 2)
    monkeypatch.setattr(
        risk_map,
        "classify_urban",
        lambda lat, _lon, **_kwargs: (
            np.zeros(len(lat), dtype=bool),
            np.full(len(lat), 30.0),
            np.full(len(lat), "Test City", dtype=object),
        ),
    )
    monkeypatch.setattr(
        risk_map,
        "find_province",
        lambda lat, _lon: np.full(len(lat), "Test", dtype=object),
    )
    metadata = {
        "features": ["signal"],
        "imminent_days": 3,
        "probability_tier_thresholds": {"HIGH": 0.8, "WATCH": 0.5},
    }

    predicted, _, _ = risk_map.build_predicted(
        frame, _PlateauSignalModel(), base_date, metadata
    )

    assert predicted["lat_grid"].tolist() == [15.0, 10.0]
    assert predicted["probability"].tolist() == [0.01, 0.01]
    assert set(predicted["urgency_level"]) == {"LOW"}


def test_background_is_not_forced_onto_map_and_alerts_are_never_capped(monkeypatch):
    ranked = pd.DataFrame({
        "lat_grid": [1.0, 2.0, 3.0, 4.0],
        "lon_grid": [100.0] * 4,
        "urgency_level": ["HIGH", "MEDIUM", "HIGH", "LOW"],
        "rank": [1, 2, 3, 4],
    })

    selected, policy = risk_map._select_operational_and_context(
        ranked, background_limit=0
    )

    assert selected["lat_grid"].tolist() == [1.0, 2.0, 3.0]
    assert policy["operational_alerts_retained"] == 3
    assert policy["operational_alerts_suppressed"] == 0
    assert policy["background_context_retained"] == 0


def test_background_context_limit_never_splits_an_exact_tie():
    ranked = pd.DataFrame({
        "lat_grid": [5.0, 10.0, 15.0, 20.0],
        "lon_grid": [100.0] * 4,
        "urgency_level": ["LOW"] * 4,
        # The first three cells are scientifically unresolved and therefore
        # share a competition rank. A limit of two must not choose the two
        # southern coordinates from that group.
        "rank": [1, 1, 1, 4],
    })

    selected, policy = risk_map._select_operational_and_context(
        ranked, background_limit=2
    )

    assert selected.empty
    assert policy["boundary_tie_omitted"] == 3
    assert policy["ties_split"] is False


def test_operational_alerts_are_not_subject_to_a_legacy_hundred_cell_cap():
    count = 151
    ranked = pd.DataFrame({
        "lat_grid": np.linspace(5.0, 20.0, count),
        "lon_grid": np.full(count, 100.0),
        "urgency_level": ["HIGH"] * 101 + ["MEDIUM"] * 50,
        "rank": np.arange(1, count + 1),
    })

    selected, policy = risk_map._select_operational_and_context(
        ranked, background_limit=0
    )

    assert len(selected) == count
    assert policy["operational_alerts_retained"] == count
    assert policy["operational_alerts_suppressed"] == 0


def test_append_geojson_versions_new_snapshot_without_relabelling_legacy(
    tmp_path, monkeypatch,
):
    geojson_path = tmp_path / "riskmap" / "fire_dates_all.geojson"
    latest_path = tmp_path / "riskmap" / "latest.json"
    geojson_path.parent.mkdir()
    legacy_properties = {
        "source": "predicted",
        "base_date": "2026-08-04",
        "urgency_level": "HIGH",
        "probability": 0.2,
        "lat": 18.4,
        "lon": 98.4,
    }
    geojson_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [98.4, 18.4]},
            "properties": legacy_properties,
        }, {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [99.0, 19.0]},
            "properties": {
                **legacy_properties,
                "base_date": "2026-08-06",
                "lat": 19.0,
                "lon": 99.0,
            },
        }],
    }), encoding="utf-8")

    monkeypatch.setattr(risk_map, "GEOJSON_PATH", str(geojson_path))
    monkeypatch.setattr(risk_map, "LATEST_PATH", str(latest_path))
    monkeypatch.setattr(risk_map, "_latest_data_time", lambda *_args: None)
    monkeypatch.setattr(risk_map, "_load_metadata", lambda: {
        "trained_at": "model-2026-07-16",
        "imminent_days": 3,
    })

    base_date = date(2026, 8, 5)
    metrics = {"roc_auc": 0.905}
    thresholds = {"CRITICAL": 1.01, "HIGH": 0.52, "MEDIUM": 0.43, "LOW": 0.0}
    predicted = _predicted_frame(base_date)
    predicted.loc[:, "tree_loss_pct_recent"] = np.nan
    predicted.loc[:, "fire_days_per_year"] = np.nan
    risk_map.append_geojson(
        pd.DataFrame(),
        predicted,
        base_date,
        thresholds,
        metrics,
    )

    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert metadata["model_version"] == "model-2026-07-16"
    assert metadata["thresholds_apply_to_base_date"] == "2026-08-05"
    assert metadata["metrics"] == metrics
    assert metadata["selection_policy"]["total_scored_cells"] == 100
    assert metadata["selection_policy"]["retained_map_candidate_cells"] == 1
    assert metadata["selection_policy"]["selection_changes_probability"] is False
    assert metadata["background_context_omitted"] == 3
    assert metadata["suppressed_by_display_cap"] == 3
    assert "Deprecated legacy alias" in metadata["suppressed_by_display_cap_note"]

    by_date = {
        feature["properties"]["base_date"]: feature["properties"]
        for feature in payload["features"]
        if feature["properties"].get("source") == "predicted"
    }
    # New snapshots identify their generating model.
    assert by_date["2026-08-05"]["model_version"] == "model-2026-07-16"
    assert by_date["2026-08-05"]["tree_cover_pct_2000"] == 40.0
    assert by_date["2026-08-05"]["tree_loss_pct_recent"] is None
    assert by_date["2026-08-05"]["fire_days_per_year"] is None
    # Retained legacy snapshots remain untouched; unknown provenance is not
    # guessed from the current top-level metadata.
    assert by_date["2026-08-04"] == legacy_properties
    assert "2026-08-06" not in by_date

    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["model_version"] == "model-2026-07-16"
    assert latest["thresholds_apply_to_base_date"] == "2026-08-05"
    assert latest["observed_scope"]["scope"] == "Thailand"
    assert latest["metrics"] == metrics
    assert latest["background_context_omitted"] == 3
    assert latest["suppressed_by_display_cap"] == 3
    assert "Deprecated legacy alias" in latest["suppressed_by_display_cap_note"]
