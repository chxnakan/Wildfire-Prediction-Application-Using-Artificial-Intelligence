"""Atomic-publication guarantees for locally served pipeline artifacts."""

import json
import os

import pandas as pd
import pytest

import storage


def _temp_artifacts_for(path):
    return list(path.parent.glob(f".{path.name}.*.tmp*"))


def test_write_json_same_directory_replace_never_exposes_partial_file(
    tmp_path, monkeypatch
):
    destination = tmp_path / "risk_map.geojson"
    destination.write_text('{"snapshot":"old"}', encoding="utf-8")
    real_replace = os.replace
    replacements = []

    def observed_replace(source, target):
        # Until the single atomic publication point, a concurrent reader still
        # sees the complete old snapshot.  The unpublished temp is complete too.
        assert destination.read_text(encoding="utf-8") == '{"snapshot":"old"}'
        with open(source, encoding="utf-8") as unpublished:
            assert json.load(unpublished) == {
                "snapshot": "new",
                "cells": [1, 2, 3],
            }
        assert os.path.dirname(source) == os.path.dirname(target)
        replacements.append((source, target))
        real_replace(source, target)

    monkeypatch.setattr(storage.os, "replace", observed_replace)
    storage.write_json(
        {"snapshot": "new", "cells": [1, 2, 3]}, str(destination)
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "snapshot": "new",
        "cells": [1, 2, 3],
    }
    assert len(replacements) == 1
    assert _temp_artifacts_for(destination) == []


def test_write_json_serialization_failure_preserves_previous_file(
    tmp_path, monkeypatch
):
    destination = tmp_path / "risk_map.geojson"
    destination.write_text('{"snapshot":"trusted"}', encoding="utf-8")

    def fail_after_partial_write(_obj, stream, **_kwargs):
        stream.write('{"snapshot":')
        stream.flush()
        raise RuntimeError("serialization interrupted")

    monkeypatch.setattr(storage.json, "dump", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="serialization interrupted"):
        storage.write_json({"snapshot": "new"}, str(destination))

    assert destination.read_text(encoding="utf-8") == '{"snapshot":"trusted"}'
    assert _temp_artifacts_for(destination) == []


def test_write_table_failure_preserves_previous_parquet(tmp_path, monkeypatch):
    destination = tmp_path / "full_features.parquet"
    trusted = pd.DataFrame({"cell": [1], "probability": [0.25]})
    trusted.to_parquet(destination, index=False)
    before = destination.read_bytes()

    def fail_after_partial_write(_df, path, **_kwargs):
        with open(path, "wb") as f:
            f.write(b"PAR1-partial")
        raise RuntimeError("table write interrupted")

    monkeypatch.setattr(storage.io_utils, "write_table", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="table write interrupted"):
        storage.write_table(
            pd.DataFrame({"cell": [2], "probability": [0.5]}),
            str(destination),
        )

    assert destination.read_bytes() == before
    assert pd.read_parquet(destination).to_dict("records") == [
        {"cell": 1, "probability": 0.25}
    ]
    assert _temp_artifacts_for(destination) == []


def test_write_table_atomically_publishes_complete_parquet(tmp_path):
    destination = tmp_path / "full_features.parquet"
    expected = pd.DataFrame({"cell": [1, 2], "probability": [0.25, 0.5]})

    storage.write_table(expected, str(destination))

    pd.testing.assert_frame_equal(pd.read_parquet(destination), expected)
    assert _temp_artifacts_for(destination) == []
