"""Documentation-contract test (finding 17).

docs/VARIABLE_NETWORK_TH.md distinguishes the current deployed contract from
the next design contract (FEATURES_CORE + FEATURES_WEATHER). This test fails CI
if the documented counts drift
from the code's feature definitions, or if the model metadata's feature list
diverges from the core contract.

    cd src && ../.venv/bin/python -m pytest tests -q
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import features as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOC = os.path.join(ROOT, "docs", "VARIABLE_NETWORK_TH.md")
META = os.path.join(ROOT, "outputs", "metadata", "dataset_info.json")

CORE = len(F.FEATURES_CORE)
WEATHER = len(F.FEATURES_WEATHER)
TOTAL = CORE + WEATHER


def test_feature_definition_counts_are_the_expected_contract():
    # The design contract is 135 core + 37 weather = 172. If these change, the
    # doc AND this expectation must be updated together (that's the point).
    assert CORE == 135, f"FEATURES_CORE is {CORE}, doc/contract expects 135"
    assert TOTAL == 172, f"total features is {TOTAL}, doc/contract expects 172"


@pytest.mark.skipif(not os.path.exists(DOC), reason="doc not present")
def test_doc_counts_match_code():
    text = open(DOC, encoding="utf-8").read()
    nums = set(int(n) for n in re.findall(r"\b(\d{2,3})\b", text))
    # The doc must reference BOTH the current-model count and the design total,
    # and those must equal what the code defines.
    assert CORE in nums, f"doc never mentions the current core count {CORE}"
    assert TOTAL in nums, f"doc never mentions the design total {TOTAL}"


@pytest.mark.skipif(not os.path.exists(META), reason="no trained model metadata")
def test_model_metadata_features_are_within_the_contract():
    meta = json.load(open(META))
    feats = meta.get("features") or []
    if not feats:
        pytest.skip("metadata has no persisted feature list")
    allowed = set(F.FEATURES_CORE) | set(F.FEATURES_WEATHER)
    unknown = [f for f in feats if f not in allowed]
    assert not unknown, f"model uses features absent from the contract: {unknown[:5]}"
    # Active count must be either the core-only (weather dropped) or the full set.
    assert CORE <= len(feats) <= TOTAL, (
        f"model feature_count {len(feats)} outside [{CORE}, {TOTAL}]")
