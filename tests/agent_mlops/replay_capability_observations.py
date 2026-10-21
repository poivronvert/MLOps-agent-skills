"""Explicit local evidence replay; excluded from ordinary test discovery.

This evaluates saved observations, not a new acting-agent session.
"""

import json
import os
from pathlib import Path

import pytest

import test_evaluation_datasets as evaluation_tests

# Reuse the same temporary SQL store as the ordinary integration tests.
store = evaluation_tests.store


def test_saved_capability_observations_in_official_evaluation(store):
    location = os.environ.get("AGENT_MLOPS_CAPABILITY_EVIDENCE_DIR")
    if not location:
        pytest.fail("Set AGENT_MLOPS_CAPABILITY_EVIDENCE_DIR to the local capabilities evidence directory")
    evidence = Path(location).expanduser().resolve()
    if not (evidence / "traces.json").is_file():
        pytest.fail(f"Capability evidence is missing traces.json: {evidence}")
    fixtures = json.loads(
        (evaluation_tests.ROOT / "tests/fixtures/agent_mlops/capabilities.json").read_text()
    )
    traces = json.loads((evidence / "traces.json").read_text())
    receipt = evaluation_tests.publish(fixtures, store, name="CAP01-CAP12")
    result = evaluation_tests.assess(
        receipt["dataset_id"], receipt["dataset_sha256"], traces, store, evidence
    )
    assert result["counts"] == {"passed": 12, "failed": 0, "unverified": 0}
    assert result["all_verified_passed"]
