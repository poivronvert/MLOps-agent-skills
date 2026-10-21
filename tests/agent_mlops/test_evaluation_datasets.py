"""Real official SQL-store integration, without a server or external model calls."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("sqlalchemy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manage_agent_evaluations import (  # noqa: E402
    assess,
    content_sha256,
    fixture_records,
    publish,
    snapshot,
)
from mlflow.genai.datasets import get_dataset, search_datasets  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = [
    {
        "id": "CASE",
        "prompt": "inspect data",
        "visible_context": "test.csv",
        "required_actions": ["inspect"],
        "forbidden_actions": ["fit"],
        "tool_actions": ["inspect"],
    }
]


@pytest.fixture
def store(tmp_path, monkeypatch):
    previous = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'official.db'}")
    experiment = mlflow.create_experiment(
        "test-evaluation", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    yield experiment
    mlflow.set_tracking_uri(previous)


def trace():
    return {
        "id": "CASE",
        "prompt": "inspect data",
        "visible_context": "test.csv",
        "assistant_messages": ["inspected"],
        "tool_calls": [{"result": "rows: 3"}],
        "observed_actions": ["inspect"],
        "evidence": {"inspect": ["/tool_calls/0"]},
    }


def test_official_dataset_roundtrip_merge_update_and_discovery(store):
    receipt = publish(FIXTURE, store, name="suite")
    repeated = publish(FIXTURE, store, dataset_id=receipt["dataset_id"])
    # Native digest can change on a merge even when selected content is identical.
    for field in ("dataset_id", "dataset_sha256", "records"):
        assert repeated[field] == receipt[field]
    found = search_datasets(experiment_ids=[store])
    assert [d.dataset_id for d in found] == [receipt["dataset_id"]]
    rows = snapshot(get_dataset(dataset_id=receipt["dataset_id"]))
    assert rows == fixture_records(FIXTURE)
    # Rubric belongs to evaluator expectations, never the acting agent's inputs.
    assert "rubric" not in rows[0]["inputs"]
    assert "required_actions" not in rows[0]["inputs"]
    changed = copy.deepcopy(FIXTURE)
    changed[0]["required_actions"].append("report")
    updated = publish(changed, store, dataset_id=receipt["dataset_id"])
    assert updated["records"] == 1
    assert updated["dataset_sha256"] != receipt["dataset_sha256"]
    with pytest.raises(ValueError, match="content changed"):
        assess(receipt["dataset_id"], receipt["dataset_sha256"], [trace()], store)
    changed[0]["prompt"] = "different input"
    with pytest.raises(ValueError, match="inputs changed"):
        publish(changed, store, dataset_id=receipt["dataset_id"])


def test_native_evaluation_preserves_failed_and_unverified_and_pins_snapshot(
    store, tmp_path
):
    receipt = publish(FIXTURE, store, name="suite")
    good = assess(receipt["dataset_id"], receipt["dataset_sha256"], [trace()], store)
    assert good["counts"] == {"passed": 1, "failed": 0, "unverified": 0}
    run = MlflowClient().get_run(good["run_id"])
    assert run.inputs.dataset_inputs
    assert run.data.tags["evaluation.dataset_id"] == receipt["dataset_id"]
    assert run.data.metrics["cases_passed"] == 1
    bad = trace()
    bad["observed_actions"].append("fit")
    bad["evidence"]["fit"] = ["/tool_calls/0"]
    failed = assess(receipt["dataset_id"], receipt["dataset_sha256"], [bad], store)
    assert failed["counts"]["failed"] == 1
    missing = assess(receipt["dataset_id"], receipt["dataset_sha256"], [], store)
    assert missing["counts"]["unverified"] == 1
    assert not missing["all_verified_passed"]
    changed = copy.deepcopy(FIXTURE)
    changed[0]["required_actions"].append("report")
    publish(changed, store, dataset_id=receipt["dataset_id"])
    path = MlflowClient().download_artifacts(
        good["run_id"], "evaluation/dataset_snapshot.json", str(tmp_path)
    )
    saved = json.loads(Path(path).read_text())
    assert content_sha256(saved) == receipt["dataset_sha256"]
    assert saved[0]["expectations"]["rubric"]["required_actions"] == ["inspect"]
    # The official evaluation engine must also persist the scorer's feedback.
    traces = mlflow.search_traces(locations=[store], run_id=missing["run_id"])
    assert len(traces) == 1
    assert "unverified" in traces.to_json()


def test_identity_evidence_and_duplicates_fail_before_run(store, tmp_path):
    receipt = publish(FIXTURE, store, name="suite")
    args = (receipt["dataset_id"], receipt["dataset_sha256"])
    with pytest.raises(ValueError, match="duplicate"):
        assess(*args, [trace(), trace()], store)
    bad = trace()
    bad["prompt"] = "something else"
    with pytest.raises(ValueError, match="prompt"):
        assess(*args, [bad], store)
    bad = trace()
    bad["visible_context"] = "another file"
    with pytest.raises(ValueError, match="visible context"):
        assess(*args, [bad], store)
    raw = tmp_path / "trace.jsonl"
    raw.write_text('{"type":"test"}\n')
    bad = trace()
    bad.update(
        raw_events=raw.name, raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="evidence-dir"):
        assess(*args, [bad], store)
    raw.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        assess(*args, [bad], store, tmp_path)
    bad["raw_events"] = "../not-allowed"
    with pytest.raises(ValueError, match="inside evidence-dir"):
        assess(*args, [bad], store, tmp_path)
    assert not MlflowClient().search_runs([store])


def test_official_evaluator_missing_feedback_cannot_report_success(store, monkeypatch):
    receipt = publish(FIXTURE, store, name="suite")
    evaluate = mlflow.genai.evaluate

    def missing_feedback(**kwargs):
        result = evaluate(**kwargs)
        result.result_df = None
        return result

    monkeypatch.setattr(mlflow.genai, "evaluate", missing_feedback)
    with pytest.raises(RuntimeError, match="feedback"):
        assess(receipt["dataset_id"], receipt["dataset_sha256"], [trace()], store)
    runs = MlflowClient().search_runs([store])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert "cases_passed" not in runs[0].data.metrics


def test_variants_remain_distinct_and_empty_or_duplicate_cases_rejected():
    fixtures = copy.deepcopy(FIXTURE)
    fixtures[0]["variants"] = [
        {
            "prompt": "ask about data",
            "required_actions": ["ask"],
            "forbidden_actions": ["fit"],
        }
    ]
    rows = fixture_records(fixtures)
    assert len(rows) == 2
    assert rows[1]["inputs"]["variant"] == 0
    assert rows[1]["inputs"]["visible_context"] == "test.csv"
    with pytest.raises(ValueError, match="empty"):
        fixture_records([])
    with pytest.raises(ValueError, match="duplicate"):
        fixture_records(FIXTURE + FIXTURE)


def test_synthetic_observations_preserve_raw_evidence(store, tmp_path):
    # This fixture tests storage contracts, not actual agent behavior.
    evidence = tmp_path / "evidence"
    raw = evidence / "sessions/fixture.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"type":"synthetic-test-fixture","result":"rows: 3"}\n')
    observation = trace()
    observation.update(
        raw_events="sessions/fixture.jsonl",
        raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
    )
    receipt = publish(FIXTURE, store, name="synthetic-storage-contract")
    result = assess(
        receipt["dataset_id"], receipt["dataset_sha256"], [observation], store, evidence
    )
    assert result["counts"] == {"passed": 1, "failed": 0, "unverified": 0}
    assert result["all_verified_passed"]
    client = MlflowClient()
    downloaded = Path(client.download_artifacts(
        result["run_id"], "evaluation/raw/sessions/fixture.jsonl", str(tmp_path)
    ))
    assert downloaded.read_bytes() == raw.read_bytes()
    assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == observation["raw_sha256"]
    reviewed = Path(client.download_artifacts(
        result["run_id"], "evaluation/reviewed_traces.json", str(tmp_path)
    ))
    assert json.loads(reviewed.read_text()) == [observation]
