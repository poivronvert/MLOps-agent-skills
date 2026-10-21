"""Dataset lineage against the official SDK store; no live service required."""

import pandas as pd
import pytest

from agent_mlops.dataset_tracking import DatasetSplit, frame_sha256, record_split
from agent_mlops.tracking import Tracker
from agent_mlops.training import train
from test_contract import spec_for

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("sqlalchemy")
from mlflow.tracking import MlflowClient  # noqa: E402


@pytest.fixture
def store(tmp_path):
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'datasets.db'}")
    exp = client.create_experiment(
        "dataset-tests", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    run_id = client.create_run(exp).info.run_id
    return client, exp, run_id


def sample():
    return DatasetSplit(
        name="train",
        context="training",
        frame=pd.DataFrame(
            {"x": [2.5, float("nan")], "label": ["yes", "no"]}, index=[7, 2]
        ),
        target="label",
        source="source.csv",
        source_sha256="a" * 64,
    )


@pytest.mark.parametrize(
    "inputs,snapshot", [(True, True), (True, False), (False, True)]
)
def test_independent_policies_and_roundtrip(store, inputs, snapshot, tmp_path):
    client, _, run_id = store
    split = sample()
    result = record_split(client, run_id, split, log_inputs=inputs, snapshot=snapshot)
    assert bool(client.get_run(run_id).inputs.dataset_inputs) == inputs
    assert result["source_sha256"] == "a" * 64
    assert result["rows"] == 2
    assert bool(result["snapshot_artifact"]) == snapshot
    if snapshot:
        path = client.download_artifacts(
            run_id, result["snapshot_artifact"], str(tmp_path)
        )
        pd.testing.assert_frame_equal(pd.read_parquet(path), split.frame)
    else:
        files = client.list_artifacts(
            run_id, f"datasets/train/{result['frame_sha256']}"
        )
        assert [f.path.rsplit("/", 1)[-1] for f in files] == ["manifest.json"]


def test_hash_covers_late_rows_labels_order_and_schema():
    frame = pd.DataFrame({"x": range(20000), "y": ["a"] * 20000})
    original = frame_sha256(frame)
    changed = frame.copy()
    changed.loc[19999, "y"] = "b"
    assert frame_sha256(changed) != original
    assert frame_sha256(frame.iloc[::-1]) != original
    assert frame_sha256(frame.astype({"x": "float64"})) != original


def tracker_for(store):
    client, experiment_id, run_id = store

    class StoreTracker(Tracker):
        def start(self):
            self.client = client
            self.experiment_id = experiment_id
            self.run_id = run_id
            self.status = "active"

        def finish(self, output, flavor, options):
            client.set_terminated(self.run_id)
            self.status = "complete"

    return StoreTracker


def test_training_records_exact_pipeline_arguments(store, table, tmp_path, monkeypatch):
    from agent_mlops.models import TabularPredictor

    original = TabularPredictor.fit
    fitted = []

    def fit(self, X, y):
        fitted.append(pd.concat([X, y], axis=1))
        return original(self, X, y)

    monkeypatch.setattr(TabularPredictor, "fit", fit)
    report = train(
        spec_for(table, tracking="remote"),
        tmp_path / "out",
        tracker_factory=tracker_for(store),
    )
    assert report["training_status"] == "completed", report
    records = report["tracking"]["datasets"]
    assert [r["context"] for r in records] == [
        "training",
        "validation",
        "training",
        "validation",
        "training",
        "test",
    ]
    assert [r["frame_sha256"] for r in records if r["context"] == "training"] == [
        frame_sha256(frame) for frame in fitted
    ]
    train_records = [r for r in records if r["context"] == "training"]
    assert [r["rows"] for r in train_records] == [24, 24, 48]
    assert records[-1]["rows"] == 12
    assert records[0]["run_id"] != records[-1]["run_id"]


@pytest.mark.parametrize("failure", ["log_inputs", "download_artifacts"])
def test_dataset_failure_stops_before_fit(store, table, tmp_path, monkeypatch, failure):
    from agent_mlops.models import TabularPredictor

    client, _, _ = store

    def fail(*args, **kwargs):
        raise RuntimeError("Authorization: sentinel-secret")

    monkeypatch.setattr(client, failure, fail)
    monkeypatch.setattr(
        TabularPredictor,
        "fit",
        lambda *a, **kw: pytest.fail("fit after dataset failure"),
    )
    report = train(
        spec_for(table, tracking="remote"),
        tmp_path / "out",
        tracker_factory=tracker_for(store),
    )
    assert report["training_status"] == "not_started"
    assert report["tracking"]["failed_stage"] == "dataset_recording"
    assert "sentinel-secret" not in (tmp_path / "out/training_report.json").read_text()


@pytest.mark.parametrize("mode", ["future_event", "forecast"])
def test_temporal_split_inputs(store, tmp_path, mode):
    from test_temporal import panel, temporal_spec

    if mode == "forecast":
        pytest.importorskip("mlforecast")
    path, _ = panel(tmp_path)
    spec = temporal_spec(path, mode, tracking="remote")
    report = train(spec, tmp_path / "out", tracker_factory=tracker_for(store))
    assert report["training_status"] == "completed", report
    records = report["tracking"]["datasets"]
    assert {r["context"] for r in records} == {"training", "validation", "test"}
    for receipt in records:
        saved = store[0].download_artifacts(
            receipt["run_id"], receipt["snapshot_artifact"], str(tmp_path)
        )
        frame = pd.read_parquet(saved)
        assert frame_sha256(frame) == receipt["frame_sha256"]
        assert frame[receipt["target"]].notna().all()
        assert receipt["source_sha256"] == spec.data_sha256
        if mode == "forecast":
            assert set(
                spec.temporal.entity_keys + [spec.temporal.time_column]
            ).issubset(frame)
    assert records[-1]["rows"] == report["evaluation"]["holdout"]["n"]


def test_download_corruption_is_detected(store, tmp_path, monkeypatch):
    client, _, run_id = store
    original = client.download_artifacts

    def corrupt(run_id, path, destination):
        result = original(run_id, path, destination)
        if path.endswith(".parquet"):
            from pathlib import Path

            Path(result).write_bytes(b"corrupt")
        return result

    monkeypatch.setattr(client, "download_artifacts", corrupt)
    with pytest.raises(RuntimeError, match="byte hash mismatch"):
        record_split(client, run_id, sample())
