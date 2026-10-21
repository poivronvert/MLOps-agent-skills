"""Real-service acceptance. The test platform supplies the server; no local substitute."""

import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

import mlflow.pyfunc
import mlflow.artifacts
from agent_mlops.training import train
from test_temporal import TOL, panel, temporal_spec


@pytest.mark.remote
@pytest.mark.parametrize("mode", ["future_event", "forecast"])
def test_M02_C03_real_service_upload_download_and_clean_client(
    tmp_path, mode, monkeypatch
):
    uri = os.environ.get("AGENT_MLOPS_TEST_TRACKING_URI")
    clean_python = os.environ.get("AGENT_MLOPS_CLEAN_CLIENT_PYTHON")
    if not uri or not clean_python:
        pytest.skip(
            "platform must provide AGENT_MLOPS_TEST_TRACKING_URI and a clean locked AGENT_MLOPS_CLEAN_CLIENT_PYTHON"
        )
    pytest.importorskip("mlflow")
    pytest.importorskip("mlforecast")
    from mlflow.tracking import MlflowClient

    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    path, df = panel(tmp_path)
    spec = temporal_spec(
        path,
        mode,
        tracking="remote",
        experiment=os.environ.get("AGENT_MLOPS_TEST_EXPERIMENT", "agent-mlops-bdd"),
        hpo=True,
        n_trials=2,
    )
    # A deterministic model computation failure exercises real failed-trial logging.
    import agent_mlops.training as training

    original = training.fit_model
    calls = []

    def first_trial_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("BDD deterministic failed trial")
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "fit_model", first_trial_fails)
    output = tmp_path / "local"
    report = train(spec, output)
    assert report["training_status"] == "completed", report
    assert report["tracking"]["status"] == "complete", report
    assert report["artifact_status"] == "complete", report
    assert report["tracking"]["model_uri"]
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(report["tracking"]["run_id"])
    assert {
        tag.value
        for item in run.inputs.dataset_inputs
        for tag in item.tags
        if tag.key == "mlflow.data.context"
    } == {"training", "test"}
    assert report["tracking"]["datasets"]
    assert all(item["snapshot_artifact"] for item in report["tracking"]["datasets"])
    children = client.search_runs(
        [run.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'",
    )
    assert {r.info.status for r in children} == {"FINISHED", "FAILED"}
    (tmp_path / "download").mkdir()
    downloaded = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=report["tracking"]["model_uri"],
            dst_path=str(tmp_path / "download"),
            tracking_uri=uri,
        )
    )
    model = mlflow.pyfunc.load_model(str(output / "model"))
    origin = spec.temporal.timestamp(spec.temporal.test_start)
    future = df[
        (pd.to_datetime(df.time) >= origin)
        & (pd.to_datetime(df.time) < origin + 4 * spec.temporal.step)
    ].sort_values(["id", "time"])
    future.to_csv(tmp_path / "fixture.csv", index=False)
    script = """import sys, mlflow.pyfunc, pandas as pd
model = mlflow.pyfunc.load_model(sys.argv[1])
model.predict(pd.read_csv(sys.argv[2])).to_json(sys.argv[3], orient="table", index=False, double_precision=15)
"""
    subprocess.run(
        [
            clean_python,
            "-I",
            "-c",
            script,
            str(downloaded),
            str(tmp_path / "fixture.csv"),
            str(tmp_path / "actual.json"),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = pd.read_json(tmp_path / "actual.json", orient="table")
    expected = model.predict(future)
    pd.testing.assert_frame_equal(
        expected, actual, check_dtype=False, check_exact=False, atol=TOL, rtol=0
    )
    if mode != "forecast":
        np.testing.assert_array_equal(expected.prediction, actual.prediction)
    logged = client.get_logged_model(report["tracking"]["model_uri"].split("/")[-1])
    assert logged.source_run_id == run.info.run_id
    assert run.info.status == "FINISHED"
