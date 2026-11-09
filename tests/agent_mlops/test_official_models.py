"""Official delivery acceptance against newly fitted in-memory objects."""

import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest

from agent_mlops.inference import prediction_frame
from agent_mlops.model_export import model_definition, input_example
from agent_mlops.models import TabularPredictor, ForecastPredictor
from agent_mlops.temporal import normalize_time, ID, TIME
from agent_mlops.training import train
from test_contract import spec_for
from test_temporal import panel, temporal_spec


def test_OM01_training_delivers_official_model(table, tmp_path):
    report = train(spec_for(table), tmp_path / "out")
    assert report["artifact_status"] == "complete", report
    assert (tmp_path / "out/model/MLmodel").is_file()
    model = mlflow.pyfunc.load_model(report["model_path"])
    result = model.predict(pd.DataFrame({"x": [1, 4], "kind": ["unseen", "a"]}))
    assert list(result) == ["prediction", "probability_0", "probability_1"]
    assert len(list((tmp_path / "out/model").rglob("model.joblib"))) == 1
    assert not list((tmp_path / "out/model").rglob("model.pkl"))
    assert not list((tmp_path / "out/model").rglob("manifest.json"))
    assert set(model.metadata.flavors) == {"python_function"}
    assert not (tmp_path / "out/manifest.json").exists()


def assert_predictions(expected, actual, *, check_dtype=True):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True),
            actual.reset_index(drop=True),
            check_dtype=check_dtype,
            check_exact=False,
            atol=1e-10,
            rtol=0,
        )
        if "prediction" in expected:
            np.testing.assert_array_equal(expected.prediction, actual.prediction)
    else:
        np.testing.assert_allclose(expected, actual, atol=1e-10, rtol=0)


@pytest.mark.parametrize("model_name", ["rf", "svm", "xgb", "lgbm"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_OM02_joblib_matrix_without_refit(
    table, tmp_path, monkeypatch, model_name, task
):
    if model_name in {"xgb", "lgbm"}:
        pytest.importorskip({"xgb": "xgboost", "lgbm": "lightgbm"}[model_name])
    data = pd.read_csv(table)
    # Multiclass labels also exercise encoded class order.
    y = (
        np.array(["third", "first", "second"] * 20)
        if task == "classification"
        else data.x * 1.5
    )
    params = {} if model_name == "svm" else {"n_estimators": 8, "max_depth": 3}
    spec = spec_for(table, task=task, model=model_name, fixed_params=params)
    model = TabularPredictor(spec, params).fit(data[["x", "kind"]], y)
    x = pd.DataFrame({"x": [1.0, np.nan, 4.5], "kind": ["new", None, "a"]})
    expected = (
        prediction_frame(model, x) if task == "classification" else model.predict(x)
    )
    monkeypatch.setattr(
        TabularPredictor, "fit", lambda *a, **k: pytest.fail("refit during packaging")
    )
    with model_definition(model, x) as (flavor, options):
        flavor.save_model(path=str(tmp_path / "model"), **options)
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))
    actual = loaded.predict(x)
    assert type(actual) is type(expected)
    if task == "classification":
        pd.testing.assert_series_equal(expected.dtypes, actual.dtypes)
    else:
        assert actual.dtype == expected.dtype
    assert_predictions(expected, actual)
    # Integer inputs remain legal too; no caller-side casting requirement.
    integers = pd.DataFrame({"x": [1, 2], "kind": ["a", "b"]})
    expected_int = (
        prediction_frame(model, integers)
        if task == "classification"
        else model.predict(integers)
    )
    assert_predictions(expected_int, loaded.predict(integers))
    with pytest.raises(Exception, match="schema|numeric"):
        loaded.predict(x.assign(x="bad"))
    with pytest.raises(Exception, match="inputs|column|[Ss]chema"):
        loaded.predict(x.drop(columns="kind"))
    assert loaded.metadata.get_input_schema().input_names() == ["x", "kind"]
    if task == "classification":
        assert loaded.metadata.metadata["label_mapping"] == model.classes_.tolist()


@pytest.mark.parametrize("model_name", ["rf", "svm", "xgb", "lgbm"])
def test_OM03_event_threshold_boundary_and_positive_label(table, tmp_path, model_name):
    if model_name in {"xgb", "lgbm"}:
        pytest.importorskip({"xgb": "xgboost", "lgbm": "lightgbm"}[model_name])
    data = pd.read_csv(table)
    params = {} if model_name == "svm" else {"n_estimators": 8, "max_depth": 3}
    spec = spec_for(table, model=model_name, fixed_params=params)
    model = TabularPredictor(spec, params).fit(data[["x", "kind"]], data.y)
    # Isolate threshold semantics with a non-default positive label/order.
    model.spec.mode = "future_event"
    model.positive_index = 0
    model.positive_label, model.negative_label = model.classes_
    x = data[["x", "kind"]].head(3)
    p = model.predict_proba(x)[0, 0]
    assert 0 < p < 1
    for name, threshold in [
        ("below", np.nextafter(p, type(p)(0))),
        ("equal", p),
        ("above", np.nextafter(p, type(p)(1))),
    ]:
        model.spec.threshold = float(threshold)
        expected = prediction_frame(model, x)
        assert expected.prediction.iloc[0] == (
            model.negative_label if name == "above" else model.positive_label
        )
        with model_definition(model, x) as (flavor, options):
            flavor.save_model(path=str(tmp_path / name), **options)
        loaded = mlflow.pyfunc.load_model(str(tmp_path / name))
        assert_predictions(expected, loaded.predict(x))


@pytest.mark.parametrize("mode", ["tabular", "regression", "future_event", "forecast"])
def test_OM01_OM04_isolated_sdk_and_official_cli(tmp_path, table, monkeypatch, mode):
    import agent_mlops.training as training

    if mode == "forecast":
        pytest.importorskip("mlforecast")
    if mode in {"tabular", "regression"}:
        raw = pd.read_csv(table)
        if mode == "regression":
            raw.y = raw.x * 1.5
            raw.to_csv(table, index=False)
        spec = spec_for(
            table, task="regression" if mode == "regression" else "classification"
        )
    else:
        path, raw = panel(tmp_path)
        spec = temporal_spec(path, mode)
    captured = []
    fit = training.fit_model

    def capture(*args, **kwargs):
        model = fit(*args, **kwargs)
        captured.append(model)
        return model

    monkeypatch.setattr(training, "fit_model", capture)
    report = train(spec, tmp_path / "run")
    assert report["artifact_status"] == "complete", report
    x = input_example(captured[-1], raw)
    expected = (
        captured[-1].predict(x)
        if mode == "regression"
        else prediction_frame(captured[-1], x)
    )
    assert spec.target not in x
    x.to_csv(tmp_path / "input.csv", index=False)
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    env.pop("PYTHONPATH", None)
    script = """
import sys, mlflow.pyfunc, pandas as pd, numpy as np
from pathlib import Path
m = mlflow.pyfunc.load_model(sys.argv[1])
result = m.predict(pd.read_csv(sys.argv[2]))
if sys.argv[4] == "regression":
    assert isinstance(result, np.ndarray)
    np.save(sys.argv[3], result)
else:
    result.to_json(sys.argv[3], orient="table", index=False, double_precision=15)
import agent_mlops.inference
assert Path(agent_mlops.inference.__file__).is_relative_to(Path(sys.argv[1]) / 'code')
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            report["model_path"],
            str(tmp_path / "input.csv"),
            str(tmp_path / "sdk.json"),
            mode,
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    sdk_result = (
        np.load(tmp_path / "sdk.json.npy")
        if mode == "regression"
        else pd.read_json(tmp_path / "sdk.json", orient="table")
    )
    if mode == "regression":
        assert sdk_result.dtype == expected.dtype
    assert_predictions(expected, sdk_result, check_dtype=False)
    subprocess.run(
        [
            str(Path(sys.executable).parent / "mlflow"),
            "models",
            "predict",
            "--model-uri",
            report["model_path"],
            "--input-path",
            str(tmp_path / "input.csv"),
            "--output-path",
            str(tmp_path / "cli.json"),
            "--content-type",
            "csv",
            "--env-manager",
            "local",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    predictions = json.loads((tmp_path / "cli.json").read_text())["predictions"]
    if mode == "regression":
        actual = np.asarray(predictions)
    else:
        actual = pd.DataFrame(predictions)
        for c in expected:
            if pd.api.types.is_datetime64_any_dtype(expected[c]):
                actual[c] = pd.to_datetime(actual[c], utc=True)
    assert_predictions(expected, actual, check_dtype=False)


def test_OM04_forecast_historical_exogenous_and_invalid_future(tmp_path, monkeypatch):
    pytest.importorskip("mlforecast")
    path, raw = panel(tmp_path)
    spec = temporal_spec(path, availability={"sensor": "2h"})
    spec.temporal.timezone = "Asia/Taipei"
    raw.time = raw.time.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    spec.temporal.historical_features = {"sensor": [6, 8]}
    normalized = normalize_time(raw, spec.temporal)
    origin = spec.temporal.timestamp(spec.temporal.test_start)
    model = ForecastPredictor(spec, spec.fixed_params).fit(
        normalized[normalized[TIME] < origin]
    )
    history = model.history.copy(deep=True)
    mapping = model.mapping.copy(deep=True)
    forecaster = model.forecaster
    monkeypatch.setattr(
        ForecastPredictor, "fit", lambda *a, **k: pytest.fail("refit during packaging")
    )
    x = input_example(model, raw)
    with model_definition(model, x) as (flavor, options):
        assert flavor is mlflow.pyfunc
        assert set(options["artifacts"]) == {"model"}
        saved = joblib.load(options["artifacts"]["model"])
        pd.testing.assert_frame_equal(
            saved.history, history[[ID, TIME, *spec.temporal.historical_features]]
        )
        pd.testing.assert_frame_equal(saved.mapping, mapping)
        flavor.save_model(path=str(tmp_path / "model"), **options)
    pd.testing.assert_frame_equal(model.history, history)
    pd.testing.assert_frame_equal(model.mapping, mapping)
    assert model.forecaster is forecaster
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))
    shuffled = x.sample(frac=1, random_state=1)
    assert_predictions(model.predict(shuffled), loaded.predict(shuffled))
    for bad in [
        x.iloc[1:],
        pd.concat([x, x.iloc[:1]]),
        x.assign(calendar=np.nan),
        x.assign(time="invalid"),
        x.assign(calendar="invalid"),
    ]:
        with pytest.raises(Exception):
            loaded.predict(bad)


def test_OM04_forecast_without_future_features_accepts_empty_frame(tmp_path):
    pytest.importorskip("mlforecast")
    path, raw = panel(tmp_path)
    spec = temporal_spec(path)
    spec.temporal.known_future = {}
    spec.temporal.horizons = list(range(1, 25))
    history = normalize_time(raw, spec.temporal)
    history = history[history[TIME] < spec.temporal.timestamp(spec.temporal.test_start)]
    model = ForecastPredictor(spec, spec.fixed_params).fit(history)
    with model_definition(model, input_example(model, raw)) as (flavor, options):
        flavor.save_model(path=str(tmp_path / "model"), **options)
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))
    assert_predictions(model.predict(), loaded.predict(pd.DataFrame()))


@pytest.mark.parametrize("failure", ["official_save", "joblib_dump"])
def test_OM05_save_failure_and_cli_exit(table, tmp_path, monkeypatch, failure):
    import agent_mlops.cli as cli

    monkeypatch.setattr(
        mlflow.pyfunc if failure == "official_save" else joblib,
        "save_model" if failure == "official_save" else "dump",
        lambda *args, **kw: (_ for _ in ()).throw(OSError("disk full")),
    )
    report = train(spec_for(table), tmp_path / "out")
    assert report["training_status"] == "completed"
    assert report["artifact_status"] == "failed"
    assert report["model_path"] is None
    assert (
        json.loads((tmp_path / "out/training_report.json").read_text())[
            "artifact_status"
        ]
        == "failed"
    )
    monkeypatch.setattr(cli, "run_training", lambda args: report)
    monkeypatch.setattr(sys, "argv", ["train"])
    assert cli.main() == 1


def test_OM05_report_update_does_not_save_again(table, tmp_path, monkeypatch):
    save = mlflow.pyfunc.save_model
    dump = joblib.dump
    calls = []
    serializations = []

    def record(**kwargs):
        calls.append(kwargs["path"])
        return save(**kwargs)

    monkeypatch.setattr(mlflow.pyfunc, "save_model", record)

    def record_dump(model, path, **kwargs):
        serializations.append(path)
        return dump(model, path, **kwargs)

    monkeypatch.setattr(joblib, "dump", record_dump)
    report = train(spec_for(table), tmp_path / "out")
    assert report["artifact_status"] == "complete"
    assert calls == [str(tmp_path / "out/model")]
    assert len(serializations) == 1


def test_OM05_corrupt_artifact_fails_official_load(table, tmp_path):
    report = train(spec_for(table), tmp_path / "out")
    assert report["artifact_status"] == "complete"
    artifact, = (tmp_path / "out/model").rglob("model.joblib")
    artifact.write_bytes(b"")
    with pytest.raises(EOFError):
        mlflow.pyfunc.load_model(report["model_path"])


def test_OM04_classifier_without_probability_keeps_label_only_contract(table, tmp_path):
    frame = pd.read_csv(table)
    spec = spec_for(table, model="svm", fixed_params={"probability": False})
    model = TabularPredictor(spec, spec.fixed_params).fit(frame[["x", "kind"]], frame.y)
    x = frame[["x", "kind"]].head(3)
    with model_definition(model, x) as (flavor, options):
        flavor.save_model(path=str(tmp_path / "model"), **options)
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))
    result = loaded.predict(x)
    assert list(result) == ["prediction"]
    assert loaded.metadata.metadata["probability_columns"] is None
    np.testing.assert_array_equal(result.prediction, model.predict(x))


def test_OM06_off_does_not_access_tracking(table, tmp_path, monkeypatch):
    import mlflow.tracking

    monkeypatch.setattr(
        mlflow.tracking,
        "MlflowClient",
        lambda *a, **k: pytest.fail("tracking off accessed client"),
    )
    monkeypatch.setattr(
        mlflow, "start_run", lambda *a, **k: pytest.fail("tracking off started run")
    )
    monkeypatch.chdir(tmp_path)
    report = train(spec_for(table), tmp_path / "out")
    assert report["artifact_status"] == "complete", report
    assert not (tmp_path / "mlruns").exists()
    assert not (tmp_path / "mlflow.db").exists()


@pytest.mark.parametrize(
    "failure", [None, "model_upload", "provenance_upload", "report_upload"]
)
def test_OM06_official_logging_links_parent_and_restores_context(
    table, tmp_path, monkeypatch, failure
):
    import agent_mlops.tracking as tracking

    previous_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'test.db'}"
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.create_experiment(
            "model-test", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        unrelated_uri = f"sqlite:///{tmp_path / 'unused.db'}"
        mlflow.set_tracking_uri(unrelated_uri)
        monkeypatch.setattr(tracking, "remote_uri", lambda: uri)
        log_model = mlflow.pyfunc.log_model
        publications = []

        def observed_log(**kwargs):
            publications.append(True)
            if failure == "model_upload":
                raise OSError("injected model upload failure")
            return log_model(**kwargs)

        monkeypatch.setattr(mlflow.pyfunc, "log_model", observed_log)
        if failure == "provenance_upload":
            original = mlflow.MlflowClient.log_artifact

            def fail_provenance(self, run_id, local_path, *args, **kwargs):
                if Path(local_path).name == "environment.json":
                    raise OSError("injected provenance upload failure")
                return original(self, run_id, local_path, *args, **kwargs)

            monkeypatch.setattr(mlflow.MlflowClient, "log_artifact", fail_provenance)
        elif failure == "report_upload":
            original = mlflow.MlflowClient.log_dict

            def fail_report(self, run_id, dictionary, artifact_file):
                if artifact_file == "training_report.json":
                    raise OSError("injected report upload failure")
                return original(self, run_id, dictionary, artifact_file)

            monkeypatch.setattr(mlflow.MlflowClient, "log_dict", fail_report)
        report = train(
            spec_for(table, tracking="remote", experiment="model-test"),
            tmp_path / "out",
        )
        assert report["tracking"]["status"] == ("failed" if failure else "complete"), (
            report
        )
        assert report["tracking"]["failed_stage"] == failure
        assert publications == [True]
        assert report["artifact_status"] == "complete", report
        assert mlflow.active_run() is None
        assert mlflow.get_tracking_uri() == unrelated_uri
        assert not (tmp_path / "unused.db").exists()
        client = mlflow.MlflowClient(tracking_uri=uri)
        if failure == "model_upload":
            assert report["tracking"]["model_uri"] is None
            assert client.get_run(report["tracking"]["run_id"]).info.status == "FAILED"
            x = pd.read_csv(table)[["x", "kind"]].head(3)
            assert len(mlflow.pyfunc.load_model(report["model_path"]).predict(x)) == 3
            return
        info = client.get_logged_model(report["tracking"]["model_uri"].split("/")[-1])
        assert info.source_run_id == report["tracking"]["run_id"]
        assert client.get_run(info.source_run_id).info.status == (
            "FAILED" if failure else "FINISHED"
        )
        x = pd.read_csv(table)[["x", "kind"]].head(3)
        mlflow.set_tracking_uri(uri)
        assert_predictions(
            mlflow.pyfunc.load_model(report["model_path"]).predict(x),
            mlflow.pyfunc.load_model(report["tracking"]["model_uri"]).predict(x),
        )
        if failure:
            assert "training_report.json" not in {
                artifact.path for artifact in client.list_artifacts(info.source_run_id)
            }
            return
        saved = client.download_artifacts(
            info.source_run_id, "training_report.json", str(tmp_path)
        )
        assert json.loads(Path(saved).read_text())["tracking"] == report["tracking"]
    finally:
        mlflow.set_tracking_uri(previous_uri)
