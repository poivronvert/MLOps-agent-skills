"""Fit spies prove fold isolation, HPO failure handling and tracking stop boundaries."""

import importlib.util
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agent_mlops.tracking import Tracker, TrackingError
from agent_mlops.training import train
from test_contract import spec_for


def test_C01_preprocessors_fit_only_each_fold_then_refit(table, tmp_path, monkeypatch):
    sizes = {"imputer": [], "encoder": [], "scaler": [], "estimator": []}
    for cls, label in [
        (SimpleImputer, "imputer"),
        (OneHotEncoder, "encoder"),
        (StandardScaler, "scaler"),
        (RandomForestClassifier, "estimator"),
    ]:
        old = cls.fit

        def spy(self, X, *args, _old=old, _label=label, **kwargs):
            sizes[_label].append(len(X))
            return _old(self, X, *args, **kwargs)

        monkeypatch.setattr(cls, "fit", spy)
    train(spec_for(table), tmp_path / "out")
    assert sizes == {
        "imputer": [24, 24, 24, 24, 48, 48],
        "encoder": [24, 24, 48],
        "scaler": [24, 24, 48],
        "estimator": [24, 24, 48],
    }


class FailingTracker(Tracker):
    failure = "trial_metrics"

    def dataset(self, *args, **kwargs):
        pass

    def start(self):
        self.status = "active"
        self.run_id = "test-parent"

    def start_trial(self, number, params):
        self.trial_id = f"test-{number}"

    def fold(self, number, result):
        if self.failure == "fold_metrics":
            self.fail()

    def fail(self):
        self.status = "failed"
        self.failed_stage = self.failure
        raise TrackingError(f"injected tracking failure at {self.failure}")

    def trial(self, result):
        if self.failure == "trial_metrics":
            self.fail()

    def finish(self, output, flavor, options):
        self.fail()


@pytest.mark.parametrize(
    "stage,expected_fits,expected_training",
    [
        ("fold_metrics", 1, "interrupted"),
        ("trial_metrics", 2, "interrupted"),
        ("model_upload", 7, "completed"),
    ],
)
def test_M04_stop_new_computation_and_preserve_results(
    table, tmp_path, monkeypatch, stage, expected_fits, expected_training
):
    monkeypatch.setattr(FailingTracker, "failure", stage)
    fits = []
    original = RandomForestClassifier.fit

    def fit(self, X, y, **kw):
        fits.append(len(X))
        return original(self, X, y, **kw)

    monkeypatch.setattr(RandomForestClassifier, "fit", fit)
    spec = spec_for(table, hpo=True, n_trials=3)
    from agent_mlops.models import TabularPredictor

    prediction = TabularPredictor.predict
    predicted_after_failure = []

    class ObservedTracker(FailingTracker):
        def fail(self):
            def forbidden_prediction(*args, **kwargs):
                predicted_after_failure.append(True)
                raise AssertionError("inference after tracking failure")

            monkeypatch.setattr(TabularPredictor, "predict", forbidden_prediction)
            super().fail()

    report = train(spec, tmp_path / "out", tracker_factory=ObservedTracker)
    assert not predicted_after_failure
    monkeypatch.setattr(TabularPredictor, "predict", prediction)
    assert len(fits) == expected_fits
    assert report["training_status"] == expected_training
    assert report["tracking"]["status"] == "failed"
    assert report["tracking"]["model_uri"] is None
    assert (tmp_path / "out" / "model/MLmodel").is_file()
    import mlflow.pyfunc

    assert (
        len(
            mlflow.pyfunc.load_model(str(tmp_path / "out/model")).predict(
                pd.DataFrame({"x": [1], "kind": ["unseen"]})
            )
        )
        == 1
    )
    assert len(fits) == expected_fits
    if stage == "trial_metrics":
        assert report["trials"][0]["status"] == "complete"
        assert np.isfinite(report["trials"][0]["score"])
        assert report["search_status"] == "interrupted"
        assert report["best_trial"] is None


def test_H02_timeout_prevents_new_trials(table, tmp_path, monkeypatch):
    original = RandomForestClassifier.fit
    calls = []

    def slow(self, X, y, **kw):
        calls.append(1)
        time.sleep(0.03)
        return original(self, X, y, **kw)

    monkeypatch.setattr(RandomForestClassifier, "fit", slow)
    report = train(
        spec_for(table, hpo=True, n_trials=3, timeout=0.01), tmp_path / "out"
    )
    assert len(report["trials"]) == 1
    assert len(calls) == 3  # two CV fits plus allowed final refit
    assert report["search_status"] == "budget_exhausted"


def test_H03_no_valid_trial_not_arbitrarily_selected(table, tmp_path, monkeypatch):
    def fail(*args, **kw):
        raise ValueError("fixture model failure")

    monkeypatch.setattr(RandomForestClassifier, "fit", fail)
    report = train(spec_for(table, hpo=True, n_trials=2), tmp_path / "out")
    assert report["best_trial"] is None
    assert report["training_status"] == "failed"
    assert report["artifact_status"] == "diagnostics_only"
    assert report["model_path"] is None
    assert not (tmp_path / "out/model/MLmodel").exists()
    assert all(
        t["status"] == "failed" and "fixture model failure" in t["error"]
        for t in report["trials"]
    )


@pytest.mark.parametrize("model", ["rf", "svm", "xgb", "lgbm"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_Q01_model_matrix(table, tmp_path, model, task):
    modules = {"xgb": "xgboost", "lgbm": "lightgbm"}
    if model in modules and importlib.util.find_spec(modules[model]) is None:
        pytest.skip(f"requires {modules[model]} extra")
    if task == "regression":
        df = pd.read_csv(table)
        df.y = df.x * 1.5 + np.sin(df.x)
        df.to_csv(table, index=False)
    params = {} if model == "svm" else {"n_estimators": 8, "max_depth": 3}
    report = train(
        spec_for(table, model=model, task=task, fixed_params=params), tmp_path / "out"
    )
    assert report["training_status"] == "completed", report


def test_M03_service_auth_failure_precedes_fit_and_redacts_secrets(
    table, tmp_path, monkeypatch
):
    pytest.importorskip("mlflow")
    import mlflow.tracking

    class RejectedClient:
        def __init__(self, **kwargs):
            pass

        def get_experiment_by_name(self, name):
            raise RuntimeError(
                "401 Authorization Bearer sentinel-secret https://user:password@server"
            )

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid")
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", RejectedClient)
    monkeypatch.setattr(
        RandomForestClassifier,
        "fit",
        lambda *a, **k: pytest.fail("fit before auth preflight"),
    )
    report = train(spec_for(table, tracking="remote"), tmp_path / "out")
    assert report["training_status"] == "not_started"
    assert report["tracking"]["failed_stage"] == "preflight"
    assert "sentinel-secret" not in (tmp_path / "out/training_report.json").read_text()
    assert "password" not in (tmp_path / "out/task_spec.json").read_text()


def test_H03_nonfinite_metric_cannot_win(table, tmp_path, monkeypatch):
    import agent_mlops.training as training

    original = training.evaluate

    def nonfinite(*args, **kwargs):
        result = original(*args, **kwargs)
        result["score"] = float("nan")
        return result

    monkeypatch.setattr(training, "evaluate", nonfinite)
    report = train(spec_for(table, hpo=True, n_trials=2), tmp_path / "out")
    assert report["best_trial"] is None
    assert report["training_status"] == "failed"


def test_C01_selector_fits_only_development_rows(table, tmp_path, monkeypatch):
    from sklearn.feature_selection import SelectKBest

    original = SelectKBest.fit
    sizes = []

    def record(self, X, y):
        sizes.append(len(X))
        return original(self, X, y)

    monkeypatch.setattr(SelectKBest, "fit", record)
    train(spec_for(table, select_k=2), tmp_path / "out")
    assert sizes == [24, 24, 48]


def test_M02_failed_calculation_closes_parent_as_failed(table, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import mlflow.tracking
    from types import SimpleNamespace

    statuses = {}

    class RecordingClient:
        def __init__(self, **kwargs):
            self.count = 0

        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="test")

        def create_run(self, experiment_id, tags):
            self.count += 1
            return SimpleNamespace(info=SimpleNamespace(run_id=str(self.count)))

        def log_param(self, *args, **kwargs):
            pass

        def set_tag(self, *args, **kwargs):
            pass

        def log_artifact(self, *args, **kwargs):
            pass

        def set_terminated(self, run_id, status):
            statuses[run_id] = status

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid")
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", RecordingClient)

    def fail(*args, **kwargs):
        raise ValueError("deliberately invalid model calculation")

    monkeypatch.setattr(RandomForestClassifier, "fit", fail)
    report = train(
        spec_for(table, tracking="remote", dataset_inputs=False, dataset_snapshots=[]),
        tmp_path / "out",
    )
    assert report["training_status"] == "failed"
    assert report["tracking"]["status"] == "complete"
    assert report["tracking"]["model_uri"] is None
    assert statuses == {"1": "FAILED", "2": "FAILED"}
