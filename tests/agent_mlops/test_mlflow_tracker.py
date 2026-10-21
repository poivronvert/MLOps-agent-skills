"""Model logging contracts for the standalone experiment tracking wrapper."""

import builtins
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/mlops/ml-experiment-tracking/scripts/mlflow_tracker.py"
)
SPEC = importlib.util.spec_from_file_location("model_logging_tracker", SCRIPT)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


class ReviewedTree(DecisionTreeClassifier):
    """Test-owned top-level type, explicitly reviewed as a plain tree subclass."""


@pytest.fixture
def data():
    features = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
    return features, np.array([0, 0, 1, 1])


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    pytest.importorskip("skops", reason="Real serialization requires the tracking extra")
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment("wrapper", artifact_location=(tmp_path / "artifacts").as_uri())
    instance = wrapper.ExperimentTracker("wrapper", uri)
    instance.start_run("model")
    yield instance
    if mlflow.active_run():
        instance.end_run()
    mlflow.set_tracking_uri(previous)


def logged_models(tracker):
    return tracker.client.search_logged_models(
        experiment_ids=[tracker._active_run.info.experiment_id],
    )


@pytest.mark.parametrize("with_example", [False, True])
def test_default_skops_real_roundtrip(tracker, data, with_example):
    features, labels = data
    model = LogisticRegression(random_state=7).fit(features, labels)
    assert tracker.log_model(
        model, input_example=features if with_example else None,
    ) is None
    saved = logged_models(tracker)
    assert len(saved) == 1
    uri = f"models:/{saved[0].model_id}"
    metadata = Model.load(uri)
    assert metadata.flavors["sklearn"]["serialization_format"] == "skops"
    assert (metadata.signature is not None) == with_example
    for loaded in (mlflow.sklearn.load_model(uri), mlflow.pyfunc.load_model(uri)):
        np.testing.assert_array_equal(loaded.predict(features), model.predict(features))


def test_unreviewed_type_rejected_then_explicitly_trusted(tracker, data, caplog):
    features, labels = data
    model = ReviewedTree(random_state=7).fit(features, labels)
    with caplog.at_level(logging.INFO), pytest.raises(MlflowException, match="untrusted types"):
        tracker.log_model(model, serialization_format="skops")
    assert "Logged model to" not in caplog.text
    # Trust only this test-owned class and the tree built locally by fit(), never
    # all types discovered in an arbitrary artifact. skops 0.16 requires explicit
    # trust for Tree's native node storage, even in a standard sklearn estimator.
    trusted = [f"{__name__}.ReviewedTree", "sklearn.tree._tree.Tree"]
    tracker.log_model(
        model, artifact_path="reviewed", serialization_format="skops",
        skops_trusted_types=trusted,
    )
    saved = [model for model in logged_models(tracker) if model.name == "reviewed"]
    assert len(saved) == 1
    uri = f"models:/{saved[0].model_id}"
    assert Model.load(uri).flavors["sklearn"]["skops_trusted_types"] == trusted
    np.testing.assert_array_equal(mlflow.pyfunc.load_model(uri).predict(features), labels)


def test_real_sdk_missing_skops_is_reported_and_raised(tracker, data, monkeypatch, caplog):
    features, labels = data
    model = DecisionTreeClassifier(random_state=7).fit(features, labels)
    original_import = builtins.__import__
    error = ModuleNotFoundError("No module named 'skops'", name="skops")

    def without_skops(name, *args, **kwargs):
        if name == "skops" or name.startswith("skops."):
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_skops)
    with caplog.at_level(logging.INFO), pytest.raises(ModuleNotFoundError) as raised:
        tracker.log_model(model)
    assert raised.value is error
    assert "--extra tracking" in caplog.text
    assert "Logged model to" not in caplog.text
    assert mlflow.active_run().info.run_id == tracker._active_run.info.run_id


def test_random_forest_requires_explicit_tree_trust(tracker, data, caplog):
    features, labels = data
    model = RandomForestClassifier(n_estimators=5, random_state=7).fit(features, labels)
    with caplog.at_level(logging.INFO), pytest.raises(MlflowException, match="untrusted types"):
        tracker.log_model(model, artifact_path="unreviewed-forest")
    assert "Logged model to" not in caplog.text
    # The tree nodes belong to this locally fitted forest. Do not grant trust
    # based solely on types discovered inside an artifact from another source.
    trusted = ["sklearn.tree._tree.Tree"]
    tracker.log_model(
        model, artifact_path="reviewed-forest", input_example=features,
        skops_trusted_types=trusted,
    )
    saved = [saved for saved in logged_models(tracker) if saved.name == "reviewed-forest"]
    assert len(saved) == 1
    uri = f"models:/{saved[0].model_id}"
    metadata = Model.load(uri)
    assert metadata.flavors["sklearn"]["serialization_format"] == "skops"
    assert metadata.flavors["sklearn"]["skops_trusted_types"] == trusted
    for loaded in (mlflow.sklearn.load_model(uri), mlflow.pyfunc.load_model(uri)):
        np.testing.assert_array_equal(loaded.predict(features), model.predict(features))
    np.testing.assert_allclose(
        mlflow.sklearn.load_model(uri).predict_proba(features),
        model.predict_proba(features), atol=1e-10, rtol=0,
    )


def fake_tracker():
    instance = wrapper.ExperimentTracker.__new__(wrapper.ExperimentTracker)
    instance.mlflow = SimpleNamespace(
        sklearn=SimpleNamespace(log_model=Mock()),
        xgboost=SimpleNamespace(log_model=Mock()),
        pytorch=SimpleNamespace(log_model=Mock()),
    )
    instance._active_run = object()
    return instance


def framework_model(module):
    return type("Model", (), {"__module__": module})()


@pytest.mark.parametrize(
    ("module", "flavour"), [
        ("sklearn.tree", "sklearn"), ("custom.models", "sklearn"),
        ("xgboost.core", "xgboost"), ("torch.nn", "pytorch"),
    ],
)
def test_dispatch_only_sklearn_receives_serialization_options(module, flavour):
    tracker = fake_tracker()
    model = framework_model(module)
    tracker.log_model(model, serialization_format="skops", skops_trusted_types=["reviewed.Type"])
    for name in ("sklearn", "xgboost", "pytorch"):
        call = getattr(tracker.mlflow, name).log_model
        assert call.call_count == (1 if name == flavour else 0)
    kwargs = getattr(tracker.mlflow, flavour).log_model.call_args.kwargs
    if flavour == "sklearn":
        assert kwargs["serialization_format"] == "skops"
        assert kwargs["skops_trusted_types"] == ["reviewed.Type"]
    else:
        assert "serialization_format" not in kwargs
        assert "skops_trusted_types" not in kwargs


def test_omitted_serialization_options_preserve_sdk_defaults():
    tracker = fake_tracker()
    tracker.log_model(framework_model("sklearn.tree"))
    kwargs = tracker.mlflow.sklearn.log_model.call_args.kwargs
    assert "serialization_format" not in kwargs
    assert "skops_trusted_types" not in kwargs


@pytest.mark.parametrize(
    ("module", "flavour"), [
        ("sklearn.tree", "sklearn"), ("custom.models", "sklearn"),
        ("xgboost.core", "xgboost"), ("torch.nn", "pytorch"),
    ],
)
@pytest.mark.parametrize("failure", ["skops", "other_dependency", "serialization", "upload"])
def test_failures_propagate_original_without_retry_or_run_mutation(
    module, flavour, failure, caplog,
):
    tracker = fake_tracker()
    run = tracker._active_run
    error = (
        ModuleNotFoundError(f"No module named '{failure}'", name=failure)
        if failure in {"skops", "other_dependency"}
        else RuntimeError(f"{failure} failed")
    )
    call = getattr(tracker.mlflow, flavour).log_model
    call.side_effect = error
    with caplog.at_level(logging.INFO), pytest.raises(type(error)) as raised:
        tracker.log_model(framework_model(module))
    assert raised.value is error
    assert call.call_count == 1
    assert sum(getattr(tracker.mlflow, name).log_model.call_count
               for name in ("sklearn", "xgboost", "pytorch")) == 1
    assert tracker._active_run is run
    assert "Logged model to" not in caplog.text
    assert "Failed to log model" in caplog.text
    assert ("--extra tracking" in caplog.text) == (failure == "skops")


def test_signature_failure_remains_optional(data, caplog):
    tracker = fake_tracker()
    model = framework_model("sklearn.tree")
    model.predict = Mock(side_effect=ValueError("cannot infer example"))
    with caplog.at_level(logging.WARNING):
        tracker.log_model(model, input_example=data[0])
    tracker.mlflow.sklearn.log_model.assert_called_once()
    assert tracker.mlflow.sklearn.log_model.call_args.kwargs["signature"] is None
    assert "Could not infer signature" in caplog.text
