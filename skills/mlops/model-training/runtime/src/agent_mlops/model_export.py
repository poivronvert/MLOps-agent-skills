"""Bind fitted prediction semantics to an official MLflow PythonModel."""

from contextlib import contextmanager
from copy import copy
from importlib.metadata import version
from pathlib import Path
import tempfile

import joblib
import mlflow.pyfunc
from mlflow.models import infer_signature, ModelSignature
from mlflow.types import ColSpec, Schema
from mlflow.types.schema import AnyType
import pandas as pd

from .inference import prediction_frame
from .temporal import ID, TIME, normalize_time


def input_example(model, raw):
    """Select legal prediction inputs, excluding truth and unused columns."""
    spec = model.spec
    if spec.mode == "tabular" or (
        spec.mode == "future_event" and model.history is None
    ):
        return raw[model.features].head(5).copy()
    t = spec.temporal
    columns = list(
        dict.fromkeys(
            t.entity_keys
            + [t.time_column]
            + (
                list(t.known_future)
                if spec.mode == "forecast"
                else spec.features
                + list(spec.history_lags)
                + list(spec.rolling_windows)
            )
        )
    )
    normalized = normalize_time(raw, t)
    if spec.mode == "forecast":
        future = model.forecaster.make_future_dataframe(max(t.horizons))
        future[TIME] = future[TIME].dt.tz_localize("UTC")
        if t.known_future:
            selected = future.merge(normalized, on=[ID, TIME], validate="one_to_one")
        else:
            selected = future.merge(model.mapping, on=ID, validate="many_to_one")
            selected[t.time_column] = selected[TIME]
    else:
        origin = t.timestamp(t.test_start)
        selected = (
            normalized[normalized[TIME] >= origin].groupby(ID, sort=False).head(5)
        )
    example = selected[columns].copy()
    # Retain timezone information; the temporal predictor validates/normalizes it.
    example[t.time_column] = selected[TIME]
    return example.reset_index(drop=True)


def requirements(spec):
    packages = [
        "mlflow-skinny",
        "scikit-learn",
        "numpy",
        "pandas",
        "scipy",
        "joblib",
        "cloudpickle",
        "pydantic",
        "pyarrow",
    ]
    if spec.mode == "forecast":
        packages.append("mlforecast")
    if spec.model in {"xgb", "lgbm"}:
        packages.append({"xgb": "xgboost", "lgbm": "lightgbm"}[spec.model])
    return [f"{name}=={version(name)}" for name in packages]


@contextmanager
def model_definition(model, example, *, output_signature=True):
    """One definition shared by official save_model and log_model."""
    spec = model.spec
    code = str(Path(__file__).resolve().parent)
    dependencies = requirements(spec)
    regression = spec.mode == "tabular" and spec.task == "regression"
    # Interrupted jobs retain a usable partial artifact without further inference.
    expected = None
    if output_signature:
        expected = (
            model.predict(example) if regression else prediction_frame(model, example)
        )
    signature = infer_signature(example, expected)
    # MLflow's scalar double rejects pandas int64; long rejects NaN. Preserve both
    # numeric representations through official AnyType, with predictor dtype validation.
    schema = [
        ColSpec(AnyType(), name=column.name)
        if (
            pd.api.types.is_numeric_dtype(example[column.name])
            and not pd.api.types.is_bool_dtype(example[column.name])
        )
        or pd.api.types.is_datetime64_any_dtype(example[column.name])
        else ColSpec(column.type, name=column.name)
        for column in signature.inputs
    ]
    if spec.mode == "forecast" and not spec.temporal.known_future:
        schema = [ColSpec(c.type, name=c.name, required=False) for c in schema]
    options = dict(
        code_paths=[code],
        pip_requirements=dependencies,
        signature=ModelSignature(inputs=Schema(schema), outputs=signature.outputs),
        input_example=example,
        metadata=model.metadata(),
    )
    with tempfile.TemporaryDirectory(prefix="mlflow-model-") as directory:
        directory = Path(directory)
        artifact = directory / "model.joblib"
        saved_model = model
        if spec.mode == "forecast":
            # Preserve the existing artifact's history scope without changing the
            # fitted object used by evaluation or subsequent predictions.
            saved_model = copy(model)
            saved_model.history = model.history[
                [ID, TIME, *spec.temporal.historical_features]
            ].copy()
        joblib.dump(saved_model, artifact)
        yield (
            mlflow.pyfunc,
            dict(
                python_model=str(Path(__file__).with_name("model_code.py")),
                artifacts={"model": str(artifact)},
                **options,
                uv_project_path=Path(__file__).resolve().parents[2],
                uv_groups=[],
                uv_extras=sorted(
                    ({"forecasting"} if spec.mode == "forecast" else set())
                    | (
                        {{"xgb": "xgboost", "lgbm": "lightgbm"}[spec.model]}
                        if spec.model in {"xgb", "lgbm"}
                        else set()
                    )
                ),
            ),
        )
