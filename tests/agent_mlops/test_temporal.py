"""Temporal integration fixtures: real MLForecast, isolated entities, reload tolerances."""

import hashlib

import numpy as np
import pandas as pd
import pytest

import mlflow.pyfunc
from agent_mlops.config import TaskSpec, TemporalSpec
from agent_mlops.models import ForecastPredictor
from agent_mlops.temporal import ID, TIME, normalize_time
from agent_mlops.training import train

TOL = 1e-10  # Declared before any model calculation.


def panel(tmp_path, entities=("A", "B")):
    frames = []
    for entity in entities:
        x = np.arange(100)
        frames.append(
            pd.DataFrame(
                {
                    "id": entity,
                    "time": pd.date_range(
                        "2025-01-01", periods=100, freq="h", tz="UTC"
                    ),
                    "y": np.sin(x / 4) + (entity == "B") * 8,
                    "sensor": x.astype(float),
                    "calendar": (x % 24).astype(float),
                    "static": float(entity == "B"),
                    "event": (x % 9 == 0).astype(int),
                }
            )
        )
    df = pd.concat(frames).sample(frac=1, random_state=17).reset_index(drop=True)
    path = tmp_path / "panel.csv"
    df.to_csv(path, index=False)
    return path, df


def temporal_spec(path, mode="forecast", **overrides):
    t = TemporalSpec(
        time_column="time",
        entity_keys=["id"],
        timezone="UTC",
        frequency="h",
        origins=["2025-01-02T06:00Z", "2025-01-02T18:00Z"],
        test_start="2025-01-04T00:00Z",
        horizons=[1, 2, 4],
        validation_steps=8,
        window="4h",
        reporting_delay="2h",
        lags=[1, 2, 4],
        known_future={"calendar": "fixture: published calendar"},
        static_features=["static"],
    )
    base = dict(
        input=str(path),
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        target="y" if mode == "forecast" else "event",
        task="regression" if mode == "forecast" else "classification",
        mode=mode,
        temporal=t,
        model="rf",
        fixed_params={"n_estimators": 12, "max_depth": 4},
        hpo=False,
        metric="mae" if mode == "forecast" else "f1_weighted",
        extras=["forecasting"] if mode == "forecast" else [],
    )
    if mode == "future_event":
        base.update(
            features=["sensor"],
            availability={"sensor": "0h"},
            history_lags={"sensor": [1, 2]},
            threshold=0.6,
            positive_label=1,
        )
    base.update(overrides)
    return TaskSpec(**base)


@pytest.mark.forecasting
def test_T07_T08_T13_recursive_truth_invariance_future_alignment_and_roles(tmp_path):
    pytest.importorskip("mlforecast")
    path, df = panel(tmp_path)
    spec = temporal_spec(path)
    normalized = normalize_time(df, spec.temporal)
    origin = spec.temporal.timestamp(spec.temporal.test_start)
    history = normalized[normalized[TIME] < origin]
    future = df[
        (pd.to_datetime(df.time) >= origin)
        & (pd.to_datetime(df.time) < origin + 4 * spec.temporal.step)
    ]
    model = ForecastPredictor(spec, spec.fixed_params).fit(history)
    original = model.predict(future)
    changed = future.sample(frac=1, random_state=2).copy()
    changed["y"] = 1e9
    np.testing.assert_allclose(
        original.model, model.predict(changed).model, atol=TOL, rtol=0
    )
    changed.loc[changed.id == "A", "calendar"] += 100
    after = model.predict(changed)
    np.testing.assert_allclose(
        original[original.id == "B"].model,
        after[after.id == "B"].model,
        atol=TOL,
        rtol=0,
    )
    for bad in [
        future.drop(columns="id"),
        pd.concat([future, future.iloc[:1]]),
        future.iloc[1:],
        future.assign(calendar=np.nan),
    ]:
        with pytest.raises(ValueError):
            model.predict(bad)
    with pytest.raises(ValueError, match="missing"):
        model.predict(None)
    corrupted = history.copy()
    corrupted.loc[corrupted.index[0], "static"] += 1
    with pytest.raises(ValueError, match="static"):
        ForecastPredictor(spec, spec.fixed_params).fit(corrupted)


@pytest.mark.forecasting
def test_C03_T09_T10_T12_forecast_train_reload_and_boundaries(tmp_path):
    pytest.importorskip("mlforecast")
    path, df = panel(tmp_path)
    spec = temporal_spec(path)
    output = tmp_path / "out"
    report = train(spec, output)
    assert report["training_status"] == "completed", report
    assert report["evaluation"]["refit"]["end"] < str(
        spec.temporal.timestamp(spec.temporal.test_start)
    )
    groups = report["evaluation"]["holdout"]["horizons"]
    assert [g["horizon"] for g in groups] == [1, 2, 4]
    assert all(g["n"] == 2 for g in groups)
    assert report["evaluation"]["holdout"]["score"] == pytest.approx(
        np.mean([g["score"] for g in groups])
    )
    bundle = mlflow.pyfunc.load_model(str(output / "model"))
    origin = spec.temporal.timestamp(spec.temporal.test_start)
    future = df[
        (pd.to_datetime(df.time) >= origin)
        & (pd.to_datetime(df.time) < origin + 4 * spec.temporal.step)
    ]
    prediction = bundle.predict(future)
    for group in groups:
        np.testing.assert_allclose(
            prediction[prediction.horizon == group["horizon"]].model,
            group["y_pred"],
            atol=TOL,
            rtol=0,
        )
    # Last-value baseline calculated from exactly the same origin and entities.
    truth = normalize_time(df, spec.temporal)
    last = truth[truth[TIME] < origin].groupby(ID).y.last()
    for group in groups:
        manual = np.mean(
            np.abs(np.array(group["y_true"]) - last.loc[group["entities"]].to_numpy())
        )
        assert group["baseline_score"] == pytest.approx(manual)


@pytest.mark.parametrize("entities", [("A",), ("A", "B")])
def test_C03_T06_T09_T12_event_threshold_history_reload(tmp_path, entities):
    path, df = panel(tmp_path, entities)
    spec = temporal_spec(path, "future_event")
    out = tmp_path / "event"
    report = train(spec, out)
    assert report["training_status"] == "completed", report
    bundle = mlflow.pyfunc.load_model(str(out / "model"))
    origin = spec.temporal.timestamp(spec.temporal.test_start)
    future = df[
        (pd.to_datetime(df.time) >= origin)
        & (pd.to_datetime(df.time) < origin + 4 * spec.temporal.step)
    ].sort_values(["id", "time"])
    probabilities = bundle.predict(future).filter(like="probability_").to_numpy()
    np.testing.assert_array_equal(
        bundle.predict(future).prediction,
        (probabilities[:, 1] >= 0.6).astype(float),
    )
    assert (
        pd.Timestamp(report["evaluation"]["refit"]["end"]) + pd.Timedelta("6h")
        <= origin
    )
    for fold in report["evaluation"]["cv"]:
        assert pd.Timestamp(fold["train"]["end"]) + pd.Timedelta("6h") <= pd.Timestamp(
            fold["origin"]
        )
        assert "baseline_score" in fold


@pytest.mark.forecasting
def test_T07_24_step_forecast_without_future_target(tmp_path):
    pytest.importorskip("mlforecast")
    path, df = panel(tmp_path, ("A",))
    spec = temporal_spec(path)
    t = spec.temporal.model_copy(
        update={"horizons": list(range(1, 25)), "known_future": {}}
    )
    spec = spec.model_copy(update={"temporal": t})
    history = normalize_time(df, t).iloc[:60]
    model = ForecastPredictor(spec, spec.fixed_params).fit(history)
    predictions = model.predict()
    assert len(predictions) == 24
    assert predictions.horizon.tolist() == list(range(1, 25))
    assert predictions[TIME].min() > history[TIME].max()


def test_C03_event_prediction_preserves_requested_row_order(tmp_path):
    path, df = panel(tmp_path)
    spec = temporal_spec(path, "future_event")
    train(spec, tmp_path / "out")
    bundle = mlflow.pyfunc.load_model(str(tmp_path / "out/model"))
    origin = spec.temporal.timestamp(spec.temporal.test_start)
    future = df[
        (pd.to_datetime(df.time) >= origin)
        & (pd.to_datetime(df.time) < origin + 4 * spec.temporal.step)
    ].sort_values(["id", "time"])
    sorted_probs = bundle.predict(future).filter(like="probability_").to_numpy()
    perm = np.array([6, 1, 4, 0, 2, 7, 5, 3])
    np.testing.assert_allclose(
        bundle.predict(future.iloc[perm]).filter(like="probability_").to_numpy(),
        sorted_probs[perm],
        atol=TOL,
        rtol=0,
    )


@pytest.mark.forecasting
def test_T09_holdout_truth_never_changes_selection_or_refit(tmp_path):
    pytest.importorskip("mlforecast")
    path, df = panel(tmp_path)
    spec = temporal_spec(
        path,
        hpo=True,
        n_trials=2,
        fixed_params={"n_estimators": 8},
        search_space={"max_depth": {"type": "int", "low": 2, "high": 3}},
    )
    first = train(spec, tmp_path / "first")
    changed = df.copy()
    changed.loc[
        pd.to_datetime(changed.time)
        >= spec.temporal.timestamp(spec.temporal.test_start),
        "y",
    ] += 1e5
    changed.to_csv(path, index=False)
    second_spec = spec.model_copy(
        update={"data_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
    second = train(second_spec, tmp_path / "second")
    assert first["best_params"] == second["best_params"]
    assert first["best_cv_score"] == second["best_cv_score"]
    for before, after in zip(
        first["evaluation"]["holdout"]["horizons"],
        second["evaluation"]["holdout"]["horizons"],
    ):
        np.testing.assert_allclose(before["y_pred"], after["y_pred"], atol=TOL, rtol=0)
        assert before["score"] != after["score"]
