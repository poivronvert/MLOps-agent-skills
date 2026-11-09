"""Executable BDD: fixed data, hand-computable boundaries and reload oracles."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from agent_mlops.config import TaskSpec
from agent_mlops.data import read_snapshot
from agent_mlops.temporal import (
    normalize_time,
    history_features,
    event_labels,
    mature_mask,
)
from agent_mlops.evaluation import aggregate_scores, score
from agent_mlops.training import train
import mlflow.pyfunc


def spec_for(path, **updates):
    values = dict(
        input=str(path),
        target="y",
        task="classification",
        features=["x", "kind"],
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model="rf",
        fixed_params={"n_estimators": 8, "max_depth": 3},
        hpo=False,
        cv_folds=2,
        holdout_fraction=0.2,
        provenance={
            "target": {"value": "y", "source": "fixture:1", "status": "confirmed"}
        },
    )
    values.update(updates)
    return TaskSpec(**values)


def time_spec(**updates):
    values = dict(
        time_column="time",
        entity_keys=["id"],
        timezone="UTC",
        frequency="h",
        horizons=[1, 2],
        origins=["2025-01-02T00:00:00Z"],
        test_start="2025-01-03T00:00:00Z",
        window="24h",
    )
    values.update(updates)
    from agent_mlops.config import TemporalSpec

    return TemporalSpec(**values)


def test_A11_invalid_metric_and_missing_target_stop_before_fit(
    table, tmp_path, monkeypatch
):
    from sklearn.ensemble import RandomForestClassifier

    monkeypatch.setattr(
        RandomForestClassifier, "fit", lambda *a, **k: pytest.fail("fit called")
    )
    with pytest.raises(ValidationError, match="metric"):
        spec_for(table, task="regression", metric="f1_weighted")
    with pytest.raises(ValueError, match="target"):
        train(spec_for(table, target="missing"), tmp_path / "out")


def test_A09_changed_snapshot_requires_revalidation(table):
    spec = spec_for(table)
    table.write_text(table.read_text() + "70,a,ok\n")
    with pytest.raises(ValueError, match="content"):
        read_snapshot(spec)


def test_C02_C03_C04_H04_M01_official_model_and_score_boundaries(
    table, tmp_path, monkeypatch
):
    import optuna

    monkeypatch.setattr(
        optuna, "create_study", lambda **kw: pytest.fail("HPO disabled")
    )
    out = tmp_path / "out"
    report = train(spec_for(table), out)
    bundle = mlflow.pyfunc.load_model(str(out / "model"))
    inputs = pd.DataFrame({"x": [3, 9], "kind": ["unseen", "a"]})
    expected = bundle.predict(inputs)
    assert set(expected.prediction) <= {"ok", "fault"}
    assert set(bundle.metadata.metadata["label_mapping"]) == {"ok", "fault"}
    assert report["tracking"]["status"] == "off"
    assert report["evaluation"]["train"]["n"] == 48
    assert report["evaluation"]["holdout"]["n"] == 12
    assert (
        report["evaluation"]["holdout"]["independence"] == "independent_within_this_run"
    )
    assert report["best_cv_score"] == np.mean(
        [f["score"] for f in report["trials"][0]["folds"]]
    )
    assert report["training_status"] == "completed"
    with pytest.raises(Exception, match="[Ss]chema|inputs|columns"):
        bundle.predict(inputs.drop(columns="kind"))
    (out / "task_spec.json").unlink()
    # Reports are independent artifacts, not prerequisites for official loading.
    pd.testing.assert_frame_equal(
        mlflow.pyfunc.load_model(str(out / "model")).predict(inputs), expected
    )


def test_C02_regression_does_not_encode_text(table, tmp_path):
    with pytest.raises(ValueError, match="numeric"):
        train(spec_for(table, task="regression"), tmp_path / "out")


def test_T01_time_rules():
    spec = time_spec()
    df = pd.DataFrame(
        {"id": ["A", "A"], "time": ["2025-01-01", "2025-01-01"], "v": [1, 2]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        normalize_time(df, spec)
    df["time"] = ["2025-01-01 00:00", "2025-01-01 02:00"]
    with pytest.raises(ValueError, match="missing periods"):
        normalize_time(df, spec)


def test_T02_T03_T04_entity_history_and_availability():
    df = pd.DataFrame(
        {
            "id": ["B", "A", "B", "A"],
            "time": [
                "2025-01-01 00:00",
                "2025-01-01 00:00",
                "2025-01-01 01:00",
                "2025-01-01 01:00",
            ],
            "v": [100, 10, 200, 20],
        }
    )
    spec = time_spec(history_policy="exclude")
    df = normalize_time(df, spec)
    result, excluded = history_features(df, spec, {"v": [1]}, {}, {"v": "0h"})
    assert excluded == 2
    assert result.set_index("id")["v_lag1"].to_dict() == {"A": 10, "B": 100}
    with pytest.raises(ValueError, match="history"):
        history_features(
            df, time_spec(history_policy="reject"), {"v": [1]}, {}, {"v": "0h"}
        )
    # Event at 00:00, available at 01:30: cannot enter origin 01:00.
    result, _ = history_features(df, spec, {"v": [1]}, {}, {"v": "90min"})
    assert result.empty


def test_T05_T06_open_closed_window_unknown_tail_and_maturity():
    spec = time_spec()
    df = pd.DataFrame(
        {
            "id": "A",
            "time": pd.date_range("2025-01-01", periods=49, freq="h", tz="UTC"),
            "event": [1] + [0] * 23 + [1] + [0] * 24,
        }
    )
    labels = event_labels(df, spec, "event", 1)
    assert labels.iloc[0] == 1
    assert labels.iloc[24] == 0  # excludes event at t
    assert labels.iloc[25:].isna().all()
    times = pd.Series(pd.to_datetime(["2025-01-08T22:00Z", "2025-01-08T23:00Z"]))
    assert mature_mask(
        times, pd.Timestamp("2025-01-10T00:00Z"), "24h", "2h"
    ).tolist() == [True, False]


def test_T10_hand_computed_aggregation_and_missing_policy():
    horizons = [
        [score("mae", [0, 0], errors) for errors in fold]
        for fold in [[[0, 2], [4, 8]], [[2, 4], [6, 10]]]
    ]
    assert horizons == [[1, 6], [3, 8]]
    assert aggregate_scores(horizons) == 4.5
    with pytest.raises(ValueError, match="invalid"):
        aggregate_scores([[1, np.nan], [3, 8]])
    with pytest.raises(ValueError, match="invalid"):
        aggregate_scores([[]])


def test_H01_invalid_search_space_before_fit(table):
    for update in [
        dict(search_space={"bogus": {"type": "int", "low": 1, "high": 2}}),
        dict(search_space={"n_estimators": {"type": "int", "low": 9, "high": 1}}),
        dict(search_space={"n_estimators": {"type": "int", "low": 1, "high": 4}}),
    ]:
        with pytest.raises((ValueError, ValidationError)):
            spec_for(table, hpo=True, **update).validate_model()


@pytest.mark.parametrize(
    "uri",
    [
        "file:/tmp/mlruns",
        "sqlite:///ml.db",
        "/tmp/mlruns",
        "https://user:secret@host",
        "https://host?token=secret",
    ],
)
def test_M03_reject_local_and_credential_uris(table, tmp_path, monkeypatch, uri):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with pytest.raises(ValueError, match="HTTP|credential"):
        train(spec_for(table, tracking="remote"), tmp_path / "out")
    assert not (tmp_path / "ml.db").exists()


def test_T11_exposed_holdout_is_development(table, tmp_path):
    report = train(spec_for(table, holdout_used_for_selection=True), tmp_path / "out")
    assert report["evaluation"]["holdout"]["independence"] == "development_only"


def test_H01_H02_validation_objective_and_trial_budget(table, tmp_path):
    spec = spec_for(
        table,
        hpo=True,
        n_trials=3,
        timeout=60,
        fixed_params={"n_estimators": 8},
        search_space={"max_depth": {"type": "int", "low": 1, "high": 3}},
    )
    report = train(spec, tmp_path / "out")
    assert len(report["trials"]) == 3
    from sklearn.metrics import f1_score

    for trial in report["trials"]:
        assert 1 <= trial["params"]["max_depth"] <= 3
        expected = [
            f1_score(f["y_true"], f["y_pred"], average="weighted")
            for f in trial["folds"]
        ]
        assert trial["score"] == pytest.approx(np.mean(expected))
    assert (
        json.loads((tmp_path / "out" / "task_spec.json").read_text())["timeout"] == 60
    )


def test_Q01_parquet_snapshot_trains_and_reloads(table, tmp_path):
    frame = pd.read_csv(table)
    parquet = tmp_path / "table.parquet"
    frame.to_parquet(parquet, index=False)
    spec = spec_for(parquet)
    report = train(spec, tmp_path / "parquet-output")
    assert report["training_status"] == "completed"
    model = mlflow.pyfunc.load_model(str(tmp_path / "parquet-output/model"))
    assert len(model.predict(frame[["x", "kind"]].iloc[:3])) == 3
