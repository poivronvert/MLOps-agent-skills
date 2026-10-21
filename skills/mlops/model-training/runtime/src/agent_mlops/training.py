"""Sequential fold/trial execution with a hard boundary after tracking failures."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .artifacts import write_run_documents
from .model_export import input_example, model_definition
from .data import inspect_data, read_snapshot
from .evaluation import aggregate_scores, metrics_for, score
from .models import ForecastPredictor, TabularPredictor, optional_import
from .temporal import (
    ID,
    LABEL,
    TIME,
    event_labels,
    history_features,
    mature_mask,
    normalize_time,
)
from .tracking import Tracker, TrackingError


def prepare(spec, raw):
    exclusions = dict(spec.exclusions)
    if spec.mode == "tabular":
        features = (
            spec.features
            if spec.features is not None
            else [c for c in raw if c != spec.target and c not in exclusions]
        )
        missing = set(features) - set(raw)
        if missing or not features:
            raise ValueError(f"feature schema missing or empty: {sorted(missing)}")
        df = raw.copy()
        df[LABEL] = df[spec.target]
        return df, features, exclusions, 0
    t = spec.temporal
    df = normalize_time(raw, t)
    for reserved in [ID, TIME, LABEL]:
        if reserved in raw:
            raise ValueError(f"reserved input column {reserved}")
    for boundary in [*t.origins, t.test_start]:
        if t.timestamp(boundary) not in set(df[TIME]):
            raise ValueError(
                "fold/test boundaries must be present on the common time grid"
            )
    if spec.mode == "forecast":
        optional_import("mlforecast", "forecasting")
        roles = t.static_features + list(t.known_future) + list(t.historical_features)
        if set(roles) & {spec.target, t.time_column, *t.entity_keys}:
            raise ValueError("forecast target/time/entity cannot be exogenous features")
        if spec.features is not None and set(spec.features) != set(roles):
            raise ValueError("forecast features must match explicit roles")
        if set(roles) - set(df):
            raise ValueError("missing forecast feature columns")
        for c in t.static_features:
            if df.groupby(ID)[c].nunique(dropna=False).gt(1).any():
                raise ValueError(f"static feature role varies within entity: {c}")
        for col, lags in t.historical_features.items():
            if col not in spec.availability or any(
                lag * t.step
                < max(t.horizons) * t.step + pd.Timedelta(spec.availability[col])
                for lag in lags
            ):
                raise ValueError(
                    f"historical feature {col} not available for all recursive horizons"
                )
        for c in raw:
            if c not in roles + [spec.target, t.time_column, *t.entity_keys]:
                exclusions.setdefault(c, "not assigned a forecast feature role")
        return df, roles, exclusions, 0
    if spec.features is None:
        raise ValueError(
            "future_event requires explicit features and availability evidence"
        )
    if t.event_value not in set(df[spec.target]):
        raise ValueError("confirmed event_value is absent from target")
    df[LABEL] = event_labels(df, t, spec.target, t.event_value)
    df, excluded = history_features(
        df, t, spec.history_lags, spec.rolling_windows, spec.availability
    )
    generated = [
        f"{c}_lag{lag}" for c, lags in spec.history_lags.items() for lag in lags
    ]
    generated += [
        f"{c}_rolling{w}"
        for c, windows in spec.rolling_windows.items()
        for w in windows
    ]
    features = spec.features + generated
    if len(features) != len(set(features)):
        raise ValueError("duplicate configured features")
    for col in spec.features:
        if col not in df or col in [
            spec.target,
            t.time_column,
            *t.entity_keys,
            ID,
            TIME,
            LABEL,
        ]:
            raise ValueError(f"invalid raw event feature: {col}")
        if (
            col not in spec.availability
            or pd.Timedelta(spec.availability[col]) != t.step * 0
        ):
            raise ValueError(
                f"raw feature {col} unavailable at prediction time; use legal historical lags"
            )
    for c in raw:
        if c not in features + [spec.target, t.time_column, *t.entity_keys]:
            exclusions.setdefault(c, "excluded from direct prediction features")
    return df, features, exclusions, excluded


def splits(spec, df):
    if spec.mode == "tabular":
        indices = np.arange(len(df))
        dev, test = indices, np.array([], dtype=int)
        if spec.holdout_fraction:
            dev, test = train_test_split(
                indices,
                test_size=spec.holdout_fraction,
                random_state=spec.seed,
                stratify=df[LABEL] if spec.task == "classification" else None,
            )
        splitter = (
            StratifiedKFold
            if spec.cv_strategy == "stratified" and spec.task == "classification"
            else KFold
        )
        cv = splitter(n_splits=spec.cv_folds, shuffle=True, random_state=spec.seed)
        folds = [
            (df.iloc[dev[a]], df.iloc[dev[b]], None)
            for a, b in cv.split(dev, df.iloc[dev][LABEL])
        ]
        return folds, df.iloc[dev], df.iloc[test]
    t = spec.temporal
    test_start = t.timestamp(t.test_start)

    def training_before(cutoff):
        mask = df[TIME] < cutoff
        if spec.mode == "future_event":
            mask &= df[LABEL].notna() & mature_mask(
                df[TIME], cutoff, t.window, t.reporting_delay
            )
        return df[mask]

    folds = []
    for origin in t.origins:
        cutoff = t.timestamp(origin)
        end = cutoff + t.step * (
            max(t.horizons) if spec.mode == "forecast" else t.validation_steps
        )
        if end > test_start:
            raise ValueError("development validation overlaps final test interval")
        valid = df[(df[TIME] >= cutoff) & (df[TIME] < end)]
        if spec.mode == "future_event":
            valid = valid[
                valid[LABEL].notna()
                & mature_mask(valid[TIME], test_start, t.window, t.reporting_delay)
            ]
        folds.append((training_before(cutoff), valid, cutoff))
    test = df[df[TIME] >= test_start]
    if t.test_end:
        test = test[test[TIME] <= t.timestamp(t.test_end)]
    if spec.mode == "forecast":
        test = test[test[TIME] < test_start + max(t.horizons) * t.step]
    else:
        test = test[test[LABEL].notna()]
    return folds, training_before(test_start), test


def bounds(df):
    return {
        "n": len(df),
        "row_indices": df.index.tolist(),
        "start": str(df[TIME].min()) if TIME in df and len(df) else None,
        "end": str(df[TIME].max()) if TIME in df and len(df) else None,
    }


def fit_model(spec, params, train_df, features):
    if train_df.empty:
        raise ValueError("no eligible training samples")
    if spec.mode == "forecast":
        return ForecastPredictor(spec, params).fit(train_df)
    return TabularPredictor(spec, params).fit(train_df[features], train_df[LABEL])


def evaluate(spec, model, training, valid, features, origin=None):
    if valid.empty:
        raise ValueError("invalid fold/horizon: no validation samples")
    result = {
        "train": bounds(training),
        "validation": bounds(valid),
        "origin": str(origin) if origin is not None else None,
    }
    if spec.mode == "forecast":
        t = spec.temporal
        # Only explicitly known columns enter the forecast API; target truth stays here.
        future = (
            valid[[*t.entity_keys, t.time_column, *t.known_future]]
            if t.known_future
            else None
        )
        pred = model.predict(future)
        paired = pred.merge(
            valid[[ID, TIME, spec.target]],
            on=[ID, TIME],
            how="left",
            validate="one_to_one",
        )
        if paired[spec.target].isna().any():
            raise ValueError("invalid horizon: missing validation truth")
        paired["baseline"] = paired[ID].map(training.groupby(ID)[spec.target].last())
        groups = []
        for horizon in t.horizons:
            rows = paired[paired.horizon == horizon]
            groups.append(
                {
                    "horizon": horizon,
                    "n": len(rows),
                    "times": rows[TIME].astype(str).tolist(),
                    "entities": rows[ID].tolist(),
                    "y_true": rows[spec.target].tolist(),
                    "y_pred": rows.model.tolist(),
                    "score": score(spec.metric, rows[spec.target], rows.model),
                    "baseline_score": score(
                        spec.metric, rows[spec.target], rows.baseline
                    ),
                }
            )
        result.update(
            horizons=groups,
            score=aggregate_scores([[g["score"] for g in groups]]),
            baseline_score=aggregate_scores([[g["baseline_score"] for g in groups]]),
            excluded_entities=model.excluded_entities,
        )
    else:
        predictions = model.predict(valid[features])
        truth = valid[LABEL]
        baseline = (
            training[LABEL].mode().iloc[0]
            if spec.task == "classification"
            else training[LABEL].mean()
        )
        result.update(
            score=score(spec.metric, truth, predictions),
            baseline_score=score(spec.metric, truth, np.repeat(baseline, len(truth))),
            y_true=truth.tolist(),
            y_pred=predictions.tolist(),
            metrics=metrics_for(spec.task, truth, predictions),
        )
    return result


def train(spec, output, tracker_factory=Tracker):
    started = time.monotonic()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "output must be empty; use a new directory to preserve previous results"
        )
    spec.validate_model()
    raw = read_snapshot(spec)
    df, features, exclusions, excluded = prepare(spec, raw)
    folds, refit, holdout = splits(spec, df)
    if any(a.empty or b.empty for a, b, _ in folds) or refit.empty:
        raise ValueError(
            "empty training/validation fold; adjust time boundaries or history policy"
        )
    if spec.mode != "tabular" and holdout.empty:
        raise ValueError("empty final test interval")
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": spec.model,
        "task": spec.task,
        "mode": spec.mode,
        "input_file": spec.input,
        "data_sha256": spec.data_sha256,
        "target_column": spec.target,
        "dataset_rows": len(raw),
        "dataset_features": len(features),
        "features": features,
        "exclusions": exclusions,
        "data_check": inspect_data(raw),
        "excluded_history_samples": excluded,
        "unknown_event_labels": int(df[LABEL].isna().sum())
        if spec.mode == "future_event"
        else 0,
        "cv_folds": len(folds),
        "cv_strategy": "temporal" if spec.mode != "tabular" else spec.cv_strategy,
        "metric": spec.metric,
        "direction": spec.direction,
        "aggregation": spec.aggregation,
        "n_trials": spec.n_trials if spec.hpo else 0,
        "trials": [],
        "best_trial": None,
        "best_cv_score": None,
        "best_params": None,
        "train_resubstitution_metrics": {},
        "evaluation": {},
        "training_status": "not_started",
        "search_status": "pending" if spec.hpo else "disabled",
        "artifact_status": "diagnostics_only",
        "model_path": None,
        "holdout_history": spec.holdout_history,
        "limitations": [
            "timeout prevents new trials; it does not terminate an active calculation"
        ],
    }
    tracker = tracker_factory(spec)
    state = {"model": None, "best": None, "final": False}

    def compute_trial(number, params):
        result = {"number": number, "params": params, "status": "running", "folds": []}
        report["trials"].append(result)
        tracker.start_trial(number, params)
        report["training_status"] = "running"
        try:
            for i, (training, valid, origin) in enumerate(folds):
                tracker.dataset(
                    training, features, name=f"fold-{i}-train", context="training"
                )
                model = fit_model(spec, params, training, features)
                state["model"] = model  # preserve even if evaluation or tracking fails
                tracker.dataset(
                    valid, features, name=f"fold-{i}-validation", context="validation"
                )
                fold = evaluate(spec, model, training, valid, features, origin)
                result["folds"].append(fold)
                if not np.isfinite(fold["score"]):
                    fold["score"] = None
                    fold["error"] = "invalid nonfinite score"
                    raise ValueError("invalid nonfinite score")
                tracker.fold(i, fold)  # no next fit/predict after a failure
            result.update(
                status="complete",
                score=aggregate_scores([[f["score"]] for f in result["folds"]]),
            )
        except TrackingError:
            raise
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            tracker.trial(result)
            raise ValueError(result["error"]) from exc
        tracker.trial(result)
        return result["score"]

    try:
        tracker.start()
        if spec.hpo:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.ERROR)
            study = optuna.create_study(
                direction=spec.direction,
                sampler=optuna.samplers.TPESampler(seed=spec.seed),
            )

            def objective(trial):
                params = {
                    **spec.fixed_params,
                    **{
                        name: dist.suggest(trial, name)
                        for name, dist in spec.search_space.items()
                    },
                }
                return compute_trial(trial.number, params)

            study.optimize(
                objective,
                n_trials=spec.n_trials,
                timeout=spec.timeout,
                n_jobs=1,
                catch=(ValueError,),
            )
            report["search_status"] = (
                "completed"
                if len(report["trials"]) == spec.n_trials
                else "budget_exhausted"
            )
        else:
            try:
                compute_trial(0, spec.fixed_params)
            except ValueError:
                pass
        valid_trials = [
            trial for trial in report["trials"] if trial["status"] == "complete"
        ]
        if not valid_trials:
            report.update(
                training_status="failed", error="no valid best model; see failed trials"
            )
        else:
            best = (min if spec.direction == "minimize" else max)(
                valid_trials, key=lambda trial: trial["score"]
            )
            state["best"] = best
            report.update(
                best_trial=best["number"],
                best_cv_score=best["score"],
                best_params=best["params"],
            )
            tracker.dataset(refit, features, name="refit-train", context="training")
            model = fit_model(spec, best["params"], refit, features)
            state.update(model=model, final=True)
            report["evaluation"]["refit"] = bounds(refit)
            if spec.mode == "future_event":
                # Save observations available before test start, including history of purged label rows.
                normalized = normalize_time(raw, spec.temporal)
                model.history = normalized.loc[
                    normalized[TIME] < spec.temporal.timestamp(spec.temporal.test_start)
                ].drop(columns=[ID, TIME])
            if spec.mode != "forecast":
                train_predictions = model.predict(refit[features])
                train_metrics = metrics_for(spec.task, refit[LABEL], train_predictions)
                report["train_resubstitution_metrics"] = train_metrics
                report["evaluation"]["train"] = {
                    **bounds(refit),
                    "metrics": train_metrics,
                    "kind": "resubstitution",
                }
            else:
                report["evaluation"]["train"] = {
                    **bounds(refit),
                    "metrics": None,
                    "kind": "not_computed",
                }
            report["evaluation"]["cv"] = best["folds"]
            if not holdout.empty:
                tracker.dataset(holdout, features, name="holdout-test", context="test")
                test_result = evaluate(
                    spec,
                    model,
                    refit,
                    holdout,
                    features,
                    spec.temporal.timestamp(spec.temporal.test_start)
                    if spec.temporal
                    else None,
                )
                report["evaluation"]["holdout"] = {
                    **test_result,
                    **bounds(holdout),
                    "independence": "development_only"
                    if spec.holdout_used_for_selection
                    else "independent_within_this_run",
                }
            else:
                report["evaluation"]["holdout"] = None
            report["training_status"] = "completed"
    except TrackingError as exc:
        report["error"] = str(exc)
        report["training_status"] = (
            "interrupted" if state["model"] is not None else "not_started"
        )
        if spec.hpo:
            report["search_status"] = "interrupted"
    except Exception as exc:
        # Configuration errors (including missing remote prerequisites) are actionable before fit.
        if state["model"] is None:
            raise
        report.update(
            training_status="failed",
            error=f"final fit/evaluation failed: {type(exc).__name__}: {exc}; available model retained",
        )
    report["training_time_seconds"] = round(time.monotonic() - started, 6)
    report["artifact_status"] = (
        "complete"
        if state["final"] and report["training_status"] == "completed"
        else ("partial" if state["model"] is not None else "diagnostics_only")
    )
    report["model_role"] = "final_refit" if state["final"] else "last_completed_fold"
    report["model_path"] = None
    report["tracking"] = tracker.snapshot()
    if state["model"] is not None:
        try:
            example = input_example(state["model"], raw if state["final"] else df)
            with model_definition(
                state["model"],
                example,
                output_signature=report["training_status"] == "completed",
            ) as (flavor, options):
                flavor.save_model(path=str(output / "model"), **options)
                report["model_path"] = str((output / "model").resolve())
                write_run_documents(output, spec, report)
                if report["training_status"] == "completed":
                    try:
                        tracker.finish(output, flavor, options)
                    except TrackingError as exc:
                        report["tracking_error"] = str(exc)
        except Exception as exc:
            report["artifact_status"] = "failed"
            report["artifact_error"] = f"model save failed: {type(exc).__name__}"
    report["tracking"] = tracker.snapshot()
    write_run_documents(output, spec, report)
    if tracker.status == "active" and (
        report["training_status"] != "completed"
        or report["artifact_status"] == "failed"
    ):
        try:
            tracker.close_failed(output)
        except TrackingError as exc:
            report["tracking_error"] = str(exc)
        report["tracking"] = tracker.snapshot()
        write_run_documents(output, spec, report)
    return report
