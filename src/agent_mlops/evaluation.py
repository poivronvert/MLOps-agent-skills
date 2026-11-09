"""Scores are in their natural units; no hidden sign changes or dropped groups."""

import numpy as np
from sklearn import metrics


def score(metric, truth, predictions):
    if not len(truth):
        raise ValueError("invalid score: no samples")
    if metric == "accuracy":
        value = metrics.accuracy_score(truth, predictions)
    elif metric.startswith("f1_"):
        value = metrics.f1_score(
            truth, predictions, average=metric.removeprefix("f1_"), zero_division=0
        )
    elif metric == "mae":
        value = metrics.mean_absolute_error(truth, predictions)
    elif metric == "rmse":
        value = np.sqrt(metrics.mean_squared_error(truth, predictions))
    elif metric == "r2":
        value = metrics.r2_score(truth, predictions)
    else:
        raise ValueError(f"unknown metric {metric}")
    if not np.isfinite(value):
        raise ValueError("invalid nonfinite score")
    return float(value)


def aggregate_scores(folds):
    if not folds or any(
        not len(horizons) or not np.isfinite(horizons).all() for horizons in folds
    ):
        raise ValueError("invalid fold/horizon score; trial cannot be compared")
    return float(np.mean([np.mean(horizons) for horizons in folds]))


def metrics_for(task, truth, predictions):
    names = (
        ["accuracy", "f1_weighted"]
        if task == "classification"
        else ["rmse", "mae", "r2"]
    )
    return {
        m: score(m, truth, predictions) for m in names if m != "r2" or len(truth) > 1
    }
