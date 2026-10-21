"""Compatible training CLI; --config adds temporal and provenance settings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def build_parser():
    parser = argparse.ArgumentParser(
        description="Locked, fold-local tabular/event/forecast model training"
    )
    parser.add_argument("--config", help="Internal task specification in JSON or YAML")
    parser.add_argument(
        "--describe-builtins",
        action="store_true",
        help="Describe this entrypoint's implementation (not the agent's capability limit); no training",
    )
    parser.add_argument("--input", help="CSV or Parquet path")
    parser.add_argument("--target", help="Target column")
    parser.add_argument("--model-type", choices=["rf", "xgb", "lgbm", "svm"])
    parser.add_argument("--task", choices=["classification", "regression"])
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--cv-folds", type=int)
    parser.add_argument("--cv-strategy", choices=["kfold", "stratified"])
    parser.add_argument("--output", default="./output")
    parser.add_argument("--no-hpo", action="store_true")
    parser.add_argument("--fixed-params", help="JSON object of fixed model parameters")
    parser.add_argument("--search-space", help="JSON object of Optuna distributions")
    parser.add_argument("--metric")
    parser.add_argument("--direction", choices=["minimize", "maximize"])
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--holdout-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tracking", choices=["off", "remote"])
    parser.add_argument(
        "--dataset-inputs", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--dataset-snapshots",
        nargs="*",
        choices=["training", "validation", "test"],
        help="Contexts to upload; pass no values for metadata only",
    )
    parser.add_argument("--experiment")
    parser.add_argument(
        "--extra",
        action="append",
        dest="extras",
        choices=["forecasting", "tracking", "xgboost", "lightgbm"],
    )
    return parser


def default_space(model):
    if model == "rf":
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 30},
        }
    if model in {"xgb", "lgbm"}:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 600},
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
        }
    return {
        "C": {"type": "float", "low": 0.001, "high": 100, "log": True},
        "kernel": {"type": "categorical", "choices": ["linear", "rbf"]},
    }


def run_training(args):
    from .config import TaskSpec
    from .training import train
    import yaml

    config_path = getattr(args, "config", None)
    values = yaml.safe_load(Path(config_path).read_text()) if config_path else {}
    if not isinstance(values, dict):
        raise ValueError("config must contain a mapping")
    for key in [
        "input",
        "target",
        "task",
        "n_trials",
        "cv_folds",
        "cv_strategy",
        "metric",
        "direction",
        "timeout",
        "holdout_fraction",
        "seed",
        "tracking",
        "dataset_inputs",
        "dataset_snapshots",
        "experiment",
        "extras",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    if getattr(args, "model_type", None):
        values["model"] = args.model_type
    for key in ["fixed_params", "search_space"]:
        if getattr(args, key, None) is not None:
            values[key] = json.loads(getattr(args, key))
    if getattr(args, "no_hpo", False):
        values["hpo"] = False
    if "AGENT_MLOPS_RUNTIME_EXTRAS" in os.environ:
        runtime_extras = json.loads(os.environ["AGENT_MLOPS_RUNTIME_EXTRAS"])
        required = set(values.get("extras", []))
        if not required.issubset(runtime_extras):
            raise ValueError(
                "config extras require launcher flags before 'train': "
                + " ".join(f"--extra {extra}" for extra in sorted(required))
            )
        values["extras"] = runtime_extras
    missing = {"input", "target", "task", "model"} - set(values)
    if missing:
        raise ValueError(f"missing required CLI/config values: {sorted(missing)}")
    if "data_sha256" not in values:
        values["data_sha256"] = hashlib.sha256(
            Path(values["input"]).read_bytes()
        ).hexdigest()
    if not config_path and values.get("hpo", True) and "search_space" not in values:
        values["search_space"] = {
            k: v
            for k, v in default_space(values["model"]).items()
            if k not in values.get("fixed_params", {})
        }
    return train(TaskSpec(**values), args.output)


def main():
    args = build_parser().parse_args()
    if args.describe_builtins:
        from .capabilities import describe_builtins

        print(json.dumps(describe_builtins(), indent=2))
        return 0
    try:
        report = run_training(args)
    except Exception as exc:
        # ValidationError normally includes original input values; avoid printing arbitrary config secrets.
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            message = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                for e in exc.errors(include_input=False)
            )
        else:
            message = str(exc)
        print(f"Training stopped: {message}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "training_status",
                    "artifact_status",
                    "model_path",
                    "best_cv_score",
                    "metric",
                    "tracking",
                ]
            },
            indent=2,
        )
    )
    return (
        0
        if report["training_status"] == "completed"
        and report["artifact_status"] == "complete"
        and report["tracking"]["status"] in {"off", "complete"}
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
