"""Describe the shipped entrypoint, never an allowlist of what an agent can develop."""

from typing import get_args

from .config import FORECAST_MODELS, TaskSpec, TemporalSpec
from .data import INPUT_FORMATS


def describe_builtins():
    """Return implementation facts. Validation status belongs to execution evidence."""
    models = list(get_args(TaskSpec.model_fields["model"].annotation))
    return {
        "scope": "builtin_entrypoint_only",
        "agent_method_allowlist": False,
        "input_formats": list(INPUT_FORMATS),
        "modes": {
            "tabular": {
                "tasks": list(get_args(TaskSpec.model_fields["task"].annotation)),
                "models": models,
                "development": "shuffled_kfold_or_stratified_kfold",
                "final_refit": "all_development_rows",
                "holdout": "optional_random_split_excluded_from_selection_and_refit",
                "baseline": {
                    "classification": "training_majority",
                    "regression": "training_mean",
                },
            },
            "future_event": {
                "tasks": ["classification"],
                "models": models,
                "label": "event_in_(t,t+window];_incomplete_window_is_unknown",
                "definitions": list(
                    get_args(TemporalSpec.model_fields["event_definition"].annotation)
                ),
                "development": "configured_origins_with_mature_labels",
                "final_refit": "pre_test_rows_with_mature_labels",
                "holdout": "eligible_rows_from_test_start_through_optional_test_end",
                "baseline": "training_majority",
            },
            "forecast": {
                "tasks": ["regression"],
                "models": list(FORECAST_MODELS),
                "strategies": list(
                    get_args(TemporalSpec.model_fields["strategy"].annotation)
                ),
                "development": "configured_origins",
                "final_refit": "observations_before_test_start",
                "holdout": "single_origin_at_test_start",
                "baseline": "last_observation_per_entity",
                "aggregation": TaskSpec.model_fields["aggregation"].default,
                "future_inputs": "known_future_features_at_every_step_through_max_horizon",
                "feature_generation": {
                    "target": "configured_lags",
                    "historical_exogenous": "lags_covering_horizon_and_availability_delay",
                    "static": "explicit_entity_constant_columns",
                    "known_future": "explicit_columns_with_availability_evidence",
                    "target_rolling_transforms": "additional_implementation",
                },
                "additional_evaluation_examples": [
                    "previous_day_same_hour_baseline",
                    "equal_entity_reporting",
                ],
                "prediction_interface": "mlflow.pyfunc.load_model(model_uri).predict(future_frame)",
                "history_update_interface": None,
            },
        },
        "entrypoint": "scripts/run.py train",
        "reload": "mlflow.pyfunc.load_model(model_uri); mlflow models predict",
        "additional_work": (
            "Assess composition or explicit implementation for unmet requirements; "
            "preserve requested semantics and validate the actual execution path. "
            "Package APIs and teaching examples are development resources, not evidence "
            "that a workflow is already delivered. No prior test does not mean impossible."
        ),
        "evidence": (
            "This query describes implementation, not a successful run or an environment probe. "
            "Report planned, executed and verified work separately."
        ),
    }
