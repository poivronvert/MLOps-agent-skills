"""Validated internal settings; semantic evidence is recorded, never inferred here."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

FORECAST_MODELS = ("rf",)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    value: Any
    source: str
    status: Literal["confirmed", "default", "assumption"]
    reason: str | None = None
    overrides: str | None = None


class Distribution(StrictModel):
    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def check(self):
        if self.type == "categorical":
            if (
                not self.choices
                or self.low is not None
                or self.high is not None
                or self.log
                or self.step
            ):
                raise ValueError(
                    "categorical distribution requires only nonempty choices"
                )
        else:
            if (
                self.low is None
                or self.high is None
                or self.low > self.high
                or self.choices
            ):
                raise ValueError("distribution requires low <= high")
            if not all(float("-inf") < v < float("inf") for v in [self.low, self.high]):
                raise ValueError("distribution bounds must be finite")
            if self.log and (
                self.low <= 0 or (self.step is not None and self.step != 1)
            ):
                raise ValueError(
                    "log distribution requires positive bounds and no custom step"
                )
            if self.step is not None and self.step <= 0:
                raise ValueError("distribution step must be positive")
            if self.type == "int" and any(
                v is not None and int(v) != v for v in [self.low, self.high, self.step]
            ):
                raise ValueError("int distribution requires integer bounds/step")
        return self

    def suggest(self, trial, name):
        if self.type == "categorical":
            return trial.suggest_categorical(name, self.choices)
        kw = {"log": self.log}
        if self.step is not None:
            kw["step"] = int(self.step) if self.type == "int" else self.step
        if self.type == "int":
            return trial.suggest_int(name, int(self.low), int(self.high), **kw)
        return trial.suggest_float(name, self.low, self.high, **kw)


class TemporalSpec(StrictModel):
    time_column: str
    entity_keys: list[str] = Field(default_factory=list)
    timezone: str
    frequency: str
    origins: list[str]
    test_start: str
    test_end: str | None = None
    horizons: list[int] = Field(default_factory=lambda: [1])
    validation_steps: int = Field(default=1, ge=1)
    window: str = "24h"
    reporting_delay: str = "0h"
    event_definition: Literal["window"] = "window"
    event_value: Any = 1
    history_policy: Literal["reject", "exclude"] = "exclude"
    duplicate_policy: Literal["reject"] = "reject"
    missing_period_policy: Literal["reject"] = "reject"
    strategy: Literal["recursive"] = "recursive"
    lags: list[int] = Field(default_factory=lambda: [1])
    static_features: list[str] = Field(default_factory=list)
    historical_features: dict[str, list[int]] = Field(default_factory=dict)
    known_future: dict[str, str] = Field(
        default_factory=dict
    )  # column -> evidence source

    @model_validator(mode="after")
    def check(self):
        from zoneinfo import ZoneInfo

        ZoneInfo(self.timezone)
        try:
            step = pd.tseries.frequencies.to_offset(self.frequency).nanos
        except ValueError as exc:
            raise ValueError("only fixed-duration frequency is supported") from exc
        if step <= 0 or not self.origins or not self.horizons or not self.lags:
            raise ValueError(
                "positive frequency, origins, horizons and lags are required"
            )
        if any(v < 1 for v in self.horizons + self.lags) or len(
            set(self.horizons)
        ) != len(self.horizons):
            raise ValueError("horizons and lags must be positive and horizons unique")
        if pd.Timedelta(self.window) <= pd.Timedelta(0) or pd.Timedelta(
            self.reporting_delay
        ) < pd.Timedelta(0):
            raise ValueError(
                "window must be positive; reporting delay must be nonnegative"
            )
        boundaries = [self.timestamp(v) for v in self.origins]
        test = self.timestamp(self.test_start)
        if boundaries != sorted(set(boundaries)) or any(t >= test for t in boundaries):
            raise ValueError("origins must increase strictly before test_start")
        if self.test_end and self.timestamp(self.test_end) < test:
            raise ValueError("test_end must follow test_start")
        roles = (
            self.static_features
            + list(self.historical_features)
            + list(self.known_future)
        )
        if len(roles) != len(set(roles)):
            raise ValueError("forecast feature roles overlap")
        if any(not source for source in self.known_future.values()):
            raise ValueError("known future features require availability evidence")
        for lags in self.historical_features.values():
            if not lags or min(lags) < max(self.horizons):
                raise ValueError(
                    "historical exogenous lags must cover the maximum horizon"
                )
        return self

    def timestamp(self, value):
        ts = pd.Timestamp(value)
        return (ts.tz_localize(self.timezone) if ts.tzinfo is None else ts).tz_convert(
            "UTC"
        )

    @property
    def step(self):
        return pd.Timedelta(
            pd.tseries.frequencies.to_offset(self.frequency).nanos, unit="ns"
        )


class TaskSpec(StrictModel):
    input: str
    data_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    target: str
    task: Literal["classification", "regression"]
    mode: Literal["tabular", "future_event", "forecast"] = "tabular"
    sample_grain: str = "one input row"
    features: list[str] | None = None
    exclusions: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, Evidence] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    availability: dict[str, str] = Field(
        default_factory=dict
    )  # duration after event time
    history_lags: dict[str, list[int]] = Field(default_factory=dict)
    rolling_windows: dict[str, list[int]] = Field(default_factory=dict)
    temporal: TemporalSpec | None = None
    model: Literal["rf", "svm", "xgb", "lgbm"] = "rf"
    fixed_params: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, Distribution] = Field(default_factory=dict)
    seed: int = 42
    hpo: bool = True
    n_trials: int = Field(default=50, ge=1)
    timeout: float | None = Field(default=None, gt=0)
    metric: str | None = None
    direction: Literal["maximize", "minimize"] | None = None
    cv_folds: int = Field(default=5, ge=2)
    cv_strategy: Literal["kfold", "stratified"] = "stratified"
    holdout_fraction: float = Field(default=0.0, ge=0, lt=0.5)
    holdout_used_for_selection: bool = False
    holdout_history: str = "prior use outside visible records is unknown"
    aggregation: Literal["pooled_samples_then_equal_horizons_then_equal_folds"] = (
        "pooled_samples_then_equal_horizons_then_equal_folds"
    )
    missing_score_policy: Literal["invalidate"] = "invalidate"
    select_k: int | None = Field(default=None, ge=1)
    positive_label: Any = None
    threshold: float = Field(default=0.5, ge=0, le=1)
    tracking: Literal["off", "remote"] = "off"
    dataset_inputs: bool = True
    dataset_snapshots: list[Literal["training", "validation", "test"]] = Field(
        default_factory=lambda: ["training", "validation", "test"]
    )
    experiment: str = "agent-mlops"
    extras: list[Literal["forecasting", "tracking", "xgboost", "lightgbm"]] = Field(
        default_factory=list
    )
    groups: list[Literal["dev"]] = Field(default_factory=lambda: ["dev"])

    @model_validator(mode="after")
    def check(self):
        if self.unresolved:
            raise ValueError(
                "unresolved critical semantics: " + "; ".join(self.unresolved)
            )
        if self.mode != "tabular" and self.temporal is None:
            raise ValueError("temporal settings required")
        if self.mode == "future_event" and self.task != "classification":
            raise ValueError("future_event requires classification")
        if self.mode == "forecast" and (
            self.task != "regression" or self.model not in FORECAST_MODELS
        ):
            raise ValueError("built-in forecast entrypoint supports RF regression only")
        if self.mode == "tabular" and self.temporal is not None:
            raise ValueError("temporal settings require a temporal mode")
        if self.features and self.target in self.features:
            raise ValueError("target cannot be a raw feature")
        if self.features and set(self.features) & set(self.exclusions):
            raise ValueError("included and excluded features overlap")
        if not self.metric:
            self.metric = "f1_weighted" if self.task == "classification" else "rmse"
            self.provenance.setdefault(
                "metric",
                Evidence(
                    value=self.metric,
                    source="baseline default",
                    status="default",
                    reason="descriptive baseline; not a deployment threshold",
                ),
            )
        allowed = (
            {"accuracy", "f1_weighted", "f1_macro"}
            if self.task == "classification"
            else {"rmse", "mae", "r2"}
        )
        if self.metric not in allowed:
            raise ValueError(
                f"metric {self.metric!r} is incompatible with {self.task}; choose {sorted(allowed)}"
            )
        expected_direction = (
            "minimize" if self.metric in {"mae", "rmse"} else "maximize"
        )
        if self.direction and self.direction != expected_direction:
            raise ValueError("metric direction contradicts the score definition")
        self.direction = expected_direction
        if set(self.fixed_params) & set(self.search_space):
            raise ValueError("fixed and search parameters conflict")
        if not self.hpo and self.search_space:
            raise ValueError("search_space requires HPO")
        for delay in self.availability.values():
            if pd.Timedelta(delay) < pd.Timedelta(0):
                raise ValueError("availability delay must be nonnegative")
        for lags in list(self.history_lags.values()) + list(
            self.rolling_windows.values()
        ):
            if not lags or min(lags) < 1:
                raise ValueError("history lags and rolling windows must be positive")
        return self

    def validate_model(self):
        from .models import estimator

        candidate = estimator(self, self.fixed_params)
        if self.mode == "future_event" and not hasattr(candidate, "predict_proba"):
            raise ValueError(
                "event classification requires predict_proba; enable model probability support"
            )
        known = estimator(self, {}).get_params()
        unknown = (set(self.fixed_params) | set(self.search_space)) - set(known)
        if unknown:
            raise ValueError(f"unknown model parameters: {sorted(unknown)}")
        if self.model == "xgb" and "early_stopping_rounds" in set(
            self.fixed_params
        ) | set(self.search_space):
            raise ValueError(
                "early_stopping_rounds requires an explicit development eval_set; unsupported here"
            )
        if hasattr(candidate, "_parameter_constraints"):
            candidate._validate_params()
        # Validate both numeric endpoints / all categorical values before any fit.
        for name, dist in self.search_space.items():
            values = (
                dist.choices if dist.type == "categorical" else [dist.low, dist.high]
            )
            for value in values:
                if dist.type == "int":
                    value = int(value)
                probe = estimator(self, {**self.fixed_params, name: value})
                if hasattr(probe, "_parameter_constraints"):
                    probe._validate_params()
        return self
