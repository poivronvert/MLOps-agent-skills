"""Fold-local preprocessing and complete, reloadable prediction objects."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.metaestimators import available_if

from .temporal import ID, TIME, history_features


def optional_import(module, extra):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ValueError(
            f"missing optional capability {extra}; use scripts/run.py --extra {extra} train ... "
            f"from the installed skill, or uv sync --locked --extra {extra} --group dev "
            "in a development checkout"
        ) from exc


def estimator(spec, params):
    p = dict(params)
    if spec.model != "svm":
        p.setdefault("random_state", spec.seed)
        p.setdefault("n_jobs", 1)
    if spec.model == "rf":
        p.setdefault("n_estimators", 100)
        return (
            RandomForestClassifier
            if spec.task == "classification"
            else RandomForestRegressor
        )(**p)
    if spec.model == "svm":
        if spec.task == "classification":
            p.setdefault("probability", True)
            p.setdefault("random_state", spec.seed)
        return (SVC if spec.task == "classification" else SVR)(**p)
    if spec.model == "xgb":
        module = optional_import("xgboost", "xgboost")
        cls = (
            module.XGBClassifier
            if spec.task == "classification"
            else module.XGBRegressor
        )
    else:
        module = optional_import("lightgbm", "lightgbm")
        p.setdefault("verbosity", -1)
        cls = (
            module.LGBMClassifier
            if spec.task == "classification"
            else module.LGBMRegressor
        )
    return cls(**p)


def pipeline(spec, params):
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    prep = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
        ]
    )
    steps = [("preprocessor", prep)]
    if spec.select_k:
        steps.append(
            (
                "selector",
                SelectKBest(
                    f_classif if spec.task == "classification" else f_regression,
                    k=spec.select_k,
                ),
            )
        )
    return Pipeline(steps + [("estimator", estimator(spec, params))])


class TabularPredictor:
    def __init__(self, spec, params):
        self.spec = spec
        self.pipeline = pipeline(spec, params)
        self.encoder = LabelEncoder() if spec.task == "classification" else None
        self.history = None

    def fit(self, X, y):
        self.features = list(X.columns)
        self.schema = {
            c: "numeric" if pd.api.types.is_numeric_dtype(X[c]) else "categorical"
            for c in X
        }
        if self.encoder:
            y = self.encoder.fit_transform(y)
            self.classes_ = self.encoder.classes_
            if len(self.classes_) < 2:
                raise ValueError(
                    "classification fold has fewer than two training classes"
                )
            if self.spec.mode == "future_event":
                self.positive_label = (
                    1 if self.spec.positive_label is None else self.spec.positive_label
                )
                if self.positive_label not in self.classes_ or len(self.classes_) != 2:
                    raise ValueError(
                        "future event requires binary classes and a valid positive_label"
                    )
                self.positive_index = list(self.classes_).index(self.positive_label)
                self.negative_label = self.classes_[1 - self.positive_index]
        self.pipeline.fit(X, y)
        return self

    def _inputs(self, frame):
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame, columns=self.features)
        missing = set(self.features) - set(frame)
        if missing and self.history is not None and self.spec.temporal:
            from .temporal import normalize_time

            raw = pd.concat([self.history, frame], ignore_index=True)
            keys = self.spec.temporal.entity_keys + [self.spec.temporal.time_column]
            raw = raw.drop_duplicates(keys, keep="last")
            normalized = normalize_time(raw, self.spec.temporal)
            transformed, _ = history_features(
                normalized,
                self.spec.temporal,
                self.spec.history_lags,
                self.spec.rolling_windows,
                self.spec.availability,
            )
            requested = frame.copy()
            requested["__request_order"] = np.arange(len(frame))
            requests = normalize_time(requested, self.spec.temporal)[
                [ID, TIME, "__request_order"]
            ]
            frame = requests.merge(
                transformed, on=[ID, TIME], how="left", validate="one_to_one"
            ).sort_values("__request_order")
            if frame[self.features].isna().all(axis=1).any():
                raise ValueError("insufficient history for prediction schema")
        missing = set(self.features) - set(frame)
        if missing:
            raise ValueError(f"input schema missing columns: {sorted(missing)}")
        for col, kind in self.schema.items():
            if kind == "numeric" and not pd.api.types.is_numeric_dtype(frame[col]):
                raise ValueError(f"input schema: {col} must be numeric")
        return frame[self.features]

    def predict(self, frame):
        X = self._inputs(frame)
        if self.spec.mode == "future_event":
            proba = self.pipeline.predict_proba(X)[:, self.positive_index]
            return np.where(
                proba >= self.spec.threshold, self.positive_label, self.negative_label
            )
        pred = self.pipeline.predict(X)
        return (
            self.encoder.inverse_transform(pred.astype(int)) if self.encoder else pred
        )

    @available_if(
        lambda self: (
            self.encoder is not None and hasattr(self.pipeline, "predict_proba")
        )
    )
    def predict_proba(self, frame):
        if not self.encoder:
            raise ValueError("regression model has no class probabilities")
        return self.pipeline.predict_proba(self._inputs(frame))

    def metadata(self):
        return {
            "features": self.features,
            "schema": self.schema,
            "unknown_category_policy": "one-hot all zeros",
            "label_mapping": self.classes_.tolist() if self.encoder else None,
            "threshold": self.spec.threshold
            if self.spec.mode == "future_event"
            else None,
            "positive_label": getattr(self, "positive_label", None),
            "probability_columns": self.classes_.tolist()
            if self.encoder and hasattr(self, "predict_proba")
            else None,
        }


class ForecastPredictor:
    def __init__(self, spec, params):
        self.spec = spec
        mlforecast = optional_import("mlforecast", "forecasting")
        self.forecaster = mlforecast.MLForecast(
            models={"model": pipeline(spec, params)},
            freq=spec.temporal.frequency,
            lags=spec.temporal.lags,
            num_threads=1,
        )

    def fit(self, frame):
        t = self.spec.temporal
        self.history = frame.copy()
        self.mapping = frame[[ID, *t.entity_keys]].drop_duplicates(ID)
        for col in t.static_features:
            if frame.groupby(ID)[col].nunique(dropna=False).gt(1).any():
                raise ValueError(f"static feature role varies within entity: {col}")
        self.columns = list(t.static_features) + list(t.known_future)
        df = frame[[ID, TIME, self.spec.target, *self.columns]].copy()
        for col, lags in t.historical_features.items():
            for lag in lags:
                name = f"{col}_history_lag{lag}"
                df[name] = frame.groupby(ID)[col].shift(lag)
                self.columns.append(name)
        self.excluded_history = (
            int(df[self.columns].isna().any(axis=1).sum()) if self.columns else 0
        )
        if self.excluded_history and t.history_policy == "reject":
            raise ValueError("insufficient forecast exogenous history")
        df = df.dropna(subset=self.columns)
        counts = df.groupby(ID).size()
        enough = counts[counts > max(t.lags)].index
        dropped = set(frame[ID]) - set(enough)
        if dropped and t.history_policy == "reject":
            raise ValueError("insufficient forecast target history")
        self.excluded_entities = sorted(dropped)
        df = df[df[ID].isin(enough)]
        self.mapping = self.mapping[self.mapping[ID].isin(enough)]
        if df.empty:
            raise ValueError("insufficient forecast history for all entities")
        if df.groupby(ID)[TIME].max().nunique() != 1:
            raise ValueError(
                "entities require the same forecast origin; missing recent history"
            )
        df[TIME] = df[TIME].dt.tz_localize(None)
        self.forecaster.fit(
            df,
            id_col=ID,
            time_col=TIME,
            target_col=self.spec.target,
            static_features=t.static_features,
        )
        return self

    def predict(self, future=None):
        from .temporal import normalize_time

        t = self.spec.temporal
        h = max(t.horizons)
        expected = self.forecaster.make_future_dataframe(h)
        X = expected.copy()
        if t.known_future:
            required = set(t.entity_keys + [t.time_column] + list(t.known_future))
            if future is None or not required.issubset(future.columns):
                raise ValueError(
                    f"missing legal future input schema: {sorted(required)}"
                )
            # Normalization validates unique keys and the complete recursive grid.
            supplied = normalize_time(future, t)
            supplied[TIME] = supplied[TIME].dt.tz_localize(None)
            X = X.merge(
                supplied[[ID, TIME, *t.known_future]],
                on=[ID, TIME],
                how="left",
                validate="one_to_one",
            )
            if X[list(t.known_future)].isna().any().any():
                raise ValueError(
                    "missing future values or horizon coverage; historical observations cannot fill them"
                )
        for col, lags in t.historical_features.items():
            history = self.history.set_index([ID, TIME])[col]
            for lag in lags:
                keys = pd.MultiIndex.from_arrays(
                    [X[ID], X[TIME].dt.tz_localize("UTC") - lag * t.step]
                )
                X[f"{col}_history_lag{lag}"] = history.reindex(keys).to_numpy()
                if X[f"{col}_history_lag{lag}"].isna().any():
                    raise ValueError("missing available historical exogenous input")
        dynamic = list(t.known_future) + [
            f"{c}_history_lag{lag}"
            for c, lags in t.historical_features.items()
            for lag in lags
        ]
        pred = self.forecaster.predict(h=h, X_df=X if dynamic else None)
        pred["horizon"] = pred.groupby(ID).cumcount() + 1
        pred = pred[pred.horizon.isin(t.horizons)].copy()
        pred[TIME] = pred[TIME].dt.tz_localize("UTC")
        pred = pred.merge(self.mapping, on=ID, validate="many_to_one")
        pred[t.time_column] = pred[TIME]
        return pred.sort_values([ID, TIME]).reset_index(drop=True)

    def metadata(self):
        t = self.spec.temporal
        return {
            "frequency": t.frequency,
            "strategy": t.strategy,
            "horizons": t.horizons,
            "target_lags": t.lags,
            "static_features": t.static_features,
            "historical_features": t.historical_features,
            "known_future": t.known_future,
            "history_rows": len(self.history),
            "excluded_entities": self.excluded_entities,
            "excluded_history": self.excluded_history,
            "origin": str(self.history[TIME].max() + t.step),
        }
