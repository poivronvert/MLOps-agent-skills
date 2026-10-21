"""Fixed-frequency panel validation and availability-aware historical features."""

import json
import math

import numpy as np
import pandas as pd

ID = "__entity"
TIME = "__time"
LABEL = "__label"


def normalize_time(frame, spec):
    df = frame.copy()
    required = [spec.time_column, *spec.entity_keys]
    if any(c not in df for c in required) or df[required].isna().any().any():
        raise ValueError("missing temporal keys or timestamp")
    try:
        df[TIME] = pd.DatetimeIndex([spec.timestamp(v) for v in df[spec.time_column]])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "invalid timezone/time; explicitly resolve ambiguous or nonexistent local times"
        ) from exc
    df[ID] = (
        [
            json.dumps(list(row), default=str, separators=(",", ":"))
            for row in df[spec.entity_keys].itertuples(index=False, name=None)
        ]
        if spec.entity_keys
        else "[]"
    )
    if df.duplicated([ID, TIME]).any():
        raise ValueError(
            "duplicate entity/time; explicit upstream aggregation required"
        )
    df = df.sort_values([ID, TIME]).reset_index(drop=True)
    for _, group in df.groupby(ID, sort=False):
        if not group[TIME].diff().dropna().eq(spec.step).all():
            raise ValueError(
                "missing periods or irregular frequency; explicit upstream policy required"
            )
    # All series must share the same grid, not merely the same spacing.
    if (
        (df[TIME].astype("int64") - df[TIME].astype("int64").min()) % spec.step.value
    ).any():
        raise ValueError("entities do not share a common time grid")
    return df


def history_features(frame, spec, lags, windows, availability):
    df = frame.copy()
    generated = []
    for column in set(lags) | set(windows):
        if column not in df or column not in availability:
            raise ValueError(
                f"history requires column and availability evidence for {column}"
            )
        delay = pd.Timedelta(availability[column])
        grouped = df.groupby(ID, sort=False)[column]
        for lag in lags.get(column, []):
            name = f"{column}_lag{lag}"
            if name in df:
                raise ValueError(f"generated feature collides with {name}")
            df[name] = grouped.shift(lag) if lag * spec.step >= delay else np.nan
            generated.append(name)
        # Rolling ends at the most recent observation available at the origin.
        shift = math.ceil(delay / spec.step)
        for window in windows.get(column, []):
            name = f"{column}_rolling{window}"
            if name in df:
                raise ValueError(f"generated feature collides with {name}")
            df[name] = grouped.transform(
                lambda s: s.shift(shift).rolling(window, min_periods=window).mean()
            )
            generated.append(name)
    missing = (
        df[generated].isna().any(axis=1)
        if generated
        else pd.Series(False, index=df.index)
    )
    count = int(missing.sum())
    if count and spec.history_policy == "reject":
        raise ValueError(f"insufficient available history for {count} samples")
    return df.loc[~missing].copy(), count


def event_labels(frame, spec, event_column, event_value):
    labels = pd.Series(np.nan, index=frame.index, dtype=float)
    window = pd.Timedelta(spec.window)
    grouping = ID if ID in frame else (spec.entity_keys or (lambda _: 0))
    for _, group in frame.groupby(grouping, sort=False):
        group = group.sort_values(TIME if TIME in group else spec.time_column)
        times = group[TIME] if TIME in group else group[spec.time_column]
        events = group[event_column].eq(event_value).to_numpy()
        cumulative = np.r_[0, np.cumsum(events)]
        ns = pd.DatetimeIndex(times).asi8
        ends = ns + window.value
        left = np.searchsorted(ns, ns, side="right")
        right = np.searchsorted(ns, ends, side="right")
        result = (cumulative[right] - cumulative[left] > 0).astype(float)
        result[ends > ns[-1]] = np.nan
        labels.loc[group.index] = result
    return labels


def mature_mask(times, cutoff, window, reporting_delay):
    return times + pd.Timedelta(window) + pd.Timedelta(reporting_delay) <= cutoff
