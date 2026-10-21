"""Read a single verified byte snapshot, avoiding a hash/read race."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FORMATS = {".csv": ",", ".tsv": "\t", ".txt": ",", ".parquet": None}


def read_snapshot(spec):
    path = Path(spec.input)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != spec.data_sha256:
        raise ValueError(
            "input content changed; revalidate schema, types, times and affected availability/splits"
        )
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(io.BytesIO(payload))
    elif path.suffix.lower() in INPUT_FORMATS:
        frame = pd.read_csv(io.BytesIO(payload), sep=INPUT_FORMATS[path.suffix.lower()])
    else:
        raise ValueError(
            "supported inputs are CSV and Parquet (legacy TSV/TXT also accepted)"
        )
    if spec.target not in frame:
        raise ValueError(
            f"target {spec.target!r} not found; available columns: {list(frame)}"
        )
    if frame.empty or frame[spec.target].isna().any():
        raise ValueError("empty input or missing target values")
    if spec.task == "regression" and not pd.api.types.is_numeric_dtype(
        frame[spec.target]
    ):
        raise ValueError(
            "regression target must be numeric; text labels are not encoded"
        )
    if spec.task == "regression" and not np.isfinite(frame[spec.target]).all():
        raise ValueError("regression target must be finite")
    return frame


def inspect_data(frame):
    return {
        "rows": len(frame),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_fraction": frame.isna().mean().to_dict(),
    }
