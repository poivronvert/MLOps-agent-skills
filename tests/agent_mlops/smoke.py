"""Two-process clean-environment training/reload acceptance; artifacts go outside the repo."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow.pyfunc
from agent_mlops.training import train
from test_contract import spec_for
from test_temporal import TOL, panel, temporal_spec


def run(workspace, phase, mode):
    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / "bundle"
    if phase == "train":
        if mode == "tabular":
            path = workspace / "data.csv"
            pd.DataFrame(
                {
                    "x": np.arange(60),
                    "kind": ["a", "b", "c"] * 20,
                    "y": ["ok", "fault"] * 30,
                }
            ).to_csv(path, index=False)
            spec = spec_for(path)
            fixture = pd.DataFrame({"x": [1, 4, 6], "kind": ["unseen", "a", "b"]})
        else:
            path, df = panel(workspace)
            spec = temporal_spec(path, mode)
            origin = spec.temporal.timestamp(spec.temporal.test_start)
            fixture = df[
                (pd.to_datetime(df.time) >= origin)
                & (pd.to_datetime(df.time) < origin + 4 * spec.temporal.step)
            ].sort_values(["id", "time"])
        report = train(spec, output)
        assert report["training_status"] == "completed", report
        fixture.to_csv(workspace / "fixture.csv", index=False)
    model = mlflow.pyfunc.load_model(str(output / "model"))
    fixture = pd.read_csv(workspace / "fixture.csv")
    result = model.predict(fixture)
    expected = {
        "prediction": result.model.tolist() if mode == "forecast" else result.prediction.tolist()
    }
    if mode != "forecast":
        expected["probabilities"] = result.filter(like="probability_").to_numpy().tolist()
    expected_path = workspace / "expected.json"
    if phase == "train":
        expected_path.write_text(json.dumps(expected))
    else:
        stored = json.loads(expected_path.read_text())
        if mode == "forecast":
            np.testing.assert_allclose(
                stored["prediction"], expected["prediction"], atol=TOL, rtol=0
            )
        else:
            np.testing.assert_array_equal(stored["prediction"], expected["prediction"])
            np.testing.assert_allclose(
                stored["probabilities"], expected["probabilities"], atol=TOL, rtol=0
            )
    print(
        json.dumps(
            {
                "phase": phase,
                "mode": mode,
                "status": "passed",
                "workspace": str(workspace),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "reload"], required=True)
    parser.add_argument(
        "--mode", choices=["tabular", "future_event", "forecast"], required=True
    )
    args = parser.parse_args()
    run(args.workspace, args.phase, args.mode)
