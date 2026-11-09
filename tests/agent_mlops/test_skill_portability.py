"""A copied, read-only installed skill must train/reload outside its source checkout."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_contract import spec_for
from test_dependencies import ROOT, uv_binary
from test_temporal import panel, temporal_spec

SKILL = ROOT / "skills/mlops/model-training"


def files(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def installed(tmp_path):
    copies = [
        tmp_path / "installed skills/model-training",
        tmp_path / "relocated/model-training",
    ]
    for destination in copies:
        shutil.copytree(
            SKILL, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        assert not any(p.is_symlink() for p in destination.rglob("*"))
        for path in destination.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        destination.chmod(0o555)
    workspace = tmp_path / "customer project"
    workspace.mkdir()
    # Neither malformed project configuration nor a preexisting environment may be used.
    (workspace / "pyproject.toml").write_text("this is deliberately not TOML")
    (workspace / "uv.lock").write_text("unrelated lock: must not be touched")
    env = os.environ.copy()
    env.update(
        UV_BIN=uv_binary(),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        UV_PROJECT_ENVIRONMENT=str(workspace / "must-not-create-venv"),
        VIRTUAL_ENV=str(ROOT / ".venv"),
        PYTHONPATH=str(ROOT / "src"),
        MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD="true",
    )
    # Honor the developer/CI uv cache, independently of this test's environment cache.
    env.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache/uv"))
    before = [files(copy) for copy in copies]
    try:
        yield copies, workspace, env
        assert [files(copy) for copy in copies] == before
        assert (
            workspace / "pyproject.toml"
        ).read_text() == "this is deliberately not TOML"
        assert (
            workspace / "uv.lock"
        ).read_text() == "unrelated lock: must not be touched"
        assert not (workspace / "must-not-create-venv").exists()
        assert not (workspace / ".venv").exists()
    finally:
        for copy in copies:
            copy.chmod(0o755)
            for path in copy.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)


def launch(copy, workspace, env, extras, *arguments):
    command = [
        sys.executable,
        str(copy / "scripts/run.py"),
        "--python",
        f"{sys.version_info.major}.{sys.version_info.minor}",
    ]
    for extra in extras:
        command += ["--extra", extra]
    return subprocess.run(
        command + list(arguments),
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("mode", ["tabular", "future_event", "forecast"])
def test_installed_skill_train_and_relocate_reload(installed, mode):
    if mode == "forecast":
        pytest.importorskip("mlforecast")
    copies, workspace, env = installed
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
        ]
    values = spec.model_dump(mode="json")
    values["input"] = path.name
    extras = values["extras"]
    # Exercise locked SDK installation too, without creating remote runs.
    if importlib.util.find_spec("mlflow"):
        extras.append("tracking")
    (workspace / "task.json").write_text(json.dumps(values))
    fixture.to_csv(workspace / "fixture.csv", index=False)
    trained = launch(
        copies[0],
        workspace,
        env,
        extras,
        "train",
        "--config",
        "task.json",
        "--output",
        "output/run-01",
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    output = workspace / "output/run-01"
    provenance = json.loads((output / "environment.json").read_text())
    assert provenance["revision"] is None
    assert (
        provenance["uv_lock_sha256"]
        == hashlib.sha256((SKILL / "runtime/uv.lock").read_bytes()).hexdigest()
    )
    assert provenance["extras"] == sorted(extras)
    before = files(output)
    reload_code = """
import json, os, sys
from pathlib import Path
import agent_mlops, pandas as pd
import mlflow.pyfunc
assert Path(agent_mlops.__file__).is_relative_to(Path(sys.argv[1]) / 'runtime/src')
assert os.environ['MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD'] == 'true'
if 'tracking' in sys.argv[3:]:
    from mlflow import MlflowClient
model = mlflow.pyfunc.load_model('output/run-01/model')
fixture = pd.read_csv('fixture.csv')
result = model.predict(fixture)
if sys.argv[2] == 'forecast':
    result = result.model
else:
    result = result.prediction
print(json.dumps(result.tolist()))
"""
    predictions = []
    for copy in copies:
        result = launch(
            copy,
            workspace,
            env,
            extras,
            "python",
            "-c",
            reload_code,
            str(copy),
            mode,
            *extras,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        predictions.append(json.loads(result.stdout))
    if mode == "forecast":
        np.testing.assert_allclose(*predictions, atol=1e-10, rtol=0)
    else:
        assert predictions[0] == predictions[1]
    assert files(output) == before
    assert len(list((Path(env["XDG_CACHE_HOME"]) / "agent-mlops/envs").iterdir())) == 2


def test_installed_skill_rejects_missing_config_extras(installed):
    copies, workspace, env = installed
    (workspace / "task.json").write_text(
        json.dumps({"extras": ["forecasting", "tracking"]})
    )
    result = launch(copies[0], workspace, env, [], "train", "--config", "task.json")
    assert result.returncode != 0
    assert "--extra forecasting --extra tracking" in result.stderr
    assert not (workspace / "output").exists()


def test_installed_skill_rejects_damaged_lock_before_setup(tmp_path):
    destination = tmp_path / "model-training"
    shutil.copytree(SKILL, destination, ignore=shutil.ignore_patterns("__pycache__"))
    (destination / "runtime/uv.lock").write_text("damaged")
    env = os.environ.copy()
    env["UV_BIN"] = uv_binary()
    env["XDG_CACHE_HOME"] = str(tmp_path / "must-not-create-cache")
    result = launch(destination, tmp_path, env, [], "train", "--help")
    assert result.returncode != 0
    assert "incomplete or modified: uv.lock" in result.stderr
    assert not Path(env["XDG_CACHE_HOME"]).exists()
