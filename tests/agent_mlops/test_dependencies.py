"""Exercise locked rejection, legacy CLI and optional-dependency diagnostics."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from agent_mlops.models import optional_import

ROOT = Path(__file__).resolve().parents[2]


def uv_binary():
    binary = os.environ.get("UV_BIN") or shutil.which("uv")
    if not binary:
        pytest.skip("set UV_BIN to the pinned uv executable")
    return binary


@pytest.mark.parametrize("state", ["missing", "stale"])
def test_U03_locked_rejects_missing_or_stale_lock(tmp_path, state):
    uv = uv_binary()
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nname = "locked-probe"\nversion = "0.1.0"\nrequires-python = ">=3.11,<3.13"\n'
    )
    if state == "stale":
        subprocess.run(
            [uv, "lock", "--offline", "--cache-dir", str(tmp_path / "cache")],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        project.write_text(
            project.read_text().replace('version = "0.1.0"', 'version = "0.1.1"')
        )
    before = (tmp_path / "uv.lock").read_bytes() if state == "stale" else None
    result = subprocess.run(
        [
            uv,
            "sync",
            "--locked",
            "--offline",
            "--no-install-project",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "lock" in result.stderr.lower()
    after = (
        (tmp_path / "uv.lock").read_bytes() if (tmp_path / "uv.lock").exists() else None
    )
    assert after == before


def test_U05_export_matches_lock(tmp_path):
    uv = uv_binary()
    output = tmp_path / "requirements.txt"
    subprocess.run(
        [
            uv,
            "export",
            "--locked",
            "--offline",
            "--group",
            "dev",
            "--no-emit-project",
            "--no-header",
            "--output-file",
            str(output),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    assert output.read_text() == (ROOT / "requirements-test.txt").read_text()


def test_Q01_legacy_cli_executes_and_reloads(table, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "skills/mlops/model-training/scripts/train_model.py"),
            "--input",
            str(table),
            "--target",
            "y",
            "--model-type",
            "rf",
            "--task",
            "classification",
            "--n-trials",
            "1",
            "--cv-folds",
            "2",
            "--cv-strategy",
            "stratified",
            "--output",
            str(tmp_path / "out"),
            "--fixed-params",
            '{"n_estimators": 8, "max_depth": 3}',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlflow.pyfunc, sys, pandas as pd; m=mlflow.pyfunc.load_model(sys.argv[1]); assert len(m.predict(pd.DataFrame({'x':[1], 'kind':['new']}))) == 1",
            str(tmp_path / "out/model"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "module,extra",
    [
        ("mlforecast", "forecasting"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
    ],
)
def test_U04_missing_optional_capability_names_locked_command(module, extra):
    # Simulate absence explicitly, so installed extras do not skip this contract.
    from unittest.mock import patch
    import importlib

    original = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == module:
            raise ImportError(name)
        return original(name, *args, **kwargs)

    with patch("importlib.import_module", missing):
        with pytest.raises(ValueError, match=f"uv sync --locked --extra {extra}"):
            optional_import(module, extra)


def test_U01_lock_is_present_and_project_python_supported():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert project["project"]["requires-python"].replace(" ", "") == lock[
        "requires-python"
    ].replace(" ", "")
    assert {"mlforecast", "mlflow-skinny", "optuna", "scikit-learn"} <= {
        p["name"] for p in lock["package"]
    }
    assert len(hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()) == 64
