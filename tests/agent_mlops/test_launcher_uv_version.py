"""The launcher accepts uv >= 0.11.6 and still runs everything with --locked."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "skills/mlops/model-training/scripts/run.py"
spec = importlib.util.spec_from_file_location("skill_run", RUN)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("uv 0.11.6", (0, 11, 6)),
        ("uv 0.13.7 (x86_64-unknown-linux-gnu)", (0, 13, 7)),
        ("uv 1.0.0+3 (abc 2027-01-01)", (1, 0, 0)),
        ("uv 0.12", (0, 12)),
        ("ruff 0.11.6", None),
        ("uv dev", None),
        ("", None),
    ],
)
def test_uv_version_parsing(output, expected):
    assert launcher.uv_version(output) == expected


def fake_uv(tmp_path, version):
    script = tmp_path / "uv"
    script.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "--version" ]; then echo "uv {version} (fake)"; exit 0; fi\n'
        'echo "fake-uv $*"; exit 3\n'
    )
    script.chmod(0o755)
    return script


def launch(tmp_path, version):
    env = os.environ.copy()
    env.update(UV_BIN=str(fake_uv(tmp_path, version)),
               XDG_CACHE_HOME=str(tmp_path / "cache"))
    return subprocess.run(
        [sys.executable, str(RUN), "python", "-c", "pass"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )


@pytest.mark.parametrize("version", ["0.11.5", "0.9.30"])
def test_older_uv_is_rejected_before_running(tmp_path, version):
    result = launch(tmp_path, version)
    assert result.returncode == 2
    assert "requires uv 0.11.6 or newer" in result.stderr
    assert "fake-uv" not in result.stdout


@pytest.mark.parametrize("version", ["0.11.6", "0.13.7", "1.2.0"])
def test_newer_uv_runs_locked(tmp_path, version):
    result = launch(tmp_path, version)
    assert result.returncode == 3, result.stderr  # the fake uv's own exit code
    assert "fake-uv run --no-config --project" in result.stdout
    assert " --locked " in result.stdout
