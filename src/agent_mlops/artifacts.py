"""Run reports and provenance; models use official MLflow flavors."""

import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value, indent=2, ensure_ascii=False, default=json_default, allow_nan=False
        )
        + "\n"
    )


def environment(spec):
    lock = ROOT / "uv.lock"
    sources = sorted((ROOT / "src" / "agent_mlops").glob("*.py"))
    code_hash = hashlib.sha256(
        b"".join(p.name.encode() + p.read_bytes() for p in sources)
    ).hexdigest()
    # A copied skill may live inside an unrelated user's Git project. Never attribute
    # that project's revision to this runtime. Installed bundles use their code hash.
    revision = None
    if (ROOT / ".git").exists() and shutil.which("git"):
        revision = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip()
            or None
        )
    uv = os.environ.get("UV_BIN") or shutil.which("uv")
    uv_version = (
        subprocess.run([uv, "--version"], capture_output=True, text=True).stdout.strip()
        if uv
        else None
    )
    optional_packages = {
        "forecasting": "mlforecast",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
    }
    return {
        "python": platform.python_version(),
        "uv": uv_version,
        "uv_required": "0.11.6",
        "extras": spec.extras,
        "groups": spec.groups,
        "available_optional_extras": [
            extra for extra, package in optional_packages.items() if find_spec(package)
        ],
        "uv_lock_sha256": digest(lock) if lock.is_file() else None,
        "revision": revision,
        "source_sha256": code_hash,
        "source_scope": "agent_mlops_package_only",
        "includes_workspace_code": False,
        "dependencies": "See the official model's requirements.txt and environment files.",
    }


def write_run_documents(output, spec, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "task_spec.json").exists():
        write_json(output / "task_spec.json", spec.model_dump(mode="json"))
        write_json(output / "environment.json", environment(spec))
    write_json(output / "training_report.json", report)
