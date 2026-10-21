#!/usr/bin/env python3
"""Run the self-contained training skill from the caller's working directory.

Examples:
  python3 /installed/model-training/scripts/run.py train --input data.csv ...
  python3 /installed/model-training/scripts/run.py --extra tracking train --config task.yaml
  python3 /installed/model-training/scripts/run.py python -c 'import agent_mlops'

Requires uv 0.11.6 (on PATH or set UV_BIN) and Python 3.11/3.12.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SKILL = Path(__file__).resolve().parents[1]
RUNTIME = SKILL / "runtime"
EXTRAS = ["forecasting", "tracking", "xgboost", "lightgbm"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", choices=EXTRAS, action="append", default=[])
    parser.add_argument("--python", choices=["3.11", "3.12"], default="3.12")
    parser.add_argument("command", choices=["train", "python"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    uv = os.environ.get("UV_BIN") or shutil.which("uv")
    if not uv:
        fallback = Path.home() / ".local/bin/uv"
        uv = str(fallback) if fallback.is_file() else None
    if not uv:
        parser.error("Install uv 0.11.6 on PATH, or set UV_BIN to its executable")
    version = subprocess.run(
        [uv, "--version"], capture_output=True, text=True, check=True
    )
    if version.stdout.split()[:2] != ["uv", "0.11.6"]:
        parser.error(
            "This skill requires uv 0.11.6; it will not update your dependencies"
        )

    manifest_path = RUNTIME / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        path = RUNTIME / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            parser.error(
                f"Installed runtime is incomplete or modified: {name}; reinstall the skill"
            )
    extras = sorted(set(args.extra))
    # Editable installs contain an absolute package path. Include installation location as
    # well as contents/extras/Python so two independent installations cannot cross-load.
    identity = json.dumps([str(RUNTIME), manifest, extras, args.python], sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str((cache / "agent-mlops/envs" / key).resolve())
    env["UV_BIN"] = str(uv)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable in [
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "UV_NO_SYNC",
        "UV_FROZEN",
    ]:
        env.pop(variable, None)
    command = [
        uv,
        "run",
        "--no-config",
        "--project",
        str(RUNTIME),
        "--locked",
        "--python",
        args.python,
        "--group",
        "dev",
    ]
    for extra in extras:
        command.extend(["--extra", extra])
    command.extend(["--", "python"])
    if args.command == "train":
        # Keep runtime selection and the recorded task environment identical.
        if "--extra" in args.arguments or any(
            v.startswith("--extra=") for v in args.arguments
        ):
            parser.error(
                "Put --extra before 'train' so it selects and records the runtime"
            )
        command.extend([str(SKILL / "scripts/train_model.py"), *args.arguments])
        env["AGENT_MLOPS_RUNTIME_EXTRAS"] = json.dumps(extras)
    else:
        command.extend(args.arguments)
        env.pop("AGENT_MLOPS_RUNTIME_EXTRAS", None)
    # --project selects dependencies without changing cwd. Relative data/config/output
    # paths belong to the user's working directory, never to the installed skill.
    return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Skill runtime setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
