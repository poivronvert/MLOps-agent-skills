#!/usr/bin/env python3
"""Generate/check the standalone skill runtime from the repository's canonical source.

Run after changing src/agent_mlops, pyproject.toml, uv.lock or .python-version.
CI uses --check; the installed skill never needs this packaging script.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "skills/mlops/model-training/runtime"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sources = [ROOT / name for name in ["pyproject.toml", "uv.lock", ".python-version"]]
    sources += sorted((ROOT / "src/agent_mlops").glob("*.py"))
    expected = {str(path.relative_to(ROOT)): path.read_bytes() for path in sources}
    manifest = {
        "format": "agent-mlops-skill-runtime-v1",
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in expected.items()
        },
    }
    expected["bundle_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    existing = {
        str(p.relative_to(RUNTIME))
        for p in RUNTIME.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    stale = existing - expected.keys()
    changed = [
        name
        for name, data in expected.items()
        if not (RUNTIME / name).is_file() or (RUNTIME / name).read_bytes() != data
    ]
    if args.check:
        if changed or stale:
            parser.exit(
                1,
                f"Standalone skill runtime is stale: {sorted(set(changed) | stale)}\nRun python scripts/package_model_training.py\n",
            )
        print("Standalone skill runtime matches canonical source and lock")
        return
    for name in stale:
        (RUNTIME / name).unlink()
    for name in changed:
        path = RUNTIME / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected[name])
    print(f"Packaged {len(expected)} files into {RUNTIME}")


if __name__ == "__main__":
    main()
