#!/usr/bin/env python3
"""Train tabular or temporal models; existing CLI flags remain supported.

For automatic locked environment setup from any working directory, use sibling run.py.
This lower-level entrypoint also works in an already prepared Python environment.
"""

from pathlib import Path
import sys

# The installed skill includes its own package; never search an ancestor repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "src"))
from agent_mlops.cli import build_parser, main, run_training  # noqa: E402,F401

if __name__ == "__main__":
    sys.exit(main())
