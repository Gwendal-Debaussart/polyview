#!/usr/bin/env python3
"""Convenience entry point for running the polyview test suite.

Usage
-----
    python polyview/test/run_tests.py
    python polyview/test/run_tests.py -k kmeans      # only tests matching "kmeans"
    python polyview/test/run_tests.py -x -v          # stop on first failure, verbose

Any extra command-line arguments are forwarded to pytest as-is.

This script auto-detects the project's ``.venv`` (if present) so it works
whether or not that virtualenv is currently activated.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


def _find_repo_root(start: Path) -> Path:
    """Walk upward from `start` until a directory containing pyproject.toml is found."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def _venv_python(repo_root: Path) -> Optional[Path]:
    for rel in ("bin/python", "Scripts/python.exe"):
        candidate = repo_root / ".venv" / rel
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    test_dir = Path(__file__).resolve().parent
    repo_root = _find_repo_root(test_dir)

    python = _venv_python(repo_root) or Path(sys.executable)

    cmd = [str(python), "-m", "pytest", str(test_dir), *sys.argv[1:]]
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
