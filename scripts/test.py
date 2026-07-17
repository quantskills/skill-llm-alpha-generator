"""Run the complete local test suite from the skill root."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(pytest.main([str(path) for path in sorted(root.glob("tests_*.py"))] + sys.argv[1:]))
