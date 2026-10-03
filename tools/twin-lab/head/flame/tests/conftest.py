"""Synthetic-only tests: no owner files, model data or photos are read."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flamehead  # noqa: E402, F401
