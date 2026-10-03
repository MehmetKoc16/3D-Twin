"""Synthetic-only tests: no photo, FLAME file or personal measurement is read (MakeHuman CC0 assets are)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import hybridbody  # noqa: E402, F401  (puts the sibling stage folders on sys.path)


@pytest.fixture(scope="session")
def model():
    from mh import MHModel

    return MHModel()
