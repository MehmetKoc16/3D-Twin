import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import twinrefine  # noqa: E402,F401  (puts the rig and texture stage folders on sys.path)


@pytest.fixture(scope="session")
def model():
    from mh import MHModel

    return MHModel()


@pytest.fixture(scope="session")
def mh_fit(model):
    from helpers import neutral_fit

    return neutral_fit(model)
