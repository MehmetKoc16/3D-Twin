"""Synthetic artifacts stay in the stage's ignored outputs directory.

Windows Python's mode-0700 pytest directories exclude the sandbox token. Use
inherited workspace ACLs for these non-personal fixtures instead.
"""

from pathlib import Path
from uuid import uuid4

import pytest


class OutputFactory:
    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parents[1] / "outputs" / ("tests-" + uuid4().hex)
        self.root.mkdir(parents=True)

    def mktemp(self, name: str) -> Path:
        path = self.root / (name + "-" + uuid4().hex)
        path.mkdir()
        return path


@pytest.fixture(scope="session")
def tmp_path_factory():
    return OutputFactory()


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("case")
