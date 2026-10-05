"""Synthetic files only: licence evidence and pinning of the CC0 skin source."""

import hashlib
import json

import numpy as np
import pytest
from hybridbody import skin_source
from PIL import Image


def fixture_source(tmp_path, monkeypatch, licence="CC0"):
    (tmp_path / "pack.json").write_text(json.dumps({"young_caucasian_male": {"license": licence}}))
    (tmp_path / "young_caucasian_male.mhmat").write_text("# explicitly released as CC0\n")
    Image.fromarray(np.full((16, 16, 3), [180, 132, 116], np.uint8)).save(tmp_path / "young_lightskinned_male_diffuse.png")
    monkeypatch.setattr(skin_source, "CACHE", tmp_path)
    monkeypatch.setattr(skin_source, "FILES", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()})


def test_source_requires_pinned_bytes_and_explicit_cc0_evidence(tmp_path, monkeypatch):
    fixture_source(tmp_path, monkeypatch)
    source, report = skin_source.load_skin(fetch=False)
    assert source[0].shape == (16, 16, 3) and report["license"] == "CC0"
    (tmp_path / "young_lightskinned_male_diffuse.png").write_bytes(b"tampered")
    source, report = skin_source.load_skin(fetch=False)
    assert source is None and "procedural" in report["source"]


def test_non_cc0_pack_is_rejected_even_if_its_bytes_match(tmp_path, monkeypatch):
    fixture_source(tmp_path, monkeypatch, licence="research only")
    with pytest.raises(ValueError, match="CC0"):
        skin_source.load_skin(fetch=False)
