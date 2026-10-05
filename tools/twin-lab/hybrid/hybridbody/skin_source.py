"""Pinned CC0 MakeHuman skin; downloaded files stay in the ignored stage cache.

The system pack's item licence and the material's explicit September 2020 CC0
release are both checked. No upstream normal/specular image is declared by this
material; skin.py derives subtle tangent-space detail instead.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from flamehead.colour import to_lab
from PIL import Image

from . import REPO

CACHE = Path(__file__).resolve().parents[1] / ".cache/skin"
URL = "https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip"
LICENCE_URL = "https://static.makehumancommunity.org/assets/assetpacks/makehuman_system_assets.html"
FILES = {
    "pack.json": "e86f1572432cfd062f372adddba4eb7c2e2e31280e96c29291a970320e0a5df0",
    "young_caucasian_male.mhmat": "6eda60b07ff2afbde078a2535ef3913593d84e9de92b2eb76c30f396a181fe2e",
    "young_lightskinned_male_diffuse.png": "862a26e335e958b70534cb5f0d7c47ef30ab148a56c42b3e9da969cf76f12963",
}


def load_skin(*, fetch: bool = True):
    """Return source Lab and provenance, or a procedural fallback when unavailable."""
    def valid():
        return all((CACHE / n).is_file() and hashlib.sha256((CACHE / n).read_bytes()).hexdigest() == h
                   for n, h in FILES.items())

    if not valid() and fetch:
        # Reuse the repo's range reader; do not download the entire 267 MB pack.
        sys.path.insert(0, str(REPO / "tools/asset-pipeline"))
        from fetch import _HttpRangeFile

        CACHE.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_HttpRangeFile(URL)) as archive:
            for name, digest in FILES.items():
                member = next(n for n in archive.namelist() if Path(n).name == name) if name != "pack.json" else "packs/makehuman_system_assets.json"
                data = archive.read(member)
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f"CC0 skin hash mismatch: {name}")
                (CACHE / name).write_bytes(data)
    if not valid():
        return None, {"source": "procedural world-space skin", "license": "CC0", "upstream_maps": []}
    pack = json.loads((CACHE / "pack.json").read_text(encoding="utf8"))
    material = (CACHE / "young_caucasian_male.mhmat").read_text(encoding="utf8")
    if pack["young_caucasian_male"]["license"] != "CC0" or "explicitly released as CC0" not in material:
        raise ValueError("Skin requires explicit CC0 evidence")
    with Image.open(CACHE / "young_lightskinned_male_diffuse.png") as image:
        lab = to_lab(np.asarray(image.convert("RGB")))
    # Robust skin reference excludes the atlas background, hair and dark detail.
    skin = (lab[..., 0] > 45) & (lab[..., 0] < 85) & (lab[..., 1] > 2) & (lab[..., 2] > 3)
    reference = np.median(lab[skin], axis=0)
    return (lab, reference), {
        "source": "MakeHuman young_caucasian_male / young_lightskinned_male_diffuse.png",
        "license": "CC0", "license_url": LICENCE_URL, "download_url": URL,
        "sha256": FILES, "source_size": [int(lab.shape[1]), int(lab.shape[0])],
        "reference_lab": reference.tolist(), "upstream_maps": [],
        "normal_source": "source-albedo high-pass plus analytic world-space pores (CC0)",
    }
