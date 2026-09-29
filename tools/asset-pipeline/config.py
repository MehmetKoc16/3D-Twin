"""Shared constants: pinned sources, paths, unit conversion. No logic."""

from __future__ import annotations

import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent.parent
CACHE_DIR = Path(os.environ.get("DT_ASSET_CACHE", PIPELINE_DIR / ".cache")).resolve()
VENV_DIR = PIPELINE_DIR / ".venv"
DEFAULT_OUT_DIR = REPO_ROOT / "apps" / "web" / "public" / "assets" / "body"

# Pinned upstream commits (full SHAs). Bumping these changes the outputs.
SOURCES = {
    "mh": {
        "url": "https://github.com/makehumancommunity/makehuman.git",
        "sha": "a8bc2d54ff0ac92e78ff71431b1023eda42bf482",
        "dir": CACHE_DIR / "mh",
        # Sparse checkout: only the CC0 data we read (never the AGPL code).
        "sparse": [
            "/makehuman/data/3dobjs/",
            "/makehuman/data/targets/",
            "/makehuman/data/modifiers/",
            "/LICENSE.md",
            "/LICENSE.ASSETS.md",
        ],
    },
    "mpfb2": {
        "url": "https://github.com/makehumancommunity/mpfb2.git",
        "sha": "3edf9df0551765be43563d047888cf7877eb89b4",
        "dir": CACHE_DIR / "mpfb2",
        "sparse": [
            "/src/mpfb/data/rigs/standard/",
            "/src/mpfb/data/mesh_metadata/",
            "/LICENSE.md",
            "/LICENSE.ASSETS.md",
        ],
    },
}

# MediaPipe (Apache-2.0): canonical face model + connection lists, plain files at a pinned commit (no git checkout).
MEDIAPIPE = {
    "repo": "google-ai-edge/mediapipe",
    "sha": "9519bb59bf55fc6a79ed5b9f283d72e6cdfb6678",
    "dir": CACHE_DIR / "mediapipe",
    "files": {
        "mediapipe/modules/face_geometry/data/canonical_face_model.obj":
            "8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618",
        "mediapipe/python/solutions/face_mesh_connections.py":
            "cc7171e44d0612db38720cb2fd2516c395992cdd7457efe341543be6f3c22d87",
    },
}
CANONICAL_OBJ = MEDIAPIPE["dir"] / MEDIAPIPE["sha"] / "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
FACE_CONNECTIONS_PY = MEDIAPIPE["dir"] / MEDIAPIPE["sha"] / "mediapipe/python/solutions/face_mesh_connections.py"

MH_DATA = SOURCES["mh"]["dir"] / "makehuman" / "data"
MPFB_DATA = SOURCES["mpfb2"]["dir"] / "src" / "mpfb" / "data"

RIG_NAME = "game_engine"

# MakeHuman units are decimeters; outputs are meters.
DM_TO_M = 0.1
# Morph deltas with every component below this magnitude (meters) are dropped.
DELTA_EPS_M = 1e-5

OUTPUT_FILES = ("base.glb", "morphs.bin", "manifest.json", "rig.json", "measures.json", "face-map.json")
