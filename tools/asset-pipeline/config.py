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

# ---------------------------------------------------------------------------------------------------------------
# Garment templates: MakeHuman community asset packs (https://files2.makehumancommunity.org/asset_packs/).
# Only single files are fetched (HTTP range reads of the zip's central directory) and verified by sha256; the whole
# pack zips are never required. Licences are re-checked from each asset's own `.mhclo` header AND the pack's json.
# ---------------------------------------------------------------------------------------------------------------
GARMENT_PACK_URL = "https://files2.makehumancommunity.org/asset_packs/{name}/{pack}.zip"
GARMENT_CACHE = CACHE_DIR / "garments"
DEFAULT_GARMENTS_DIR = REPO_ROOT / "apps" / "web" / "public" / "assets" / "garments"
GARMENT_TEXTURE_MAX = 1024  # px, longest side

# pack -> sha256 of the complete zip as downloaded on 2026-09-29 (provenance only; the per-file hashes are enforced).
GARMENT_PACK_ZIPS = {
    "shirts01_cc0": "a5a723b0e84a109bb190fcfeac7f1de4138d875da3e30fe5b3340eac9f38bcd3",
    "pants01_cc0": "e4e0ec60db34f279be291a83cfd7b342a7c5cf09bb7676682a5f39f4f6ac4ad9",
    "pants02_ccby": "9dbcd65e03ab100977079b6960e91c334bed92e948aeda5423f11997a133904a",
    "shoes01_cc0": "ded3f70428505eabbf1f6d7b5f61196a7366ef20757103d276ad0ed336c35ada",
    "shoes02_ccby": "1b544d87dd8b3d3a9c8317e4f059be456491be979cbc9984fc53f403046f5061",
}

# template id -> upstream asset. `files`: name inside clothes/<dir>/ -> sha256. `packJson`: sha256 of packs/<pack>.json.
GARMENT_ASSETS = {
    "tshirt": {"pack": "shirts01_cc0", "dir": "toigo_basic_tucked_t-shirt", "files": {
        "T-shirt_basic.png": "c1992ad975e14aa2cef891ff9632d28283607b2de6160d72e350f059043a063b",
        "t_shirt_basic_tucked.mhmat": "44098003a7ed47cd4c613d04adcdccaae47fe0d2b380bf3c25f105a739814f94",
        "t_shirt_basic_tucked.obj": "22159cf14d1f6831f931b49625f8f7692910bac22499c43466c0b5a2159b4d03",
        "toigo_basic_tucked_t-shirt.mhclo": "c34bd03895c5ad9ec46965ef7d5f3e5e8c567630797ab312f8762c1d9fde8f71",
    }, "packJson": "fc1bdab84824c7f36e1482f8f40d381cdeda49315003caba50a40d06b8ed2998"},
    "sweatshirt": {"pack": "shirts01_cc0", "dir": "toigo_fisherman_sweater", "files": {
        "shirt-knit.png": "a7d526797da4cdc1a79603a35abebf41e577277267f36db6641cdb6cebe6ffb1",
        "sweater_fisherman.mhmat": "254c011b3edae8cbab7adb2d082de670dd41b26324bf5772318b7557a127fd64",
        "sweater_fisherman.obj": "2844465f4e798666a90f631ea4a89ae30d3e526ef045a978d338908750d21d09",
        "toigo_fisherman_sweater.mhclo": "b454960bdd61985f246e88e547a127a75d590dc87d6c1a4062fb7c7c281d8999",
    }, "packJson": "fc1bdab84824c7f36e1482f8f40d381cdeda49315003caba50a40d06b8ed2998"},
    "pants": {"pack": "pants01_cc0", "dir": "toigo_wool_pants", "files": {
        "Pants_wool.png": "a820b0dc4f9f881956a068f26889ca677b696dbdccfdbf2a168e7b327b390ab6",
        "pants_wool.mhmat": "fedf31e98ea92b57c8f45e10ec167d420f5eb11b9410feb075c321c3798fdb53",
        "pants_wool.obj": "e02cd1e417ca3365e4dafc033f415c5186739df535d98810075cface0ce59c96",
        "toigo_wool_pants.mhclo": "ac1a57a817976d80ece8a88698873d54de7f252d01ba3dca5e361693621fcf22",
    }, "packJson": "bdb03e12282c92d8dc66e34e5698b77340baef8eff212515640c1255d3ec7261"},
    "jeans": {"pack": "pants02_ccby", "dir": "punkduck_male_classic_jeans", "files": {
        "male-classic-jeans.mhmat": "bfd98ff38b170843406a78f09823eb2c5a39efc6f94e192cf7c5ce67500f5723",
        "male-classic-jeans.obj": "847cd28a532f12a560c5cbdd7069b1fea49ee9565f372e9965f8fd3653595fbf",
        "male-classic-jeans.png": "0cf8731e05cb26c0976276b9f320acff246a26a46a4c4f263c35d111835fa5c4",
        "punkduck_male_classic_jeans.mhclo": "8e375d1fc5eae9a52f2677eb02474b116d2ce6918c8e45142d254391c6b46944",
    }, "packJson": "7dad7481cd5f4e821bf88432d353e5e76f604ceea153e672aea679e333da3d10"},
    "sneakers": {"pack": "shoes02_ccby", "dir": "culturalibre_sneakers", "files": {
        "culturalibre_sneakers.mhclo": "fdb2a1d62dce8f759d544e5022f87e7530d43b1d03b7db80f0bbcd69ba7ddfab",
        "sneaker_brown_diffuse.PNG": "cdb45a0f6d38038e8e06dceb10fddbcc4ebcdc02d2a6962b9eb3e7ca1c8008fe",
        "sneakers.mhmat": "d05d66b5890e0325eb9ca77955f7e1d6733ed80dd0463765dc36929be120dd80",
        "sneakers.obj": "a9799d8c1d96a3e737d54afc7bc66410d7539fcc08e983cd7861e5a2fb31356a",
    }, "packJson": "2788098ec06025f0c986b587e0bf4c7472428d0d87a54d3f731aa95055a2ad1f"},
    "shoes": {"pack": "shoes01_cc0", "dir": "toigo_mj_cloth_shoes", "files": {
        "MJ-shoes3.png": "fa20149c846d9cb9d4837363f82dac87892efed7a152bbf966566f623062da4b",
        "mj_shoes.mhmat": "e36cc865d562690c4391d7f78dd1ee7e5690f9219ac2438b1ec4d24442f27b86",
        "mj_shoes.obj": "adcfd0886d1e25a165511419db017075afa95ea5f3d172b1f10af7a1a8fcea48",
        "toigo_mj_cloth_shoes.mhclo": "6e81c1d6475a03b91a2c7002176883efce5e2a0b129d6c8cb00864415cdb905e",
    }, "packJson": "f59c2d5151aee4b2e9a807eb0050bce080013dc79b7d21757bbda34931bac77b"},
    "boots": {"pack": "shoes01_cc0", "dir": "culturalibre_hero_boots_2", "files": {
        "blue.png": "605c22b08c5fa0f00cef40f7db561a40493d1c8820b2e17faa12c856bb9f42e6",
        "culturalibre_hero_boots_2.mhclo": "b8046c7e7b9cf39ea93d1155fbb76f9bdfc612c7a0ea140a11aba38a68f3542f",
        "hero_boots_2.mhmat": "a18dd597d6a3cd852a0b301681c3da96a900c1441b20141ef8ad6540231fea31",
        "hero_boots_2.obj": "dd1bd5b4b5cac58ef1a624a52776be71f5f9a4b6295c9074971dee5cc8d48521",
    }, "packJson": "f59c2d5151aee4b2e9a807eb0050bce080013dc79b7d21757bbda34931bac77b"},
}

MH_DATA = SOURCES["mh"]["dir"] / "makehuman" / "data"
MPFB_DATA = SOURCES["mpfb2"]["dir"] / "src" / "mpfb" / "data"

RIG_NAME = "game_engine"

# MakeHuman units are decimeters; outputs are meters.
DM_TO_M = 0.1
# Morph deltas with every component below this magnitude (meters) are dropped.
DELTA_EPS_M = 1e-5

OUTPUT_FILES = ("base.glb", "morphs.bin", "manifest.json", "rig.json", "measures.json", "face-map.json")
