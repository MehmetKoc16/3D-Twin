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

# ---------------------------------------------------------------------------------------------------------------
# Body parts (eyes, eyebrows, eyelashes, hair): MakeHuman system assets (CC0), same mechanism as the garments.
# `files`: member path inside the pack zip -> sha256 (cached flat by file name in `.cache/parts/<id>/`).
# `packJson`: sha256 of `packs/<pack dir>.json`. Licence of each item = its own `.mhclo` line AND the pack json (ADR 0008).
# ---------------------------------------------------------------------------------------------------------------
PART_PACK_URL = GARMENT_PACK_URL
PART_CACHE = CACHE_DIR / "parts"
DEFAULT_PARTS_DIR = REPO_ROOT / "apps" / "web" / "public" / "assets" / "parts"
PART_TEXTURE_MAX = {"eyes": 512, "eyebrows": 512, "eyelashes": 512, "hair": 1024}  # px, longest side
PART_ASSETS: dict[str, dict] = {
    "eyes-default": {"pack": "makehuman_system_assets_cc0", "files": {
        "eyes/high-poly/high-poly.mhclo":
            "b183cfe37120ab726f9b3f2ea6cd3a64c44ce7b4bd91a77c841cf70c04f83a0d",
        "eyes/high-poly/high-poly.obj":
            "da2493215b708a344c33dc72f2a9a5b8fa985dcc5a70ad3b208995cf871da8e1",
        "eyes/materials/grey.mhmat":
            "65c536157ef0550b2f8461a9526c726707533dc7cfe8a65abb9b9db0b4e6b25e",
        "eyes/materials/grey_eye.png":
            "ecb05613126036a3d017880fabbd570501c5f14032c186462d8f6e2d719f6c4f",
    }, "packJson": "e86f1572432cfd062f372adddba4eb7c2e2e31280e96c29291a970320e0a5df0"},
    "eyebrows-default": {"pack": "makehuman_system_assets_cc0", "files": {
        "eyebrows/eyebrow001/eyebrow001.mhclo":
            "54c8892c2ba577f4152d779ce1f20f7cd323beb8cbd9cb701796165c221b4f5b",
        "eyebrows/eyebrow001/eyebrow001.obj":
            "88b13146395133f3901715e11706cdfd293e5d6087ad7c5e01efe1485a26b513",
        "eyebrows/eyebrow001/eyebrow001.mhmat":
            "976e5f4f1d3922d7a11c6e2db233f8b4584b068d00619d5749687e81c92d61fc",
        "eyebrows/eyebrow001/eyebrow001.png":
            "9940f7d0b1b223709a19b05156ba6f6e9e4dbdbced02d7574f4d51d72c58967b",
    }, "packJson": "e86f1572432cfd062f372adddba4eb7c2e2e31280e96c29291a970320e0a5df0"},
    "eyebrows-thick": {"pack": "makehuman_system_assets_cc0", "files": {
        "eyebrows/eyebrow009/eyebrow009.mhclo":
            "14819547ba73a98baac1f6eb445e19c22010d2ea30b8a15ee0ca8767c0f3e509",
        "eyebrows/eyebrow009/eyebrow009.obj":
            "037a2edb3252e1fe4a57e63dcb1598916609fd0339f2b453b2f71e9c4cbd481c",
        "eyebrows/eyebrow009/eyebrow009.mhmat":
            "53eca588d5de9378b60f22153d212485e1c76ea907ffc8f5f97b6ca824fe55b0",
        "eyebrows/eyebrow009/eyebrow009.png":
            "4d33c85718ca5accf44625652daf0ae6e8cc007ed8138280e4d95601717910c3",
    }, "packJson": "e86f1572432cfd062f372adddba4eb7c2e2e31280e96c29291a970320e0a5df0"},
    "eyebrows-thin": {"pack": "makehuman_system_assets_cc0", "files": {
        "eyebrows/eyebrow006/eyebrow006.mhclo":
            "3f14073a558605d78deb25b195c6db8c4ef1310ec5f3f230094b01f8cabd01ad",
        "eyebrows/eyebrow006/eyebrow006.obj":
            "7e3963bde6bc23a629eb78a9f83d050e247a5b1d370885afa0a92dede192cdb1",
        "eyebrows/eyebrow006/eyebrow006.mhmat":
            "0df50e6595ffdfcd485ba1f3d00f26b341dddb2c8e0df2c31fb72dd141b0df40",
        "eyebrows/eyebrow006/eyebrow006.png":
            "1f6d34d722ef18e73494e62f2c7f1a933ff58ce75d01fd34de21d178d5afe4d8",
    }, "packJson": "e86f1572432cfd062f372adddba4eb7c2e2e31280e96c29291a970320e0a5df0"},
    "eyelashes-default": {"pack": "makehuman_system_assets_cc0", "files": {
        "eyelashes/eyelashes01/eyelashes01.mhclo":
            "5a86b29c15649273d723084954b7fa7bf199112571d4f58539b46734d397ee6b",
        "eyelashes/eyelashes01/eyelashes01.obj":
            "f78f5b93fea1946fcae103d32c4f082dd9612a1ec2e7e2f90b03ebe59b0baf20",
        "eyelashes/eyelashes01/eyelashes01.mhmat":
            "8029c99d94f3d87a1e81825fad6b90eec773ae6955a5b2592222ef237377e190",
        "eyelashes/eyelashes01/eyelashes01.png":
            "4b69c0fff2648874460e9caf80c31413c444218a5c50afebc425aeaa65484a35",
    }, "packJson": "e86f1572432cfd062f372adddba4eb7c2e2e31280e96c29291a970320e0a5df0"},
    "hair-short": {"pack": "makehuman_system_assets_cc0", "files": {
        "hair/short02/short02.mhclo":
            "625736cdb73e6d094df6e5a2df18f371781f1d2cd37b0ae15795c5d7051c3ec2",
        "hair/short02/short02.obj":
            "48f979114adfa712165a69cc55c45a70831af1fb3cba8e2a89120f0c87407b64",
        "hair/short02/short02.mhmat":
            "2ed04c8aad9c1a35d858ca72091525d32b1da3b61b5a031634dd528fd8530f6d",
        "hair/short02/short02_diffuse.png":
            "47fe33831a3929567c733356dd66243116e05df2ace1f884ddca0080b728229f",
    }, "packJson": "e86f1572432cfd062f372adddba4eb7c2e2e31280e96c29291a970320e0a5df0"},
    "hair-tousled": {"pack": "hair01_cc0", "files": {
        "hair/culturalibre_hair_05/culturalibre_hair_05.mhclo":
            "e3888a94ebb84bc37b73cefb3af7045bc674a8c3d92ff8e910e75c283a234078",
        "hair/culturalibre_hair_05/hair_05.obj":
            "7f710a92a7c47a318bf83c66a381aecaccf7d992f77681fd4c287d0c3b209cf9",
        "hair/culturalibre_hair_05/hair_05.mhmat":
            "bd786b142ba4e2eeab426f7ea12f61bd96f3c4a5b1d6012983373be72800159a",
        "hair/culturalibre_hair_05/hair_05.png":
            "3e03935cd03bb8f34a72bf53810d9cdd74634c3a0e4d121f6866d118eda3ecfe",
    }, "packJson": "b1f3c32a877cf7fb681ed54d28715d53a4d9660656ae063c6a50229181830de7"},
    "hair-bob": {"pack": "makehuman_system_assets_cc0", "files": {
        "hair/bob02/bob02.mhclo":
            "a48dcf87c6a618f9c6e90fcb00eed07cd59ab024d8d994606fa1b070da515b96",
        "hair/bob02/bob02.obj":
            "30d44038836d786364eb4134a9259cf93ca0c707f883861b0135ebfb149caa08",
        "hair/bob02/bob02.mhmat":
            "32c1712308159f4f5b3775a633352070245bf1ea8b6f837ab4f4ba82b5e9fccd",
        "hair/bob02/bob02_diffuse.png":
            "38c88e6f71631356591f09b1f17a1bb5a99c3bb095b1a29003f1dc3d3e839b0d",
    }, "packJson": "e86f1572432cfd062f372adddba4eb7c2e2e31280e96c29291a970320e0a5df0"},
    "hair-medium": {"pack": "hair01_cc0", "files": {
        "hair/toigo_inverted_bob/toigo_inverted_bob.mhclo":
            "af32890eb389dd536afcb8f6cd5cbf1f57fd42be6a7c1ab2bcb56e0b383473c5",
        "hair/toigo_inverted_bob/bob_inverted.obj":
            "a6b7c685b8558b9983cb8a883ebacf0272a238931ca20482671fe5da64a31c10",
        "hair/toigo_inverted_bob/bob_inverted.mhmat":
            "ad2abc59a0e16f2c2ac08b5210df83b6322f85a10183f95a7b68b96b53218033",
        "hair/toigo_inverted_bob/GoldenBlondHair.png":
            "0d839379f85e757aa8d5cd92281efe830beba1e34f1ffd93954eac704d729378",
    }, "packJson": "b1f3c32a877cf7fb681ed54d28715d53a4d9660656ae063c6a50229181830de7"},
    "hair-long": {"pack": "makehuman_system_assets_cc0", "files": {
        "hair/long01/long01.mhclo":
            "94ecbebfd834da2cbee2ce90b730972590eb22a252ecd9592038ea736e063646",
        "hair/long01/long01.obj":
            "7f1d6dadbdf9e96435251955d6657e77ea5433c9bc7c85a5c5f5e4eae0cb9a8c",
        "hair/long01/long01.mhmat":
            "74ec88267e679d316f8037a074511ffc887b27e198ea381d0ae4a3f180f96760",
        "hair/long01/long01_diffuse.png":
            "e8dc25d90f8f62467e8630420e1240a400d070054fb7c3f1d3f7bf3e457d6c2c",
    }, "packJson": "e86f1572432cfd062f372adddba4eb7c2e2e31280e96c29291a970320e0a5df0"},
    "hair-ponytail": {"pack": "makehuman_system_assets_cc0", "files": {
        "hair/ponytail01/ponytail01.mhclo":
            "97fc50b12bcb3b1c7b33edddd8d427a2604ffb06fe79b31132bf47f82d7afbac",
        "hair/ponytail01/ponytail01.obj":
            "d9f5fe96fbedbc8006220d87aad0e4a33d15286eb6b21f5929661cdc490ebf5a",
        "hair/ponytail01/ponytail01.mhmat":
            "b145f5a7c9606859ad523f5113482a1cf989cf0e26dccc2ea95c377cb1c2557f",
        "hair/ponytail01/ponytail01_diffuse.png":
            "dcb364300cac06ce8bef91b2f7c0970baf310898bbdd06b0c6b884ee9306eb97",
    }, "packJson": "e86f1572432cfd062f372adddba4eb7c2e2e31280e96c29291a970320e0a5df0"},
}

MH_DATA = SOURCES["mh"]["dir"] / "makehuman" / "data"
MPFB_DATA = SOURCES["mpfb2"]["dir"] / "src" / "mpfb" / "data"

RIG_NAME = "game_engine"

# MakeHuman units are decimeters; outputs are meters.
DM_TO_M = 0.1
# Morph deltas with every component below this magnitude (meters) are dropped.
DELTA_EPS_M = 1e-5

OUTPUT_FILES = ("base.glb", "morphs.bin", "manifest.json", "rig.json", "measures.json", "face-map.json")
