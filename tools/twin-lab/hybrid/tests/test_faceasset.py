import json

import numpy as np
from hybridbody.faceasset import SCHEMA, pack_offsets, read_offsets, write_face_asset


def test_offsets_use_the_morph_binary_layout_and_roundtrip():
    offsets = np.zeros((10, 3))
    offsets[[2, 5, 9]] = [[0.001, 0, 0], [0, -0.002, 0.0005], [0.0, 0.0, 0.003]]
    offsets[4] = 1e-9  # below the threshold: dropped
    blob, count = pack_offsets(offsets)
    assert count == 3 and len(blob) == 16 * count  # 16 bytes per entry, exactly like morphs.bin
    assert np.frombuffer(blob, "<u4", 1)[0] == 2
    back = read_offsets(blob, 10)
    np.testing.assert_allclose(back, np.where(np.abs(offsets) > 1e-7, offsets, 0), atol=1e-8)
    indices = np.frombuffer(blob, np.dtype([("i", "<u4"), ("d", "<f4", 3)]))["i"]
    assert (np.diff(indices) > 0).all()  # strictly ascending, as morphs.bin requires


def test_write_face_asset_documents_every_file(tmp_path):
    offsets = np.zeros((20, 3))
    offsets[3] = [0.002, 0.001, 0]
    document = write_face_asset(
        tmp_path / "asset",
        offsets=offsets,
        manifest_sha256="abc",
        solved_body={"macros": {"gender": 0.6}, "modifiers": {}, "targetsCm": {"neck": 37.0}},
        head_texture=np.full((8, 6, 3), 90, np.uint8),
        window_px=(100, 200, 6, 8),
        atlas_size=(1024, 1280),
        skin={"lab": [58, 8, 10], "srgbHex": "#a08679"},
        parts={"hair": {"id": "hair-short", "colourHex": "#49413d"}},
        metrics={"maxDisplacementMm": 2.0},
    )
    folder = tmp_path / "asset"
    on_disk = json.loads((folder / "face-asset.json").read_text())
    assert on_disk == json.loads(json.dumps(document)) and on_disk["schema"] == SCHEMA
    assert (folder / on_disk["headOffsets"]["file"]).stat().st_size == 16 * on_disk["headOffsets"]["count"]
    window = on_disk["texture"]["uvWindow"]
    assert window["u0"] == 100 / 1024 and window["v0"] == 200 / 1024 and window["u1"] == 106 / 1024
    assert on_disk["template"]["renderVertexCount"] == 20 and "never commit" in on_disk["license"]
    assert (folder / "face-texture.png").is_file()
