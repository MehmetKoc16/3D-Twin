import numpy as np
import trimesh
from PIL import Image
from synth import VIEW_YAW, ellipsoid, make_cam, photo_from
from twinrefine.scan import Scan, load_scan, save_scan

import head as head_cli
from headrecon.photos import load_photos


def _write_inputs(tmp_path, with_clean=False):
    target, tf, _ = ellipsoid((0.078, 0.115, 0.095), 3, soup=False)
    photos = tmp_path / "photos"
    cache = tmp_path / "out" / "cache"
    photos.mkdir()
    cache.mkdir(parents=True)
    for name, yaw in VIEW_YAW.items():
        ph = photo_from(target, tf, make_cam(yaw), name)
        Image.fromarray(ph.rgb).save(photos / f"{name}.jpg", quality=95)
        Image.fromarray((ph.mask * 255).astype(np.uint8)).save(cache / f"mask_{name}.png")  # newer than the photo: no rembg
    if with_clean:
        (photos / "clean").mkdir()
        Image.fromarray(np.full((800, 640, 3), 90, np.uint8)).save(photos / "clean" / "front.jpg")
    v, f, uv = ellipsoid((0.09, 0.115, 0.10), 3, soup=True)
    atlas = np.full((256, 256, 3), (255, 0, 255), np.uint8)
    glb = tmp_path / "refined.glb"
    save_scan(glb, Scan(v, f, uv.astype(np.float32), atlas))
    return glb, photos


def test_load_photos_prefers_originals_and_uses_clean_on_request(tmp_path):
    _glb, photos = _write_inputs(tmp_path, with_clean=True)
    ph = load_photos(photos, tmp_path / "out" / "cache", use_clean=False)
    assert not ph["front"].clean and ph["front"].mask.any()
    ph = load_photos(photos, tmp_path / "out" / "cache", use_clean=True)
    assert ph["front"].clean and ph["front"].tex_rgb[0, 0, 0] == 90 and not ph["back"].clean


def test_cli_end_to_end_on_a_synthetic_head(tmp_path):
    glb, photos = _write_inputs(tmp_path)
    out = tmp_path / "out" / "head.glb"
    rc = head_cli.main(["--in", str(glb), "--photos", str(photos), "--out", str(out), "--no-previews", "--rounds", "2"])
    assert rc == 0 and out.exists()
    s = load_scan(out)
    assert s.atlas is not None and s.atlas.shape == (256, 256, 3)
    assert not np.isnan(s.verts).any()
    mesh = trimesh.Trimesh(s.verts, s.faces, process=False)
    assert mesh.area > 0.05
    # narrower head than the input (the photos come from a 15 mm narrower ellipsoid)
    v0 = load_scan(glb).verts
    assert np.ptp(s.verts[:, 0]) < np.ptp(v0[:, 0])
    # the head texture was rebuilt: the magenta sentinel of the old atlas is mostly gone
    sentinel = (np.abs(s.atlas.astype(int) - [255, 0, 255]).sum(axis=2) < 40).mean()
    assert sentinel < 0.6
    assert (tmp_path / "out" / "head_report.json").exists()
