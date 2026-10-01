"""End-to-end on a synthetic, non-personal scan: the CC0 MakeHuman render mesh with a synthetic texture, a flat
"photo" made from its silhouette, landmarks injected from the body's own face-map points, and a pickled stand-in fit."""

import json
import pickle

import numpy as np
import pytest
from helpers import neutral_fit
from PIL import Image
from twintex.camera import OrthoCamera
from twintex.raster import zbuffer

from twinrefine import facefit, frontview, meshops
from twinrefine.landmarks import FaceLandmarks
from twinrefine.pipeline import RefineConfig, run
from twinrefine.scan import Scan, load_scan, save_scan


def _stats(path):
    sc = load_scan(path)
    mc = meshops.to_corners(sc.verts, sc.faces, sc.uv)
    et = meshops.edge_table(mc.F)
    return sc, mc, np.bincount(et.count, minlength=5)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory, model):
    d = tmp_path_factory.mktemp("refine")
    fit = neutral_fit(model)
    pos = fit.rest_positions[: model.nr].astype(np.float32)
    yy, xx = np.mgrid[0:256, 0:256]
    atlas = np.stack([(xx * 0.9 + 20) % 255, (yy * 0.7 + 40) % 255, 128 + 0 * xx], axis=-1).astype(np.uint8)
    scan = Scan(pos.astype(np.float64), model.faces.astype(np.int64), model.uv.astype(np.float32), atlas)
    inp = d / "out" / "texture" / "textured.glb"
    inp.parent.mkdir(parents=True)
    save_scan(inp, scan)
    # the "photo": a flat colour silhouette of the body in an orthographic front view, RGBA with the cut-out alpha
    W, H = 360, 520
    cam = OrthoCamera.axis("front").fit_bounds(pos, W, H, margin=0.06)
    p = cam.project(pos)
    _depth, fid = zbuffer(p, model.faces, W, H)
    alpha = (fid >= 0).astype(np.uint8) * 255
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:] = (205, 160, 140)
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(d / "front.png")
    (d / "out" / "refine" / "cache").mkdir(parents=True)
    with open(d / "out" / "refine" / "cache" / "fit.pkl", "wb") as fh:
        pickle.dump(fit, fh)
    return d, inp, fit


def test_pipeline_end_to_end(synthetic, model):
    d, inp, fit = synthetic
    sc0, mc0, hist0 = _stats(inp)
    fv = frontview.build_front(d / "front.png", mc0.P, mc0.F, inp)
    fm = facefit.load_face_map()
    fmodel = facefit.FaceModel(model, fit, fm)
    P = fmodel.landmark_points(np.zeros(len(fmodel.modifiers)))
    px = fv.world_to_photo(P)
    lm = FaceLandmarks(np.vstack([px, px[:10]]), np.zeros(478), (0, 0, 10, 10))

    out = d / "out" / "refine" / "refined.glb"
    cfg = RefineConfig(inp=inp, out=out, front=d / "front.png", previews=False, detector=lambda rgb, alpha, full: lm)
    report = run(cfg)
    assert out.exists()
    assert (out.parent / "refine_report.json").exists()
    rep = json.loads((out.parent / "refine_report.json").read_text())

    # face: landmarks agree with the body, so the fit is tight and the relief small (the scan IS the MakeHuman face)
    face = report["face"]
    assert face["landmark_rms_mm"]["after"] < 1.5
    assert face["relief"]["faces_after"] > face["relief"]["faces_before"]
    assert face["relief"]["mean_move_mm"] < 4.0
    assert face["rebake"]["texels"] > 100
    # armpits: the A-pose MakeHuman body has no fused arms, nothing to cut
    assert report["armpits"]["status"].startswith("skipped")
    assert rep["output_faces"] == report["output_faces"]

    sc1, mc1, hist1 = _stats(out)
    assert sc1.atlas is not None and sc1.atlas.shape == sc0.atlas.shape
    assert len(mc1.F) > len(mc0.F)
    # topology: no new open or non-manifold edges (the base mesh itself has holes at eyes / mouth)
    assert hist1[1] <= hist0[1] + 2
    assert hist1[3:].sum() <= hist0[3:].sum()
    # same body: only the face region moved, and only along z
    assert sc1.verts[:, 1].max() == pytest.approx(sc0.verts[:, 1].max(), abs=2e-3)
    assert (sc1.atlas != sc0.atlas).any()
    # the head is the only place that changed
    below = sc1.verts[:, 1] < 1.35
    keep = meshops.weld_ids  # noqa: F841 (documentation: welded ids are position based)
    old_pts = {tuple(np.round(v, 4)) for v in sc0.verts[sc0.verts[:, 1] < 1.35]}
    new_pts = {tuple(np.round(v, 4)) for v in sc1.verts[below]}
    assert old_pts == new_pts


def test_pipeline_skips_face_without_front_photo(synthetic, tmp_path):
    d, inp, _fit = synthetic
    out = tmp_path / "o" / "refined.glb"
    (out.parent / "cache").mkdir(parents=True)
    import shutil

    shutil.copy(d / "out" / "refine" / "cache" / "fit.pkl", out.parent / "cache" / "fit.pkl")
    cfg = RefineConfig(inp=inp, out=out, front=tmp_path / "missing.png", previews=False, armpits=False)
    report = run(cfg)
    assert report["face"]["status"].startswith("skipped")
    assert out.exists()
    sc = load_scan(out)
    assert len(sc.faces) == len(load_scan(inp).faces)


def test_cli_help_and_missing_input(capsys):
    import refine

    with pytest.raises(SystemExit) as e:
        refine.main(["--help"])
    assert e.value.code == 0
    assert refine.main(["--in", "does-not-exist.glb", "--out", "x.glb"]) == 2
