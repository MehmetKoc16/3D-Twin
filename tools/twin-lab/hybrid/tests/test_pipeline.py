"""End to end on the real CC0 MakeHuman assets with a synthetic FLAME fit and gradient photos."""

import json
import pickle
import sys

import numpy as np
import pytest
from glbio import _split, read_glb
from hybridbody import BODY_ASSETS, pipeline
from hybridbody.headfit import build_template, read_face_map
from hybridbody.pipeline import verify_report
from scipy.spatial import cKDTree
from synth import write_fit_files
from synth_mh import bodyfix_solution, mh_flame, write_bodyfix_glb

SIZE = 512


def execute(model, root, **options):
    fit = root / "fit"
    fit.mkdir()
    solution = bodyfix_solution(model)
    bodyfix = root / "bodyfix"
    bodyfix.mkdir()
    write_bodyfix_glb(bodyfix / "bodyfixed.glb", model, solution)
    combined = model.shape(solution["fittedMacros"], solution["fittedModifiers"], ground=True)
    measures = json.loads((BODY_ASSETS / "measures.json").read_text())["measures"]
    neck = next(d for d in measures if d["id"] == "neck")
    face_map = read_face_map(BODY_ASSETS / "face-map.json")
    template = build_template(combined[: model.nr], model.faces, face_map, neck["verts"])
    flame = mh_flame(template, face_map, fit)
    photos = write_fit_files(fit, flame)
    original_repo, original_load = pipeline.REPO, pipeline.load_flame_fit
    pipeline.REPO = root
    pipeline.load_flame_fit = lambda *a, **k: flame
    try:
        out = root / "user-data/hybrid/hybrid.glb"
        report = pipeline.run(
            bodyfix, out, fit=fit, photos=photos, flame_assets=fit, texture_size=SIZE, previews=False, **options
        )
    finally:
        pipeline.REPO, pipeline.load_flame_fit = original_repo, original_load
    return root, out, report, solution, combined


@pytest.fixture(scope="module")
def run_result(model, tmp_path_factory):
    return execute(model, tmp_path_factory.mktemp("repo"))


def test_brows_are_optional_and_follow_the_deformed_head(model, run_result, tmp_path):
    _, _, default, _, _ = run_result
    assert "eyebrows" not in default["parts"]["slices"] and default["parts"]["eyebrows"]["enabled"] is False
    _, out, report, _, _ = execute(model, tmp_path, brows=True, hair="hair-short")
    assert "eyebrows" in report["parts"]["slices"] and report["parts"]["hair"]["id"] == "hair-short"
    clearance = report["parts"]["eyebrows"]["clearance_to_head"]
    assert clearance["median_mm"] > -2 and clearance["median_mm"] < 12  # brow cards ride on the deformed skin
    lashes = report["parts"]["eyelashes"]["clearance_to_head"]
    assert lashes["median_mm"] > -4
    assert read_glb(str(out)).prims[0].positions.shape[0] == report["mesh"]["vertices"]


def test_outputs_exist_and_the_report_passes_its_numeric_checks(run_result):
    _, out, report, _, _ = run_result
    assert out.is_file() and (out.parent / "hybrid_report.json").is_file()
    assert verify_report(report) == []
    neck = report["head"]["neck"]
    assert neck["girth_cm"] == pytest.approx(neck["solved_girth_cm"], abs=1e-6)
    assert report["head"]["displacement_mm"]["fixed_below_anchor_max"] == 0.0
    assert report["head"]["residual_to_flame_surface"]["face"]["mean_mm"] < 3.0
    check = report["texture"]["vertex_photo_check"]
    assert check["matched_mean_de"] < check["shuffled_mean_de"]
    assert report["seam"]["delta_e76"] < 2 and report["texture"]["fill_ratio"] == 1.0
    validity = report["mesh"]["validity"]
    assert validity["body_nonmanifold_edges"] == 0 and validity["finite"] and validity["uv_in_range"]
    assert report["parts"]["hair"]["clearance_to_head"]["deeper_than_10mm"] < 0.05
    assert report["hands"]["native"] is True and report["body"]["mode"] == "saved-bodyfix-exact"
    saved = json.loads((out.parent / "hybrid_report.json").read_text())
    assert saved["glb"]["bytes"] == out.stat().st_size


def test_glb_is_one_textured_primitive_with_the_hybrid_marker_and_hand_flags(run_result):
    _, out, report, solution, _ = run_result
    scene = read_glb(str(out))
    assert len(scene.prims) == 1 and scene.images and scene.prims[0].uv is not None
    extras = scene.extras
    assert extras["dtScanHandsRemoved"] is False and extras["dtHasMakeHumanHands"] is True
    assert extras["dtBodyfix"]["fittedMacros"] == solution["fittedMacros"]
    marker = extras["dtHybrid"]
    assert marker["frame"] == "MakeHuman-grounded-A-pose" and marker["ownsHands"] is True
    assert report["mesh"]["vertices"] == len(scene.prims[0].positions)
    uv = scene.prims[0].uv
    assert uv.min() >= 0 and uv.max() <= 1


def test_body_vertices_below_the_neck_are_exactly_the_solved_body(run_result, model):
    _, out, _, _, combined = run_result
    scene = read_glb(str(out))
    positions = scene.prims[0].positions.astype(np.float64)
    below = positions[positions[:, 1] < scene.extras["dtHybrid"]["cutHeightM"] - 1e-3]
    assert len(below) > 5000
    assert cKDTree(combined[: model.nr]).query(below)[0].max() < 2e-5


def test_face_asset_is_written_next_to_the_glb(run_result):
    _, out, report, _, _ = run_result
    folder = out.parent / "face_asset"
    document = json.loads((folder / "face-asset.json").read_text())
    assert document["schema"] == "dt-face-asset/1"
    assert report["face_asset"]["offsets"] == document["headOffsets"]["count"]
    assert document["headOffsets"]["count"] > 1000 and document["parts"]["hair"]["id"] == "hair-tousled"
    assert (folder / "face-offsets.bin").stat().st_size == 16 * document["headOffsets"]["count"]
    assert (folder / "face-texture.png").is_file()


def test_rig_stage_accepts_the_hybrid_as_a_verified_native_a_pose(run_result, monkeypatch, tmp_path):
    _, out, _, solution, _ = run_result
    import rig_scan

    folder = tmp_path / "rig"
    argv = ["rig_scan.py", str(out), str(folder), "--fingers", "keep", "--smooth", "0", "--samples", "20000"]
    monkeypatch.setattr(sys, "argv", argv)
    rig_scan.main()
    twin = json.loads((folder / "twin.json").read_text())
    assert twin["fittedMacros"] == solution["fittedMacros"]
    assert twin["measurementsCm"] == solution["achievedCm"]
    fit = pickle.loads((folder / "fit.pkl").read_bytes())
    assert fit.stats["pose_source"] == "verified-native-A-pose"
    js, _ = _split((folder / "rigged.glb").read_bytes())
    assert len(js["meshes"]) == 1 and len(js["skins"]) == 1 and len(js["skins"][0]["joints"]) == 53


def test_glb_uses_the_alpha_mask_material_with_an_rgba_png_atlas(run_result):
    import io

    from glbio import _split
    from PIL import Image

    _, out, report, _, _ = run_result
    js, blob = _split(out.read_bytes())
    material = js["materials"][0]
    assert (material["alphaMode"], material["alphaCutoff"], material["doubleSided"]) == ("MASK", 0.5, True)
    image = js["images"][0]
    assert image["mimeType"] == "image/png"
    view = js["bufferViews"][image["bufferView"]]
    atlas = np.asarray(Image.open(io.BytesIO(blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]])))
    assert atlas.shape[2] == 4 and atlas.shape[1] <= 4096
    body_rows = atlas.shape[1]
    assert (atlas[:body_rows, :, 3] == 255).all()  # skin is opaque
    assert report["glb"]["image_mime"] == "image/png"


def test_neck_below_the_chin_matches_the_body_skin(run_result):
    _, _, report, _, _ = run_result
    neck = report["texture"]["neck"]
    after = neck["after"]["below_chin"]
    assert after["texels"] > 0 and abs(after["delta_l_vs_body"]) < 3
    assert report["texture"]["scalp_tint"]["covered_texel_fraction"] >= 0
