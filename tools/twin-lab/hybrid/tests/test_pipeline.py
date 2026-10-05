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


def test_deglass_hook_audits_mask_and_respects_opt_out(model, tmp_path, monkeypatch):
    from hybridbody import deglass_tex, glasses_hy3d

    def guides(*args, **kwargs):
        return {"synthetic": True}

    def projection(face, report):
        assert report == {"synthetic": True}
        mask = np.zeros(face.covered.shape, bool)
        y, x = np.argwhere(face.covered)[len(face.texel_y) // 2]
        mask[max(0, y-3):y+4, max(0, x-3):x+4] = True
        return mask & face.covered

    monkeypatch.setattr(glasses_hy3d, "removal_report_from_bake", guides)
    monkeypatch.setattr(deglass_tex, "projection_from_report", projection)
    bust = tmp_path / "synthetic-accessory.glb"
    bust.write_bytes(b"synthetic presence marker; no source mesh is read")
    root = tmp_path / "enabled"
    root.mkdir()
    _, out, report, _, _ = execute(model, root, hair="hair-short", glasses_bust=bust)
    assert report["texture"]["deglass"]["enabled"]
    audit = np.load(out.parent / "deglass_audit.npz")
    assert audit["mask"].sum() > 0
    assert np.array_equal(audit["before"][~audit["mask"]], audit["after"][~audit["mask"]])
    doc, _ = _split(out.read_bytes())
    assert doc["asset"]["extras"]["dtDeglass"]["enabled"]
    root = tmp_path / "disabled"
    root.mkdir()
    monkeypatch.setattr(glasses_hy3d, "removal_report_from_bake", lambda *a, **k: pytest.fail("opt out measured frames"))
    _, out, report, _, _ = execute(model, root, hair="hair-short", glasses_bust=bust, deglass=False)
    assert report["texture"]["deglass"] == {"enabled": False, "reason": "opt out"}
    assert not (out.parent / "deglass_audit.npz").exists()
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


def test_glb_has_the_textured_body_and_a_separate_hair_node_with_the_hybrid_marker_and_hand_flags(run_result):
    _, out, report, solution, _ = run_result
    scene = read_glb(str(out))
    assert len(scene.prims) == 2 and len(scene.images) == 2 and scene.prims[0].uv is not None
    assert [p.name for p in scene.prims] == ["twin", "dtHair"]
    extras = scene.extras
    assert extras["dtScanHandsRemoved"] is False and extras["dtHasMakeHumanHands"] is True
    assert extras["dtHairNode"] == "dtHair"
    assert extras["dtBodyfix"]["fittedMacros"] == solution["fittedMacros"]
    marker = extras["dtHybrid"]
    assert marker["frame"] == "MakeHuman-grounded-A-pose" and marker["ownsHands"] is True
    assert report["mesh"]["vertices"] == len(scene.prims[0].positions)  # the body mesh carries no hair cards
    uv = scene.prims[0].uv
    assert uv.min() >= 0 and uv.max() <= 1
    hair = scene.prims[1]
    assert hair.uv.min() >= 0 and hair.uv.max() <= 1 and hair.normals is not None and hair.material == 1
    assert len(hair.indices) == report["parts"]["hair"]["procedural"]["triangles"]
    assert np.isfinite(hair.positions).all() and abs(np.linalg.norm(hair.normals, axis=1) - 1).max() < 1e-3


def test_hair_material_and_atlas_follow_the_contract(run_result):
    import io

    from glbio import _split
    from PIL import Image

    _, out, report, _, _ = run_result
    js, blob = _split(out.read_bytes())
    assert [n["name"] for n in js["nodes"] if "mesh" in n] == ["twin", "dtHair"] and js["asset"]["extras"][
        "dtHairNode"
    ] == "dtHair"
    material = js["materials"][1]
    assert (material["name"], material["alphaMode"], material["alphaCutoff"], material["doubleSided"]) == (
        "dtHair",
        "MASK",
        0.5,
        True,
    )
    info = material["extras"]["dtHair"]
    assert info["format"] == "rcov-groot-bvar/1" and info["colorHex"] == "#2a1e18"
    assert (
        info["rootHex"] < info["colorHex"] < info["tipHex"]
        and info["cardCount"] == report["parts"]["hair"]["procedural"]["cards"]
    )
    assert report["parts"]["hair"]["colour_method"] == "default" and report["parts"]["hair"]["colour_hex"] == "#2a1e18"
    texture = js["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    image = js["images"][texture["source"]]
    assert image["mimeType"] == "image/png" and texture["source"] != js["textures"][0]["source"]
    view = js["bufferViews"][image["bufferView"]]
    atlas = np.asarray(Image.open(io.BytesIO(blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]])))
    assert atlas.shape == (1024, 2048, 4)
    r, g, b, a = (atlas[..., k] / 255.0 for k in range(4))
    assert 0.2 < r.mean() < 0.5 and r.max() > 0.9 and (g[:4].mean() < 0.05)  # root rows on top
    np.testing.assert_allclose(a, np.minimum(1.0, 2.5 * r), atol=1.5 / 255)
    assert report["glb"]["hair"]["atlas_size"] == [2048, 1024] and report["glb"]["hair"]["material"] == material


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
    assert document["headOffsets"]["count"] > 1000 and document["parts"]["hair"]["id"] == "procedural"
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
    assert len(js["meshes"]) == 2 and len(js["skins"]) == 1 and len(js["skins"][0]["joints"]) == 53
    nodes = [n for n in js["nodes"] if "mesh" in n]
    assert [n["name"] for n in nodes] == ["twin", "dtHair"] and {n["skin"] for n in nodes} == {0}  # one shared skin
    assert js["asset"]["extras"]["dtHairNode"] == "dtHair"
    assert [m["primitives"][0]["material"] for m in js["meshes"]] == [0, 1] and js["materials"][1]["name"] == "dtHair"
    hair = json.loads((folder / "rig_report.json").read_text())["hair"]
    assert (
        hair["node"] == "dtHair" and hair["head_chain_weight_mean"] > 0.9 and hair["fraction_head_chain_ge_0_95"] > 0.9
    )
    assert twin["mapping"]["twinVertexCount"] == read_glb(str(out)).prims[0].positions.shape[0]  # the body only


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


def test_default_hair_is_procedural_cards_that_keep_off_the_skin_and_follow_the_head(run_result):
    _, out, report, _, _ = run_result
    hair = report["parts"]["hair"]
    assert hair["id"] == "procedural" and "procedural" in hair
    procedural = hair["procedural"]
    assert procedural["penetration"]["vertices_inside_head"] == 0
    assert procedural["penetration"]["vertices_below_clearance"] == 0
    assert 8000 < procedural["triangles"] <= 60000 and procedural["coverage"]["min_hidden_fraction"] > 0.8
    assert hair["skin_weights"]["fraction_head_chain_ge_0_95"] > 0.95
    assert hair["colour_method"] in ("default", "near-grey measurement -> dark brown", "measured", "lightness clamped")
    assert "hair" not in report["parts"]["slices"]  # the cards are not part of the body mesh
    assert report["glb"]["hair"]["triangles"] == procedural["triangles"]  # one-sided cards, no back copies
    assert report["texture"]["scalp_tint"]["method"] == "procedural hair density field"
    assert report["texture"]["atlas_size"][0] <= 4096 and report["texture"]["alpha"]["cutout_texels"] > 0
    scene = read_glb(str(out))
    assert len(scene.prims) == 2 and scene.extras["dtHybrid"]["hair"] == "procedural"
    assert scene.extras["dtHybrid"]["hairProcedural"] is True and verify_report(report) == []


def test_procedural_hair_takes_a_style_a_colour_and_fits_next_to_the_brow_tile(model, tmp_path):
    from hybridbody.hairgen import HairStyle

    style = HairStyle.with_overrides({"triangle_target": 8000, "hairline_front": 64.0, "length_side": 9.0})
    _, out, report, _, _ = execute(model, tmp_path, brows=True, hair_hex="#2b1d14", hair_style=style)
    hair = report["parts"]["hair"]
    assert hair["colour_hex"] == "#2b1d14" and hair["colour_method"] == "override"
    assert abs(hair["procedural"]["triangles"] / 8000 - 1) < 0.25
    assert hair["procedural"]["style"]["hairline_front"] == 64.0
    assert "eyebrows" in report["parts"]["slices"]
    document = json.loads((out.parent / "face_asset/face-asset.json").read_text())
    assert document["parts"]["hair"]["kind"] == "procedural-cards" and document["parts"]["hair"]["fallbackId"]
    assert document["parts"]["hair"]["style"]["hairline_front"] == 64.0


def test_makehuman_hair_parts_stay_selectable_next_to_the_procedural_one(model, tmp_path):
    _, _, report, _, _ = execute(model, tmp_path, hair="hair-tousled")
    hair = report["parts"]["hair"]
    assert hair["id"] == "hair-tousled" and "procedural" not in hair
    assert report["texture"]["scalp_tint"]["method"] == "ray march to the hair surface"
    assert hair["clearance_to_head"]["deeper_than_10mm"] < 0.05 and verify_report(report) == []


def test_hybrid_cli_builds_the_hair_style_and_rejects_style_flags_for_makehuman_hair():
    import argparse

    import hybrid

    parser = argparse.ArgumentParser()
    base = {
        "hair": "procedural",
        "hair_hex": None,
        "hairline_mm": None,
        "hair_top_mm": None,
        "hair_side_mm": None,
        "hair_seed": None,
        "hair_param": None,
    }
    assert hybrid.hair_style_from(argparse.Namespace(**base), parser) is None
    style = hybrid.hair_style_from(
        argparse.Namespace(
            **{
                **base,
                "hairline_mm": 70.0,
                "hair_top_mm": 60.0,
                "hair_side_mm": 10.0,
                "hair_param": ["sideburn_drop=4"],
            }
        ),
        parser,
    )
    assert style.hairline_front == 70.0 and style.length_front == 60.0 and style.length_mid == pytest.approx(54.0)
    assert style.length_side == 10.0 and style.sideburn_drop == 4.0
    with pytest.raises(SystemExit):
        hybrid.hair_style_from(argparse.Namespace(**{**base, "hair": "hair-short", "hairline_mm": 70.0}), parser)
    with pytest.raises(SystemExit):
        hybrid.hair_style_from(argparse.Namespace(**{**base, "hair_param": ["nope=1"]}), parser)


# ------------------------------------------------------------------------------------------- hy3d (the user's own hair)
@pytest.fixture(scope="module")
def hy3d_result(model, tmp_path_factory):
    """The whole stage with ``--hair hy3d`` on a synthetic bust (landmarks from a fake detector, a tiny shell)."""
    from hybridbody import hy3d
    from hybridbody.hairhy3d import ShellStyle
    from synth_bust import build_kit, fake_detector, write_bust
    from test_hairhy3d import SYNTHETIC

    kit = build_kit()
    root = tmp_path_factory.mktemp("hy3d_repo")
    write_bust(root / "user-data/twin/hy3d/hy3d.glb", kit, top=0.05)
    original = hy3d.detect_front_landmarks
    hy3d.detect_front_landmarks = fake_detector(kit)
    try:
        return execute(model, root, hair="hy3d", hair_style=ShellStyle.with_overrides(SYNTHETIC))
    finally:
        hy3d.detect_front_landmarks = original


def test_hy3d_hair_is_a_shell_node_with_its_own_material_colour_texture_and_normal_map(hy3d_result):
    import io

    from PIL import Image

    _, out, report, _, _ = hy3d_result
    hair = report["parts"]["hair"]
    assert hair["id"] == "hy3d" and "shell" in hair and "procedural" not in hair and hair["node"] == "dtHair"
    assert "hair" not in report["parts"]["slices"] and hair["colour_method"].startswith("Lab chroma")
    scene = read_glb(str(out))
    assert [p.name for p in scene.prims] == ["twin", "dtHair"] and scene.extras["dtHairNode"] == "dtHair"
    marker = scene.extras["dtHybrid"]
    assert marker["hair"] == "hy3d" and marker["hairProcedural"] is False and "hairAtlas" not in marker
    assert marker["hairShell"]["format"] == "shell/1" and marker["hairShell"]["triangles"] == len(scene.prims[1].indices)
    js, blob = _split(out.read_bytes())
    material = js["materials"][1]
    assert (material["name"], material["alphaMode"], material["alphaCutoff"], material["doubleSided"]) == (
        "dtHair",
        "MASK",
        0.5,
        True,
    )
    assert material["extras"]["dtHair"]["format"] == "shell/1" and material["extras"]["dtHair"]["cardCount"] == 0
    assert "baseColorFactor" not in material["pbrMetallicRoughness"]  # the colour comes from the texture alone
    colour = js["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    normal = js["textures"][material["normalTexture"]["index"]]
    assert len({js["textures"][0]["source"], colour["source"], normal["source"]}) == 3  # three different images
    images = [js["images"][t["source"]] for t in (colour, normal)]
    assert [i["mimeType"] for i in images] == ["image/png", "image/jpeg"]
    views = [js["bufferViews"][i["bufferView"]] for i in images]
    texture = np.asarray(Image.open(io.BytesIO(blob[views[0]["byteOffset"] : views[0]["byteOffset"] + views[0]["byteLength"]])))
    assert texture.shape == (256, 256, 4) and texture[..., 3].min() < 128 and texture[..., 3].max() == 255
    relief = np.asarray(Image.open(io.BytesIO(blob[views[1]["byteOffset"] : views[1]["byteOffset"] + views[1]["byteLength"]])))
    assert relief.shape == (256, 256, 3) and relief[..., 2].mean() > 200  # blue dominant: a tangent-space normal map
    assert report["glb"]["hair"]["material"] == material and report["glb"]["hair"]["normal_size"] == [256, 256]
    assert hair["skin_weights"]["fraction_head_chain_ge_0_95"] > 0.8


def test_hy3d_hair_report_passes_its_checks_up_to_the_synthetic_size_and_darkens_the_scalp_under_it(hy3d_result):
    _, _, report, _, _ = hy3d_result
    shell = report["parts"]["hair"]["shell"]
    assert shell["penetration"]["vertices_inside_head"] == 0 and shell["penetration"]["vertices_below_clearance"] == 0
    assert shell["visible_clearance"]["min_mm"] >= 1.5 and 40 <= shell["hairline"]["front_mm_above_eye_shell"] <= 100
    problems = verify_report(report)
    # the tiny synthetic shell has fewer triangles than a real one and leaves skin showing from some views
    assert all("scalp visible" in p or "triangle count" in p for p in problems)
    tint = report["texture"]["scalp_tint"]
    assert tint["method"].startswith("hy3d shell coverage") and tint["covered_texel_fraction"] > 0.02
    assert tint["tint_lab"][0] < 40  # the scalp takes the dark colour of the hair next to it
    assert report["texture"]["alpha"]["cutout_texels"] > 0  # the eyelash cards still cut out of the body atlas


def test_hy3d_face_asset_names_the_shell_and_the_rig_accepts_the_two_meshes(hy3d_result, monkeypatch, tmp_path):
    _, out, _, solution, _ = hy3d_result
    document = json.loads((out.parent / "face_asset/face-asset.json").read_text())
    hair = document["parts"]["hair"]
    assert hair["id"] == "hy3d" and hair["kind"] == "bust-shell" and hair["format"] == "shell/1" and hair["fallbackId"]
    assert hair["style"]["fit"]["clearance"] == 1.8 and "colorHex" in hair["colours"]
    import rig_scan

    folder = tmp_path / "rig"
    argv = ["rig_scan.py", str(out), str(folder), "--fingers", "keep", "--smooth", "0", "--samples", "20000"]
    monkeypatch.setattr(sys, "argv", argv)
    rig_scan.main()
    js, _ = _split((folder / "rigged.glb").read_bytes())
    nodes = [n for n in js["nodes"] if "mesh" in n]
    assert [n["name"] for n in nodes] == ["twin", "dtHair"] and {n["skin"] for n in nodes} == {0}
    shell_material = js["materials"][1]
    assert shell_material["extras"]["dtHair"]["format"] == "shell/1" and "normalTexture" in shell_material
    assert len(js["images"]) == 3 and [i["mimeType"] for i in js["images"]] == ["image/png", "image/png", "image/jpeg"]
    rig = json.loads((folder / "rig_report.json").read_text())["hair"]
    assert rig["node"] == "dtHair" and rig["head_chain_weight_mean"] > 0.9


def test_hair_defaults_to_the_users_bust_when_it_exists_and_to_cards_otherwise(tmp_path):
    assert pipeline.resolve_hair("hair-short") == "hair-short" and pipeline.resolve_hair("procedural") == "procedural"
    assert pipeline.resolve_hair(None, tmp_path) == "procedural"  # no bust in this folder
    bust = tmp_path / pipeline.BUST_RELATIVE
    bust.parent.mkdir(parents=True)
    bust.write_bytes(b"glTF")
    assert pipeline.resolve_hair(None, tmp_path) == "hy3d"


def test_hy3d_without_the_bust_is_an_error_and_the_cli_routes_hair_params(model, tmp_path):
    import argparse

    import hybrid
    from hybridbody.hairhy3d import ShellStyle

    with pytest.raises(ValueError, match="needs the Hunyuan3D bust"):
        execute(model, tmp_path, hair="hy3d")
    parser = argparse.ArgumentParser()
    base = {"hair": "hy3d", "hair_hex": None, "hairline_mm": None, "hair_top_mm": None, "hair_side_mm": None, "hair_seed": None}
    assert hybrid.hair_style_from(argparse.Namespace(**base, hair_param=None), parser) is None
    style = hybrid.hair_style_from(argparse.Namespace(**base, hair_param=["hair_lightness=40", "clearance=2.2"]), parser)
    assert isinstance(style, ShellStyle) and style.segment.hair_lightness == 40 and style.fit.clearance == 2.2
    with pytest.raises(SystemExit):  # the procedural style flags do not apply to hy3d
        hybrid.hair_style_from(argparse.Namespace(**{**base, "hairline_mm": 70.0}, hair_param=None), parser)
    with pytest.raises(SystemExit):
        hybrid.hair_style_from(argparse.Namespace(**base, hair_param=["nope=1"]), parser)
    with pytest.raises(SystemExit):
        hybrid.hair_style_from(argparse.Namespace(**base, hair_param=["clearance"]), parser)
