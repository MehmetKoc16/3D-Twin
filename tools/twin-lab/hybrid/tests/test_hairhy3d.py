"""The hy3d hair end to end on a synthetic bust (built from the generic CC0 head; landmarks come from a fake detector)."""

import numpy as np
import pytest
from synth_bust import build_kit, fake_detector, write_bust

from hybridbody import hairhy3d, hy3d
from hybridbody.hair import FORMAT_SHELL, HairMesh
from hybridbody.hairgen import HeadSurface
from hybridbody.hairhy3d import ShellField, ShellStyle, build_hy3d_hair, edge_report, visible_samples

SYNTHETIC = {
    # the synthetic head has about 5 mm vertex spacing, the real bust 1 mm
    "smooth_rings": 1,
    "open_rings": 1,
    "close_rings": 1,
    "hole_vertices": 40,
    "smooth_iterations": 3,
    "ear_margin": 6,
    "min_vertices": 60,
    "target_triangles": 1500,
    "texture_size": 256,
    "padding": 2,
    "sample_spacing": 1.5,
    "margin": 12,
    "fringe": 3,
}


@pytest.fixture(scope="module")
def kit():
    return build_kit()


@pytest.fixture(scope="module")
def build(kit, tmp_path_factory):
    path = write_bust(tmp_path_factory.mktemp("bust") / "bust.glb", kit)
    original = hy3d.detect_front_landmarks
    hy3d.detect_front_landmarks = fake_detector(kit)
    try:
        return build_hy3d_hair(
            path,
            kit.positions,
            kit.faces,
            kit.eye_y,
            kit.landmark_points,
            kit.landmark_index,
            style=ShellStyle.with_overrides(SYNTHETIC),
            coverage_pixel_mm=2.0,
        )
    finally:
        hy3d.detect_front_landmarks = original


def test_shell_style_routes_overrides_to_their_group_and_rejects_unknown_ones():
    style = ShellStyle.with_overrides(
        {"hair_lightness": 40, "target_triangles": 20000, "clearance": 2.2, "open_rings": 2}
    )
    assert style.segment.hair_lightness == 40 and style.segment.open_rings == 2
    assert style.shell.target_triangles == 20000 and style.fit.clearance == 2.2
    assert ShellStyle.with_overrides(None) == ShellStyle() and ShellStyle().to_dict()["fit"]["clearance"] == 1.8
    with pytest.raises(ValueError, match="Unknown hy3d hair parameter"):
        ShellStyle.with_overrides({"colour": 1})
    with pytest.raises(ValueError, match="non-negative"):
        ShellStyle.with_overrides({"clearance": -2})
    assert set(ShellStyle().to_dict()) == {"segment", "shell", "fit"}


def test_the_build_is_a_shell_1_hair_mesh_with_colour_and_normal_textures(build):
    mesh = build.mesh
    assert isinstance(mesh, HairMesh) and mesh.format == FORMAT_SHELL and mesh.node == "dtHair" and mesh.card_count == 0
    assert mesh.atlas.shape == (256, 256, 4) and mesh.normal_atlas.shape == (256, 256, 3)
    assert set(mesh.colours) == {"colorHex"} and mesh.colours["colorHex"].startswith("#")
    r, g, b = (int(mesh.colours["colorHex"][i : i + 2], 16) for i in (1, 3, 5))
    assert max(r, g, b) < 90  # the synthetic hair is dark
    assert abs(len(mesh.faces) / 1500 - 1) < 0.25 and build.report["triangles"] == len(mesh.faces)
    assert mesh.uv.min() >= 0 and mesh.uv.max() <= 1 and np.isfinite(mesh.positions).all()
    assert abs(np.linalg.norm(mesh.normals, axis=1) - 1).max() < 1e-6
    alpha = mesh.atlas[..., 3]
    assert alpha.max() == 255 and alpha.min() == 0 and ((alpha > 0) & (alpha < 255)).any()


def test_the_shell_clears_the_whole_head_and_its_edge_follows_the_scalp(kit, build):
    report = build.report
    penetration = report["penetration"]
    assert penetration["vertices_inside_head"] == 0 and penetration["vertices_below_clearance"] == 0
    assert penetration["samples_inside_head"] == 0 and penetration["min_mm"] >= 1.5
    assert report["visible_clearance"]["min_mm"] >= 1.5
    surface = HeadSurface(kit.positions, kit.faces)
    signed, *_ = surface.signed_distance(build.mesh.positions, candidates=24)
    assert signed.min() > 0.0015
    assert report["fit"]["clearance"]["vertices_inside_head"] == 0
    assert report["edge_gap"]["vertices"] > 0 and report["edge_gap"]["median_mm"] < 6.0
    assert report["hairline"]["front_mm_above_eye_bust"] > 30 and report["hairline"]["procedural_target_mm"] == 68.0


def test_the_build_reports_its_transform_segmentation_shell_and_coverage(build):
    report = build.report
    assert report["format"] == FORMAT_SHELL and report["kind"] == "hy3d-shell"
    assert report["source"]["file"] == "bust.glb" and report["transform"]["scale_to_metres"] == pytest.approx(
        0.588, rel=0.03
    )
    assert report["segmentation"]["vertices"] > 60 and report["fill"]["unreached_vertices"] >= 0
    assert report["shell"]["method"].startswith("Lambert") and report["shell"]["bake"]["texels_visible"] > 1000
    assert report["atlas"]["colour_size"] == [256, 256] and report["atlas"]["normal_size"] == [256, 256]
    coverage = report["coverage"]
    assert set(coverage) >= {"front", "left", "right", "back", "top", "min_hidden_fraction"}
    assert coverage["top"]["hidden_fraction"] > 0.8  # the scalp under the volume is hidden from above
    assert (
        report["style"]["shell"]["texture_size"] == 256
        and report["colours"]["colorHex"] == build.mesh.colours["colorHex"]
    )


def test_the_field_covers_the_scalp_under_the_shell_and_leaves_the_face_alone(kit, build):
    field = build.field
    assert isinstance(field, ShellField)
    head = HeadSurface(kit.positions, kit.faces)
    top = np.array([[kit.frame.x0, kit.positions[:, 1].max() - 0.002, kit.frame.zc]])
    cover = field.cover(top)
    assert cover[0] > 0.9  # the vertex of the skull is under the volume
    cheek = (
        kit.landmark_points[list(kit.landmark_index).index(234)][None, :]
        if 234 in kit.landmark_index
        else kit.positions[:1]
    )
    assert field.cover(cheek)[0] < 0.1
    normals = head.closest(top)[3]
    assert np.allclose(field.cover(top, normals), cover, atol=0.3) and field.weight(top)[0] == pytest.approx(
        cover[0], abs=0.3
    )
    tint = field.tint_lab(top)
    assert tint.shape == (1, 3) and tint[0, 0] < 30  # dark, including the lighter synthetic fade at the hairline
    assert field.tint_lab(np.vstack((top, top))).shape == (2, 3)


def test_visible_samples_keep_only_the_texels_that_survive_the_alpha_cut():
    positions = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], float)
    faces = np.array([[0, 1, 2]])
    uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]])
    colour = np.zeros((8, 8, 4), np.uint8)
    colour[..., :3] = (10, 20, 30)
    colour[..., 3] = 255
    colour[:, :4, 3] = 0  # the left half is cut out
    points, colours = visible_samples(positions, faces, uv, colour, 0.002)
    assert len(points) > 0 and (colours == (10, 20, 30)).all()
    full, _ = visible_samples(
        positions, faces, uv, np.concatenate((colour[..., :3], np.full((8, 8, 1), 255, np.uint8)), 2), 0.002
    )
    assert len(full) > len(points)


def test_edge_report_measures_the_visible_edge_ring_only():
    depth = np.array([-0.003, 0.0, 0.0008, 0.0014, 0.002, 0.05])
    distance = np.array([0.001, 0.002, 0.003, 0.006, 0.02, 0.03])
    report = edge_report(depth, distance)
    assert (
        report["vertices"] == 3
        and report["median_mm"] == pytest.approx(3.0)
        and report["share_above_4mm"] == pytest.approx(1 / 3)
    )
    assert edge_report(np.array([0.05]), np.array([0.01]))["vertices"] == 0


def test_a_failed_segmentation_is_an_error_not_an_empty_shell(kit, tmp_path, monkeypatch):
    path = write_bust(tmp_path / "bust.glb", kit)
    monkeypatch.setattr(hy3d, "detect_front_landmarks", fake_detector(kit))
    strict = ShellStyle.with_overrides({**SYNTHETIC, "min_vertices": 100000})
    with pytest.raises(ValueError, match="almost nothing"):
        hairhy3d.build_hy3d_hair(
            path, kit.positions, kit.faces, kit.eye_y, kit.landmark_points, kit.landmark_index, style=strict
        )
