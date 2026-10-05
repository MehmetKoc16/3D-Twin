"""Loading and registering the Hunyuan3D bust: only a synthetic bust built from the generic CC0 head is used."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from synth_bust import SCALE, build_kit, bust_geometry, fake_detector, to_bust_frame, write_bust

from hybridbody import hy3d
from hybridbody.hairgen import HeadSurface


@pytest.fixture(scope="module")
def kit():
    return build_kit()


@pytest.fixture(scope="module")
def bust_file(tmp_path_factory, kit):
    return write_bust(tmp_path_factory.mktemp("bust") / "bust.glb", kit)


def test_the_24_axis_rotations_are_proper_and_the_identity_comes_first():
    rotations = hy3d.axis_rotations()
    assert len(rotations) == 24 and np.allclose(rotations[0], np.eye(3))
    assert all(np.isclose(np.linalg.det(r), 1.0) and np.allclose(r @ r.T, np.eye(3)) for r in rotations)
    assert len({tuple(np.rint(r).astype(int).ravel()) for r in rotations}) == 24


def test_weld_vertices_merges_seam_copies_and_drops_collapsed_faces():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    normals = np.tile([0.0, 0.0, 1.0], (6, 1))
    colours = np.array([[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0], [50, 0, 0], [60, 0, 0]], np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5], [1, 3, 3]])
    p, n, c, f, kept = hy3d.weld_vertices(positions, normals, colours, faces)
    assert len(p) == 4 and len(f) == 2 and kept.tolist() == [0, 1]  # the degenerate face is gone
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    shared = np.flatnonzero((p == [1, 0, 0]).all(1))[0]
    assert c[shared, 0] == pytest.approx(30.0)  # seam copies average their colour (20 and 40)


def test_load_bust_bakes_the_node_rotation_welds_the_seams_and_samples_the_colours(bust_file, kit):
    bust = hy3d.load_bust(bust_file)
    assert bust.info["welded_vertices"] < bust.info["raw_vertices"] == len(kit.positions)  # seam copies merged
    # the raw file is Z-up; after the node rotation the bust is +Y up and faces +Z: the head is taller than deep
    extent = np.ptp(bust.positions, axis=0)
    assert extent[1] > extent[2]
    expected = to_bust_frame(bust_geometry(kit)[0], kit)
    assert bust.positions[:, 1].mean() == pytest.approx(expected[:, 1].mean(), abs=0.02)
    assert bust.positions[:, 2].max() == pytest.approx(expected[:, 2].max(), abs=1e-4)  # the nose is still at the front
    assert bust.info["has_normal_map"] and bust.normal_texture is not None and bust.base_texture.shape == (256, 256, 3)
    assert bust.face_uv.shape == (len(bust.faces), 3, 2) and 0 <= bust.face_uv.min() and bust.face_uv.max() <= 1
    dark = bust.lab[:, 0] < 30
    assert 0.04 < dark.mean() < 0.8  # hair and skin are both there
    assert np.allclose(np.linalg.norm(bust.normals, axis=1), 1.0, atol=1e-6)


def test_trimmed_similarity_recovers_scale_rotation_and_translation_despite_outliers():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(60, 3)) * 0.1
    rotation = Rotation.from_euler("xyz", [4, -7, 9], degrees=True).as_matrix()
    target = 0.6 * source @ rotation.T + [0.1, 1.6, -0.2]
    target[:6] += rng.normal(scale=0.02, size=(6, 3))  # glasses frames, a smile: gross outliers
    sim = hy3d.trimmed_similarity(source, target)
    assert sim["scale"] == pytest.approx(0.6, rel=0.01)
    assert np.allclose(sim["rotation"], rotation, atol=0.01) and np.allclose(
        sim["translation"], [0.1, 1.6, -0.2], atol=2e-3
    )
    assert not sim["inliers"][:6].all() and sim["inliers"][6:].mean() > 0.9
    moved = hy3d.apply_similarity(source[10:], sim)
    assert np.abs(moved - target[10:]).max() < 2e-3
    with pytest.raises(ValueError, match="implausible"):
        hy3d.trimmed_similarity(source, source * 9.0, band=(0.2, 3.0))


def test_find_orientation_returns_the_identity_for_an_upright_bust_and_the_turn_back_for_a_turned_one(
    bust_file, kit, monkeypatch
):
    bust = hy3d.load_bust(bust_file)
    monkeypatch.setattr(hy3d, "detect_front_landmarks", fake_detector(kit, upright_only=True))
    assert np.allclose(hy3d.find_orientation(bust), np.eye(3))
    turn = Rotation.from_euler("x", 90, degrees=True).as_matrix()  # the face looks up: a Z-up file nobody rotated
    found = hy3d.find_orientation(hy3d.rotated(bust, turn))
    assert np.allclose(found @ turn, np.eye(3), atol=1e-9)
    monkeypatch.setattr(hy3d, "detect_front_landmarks", lambda *a, **k: None)
    with pytest.raises(ValueError, match="No axis-aligned orientation"):
        hy3d.find_orientation(bust)


def test_small_rigid_with_a_ridge_moves_along_the_normals_and_hardly_turns():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(500, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= 0.1  # a sphere of radius 10 cm
    normals = points / 0.1
    shift = np.array([0.002, -0.001, 0.003])
    target = points + shift
    rotation, translation = hy3d.small_rigid(points, target, normals, np.ones(500), rotation_ridge=2.0)
    assert np.allclose(translation, shift, atol=2e-4) and np.degrees(np.arccos((np.trace(rotation) - 1) / 2)) < 0.3
    free_rotation, _ = hy3d.small_rigid(points + 0.01 * np.cross([0, 0, 1], points), points, normals, np.ones(500))
    assert np.allclose(free_rotation @ free_rotation.T, np.eye(3), atol=1e-9)


def test_refine_on_skin_pulls_a_displaced_patch_back_onto_the_surface(kit):
    surface = HeadSurface(kit.positions, kit.faces)
    face = (kit.positions[:, 2] > kit.frame.zc + 0.05) & (np.abs(kit.positions[:, 0]) < 0.06)
    ids = np.flatnonzero(face)
    assert len(ids) > 100
    moved = kit.positions.copy()
    moved[:] += [0.001, -0.002, 0.004]  # the whole bust is a few millimetres off
    result = hy3d.refine_on_skin(moved, ids, np.zeros(0, np.int64), surface, iterations=15)
    refined = moved[ids] @ result["R"].T + result["t"]
    signed, *_ = surface.signed_distance(refined)
    assert np.abs(signed).mean() < 0.0006 and result["report"]["rotation_deg"] < 1.0
    assert result["report"]["skin_signed_distance_mm"]["p95_abs"] < 1.5


def test_align_bust_recovers_the_metric_scale_and_puts_the_skin_on_the_twin_head(bust_file, kit, monkeypatch):
    monkeypatch.setattr(hy3d, "detect_front_landmarks", fake_detector(kit))
    bust = hy3d.load_bust(bust_file)
    surface = HeadSurface(kit.positions, kit.faces)
    alignment = hy3d.align_bust(bust, kit.landmark_points, kit.landmark_index, surface, kit.frame)
    assert alignment.scale == pytest.approx(1.0 / SCALE, rel=0.02)
    assert alignment.report["landmark_residual_mm"]["median"] < 3.0 and alignment.report["landmarks_used"] >= 20
    skin = bust.lab[:, 0] > 55
    signed, *_ = surface.signed_distance(alignment.positions[skin & (alignment.positions[:, 1] > kit.eye_y - 0.05)])
    assert np.abs(np.median(signed)) < 0.0015  # the face of the bust lies on the twin face
    assert np.allclose(np.linalg.norm(alignment.normals, axis=1), 1.0, atol=1e-6)
    # the total transform reproduces the aligned positions: p_twin = scale * (R @ p_bust) + t
    again = alignment.scale * bust.positions @ alignment.rotation.T + alignment.translation
    assert np.abs(again - alignment.positions).max() < 1e-6
    assert alignment.report["orientation_rotation"] == np.eye(3, dtype=int).tolist()
    assert "skipped" in alignment.report["icp"]  # the synthetic hair leaves too little forehead for the ICP


def test_align_bust_runs_the_icp_when_the_forehead_is_visible(tmp_path, kit, monkeypatch):
    path = write_bust(tmp_path / "high.glb", kit, top=0.075)
    monkeypatch.setattr(hy3d, "detect_front_landmarks", fake_detector(kit))
    bust = hy3d.load_bust(path)
    surface = HeadSurface(kit.positions, kit.faces)
    alignment = hy3d.align_bust(bust, kit.landmark_points, kit.landmark_index, surface, kit.frame)
    icp = alignment.report["icp"]
    assert "skipped" not in icp and icp["skin_points"] >= 60 and icp["rotation_deg"] < 1.5
    assert icp["skin_signed_distance_mm"]["p95_abs"] < 2.5


def test_align_bust_refuses_a_bust_without_a_detectable_face(bust_file, kit, monkeypatch):
    monkeypatch.setattr(hy3d, "detect_front_landmarks", lambda *a, **k: None)
    bust = hy3d.load_bust(bust_file)
    with pytest.raises(ValueError, match="No axis-aligned orientation"):
        hy3d.align_bust(bust, kit.landmark_points, kit.landmark_index, HeadSurface(kit.positions, kit.faces), kit.frame)


def test_load_bust_rejects_meshes_without_a_colour_texture(tmp_path):
    from glbio import GlbScene, Prim, write_static_glb

    prim = Prim(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), np.array([[0, 1, 2]], np.uint32))
    write_static_glb(str(tmp_path / "plain.glb"), GlbScene([prim]))
    with pytest.raises(ValueError, match="UVs"):
        hy3d.load_bust(tmp_path / "plain.glb")
