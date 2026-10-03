"""New surgery and colour regressions, using generated surfaces and pixels only."""

import numpy as np
import pytest
import trimesh
from flamehead.colour import from_lab, paired_colour, recolour_and_crossfade, to_lab
from flamehead.geometry import edge_table
from flamehead.seam import cut_level, expanded_region, nearest_curve, smooth_ring, smoothstep, transplant
from flamehead.texture import bake, pack_atlases, sample
from twinrefine.meshops import Corners


def plane(bottom, top, count=17):
    x, y = np.meshgrid(np.linspace(-0.05, 0.05, count), np.linspace(bottom, top, count))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    ids = np.arange(count * count).reshape(count, count)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    faces = np.vstack(
        (np.column_stack((a.ravel(), b.ravel(), c.ravel())), np.column_stack((b.ravel(), d.ravel(), c.ravel())))
    )
    uv = np.column_stack(((x.ravel() + 0.05) / 0.1, (y.ravel() - bottom) / (top - bottom)))
    return Corners(points, faces, uv[faces])


@pytest.mark.parametrize("positive", [True, False])
def test_level_cut_exact_curve_and_uv_interpolation(positive):
    mesh = plane(-0.1, 0.1)
    result, report = cut_level(mesh, mesh.P[:, 1] - 0.013, positive)
    new = result.P[len(mesh.P) :]
    assert report["inserted_cut_vertices"] > 0 and report["split_triangles"] > 0
    np.testing.assert_allclose(new[:, 1], 0.013, atol=1e-12)
    expected = np.stack(((result.P[:, 0] + 0.05) / 0.1, (result.P[:, 1] + 0.1) / 0.2), axis=1)
    np.testing.assert_allclose(result.C, expected[result.F], atol=1e-12)
    signed = result.P[result.F].mean(1)[:, 1] - 0.013
    assert (signed > 0).all() if positive else (signed < 0).all()
    assert (edge_table(result.F)[1] <= 2).all()


def test_smooth_curve_reduces_sawtooth_and_nearest_segment():
    theta = np.arange(64) * 2 * np.pi / 64
    radius = 0.1 + np.where(np.arange(64) % 2, 0.004, -0.004)
    curve = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros(64)))
    smoothed = smooth_ring(curve, np.arange(64))
    assert np.std(np.linalg.norm(smoothed, axis=1)) < np.std(radius) / 4
    middle = (smoothed[4] + smoothed[5]) / 2
    np.testing.assert_allclose(nearest_curve(middle[None], smoothed), middle[None], atol=1e-12)


def test_split_surgery_closed_manifold_and_geometry_band():
    scan = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    flame = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    scan_mc = Corners(scan.vertices, scan.faces, np.zeros((len(scan.faces), 3, 2)))
    flame_mc = Corners(flame.vertices, flame.faces, np.zeros((len(flame.faces), 3, 2)))
    result = transplant(scan_mc, flame_mc, flame.vertices[:, 2] - 0.019)
    faces = np.vstack((result.scan.F, result.flame.F + len(result.scan.P), result.bridge))
    vertices = np.vstack((result.scan.P, result.flame.P))
    assert (edge_table(faces)[1] == 2).all()
    welded = trimesh.Trimesh(vertices, faces, process=False)
    assert welded.is_watertight and welded.is_winding_consistent
    assert result.report["stitch_gap_max_mm"] == 0
    assert result.report["split_scan_triangles"] > 0
    assert result.report["split_flame_triangles"] > 0
    assert result.report["geometry_band_mm"] == 22
    # Opposite pole and the body side of the scan retain their exact positions.
    distant = scan.vertices[:, 2] < -0.03
    np.testing.assert_array_equal(result.scan.P[: len(scan.vertices)][distant], scan.vertices[distant])


def test_lab_roundtrip_preserves_chroma_and_diagnostics_reject_bias():
    lab = np.tile([62, 13, 19], (32, 1)).astype(np.float32)
    rgb = from_lab(lab)
    np.testing.assert_allclose(to_lab(rgb), lab, atol=0.25)
    shade = lab.copy()
    shade[:, 0] -= 10
    np.testing.assert_allclose(to_lab(from_lab(shade))[:, 1:], lab[:, 1:], atol=0.25)
    assert paired_colour(rgb, from_lab(shade), np.ones(32, bool))["chroma_pass"]
    shade[:, 1] -= 8
    assert not paired_colour(rgb, from_lab(shade), np.ones(32, bool))["chroma_pass"]


def test_bake_matches_luminance_without_rgb_white_balance_changes():
    from flamehead.camera import Camera
    from flamehead.geometry import symmetry_map

    surface = plane(-0.1, 0.1)
    uv = np.stack(((surface.P[:, 0] + 0.05) / 0.1, (surface.P[:, 1] + 0.1) / 0.2), axis=1)
    camera = Camera(
        np.array([[100, 0, 64], [0, 100, 64], [0, 0, 1.0]]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.5], [0, 0, 0, 1.0]]),
        np.array([0, 128, 0, 128.0]),
        (128, 128),
        (128, 128),
    )
    front = np.full((128, 128, 3), [180, 125, 95], np.uint8)
    darker = to_lab(front)
    darker[:, :, 0] -= 8
    right = np.rint(from_lab(darker) * 255).astype(np.uint8)
    masks = {"face": np.arange(len(surface.P)), "left_eyeball": np.array([], int), "right_eyeball": np.array([], int)}
    symmetry, _ = symmetry_map(surface.P)
    texture, report = bake(
        surface.P,
        surface.F,
        np.arange(len(surface.P)),
        surface.F,
        uv,
        {"front": surface.P, "right": surface.P},
        {"front": camera, "right": camera},
        {"front": front, "right": right},
        symmetry,
        masks,
        128,
    )
    assert report["texture_fill_ratio"] == 1 and texture.min() > 0
    assert report["gains"]["right"]["luminance_offset_lab"] == pytest.approx(8, abs=0.5)
    assert not report["gains"]["right"]["chroma_modified"]
    # Allow the accumulated RGB8 / OpenCV Lab round-trip quantisation.
    assert np.max(np.abs(report["photo_colour"]["delta_lab"][1:])) < 1


def test_photo_referenced_neck_falloff_and_beard_crossfade():
    scan, flame = plane(-0.10, 0), plane(0, 0.08)
    scan_tex = np.full((256, 256, 3), [195, 145, 135], np.uint8)
    # A dark synthetic beard, deliberately distinct from the pink scan neck.
    flame_tex = np.full((256, 256, 3), [50, 35, 25], np.uint8)
    scan_ring = np.flatnonzero(scan.P[:, 1] == 0)
    flame_ring = np.flatnonzero(flame.P[:, 1] == 0)
    target = to_lab(np.array([[175, 125, 95]], np.uint8))[0]
    corrected, blended, report = recolour_and_crossfade(scan, flame, scan_tex, flame_tex, scan_ring, flame_ring, target)
    assert report["neck_chroma_pass"] and report["texture_band_mm"] == 22
    np.testing.assert_allclose(report["neck_mean_lab"], target, atol=0.5)
    np.testing.assert_array_equal(corrected[0], scan_tex[0])  # Beyond the 75mm skin falloff.
    np.testing.assert_array_equal(blended[128], flame_tex[128])  # Inner beard remains photographed.
    outer = to_lab(blended[0, 128][None])[0]
    np.testing.assert_allclose(outer, target, atol=1.0)
    luminance = to_lab(blended[:80, 128])[:, 0]
    assert np.max(np.abs(np.diff(luminance))) < 4  # No abrupt beard/neck texture edge.
    assert smoothstep(0) == 0 and smoothstep(1) == 1


@pytest.mark.parametrize("face_size,expected_face,expected_scan", [(2048, 2048, 2032), (4096, 3072, 1008)])
def test_packing_caps_canvas_and_prioritises_face(face_size, expected_face, expected_scan):
    scan = np.full((4096, 4096, 3), [90, 100, 110], np.uint8)
    face = np.full((face_size, face_size, 3), [170, 125, 100], np.uint8)
    atlas, suv, fuv = pack_atlases(scan, face, np.array([[0, 0], [1, 1.0]]), np.array([[0, 0], [1, 1.0]]))
    assert atlas.shape == (4096, 4096, 3)
    assert suv[1, 0] * 4096 == expected_scan
    assert (fuv[1, 0] - fuv[0, 0]) * 4096 == expected_face
    np.testing.assert_array_equal(sample(atlas, np.mean(fuv, axis=0)[None] * 4096)[0], face[0, 0])


def test_cli_default_ears_and_optout(monkeypatch):
    import flame_head

    calls = []
    monkeypatch.setattr(flame_head, "run", lambda *args, **kwargs: calls.append(kwargs) or {})
    argv = [
        "--in",
        "synthetic.glb",
        "--fit",
        "fit",
        "--photos",
        "photos",
        "--flame-assets",
        "assets",
        "--out",
        "head.glb",
    ]
    assert flame_head.main(argv) == 0 and calls[-1]["include_ears"]
    assert flame_head.main(argv + ["--no-include-ears"]) == 0 and not calls[-1]["include_ears"]


def test_region_expands_into_neck_forehead_and_ear_collar():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.1)
    v = mesh.vertices
    masks = {
        key: np.array([], int)
        for key in (
            "face",
            "nose",
            "lips",
            "eye_region",
            "left_eye_region",
            "right_eye_region",
            "left_eyeball",
            "right_eyeball",
            "scalp",
            "neck",
            "boundary",
            "left_ear",
            "right_ear",
        )
    }
    masks.update(
        face=np.flatnonzero((v[:, 2] > 0.035) & (v[:, 1] > -0.04)),
        forehead=np.flatnonzero((v[:, 1] > 0.045) & (v[:, 2] > 0.035)),
        neck=np.flatnonzero(v[:, 1] < -0.04),
        boundary=np.flatnonzero(v[:, 1] < -0.09),
        left_ear=np.flatnonzero((v[:, 0] < -0.09) & (np.abs(v[:, 1]) < 0.015)),
        right_ear=np.flatnonzero((v[:, 0] > 0.09) & (np.abs(v[:, 1]) < 0.015)),
    )
    field, report = expanded_region(v, mesh.faces, masks)
    assert (field[masks["left_ear"]] > 0).all() and (field[masks["right_ear"]] > 0).all()
    assert (field[masks["boundary"]] < 0).all()
    upper_neck = (v[:, 1] < -0.045) & (v[:, 1] > -0.065) & (v[:, 2] > 0.065)
    assert (field[upper_neck] > 0).all()
    assert report["ear_collar_mm"] == 25 and report["forehead_margin_mm"] == 12
    without_ears, _ = expanded_region(v, mesh.faces, masks, ears=False)
    assert (without_ears[masks["left_ear"]] < 0).all()
    assert np.count_nonzero(field > 0) > np.count_nonzero(without_ears > 0)
