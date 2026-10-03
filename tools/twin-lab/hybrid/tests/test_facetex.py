import numpy as np
import pytest
from flamehead.colour import to_lab
from hybridbody.facetex import (
    bake_face,
    blend_unobserved,
    build_head_mesh,
    flame_masks_on_head,
    posed_vertices,
    uv_window,
    vertex_photo_check,
)
from hybridbody.headfit import build_template, fit_head
from hybridbody.register import Params
from synth import face_map_for, flame_like, sphere_template, write_fit_files

PARAMS = Params(iterations=12, stiffness_start=40, stiffness_end=0.5)
SIZE = 256


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    folder = tmp_path_factory.mktemp("fit")
    positions, faces, uv, neck = sphere_template(radius=0.1, subdivisions=4)
    face_map = face_map_for(positions, faces)
    template = build_template(positions, faces, face_map, neck)
    flame, _ = flame_like(positions, faces, face_map, folder=folder)
    photos = write_fit_files(folder, flame)
    fit = fit_head(template, flame, params=PARAMS)
    head = build_head_mesh(template)
    face = bake_face(template, fit, flame, head, uv, photos, SIZE)
    return template, flame, fit, head, face, uv, folder


def test_head_mesh_is_compact_and_symmetric(baked):
    template, _, _, head, _, _, _ = baked
    assert len(head.ids) == template.free[template.inverse].sum()
    assert head.faces.max() == len(head.ids) - 1 and head.faces.min() == 0
    # mirrored vertices are x-mirror images of each other on the undeformed template
    pos = template.positions[head.ids]
    mirrored = pos[head.symmetry]
    centre = template.base[:, 0].mean()
    near = np.abs(mirrored[:, 0] + pos[:, 0] - 2 * centre) < 1e-6
    assert near.mean() > 0.95
    assert (head.compact_of_render[head.ids] == np.arange(len(head.ids))).all()


def test_posed_head_matches_the_flame_pose_to_the_registration_residual(baked):
    template, flame, fit, head, _, _, folder = baked
    from flamehead.assets import read_mesh

    view, _ = read_mesh(folder / "fitted_views/front.ply")
    posed = posed_vertices(template, fit, flame, view, head)
    moved = (template.base + fit.displacement)[head.welded]
    s, r, t = fit.similarity["scale"], fit.similarity["rotation"], fit.similarity["translation"]
    native = ((moved - t) / s) @ r  # identity pose: the posed head is the deformed head in FLAME units
    assert np.linalg.norm(posed - native, axis=1).mean() < 2e-3 / s


def test_flame_masks_follow_the_correspondence(baked):
    template, flame, fit, head, _, _, _ = baked
    masks = flame_masks_on_head(template, fit, flame, head)
    assert set(masks) >= {"face", "eye_region", "lips", "neck", "left_eyeball", "right_eyeball"}
    assert len(masks["face"]) > 0
    z = (template.base + fit.displacement)[head.welded][masks["face"], 2]
    assert z.mean() > (template.base + fit.displacement)[head.welded][:, 2].mean()


def test_bake_fills_every_island_texel_and_reports_photo_colour(baked):
    _, _, _, _, face, _, _ = baked
    assert face.covered.sum() == face.report["island_texels"] > 1000
    assert face.texture.min() >= 1
    assert face.report["photo_observed_ratio"] > 0.2 and face.report["texture_fill_ratio"] == 1.0
    assert face.confidence.shape == face.texture.shape[:2]
    assert face.texel_points.shape == (face.report["island_texels"], 3)
    x0, y0, w, h = (face.report["uv_window"][k] for k in ("x", "y", "width", "height"))
    assert (x0, y0) == face.origin and x0 + w <= SIZE and y0 + h <= SIZE


def test_vertex_photo_check_detects_a_correct_uv_chain(baked):
    template, _, _, head, face, uv, _ = baked
    atlas = np.zeros((SIZE, SIZE, 3), np.uint8)
    x0, y0 = face.origin
    h, w = face.texture.shape[:2]
    atlas[y0 : y0 + h, x0 : x0 + w] = face.texture
    result = vertex_photo_check(face, head, template, uv[head.ids], atlas, SIZE)
    assert result["vertices"] > 50
    assert result["matched_mean_de"] < 0.5 * result["shuffled_mean_de"]
    # a chain with the window shifted must score worse than the correct one
    shifted = np.roll(atlas, 7, axis=1)
    worse = vertex_photo_check(face, head, template, uv[head.ids], shifted, SIZE)
    assert worse["matched_mean_de"] > result["matched_mean_de"]


def test_blend_unobserved_fades_to_the_tone_only_where_the_photos_say_little():
    texture = np.full((64, 64, 3), [200, 120, 90], np.uint8)
    covered = np.ones((64, 64), bool)
    confidence = np.zeros((64, 64), np.float32)
    confidence[:, :32] = 1.0
    tone = np.array([50.0, 5.0, 5.0])
    out, report = blend_unobserved(texture, covered, confidence, tone)
    lab = to_lab(out)
    assert np.abs(lab[:, 4] - to_lab(texture)[:, 4]).max() < 2  # confident side stays
    assert np.abs(lab[:, 60, 0] - 50).max() < 2  # unseen side takes the tone
    assert 0.3 < report["observed_fraction"] < 0.7


def test_uv_window_is_padded_and_a_multiple_of_32():
    uv = np.array([[0.30, 0.20], [0.55, 0.61]])
    x, y, w, h = uv_window(uv, 1024)
    assert x <= 0.30 * 1024 and y <= 0.20 * 1024 and w % 32 == 0 and h % 32 == 0
    assert x + w >= 0.55 * 1024 and y + h >= 0.61 * 1024
