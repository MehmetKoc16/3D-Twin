"""Analytic skin gradients and frame strokes; no personal images."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hybridbody.deglass_tex import FrameProjection, projection_from_report, remove_glasses_frames


@pytest.mark.parametrize("method", ["telea", "ns"])
def test_masked_line_inpainting_restores_gradient_and_preserves_alpha(method):
    x = np.linspace(135, 180, 128)
    skin = np.tile(np.stack((x, x * 0.8, x * 0.7), axis=1).astype(np.uint8)[None], (100, 1, 1))
    texture = np.dstack((skin.copy(), np.full((100, 128), 197, np.uint8)))
    mask = np.zeros(texture.shape[:2], np.uint8)
    cv2.line(mask, (25, 35), (105, 55), 255, 2)
    texture[mask > 0, :3] = [55, 53, 50]
    clean, report, final_mask = remove_glasses_frames(texture, mask, method=method, return_report=True)
    assert np.mean(np.abs(clean[..., :3][mask > 0].astype(float) - skin[mask > 0])) < 8
    assert np.array_equal(clean[..., 3], texture[..., 3])
    assert np.array_equal(clean[~final_mask], texture[~final_mask])
    assert report["mask_texels"] > np.count_nonzero(mask)
    assert not report["outside_mask_changed"]


def test_geometry_maps_seams_and_protects_eyes_and_back_of_head():
    # Two UV patches with identical XY: one front, one back, plus eye pixels.
    y, x = np.indices((80, 160))
    px = (x % 80 - 40) * 0.001
    py = (y - 40) * 0.001
    z = np.where(x < 80, 0.10, -0.10)
    points = np.column_stack((px.ravel(), py.ravel(), z.ravel()))
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    curve = np.column_stack((0.025 * np.cos(t), 0.025 * np.sin(t), np.full(len(t), 0.12)))
    texture = np.full((80, 160, 3), [165, 135, 120], np.uint8)
    eyes = np.array([[0, 0, 0.08]])
    projection = FrameProjection(
        points, y.ravel(), x.ravel(), curve, np.empty((0, 3)), eyes, np.array([0.025, 0.025]), np.ones((80, 160), bool)
    )
    _, report, mask = remove_glasses_frames(texture, projection, return_report=True)
    assert mask[:, :80].any()
    assert not mask[:, 80:].any()
    assert not mask[36:44, 32:48].any()
    assert report["projected_core_texels"] > 0


def test_empty_mask_is_identity_copy():
    image = np.full((40, 40, 3), 170, np.uint8)
    clean = remove_glasses_frames(image, np.zeros((40, 40), bool))
    assert np.array_equal(image, clean)
    assert not np.shares_memory(image, clean)


def test_shaded_upper_features_are_kept_when_indistinguishable_from_brow_hairs():
    y, x = np.indices((100, 100))
    points = np.column_stack(((x.ravel()-50)*.001, (y.ravel()-30)*.001, np.full(10000, .1)))
    curve = np.column_stack((np.linspace(-.035, .035, 100), np.full(100, .01), np.full(100, .12)))
    texture = np.full((100, 100, 3), [165, 135, 120], np.uint8)
    texture[40, 20:80] = 85  # shaded silver rim above the eye line
    texture[42, 20:80] = 40  # nearby dark brow hairs
    geometry = FrameProjection(points, y.ravel(), x.ravel(), curve, np.empty((0, 3)),
                               np.array([[0., 0., .08]]), np.array([.025, .025]))
    clean, report, mask = remove_glasses_frames(texture, geometry, method="ns", return_report=True)
    assert not mask[40, 25:75].any()
    assert not mask[42, 20:80].any()
    assert np.array_equal(clean[42, 20:80], texture[42, 20:80])
    assert not report["outside_mask_changed"]


def test_report_hook_uses_original_curves_and_preserves_upper_brows():
    points = np.array([[0.02, 0, 0.1], [0.02, 0.030, 0.1]])
    face = SimpleNamespace(
        texel_points=points, texel_y=np.array([0, 1]), texel_x=np.array([0, 1]), covered=np.ones((2, 2), bool)
    )
    report = {
        "placement": {"eyes_m": [[-0.032, 0, 0.08], [0.032, 0, 0.08]], "rim_radii_mm": [24, 23]},
        "removal_curves_m": {"front": [[0.02, 0, 0.12], [0.02, 0.030, 0.12]], "temples": []},
    }
    projection = projection_from_report(face, report)
    assert projection.protected[1, 1]
    assert np.allclose(projection.radii, [0.024, 0.023])
    assert np.array_equal(projection.front_curves[:, :2], points[:, :2])


@pytest.mark.parametrize("kwargs", [{"radius": 0}, {"method": "guess"}, {"dilate_texels": -1}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        remove_glasses_frames(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), bool), **kwargs)


def test_invalid_mask_and_texture():
    with pytest.raises(ValueError, match="mask"):
        remove_glasses_frames(np.zeros((10, 10, 3), np.uint8), np.zeros((8, 10), bool))
    with pytest.raises(ValueError, match="uint8"):
        remove_glasses_frames(np.zeros((10, 10, 3), float), np.zeros((10, 10), bool))
