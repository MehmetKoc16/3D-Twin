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


# --- lens area gain correction and temple streak (synthetic) -------------------------------------------------------


def _lens_scene(shift=(14.0, 12.0, 6.0)):
    """Flat front patch, 0.75 mm per texel; two round lenses carrying a Lab-ish offset, with an eye patch and a brow."""
    from hybridbody.deglass_tex import protection_masks  # noqa: F401  (import check)

    h, w = 200, 300
    y, x = np.indices((h, w))
    px, py = (x - 150) * 0.00075, (110 - y) * 0.00075
    points = np.column_stack((px.ravel(), py.ravel(), np.full(h * w, 0.10)))
    rng = np.random.default_rng(3)
    skin = np.array([170, 130, 115], float)
    texture = np.clip(skin + rng.normal(0, 1.0, (h, w, 1)) + np.array([0.0, 0, 0]), 0, 255)
    texture = np.repeat(texture[..., :1], 3, axis=2) * (skin / skin[0])
    eyes = np.array([[-0.033, 0.0, 0.08], [0.033, 0.0, 0.08]])
    lens = np.zeros((h, w), bool)
    eye_patch = np.zeros((h, w), bool)
    brow = np.zeros((h, w), bool)
    for ex, ey, _ in eyes:
        lens |= ((px - ex) / 0.024) ** 2 + ((py - ey) / 0.024) ** 2 < 0.97**2
        eye_patch |= ((px - ex) / 0.010) ** 2 + ((py - ey) / 0.004) ** 2 < 1
        brow |= (np.abs(px - ex) < 0.014) & (np.abs(py - (ey + 0.018)) < 0.003)
    texture[lens] += np.array(shift)
    texture[eye_patch] = [60, 40, 40]
    texture[brow] = [50, 38, 32]
    for ex, ey, _ in eyes:
        texture[(np.abs(py - (ey + 0.0105)) < 0.0003) & (np.abs(px - ex) < 0.012)] = [60, 50, 45]
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    curves = np.vstack(
        [np.column_stack((ex + 0.024 * np.cos(t), ey + 0.024 * np.sin(t), np.full(360, 0.12))) for ex, ey, _ in eyes]
    )
    covered = np.ones((h, w), bool)
    geometry = FrameProjection(
        points, y.ravel(), x.ravel(), curves, np.empty((0, 3)), eyes, np.array([0.024, 0.024]), covered
    )
    return texture, geometry, lens, eye_patch, brow, (px, py, eyes)


def _lab_median(img, mask):
    from hybridbody.deglass_tex import _lab

    return np.median(_lab(img)[mask], axis=0)


def test_lens_offset_is_removed_and_eyes_brows_stay_untouched():
    from hybridbody.deglass_tex import correct_lens_area, protection_masks

    texture, geometry, lens, eye_patch, brow, (px, py, eyes) = _lens_scene()
    prot = protection_masks(texture, geometry)
    assert prot["eyes"][eye_patch].all() and prot["brows"][brow].any()
    wire = np.zeros(lens.shape, bool)
    for ex, ey, _ in eyes:
        wire |= (np.abs(py - (ey + 0.0105)) < 0.0003) & (np.abs(px - ex) < 0.012)
    assert wire.any() and not prot["brows"][wire & ~brow].any()  # a thin dark wire is not a brow
    out, report, weight = correct_lens_area(texture, geometry, prot, skin_mask=np.ones(lens.shape, bool), min_samples=5)
    inner = np.zeros(lens.shape, bool)
    ring = np.zeros(lens.shape, bool)
    for ex, ey, _ in eyes:  # rim-hugging bands below the eye line (the model corrects the rim step, decaying inward)
        r = np.hypot((px - ex) / 0.024, (py - ey) / 0.024)
        low = py < ey - 0.012
        inner |= (r > 0.8) & (r < 0.9) & low & ~prot["protected"]
        ring |= (r > 1.12) & (r < 1.25) & low & ~prot["protected"]
    before = _lab_median(texture, inner) - _lab_median(texture, ring)
    after = _lab_median(out, inner) - _lab_median(out, ring)
    assert abs(before[0]) > 4
    assert abs(after[0]) < 0.35 * abs(before[0]) and abs(after[1]) < 1.5 and abs(after[2]) < 1.5
    for entry in report["eyes"]:
        assert entry["applied"] and abs(entry["rim_step_after"]["L"]) < abs(entry["rim_step_before"]["L"]) + 0.5
    # eyes / lashes / brows are bit-identical, and so is everything beyond the feather outside the rim
    assert np.array_equal(out[eye_patch], texture[eye_patch])
    assert np.array_equal(out[brow], texture[brow])
    assert np.array_equal(out[prot["protected"]], texture[prot["protected"]])
    far = np.ones(lens.shape, bool)
    for ex, ey, _ in eyes:
        far &= np.hypot((px - ex) / 0.024, (py - ey) / 0.024) > 1.25
    assert np.array_equal(out[far], texture[far])
    assert weight[far].max() == 0
    # no new edge at the rim: the gradient across the outline stays small
    step_after = np.abs(np.diff(_lab_median_row(out, 129))).max()  # 14 mm below the eye line: both lens outlines, no eyes
    step_before = np.abs(np.diff(_lab_median_row(texture, 129))).max()
    assert step_before > 1.6 and step_after < 0.65 * step_before


def _lab_median_row(img, y):
    from hybridbody.deglass_tex import _lab

    return _lab(img)[y - 2 : y + 3, 30:270, 0].mean(0)


def test_lens_correction_skipped_without_skin_samples():
    from hybridbody.deglass_tex import correct_lens_area, protection_masks

    texture, geometry, *_ = _lens_scene()
    prot = protection_masks(texture, geometry)
    out, report, weight = correct_lens_area(texture, geometry, prot, skin_mask=np.zeros(texture.shape[:2], bool))
    assert not any(e["applied"] for e in report["eyes"])
    assert np.array_equal(out, texture) and weight.max() == 0


def test_temple_streak_is_tracked_off_the_nominal_line_and_flattened():
    from hybridbody.deglass_tex import remove_lens_residue

    h, w = 160, 300
    y, x = np.indices((h, w))
    py = (80 - y) * 0.0004
    points = np.column_stack((np.full(h * w, 0.085), py.ravel(), 0.14 - x.ravel() * 0.0004))
    texture = np.full((h, w, 3), [170, 130, 115], np.uint8)
    streak = np.abs(py - 0.013) < 0.0012  # the photographed arm lies 2 mm above the nominal line
    texture[streak] = np.clip(texture[streak].astype(int) + 25, 0, 255)
    arm_z = np.linspace(0.14, 0.05, 80)
    arms = np.column_stack((np.full(80, 0.080), np.full(80, 0.011), arm_z))
    geometry = FrameProjection(
        points, y.ravel(), x.ravel(), np.zeros((4, 3)), arms, np.array([[-0.03, 0, 0.1], [0.03, 0, 0.1]]),
        np.array([0.024, 0.024]), np.ones((h, w), bool),
    )
    out, report, masks = remove_lens_residue(texture, geometry)
    side = next(s for s in report["temple"]["sides"] if s["applied"])
    before, after = side["streak_contrast_before"]["ridge_dE"], side["streak_contrast_after"]["ridge_dE"]
    assert before > 8 and after < 0.3 * before
    assert abs(side["track_offset_mm_range"][0] - 2.0) < 1.0
    inner = streak.copy()
    inner[:, :12] = inner[:, -12:] = False
    assert masks["temple_band"][inner].all()
    assert not masks["temple_band"][np.abs(py - 0.013) > 0.006].any()
    assert report["protected_regions"]["eyes"]["max_abs_rgb_diff"] == 0


def test_rim_wire_is_snapped_and_removed_but_not_over_the_eye():
    from hybridbody.deglass_tex import correct_lens_stage, remove_rim_traces

    h, w = 300, 560
    y, x = np.indices((h, w))
    px, py = (x - 280) * 0.00025, (150 - y) * 0.00025
    points = np.column_stack((px.ravel(), py.ravel(), np.full(h * w, 0.10)))
    skin = np.array([170, 130, 115], np.uint8)
    texture = np.tile(skin, (h, w, 1))
    eyes = np.array([[-0.033, 0.0, 0.08], [0.033, 0.0, 0.08]])
    wire = np.zeros((h, w), bool)
    for ex, ey, _ in eyes:  # the real wire sits 1.5 mm outside the fitted 24 mm outline
        wire |= np.abs(np.hypot((px - ex) / 0.0255, (py - ey) / 0.0255) - 1) < 0.02
    texture[wire] = [70, 62, 58]
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    curves = np.vstack(
        [np.column_stack((ex + 0.024 * np.cos(t), ey + 0.024 * np.sin(t), np.full(360, 0.12))) for ex, ey, _ in eyes]
    )
    geometry = FrameProjection(
        points, y.ravel(), x.ravel(), curves, np.empty((0, 3)), eyes, np.array([0.024, 0.024]), np.ones((h, w), bool)
    )
    corrected, _, ctx = correct_lens_stage(texture, geometry, reference=texture)
    assert all(r["tracked"] for r in ctx["rims"])
    assert all(1.0 < np.median(r["offset_mm"]) < 2.0 for r in ctx["rims"])
    out, report, mask = remove_rim_traces(corrected, geometry, ctx)
    free = wire & ~ctx["prot"]["eyes"] & ~ctx["prot"]["brows"]
    assert report["applied"] and mask[free].mean() > 0.9
    assert np.abs(out[free].astype(int) - skin).max() < 12
    kept = ctx["prot"]["eyes"] | ctx["prot"]["brows"]
    assert np.array_equal(out[kept], corrected[kept])
