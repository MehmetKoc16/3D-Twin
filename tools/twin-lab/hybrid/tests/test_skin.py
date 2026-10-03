import numpy as np
from flamehead.colour import to_lab
from hybridbody.skin import (
    Underwear,
    match_mean,
    pad_texture,
    paint_body,
    rasterize_bands,
    seam_blend,
    skin_lab,
    underwear_lab,
    underwear_mask,
)
from synth import sphere_template

TONE = np.array([57.8, 8.0, 10.66])


def test_skin_lab_is_the_tone_plus_fine_grain_only():
    points = np.random.default_rng(1).uniform(-0.5, 0.5, (5000, 3))
    lab = skin_lab(points, TONE)
    assert abs(lab[:, 0].mean() - TONE[0]) < 0.4 and lab[:, 0].std() < 1.5
    np.testing.assert_allclose(lab[:, 1].mean(), TONE[1], atol=0.15)
    np.testing.assert_allclose(skin_lab(points, TONE), lab)  # deterministic


def test_underwear_mask_is_a_crisp_horizontal_band_gated_by_bones():
    spec = Underwear(bottom_y=0.7, top_y=1.0)
    points = np.array([[0, y, 0] for y in (0.6, 0.69, 0.8, 0.99, 1.1)], float)
    gate = np.ones(5)
    mask = underwear_mask(points, gate, spec)
    assert mask[0] == 0 and mask[2] == 1 and mask[4] == 0 and 0 <= mask[1] < 1 and mask[3] > 0.5
    assert (underwear_mask(points, np.zeros(5), spec) == 0).all()  # arms and hands are never covered
    cloth = underwear_lab(points, mask, spec)
    assert cloth[2, 0] < 45 and cloth[3, 0] < cloth[2, 0]  # waistband darker than the cotton


def test_paint_body_covers_every_uv_texel_with_skin_and_underwear():
    positions, faces, uv, _ = sphere_template(radius=0.3, subdivisions=3, centre_y=0.9)
    size = 128
    canvas = np.zeros((size, size, 3), np.uint8)
    covered = np.zeros((size, size), bool)
    gate = np.ones(len(positions))
    report = paint_body(canvas, covered, positions, faces, uv, gate, TONE, Underwear(0.85, 0.95), size)
    assert covered.sum() > 3000 and canvas[covered].min() >= 1
    lab = to_lab(canvas[covered])
    assert report["underwear_texels"] > 50
    skin = lab[:, 1] > 4  # skin is warm; the slate underwear is not
    assert np.abs(lab[skin].mean(0) - TONE).max() < 2.5
    assert (canvas[~covered] == 0).all()


def test_pad_texture_bleeds_outward_and_fills_the_rest_with_the_mean():
    canvas = np.zeros((64, 64, 3), np.uint8)
    covered = np.zeros((64, 64), bool)
    covered[20:40, 20:40] = True
    canvas[covered] = [200, 100, 50]
    out = pad_texture(canvas, covered, pixels=5)
    assert (out[18, 30] == [200, 100, 50]).all() and (out[2, 2] == [200, 100, 50]).all()
    np.testing.assert_array_equal(out[covered], canvas[covered])


def test_match_mean_shifts_skin_samples_to_the_target_and_protects_black():
    texture = np.full((32, 32, 3), [190, 130, 110], np.uint8)
    texture[:4] = 3  # black (hair): must stay black
    covered = np.ones((32, 32), bool)
    samples = np.zeros((32, 32), bool)
    samples[10:30] = True
    current = to_lab(texture)[samples].mean(0)
    target = current + [2.0, -1.0, 1.0]
    out, report = match_mean(texture, covered, samples, target)
    assert np.abs(to_lab(out)[samples].mean(0) - target).max() < 0.6  # 8-bit quantisation
    assert out[:4].max() <= 4 and report["shift"][0] > 1.5
    limited, _ = match_mean(texture, covered, samples, current + [20, 0, 0], limit=3.0)
    assert abs(to_lab(limited)[samples][:, 0].mean() - current[0]) <= 3.3


def test_seam_blend_reaches_the_tone_at_the_seam_and_keeps_far_texels():
    texture = np.full((16, 16, 3), [220, 100, 70], np.uint8)
    seam = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[0, 0, 0.0], [0.009, 0, 0], [0.05, 0, 0]])
    ty, tx = np.array([1, 2, 3]), np.array([1, 2, 3])
    out, report = seam_blend(texture, points, ty, tx, seam, TONE, radius=0.018)
    lab = to_lab(out)
    assert np.abs(lab[1, 1] - TONE).max() < 1.0
    assert (out[3, 3] == [220, 100, 70]).all() and 0 < lab[2, 2, 0] and report["blended_texels"] == 2


def test_rasterize_bands_cover_the_same_texels_as_one_pass():
    positions, faces, uv, _ = sphere_template(subdivisions=2)
    size = 96
    banded = np.zeros((size, size), bool)
    for r0, y, x, _face, bary in rasterize_bands(uv, faces, size, rows=40):
        banded[r0 + y, x] = True
        np.testing.assert_allclose(bary.sum(1), 1, atol=1e-5)
    from twintex.raster import rasterize_uv

    fid, _ = rasterize_uv(uv * size, faces, size, size)
    np.testing.assert_array_equal(banded, fid >= 0)


def test_fade_to_tone_and_scalp_tint_only_touch_listed_texels():
    from hybridbody.skin import fade_to_tone, hair_cover, tint_scalp

    texture = np.full((8, 8, 3), 40, np.uint8)
    ys, xs = np.array([1, 2]), np.array([1, 2])
    out = fade_to_tone(texture, ys, xs, np.array([1.0, 0.0]), (70.0, 5.0, 8.0))
    assert out[1, 1].mean() > 100 and (out[2, 2] == texture[2, 2]).all() and (out[5, 5] == texture[5, 5]).all()
    dark = tint_scalp(np.full((8, 8, 3), 200, np.uint8), ys, xs, np.array([1.0, 1.0]), (10.0, 0.0, 0.0), blur_px=0.1)
    assert dark[1, 1].mean() < 60 and dark[6, 6].mean() == 200
    hair = np.array([[0.0, 0.02, 0.0]])
    cover = hair_cover(np.zeros((2, 3)), np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]), hair)
    assert cover[0] > 0.9 and cover[1] < 0.1  # hair above the texel, not below it
