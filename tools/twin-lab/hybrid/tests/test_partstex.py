import numpy as np
import pytest
from flamehead.colour import to_lab
from hybridbody import PARTS_ASSETS
from hybridbody.partstex import (
    Tile,
    card_tile,
    covered_linear_mean,
    eye_tile,
    make_tile,
    pack_strip,
    plausible_iris,
    recolor_iris,
    robust_colour,
    sample_ring,
    srgb_hex,
    triangle_coverage,
)
from hybridbody.template import load_part
from test_template import synthetic_part


def grey_eye(size=64):
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(xx + 0.5 - size / 2, yy + 0.5 - size / 2) / size
    image = np.full((size, size, 4), 235, np.uint8)  # white sclera
    image[d < 0.2] = [120, 120, 120, 255]  # grey iris
    image[d < 0.07] = [10, 10, 10, 255]  # pupil
    image[..., 3] = 255
    return image


def test_recolor_iris_tints_only_the_iris_and_keeps_pupil_and_sclera():
    eye = grey_eye()
    out = recolor_iris(eye, (0.5, 0.5), 0.2, (80, 50, 30))
    assert (out[32, 32, :3] <= 40).all()  # pupil stays dark
    assert (out[2, 2] == eye[2, 2]).all()  # sclera untouched
    ring = out[32, 32 + 9]  # inside the iris ring
    assert ring[0] > ring[2] and ring[0] > 60  # brownish
    assert out.dtype == np.uint8 and (out[..., 3] == 255).all()


def test_card_tile_keeps_the_real_alpha_and_bleeds_colour_into_cut_outs():
    part = synthetic_part()
    texture = np.zeros((8, 8, 4), np.uint8)
    texture[..., :3] = 200
    texture[:, :4, 3] = 255  # left half covered, right half empty
    part.texture = texture
    part.alpha_mode = "MASK"
    part.meta = {"material": {"alphaCutoff": 0.5}}
    tile = card_tile(part, np.array([0.05, 0.03, 0.02]))
    assert tile.shape == (8, 8, 4)
    assert (tile[:, :4, 3] == 255).all() and (tile[:, 4:, 3] == 0).all()
    assert tile[0, 0, :3].sum() < 400  # tinted dark
    assert (tile[0, 7, :3] == tile[0, 3, :3]).all()  # transparent texels carry the neighbouring strand colour
    assert 0 < covered_linear_mean(texture, 0.5) <= 1
    part.alpha_mode = "OPAQUE"
    assert (card_tile(part, np.array([0.05, 0.03, 0.02]))[..., 3] == 255).all()


def test_plausible_iris_turns_a_near_grey_measurement_into_dark_brown():
    grey = plausible_iris([0x52, 0x4C, 0x49])
    rgb = grey["srgb"]
    assert rgb[0] > rgb[1] > rgb[2] and rgb[0] - rgb[2] > 25 and rgb.max() < 140
    assert grey["method"].startswith("near-grey") and grey["measured_hex"] == "#524c49"
    blue = plausible_iris([60, 90, 140])  # chromatic: only the lightness is clamped
    assert blue["srgb"][2] > blue["srgb"][0]
    assert plausible_iris([10, 10, 10])["srgb"].max() > 25  # very dark measurement is lifted to the lightness floor
    assert srgb_hex(plausible_iris(None, "#3b2a1e")["srgb"]) == "#3b2a1e"
    assert plausible_iris(None)["method"] == "default"
    with pytest.raises(ValueError):
        plausible_iris(None, "#12")


def test_triangle_coverage_counts_strand_samples_per_triangle():
    part = synthetic_part()
    texture = np.zeros((16, 16, 4), np.uint8)
    part.texture = texture
    part.meta = {"material": {"alphaCutoff": 0.5}}
    assert triangle_coverage(part)[0] == 0
    texture[..., 3] = 255
    assert triangle_coverage(part)[0] == 1
    assert 0.0 <= triangle_coverage(part, 0.9)[0] <= 1.0


def test_tiles_remap_uv_into_the_strip_below_the_body_atlas():
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    uv = np.array([[0.25, 0.25], [0.75, 0.75]])
    tile = make_tile("t", image, uv, width=16, max_height=64)
    assert tile.image.shape[1] == 16 and tile.uv_min.min() >= 0.2 and tile.uv_max.max() <= 0.8
    other = make_tile("o", image, uv, width=16, max_height=64)
    strip = pack_strip([tile, other], 64, 32, y_offset=64)
    assert strip.shape == (32, 64, 3) and tile.origin == (0, 64) and other.origin[0] == 20
    mapped = tile.remap(uv, 64, 96)
    # the part's UV box lands inside the strip rows (64..96) and the tile's columns
    assert (mapped[:, 1] * 96 >= 64 - 1e-6).all() and (mapped[:, 1] * 96 <= 96).all()
    assert (mapped[:, 0] * 64 >= -1e-6).all() and (mapped[:, 0] * 64 <= 17).all()
    with pytest.raises(ValueError, match="do not fit"):
        pack_strip([Tile("w", np.zeros((8, 70, 3), np.uint8), np.zeros(2), np.ones(2))], 64, 32)


def test_ring_sampling_and_robust_colour_reject_glints_and_black():
    photo = np.full((60, 60, 3), [90, 60, 40], np.uint8)
    photo[30, 30] = 255
    ring = sample_ring(photo, (30, 30), 10, 0.4, 0.85)
    assert len(ring) > 100
    colour, count = robust_colour(np.vstack((ring, np.zeros((50, 3)), np.full((50, 3), 255.0))))
    np.testing.assert_allclose(colour, [90, 60, 40])
    assert count >= len(ring)
    assert robust_colour(np.zeros((5, 3)))[0] is None
    assert srgb_hex([255, 0, 128.4]) == "#ff0080"


def test_real_hair_tiles_are_rgba_cut_outs_and_eye_tiles_opaque():
    hair = load_part(PARTS_ASSETS, "hair-short")
    tile = card_tile(hair, np.array([0.05, 0.034, 0.025]))
    assert tile.shape[2] == 4 and tile.shape[:2] == hair.texture.shape[:2]
    assert (tile[..., 3] == hair.texture[..., 3]).all() and (tile[..., 3] < 128).mean() > 0.2
    lab = to_lab(tile[..., :3])
    assert 8 < lab[..., 0].mean() < 40  # dark brown overall
    solid = load_part(PARTS_ASSETS, "hair-tousled")
    assert (card_tile(solid, np.array([0.05, 0.034, 0.025]))[..., 3] == 255).all()
    eyes = load_part(PARTS_ASSETS, "eyes-default")
    iris = eyes.meta["irisUv"]
    brown = eye_tile(eyes, np.array([90.0, 60.0, 40.0]))
    assert brown.shape[2] == 4 and (brown[..., 3] == 255).all()
    h, w = brown.shape[:2]
    px = brown[int(iris["center"][1] * h), int((iris["center"][0] + iris["radius"] * 0.7) * w)]
    assert px[0] > px[2]
