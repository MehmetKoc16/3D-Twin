"""Synthetic texture signals only; no private photos or licensed data."""

import cv2
import numpy as np
import pytest
from scipy import ndimage

from hybridbody.overlap import compare_single_views, ghost_alignment, single_view_samples


def test_ghost_alignment_recovers_known_feature_displacement():
    rng = np.random.default_rng(54)
    reference = ndimage.gaussian_filter(rng.normal(50, 15, (192, 192)).astype(np.float32), 1)
    side = cv2.warpAffine(reference, np.array([[1, 0, 4], [0, 1, -3]], np.float32), (192, 192)) + 10
    overlap = np.ones(reference.shape, bool)
    report = ghost_alignment(reference, side, overlap)
    assert report["accepted_patches"] > 20
    assert report["median_shift_xy_px"] == [4, -3]
    assert report["median_displacement_px"] == pytest.approx(5)
    assert report["highpass_rms_after_local_alignment"] < report["highpass_rms_before"] / 10
    assert report["search_bound_patches"] == 0
    exact = ghost_alignment(reference, reference, overlap)
    assert exact["median_displacement_px"] == 0


def test_untextured_or_occluded_overlap_is_inconclusive():
    reference = np.full((100, 100), 50, np.float32)
    assert ghost_alignment(reference, reference, np.ones((100, 100), bool))["accepted_patches"] == 0
    noise = np.random.default_rng(1).normal(50, 10, (100, 100)).astype(np.float32)
    report = ghost_alignment(noise, noise, np.zeros((100, 100), bool))
    assert report["median_displacement_px"] is None
    assert report["accepted_patches"] == 0


def test_colour_diagnostics_do_not_confuse_cancelling_errors_with_agreement():
    y, x = np.indices((96, 96)).reshape(2, -1)
    n = len(y)
    front = np.tile([60., 9., 10.], (n, 1)).astype(np.float32)
    side = front.copy()
    side[:, 0] += np.tile([-5., 5.], n // 2)
    samples = {"front": front, "left": side, "right": front}
    weights = {name: np.ones(n) for name in samples}
    semantic = np.ones(n, bool)
    report = compare_single_views(samples, weights, semantic, semantic, (96, 96), y, x)
    assert report["left"]["after_delta_e76_of_mean"] == 0
    assert report["left"]["after_mean_paired_delta_e76"] == pytest.approx(5)
    assert report["right"]["after_mean_paired_delta_e76"] == 0
    common = {"left": np.zeros(n, bool), "right": np.ones(n, bool)}
    restricted = compare_single_views(samples, weights, semantic, semantic, (96, 96), y, x, support=common)
    assert restricted["left"]["overlap_texels"] == 0
    assert "after_mean_paired_delta_e76" not in restricted["left"]
    same_skin = {"left": np.arange(n) < 100, "right": np.ones(n, bool)}
    cohort = compare_single_views(samples, weights, semantic, semantic, (96, 96), y, x, skin_support=same_skin)
    assert cohort["left"]["skin_texels"] == 100


def test_independent_view_bake_handles_scalar_photo_masks():
    from flamehead.camera import Camera

    mesh = np.array([[-.1, -.1, .1], [.1, -.1, .1], [-.1, .1, .1]])
    faces = np.array([[0, 1, 2]])
    tri = np.tile(faces, (4, 1))
    bary = np.array([[.4, .3, .3], [.5, .25, .25], [.6, .2, .2], [.3, .35, .35]])
    extrinsic = np.eye(4)
    extrinsic[2, 3] = -.5
    camera = Camera(np.array([[300, 0, 128], [0, 300, 128], [0, 0, 1]]), extrinsic,
                    np.array([0, 256, 0, 256]), (256, 256), (256, 256))
    names = ("front", "left", "right")
    photos = {name: np.full((256, 256, 3), [150, 110, 95], np.uint8) for name in names}
    masks = {"face": np.arange(3)}
    photo_masks = {name: np.ones((256, 256), np.float32) for name in names}
    photo_masks["left"][:] = 0
    samples, weights, skin, face = single_view_samples(
        dict.fromkeys(names, mesh), faces, tri, bary, dict.fromkeys(names, camera), photos,
        masks, photo_masks, np.ones(4),
    )
    assert samples["front"].shape == (4, 3)
    assert weights["front"].shape == (4,)
    assert (weights["front"] > .08).all()
    assert (weights["left"] == 0).all()
    assert skin.all() and face.all()


def test_projection_skin_check_separates_skin_samples_from_hair_and_ignores_the_oval_mask_for_ears():
    from hybridbody.overlap import projection_skin_check

    n = 1000
    skin, hair = np.array([65., 12., 18.], np.float32), np.array([20., 2., 4.], np.float32)
    ear = np.arange(n) < 400
    colours = np.tile(skin, (n, 1))
    colours[(np.arange(n) % 4 == 0) & ear] = hair
    samples = {"left": colours, "right": np.tile(skin, (n, 1))}
    unmasked = {"left": np.ones(n), "right": np.ones(n)}
    masked = {"left": np.zeros(n), "right": np.ones(n)}  # the photo oval excludes the whole left face and ear
    report = projection_skin_check(samples, masked, unmasked, {"left": ear, "right": ear}, np.ones(n, bool))
    assert report["left"]["ear"]["texels"] == 400
    assert report["left"]["ear"]["skin_fraction"] == pytest.approx(.75)
    assert report["left"]["ear"]["dark_fraction"] == pytest.approx(.25)
    assert report["left"]["face"] == {"texels": 0, "skin_fraction": None, "dark_fraction": None}
    assert report["right"]["face"]["skin_fraction"] == 1 and report["right"]["ear"]["dark_fraction"] == 0
