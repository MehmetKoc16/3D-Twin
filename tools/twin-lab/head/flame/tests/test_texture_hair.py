import numpy as np
import pytest
from flamehead.colour import from_lab, to_lab
from flamehead.texture import (
    ear_fallback,
    hair_like_samples,
    hair_vertex_regions,
    mirrored_ear_colour,
    skin_colour_model,
    temple_dark_samples,
)
from twintex.colorspace import linear_to_srgb, srgb_to_linear


def test_chroma_guard_keeps_shadow_skin_and_rejects_moderately_lit_hair():
    lab = np.tile([60., 12., 18.], (64, 1))
    lab[:, 0] = np.linspace(45, 75, 64)
    model = skin_colour_model(lab, np.ones(64, bool), np.ones(64))
    samples = np.array([[22., 12., 18.], [30., 9., 13.5], [52., 2., 3.], [55., -5., 15.],
                        [65., 14., 20.], [20., 4., 6.], [8., .5, .5],
                        [68., 7.5, 10.],  # desaturated specular highlight on lit skin
                        [14., 10., 15.]])  # far darker than lit skin: black hair with a faint warm cast
    assert hair_like_samples(samples, model).tolist() == [
        False, False, True, True, False, False, True, False, True]
    # Each photo has its own white balance; the same skin under warmer lighting is accepted.
    warmer = lab + [0, 5, 10]
    warm_model = skin_colour_model(warmer, np.ones(64, bool), np.ones(64))
    assert not hair_like_samples(warmer, warm_model).any()
    assert skin_colour_model(lab[:16], np.ones(16, bool), np.ones(16)) is None
    assert skin_colour_model(lab, np.ones(64, bool), np.full(64, .035)) is None
    assert skin_colour_model(lab, np.ones(64, bool), np.full(64, .001)) is None
    assert not hair_like_samples(samples, None).any()


def test_temple_rule_only_rejects_samples_far_darker_than_lit_skin():
    model = {"median_lab": [65., 12., 18.], "mad_ab": [2.5, 3.]}
    lab = np.array([[60., 12., 18.], [50., 12., 18.], [42., 12., 18.], [30., 12., 18.]])
    assert temple_dark_samples(lab, model).tolist() == [False, False, True, True]
    assert not temple_dark_samples(lab, None).any()


def test_mirrored_ear_is_smoothed_and_takes_this_ears_own_skin_tone():
    xs, ys = np.meshgrid(np.arange(20) * .001, np.arange(15) * .001)
    grid = np.c_[xs.ravel(), ys.ravel(), np.zeros(300)]
    left, right = grid + [.08, 0, 0], grid * [-1, 1, 1] - [.08, 0, 0]
    points, mirrored_points = np.vstack((right, left)), np.vstack((left, right))
    lab = np.zeros((600, 3))
    lab[300:] = [60., 14., 10.]  # the clean opposite ear: pinker, fine checkerboard grain
    lab[300:, 0] += 8 * (((xs.ravel() / .001).astype(int) + (ys.ravel() / .001).astype(int)) % 2 * 2 - 1)
    lab[:200] = [58., 9., 16.]  # this ear's surviving texels
    lab[200:300] = [20., 0., 0.]  # rejected hair
    raw = srgb_to_linear(np.clip(from_lab(lab), 0, 1))
    is_right, observed = np.arange(600) < 300, np.arange(600) < 200
    observed[300:] = True
    want, matched, colour = mirrored_ear_colour(
        points, mirrored_points, raw, is_right & observed, ~is_right & observed, is_right & ~observed,
    )
    assert want.tolist() == list(range(200, 300)) and matched.all()
    filled = to_lab(linear_to_srgb(colour))
    np.testing.assert_allclose(filled[:, 1:].mean(0), [9., 16.], atol=1.)
    assert filled[:, 0].std() < 4  # the source grain has a standard deviation of 8
    # No usable partner within 8 mm: flagged unmatched, so the caller keeps its fallback.
    far = mirrored_points.copy()
    far[200:300] += [.5, 0, 0]
    assert not mirrored_ear_colour(points, far, raw, is_right & observed, ~is_right & observed,
                                   is_right & ~observed)[1].any()


def test_native_flame_subregions_protect_jaw_chin_upper_lip_and_eyes():
    vertices = np.array([[0, -.07, .05], [.04, -.04, .04], [0, -.025, .06], [0, -.01, .07],
                         [.03, .005, .05], [0, .02, .05], [0, .055, .05], [.07, .04, .02], [.063, .005, .01]])
    masks = {"face": np.arange(9), "nose": [3], "lips": [2], "eye_region": [5], "forehead": [6, 7]}
    regions = hair_vertex_regions(vertices, masks)
    assert set(regions["beard_allow"]) >= {0, 1, 2}
    assert set(regions["skin_reference"]) == {4, 6, 8}
    assert 5 not in regions["hair_skin_region"]
    assert 7 in regions["temple"]


def test_ear_fallback_prefers_other_views_then_opposite_ear_without_crossing_regions():
    points = np.array([[-.08, 0, 0], [.08, 0, 0], [0, 0, .05]])
    mirrored_points = points * [-1, 1, 1]
    raw = srgb_to_linear(from_lab(np.array([[60., 12., 18.], [65., 14., 20.], [10., 0., 0.]])))
    ears = {"right": np.array([True, False, False]), "left": np.array([False, True, False])}
    labs = [to_lab(from_lab(np.array([[48., 2., 3.], [65., 14., 20.], [10., 0., 0.]])))]
    models = {"front": {"median_lab": [60., 12., 18.], "mad_ab": [2.5, 3.], "texels": 64}}
    observed = np.array([True, True, True])  # another view sees the rejected right ear
    out, mirror, tone = ear_fallback(points, mirrored_points, raw, observed, ears, labs, [observed], models)
    np.testing.assert_array_equal(out, raw)
    assert not mirror.any() and not tone.any()
    observed[0] = False
    out, mirror, tone = ear_fallback(points, mirrored_points, raw, observed, ears, labs, [observed], models)
    np.testing.assert_allclose(out[0], raw[1], atol=5e-3)  # Lab round trip of the mirror partner
    assert mirror.tolist() == [True, False, False] and not tone.any()
    observed[1] = False  # neither ear is usable, despite a visible dark face sample
    out, mirror, tone = ear_fallback(points, mirrored_points, raw, observed, ears, labs, [observed], models)
    assert tone.tolist() == [True, True, False] and not mirror.any()
    np.testing.assert_allclose(to_lab(linear_to_srgb(out[:1])), [[60., 12., 18.]], atol=.2)
    # An ear that no photo ever saw keeps the legacy surface fill: only rejected evidence is repaired.
    lost = np.array([True, False, False])
    out, mirror, tone = ear_fallback(points, mirrored_points, raw, observed, ears, labs, [observed], models, lost=lost)
    assert tone.tolist() == [True, False, False] and not mirror.any()
    np.testing.assert_array_equal(out[1:], raw[1:])


def test_ear_rim_triangles_are_mirrored_but_never_take_the_ear_tone():
    points = np.array([[-.08, 0, 0], [.08, 0, 0], [0, 0, .05], [-.08, .002, 0], [-.07, .03, 0]])
    raw = srgb_to_linear(from_lab(np.array([[10., 0., 0.], [60., 12., 18.], [65., 14., 20.], [10., 0., 0.],
                                            [10., 0., 0.]])))
    ears = {"right": np.array([True, False, False, False, False]), "left": np.array([False, True, False, False, False])}
    rims = {"right": np.array([False, False, False, True, True])}
    observed = np.array([False, True, True, False, False])
    labs = [to_lab(linear_to_srgb(raw))]
    models = {"front": {"median_lab": [60., 12., 18.], "mad_ab": [2.5, 3.], "texels": 64}}
    out, mirror, tone = ear_fallback(points, points * [-1, 1, 1], raw, observed, ears, labs, [observed], models,
                                     rims=rims)
    # The rim texel beside the opposite ear's partner is mirrored; the far one has no partner and keeps its fill.
    assert mirror.tolist() == [True, False, False, True, False] and not tone.any()
    np.testing.assert_allclose(out[3], raw[1], atol=5e-3)
    np.testing.assert_array_equal(out[4], raw[4])


def test_last_ear_fallback_preserves_bounded_detail_and_uses_ear_tone_before_face_tone():
    points = np.c_[np.full(32, -.08), np.linspace(-.02, .02, 32), np.zeros(32)]
    lab = np.tile([45., 2., 3.], (32, 1))
    lab[:, 0] += np.sin(np.arange(32) * 1.8) * 2
    # The distant ear tip is valid skin and supplies the region's local tone.
    lab[-1] = [62., 16., 20.]
    observed = np.arange(32) == 31
    ears = {"right": np.ones(32, bool), "left": np.zeros(32, bool)}
    models = {"front": {"median_lab": [55., 8., 10.], "mad_ab": [2.5, 3.], "texels": 64}}
    raw = srgb_to_linear(from_lab(lab))
    out, mirror, toned = ear_fallback(points, points * [-1, 1, 1], raw, observed, ears, [lab], [observed], models)
    baked = to_lab(linear_to_srgb(out[:-1]))
    assert toned.sum() == 31 and not mirror.any()
    np.testing.assert_allclose(baked[:, 1:].mean(0), [16., 20.], atol=.2)
    assert .5 < baked[:, 0].std() < 3
    assert np.max(np.abs(baked[:, 0] - 62)) < 3.2
    np.testing.assert_array_equal(out[-1], raw[-1])


NAMES = ("front", "left", "right")


def synthetic_scene(monkeypatch, centres, labs, weights, masks):
    """Tiny head: one patch (40 texels per patch row) per centre, projected as flat per-patch photo colours."""
    import flamehead.texture as baker

    count = len(centres)
    neutral = (centres[:, None, :] + np.array([[-.001, 0, 0], [.001, 0, 0], [0, .001, 0]])).reshape(-1, 3)
    faces = np.arange(3 * count).reshape(-1, 3)
    fid = np.repeat(np.arange(count), 8)[:, None] * np.ones((1, 40), int)
    y, x = np.nonzero(fid >= 0)
    tri, lam = faces[fid[y, x]], np.full((len(y), 3), 1 / 3)
    photos = {n: np.rint(from_lab(labs[i]) * 255).astype(np.uint8) for i, n in enumerate(NAMES)}

    def project(points, normals, photo, *args):
        i = next(i for i, n in enumerate(NAMES) if photos[n] is photo)
        region = np.linalg.norm(points[:, None] - centres, axis=2).argmin(1)
        return photo[region], weights[i, region]

    monkeypatch.setattr(baker, "visibility", lambda *args: (None, None, None))
    monkeypatch.setattr(baker, "project_samples", project)
    masks = {"left_eyeball": np.array([], int), "right_eyeball": np.array([], int), **masks(faces)}
    symmetry = np.arange(3 * count)
    symmetry[faces[1]], symmetry[faces[2]] = faces[2], faces[1]
    args = (neutral, faces, fid, y, x, tri, lam, dict.fromkeys(NAMES, neutral), dict.fromkeys(NAMES, None),
            photos, symmetry, masks)
    return baker, args, fid


@pytest.mark.parametrize("case", ["other_view", "mirrored", "unphotographed"])
def test_baker_rejects_ear_hair_and_preserves_beard_statistics(monkeypatch, case):
    centres = np.array([[0, .04, .06], [.08, 0, 0], [-.08, 0, 0], [0, -.06, .05], [.06, .04, .02]])
    skin, beard, hair = [60., 12., 18.], [25., 1., 2.], [50., 2., 3.]
    labs = np.array([[skin, skin, skin, beard, skin], [skin, skin, skin, beard, skin],
                     [skin, skin, hair, beard, hair]])
    weights = np.ones((3, 5), np.float32)
    if case != "other_view":
        weights[:2, 2] = 0  # no other view sees the right ear
    if case == "unphotographed":
        weights[2, 2] = 0  # and neither does the side photo: nothing was rejected, nothing is repaired

    def masks(faces):
        return {"face": np.r_[faces[0], faces[3], faces[4]], "left_ear": faces[1], "right_ear": faces[2],
                "skin_reference": faces[0], "beard_allow": faces[3], "hair_skin_region": faces[[0, 1, 2, 4]].ravel()}

    baker, args, fid = synthetic_scene(monkeypatch, centres, labs, weights, masks)
    before, _ = baker.bake_texels(*args, view_names=NAMES)
    diagnostics = {}
    after, report = baker.bake_texels(*args, view_names=NAMES, reject_photo_hair=True, diagnostics=diagnostics)
    assert report["photo_hair_rejected_texels"]["right"] == (320 if case == "unphotographed" else 640)
    ear = fid == 2
    fallback = diagnostics["hair"]["ear_fallback"]
    if case == "unphotographed":
        assert not any(fallback.values())
        assert abs(to_lab(after[ear])[:, 0].mean() - 60) < 6
    else:
        assert to_lab(after[ear])[:, 0].mean() > 58
        assert to_lab(after[ear])[:, 1:].mean() > 12
        assert fallback["other_views" if case == "other_view" else "mirrored"] == 320
    np.testing.assert_allclose(to_lab(before[fid == 3]).mean(0), to_lab(after[fid == 3]).mean(0), atol=.3)
    np.testing.assert_allclose(to_lab(before[fid == 3]).std(0), to_lab(after[fid == 3]).std(0), atol=.3)


@pytest.mark.parametrize("limit", [0.12, 0.6])
def test_an_ear_is_repaired_only_when_enough_of_it_is_hair_like(monkeypatch, limit):
    import flamehead.texture as baker

    monkeypatch.setattr(baker, "EAR_MIN_SEEN", 100)
    monkeypatch.setattr(baker, "EAR_HAIR_SHARE", limit)
    # The left ear is half skin (patch 1), half hair-like in the left photo only (patch 5): a 50% share.
    centres = np.array([[0, .04, .06], [.08, 0, 0], [-.08, 0, 0], [0, -.06, .05], [.06, .04, .02], [.085, 0, 0]])
    skin, beard, hair = [60., 12., 18.], [25., 1., 2.], [50., 2., 3.]
    labs = np.array([[skin, skin, skin, beard, skin, skin], [skin, skin, skin, beard, skin, hair],
                     [skin, skin, skin, beard, skin, skin]])

    def masks(faces):
        return {"face": np.r_[faces[0], faces[3], faces[4]], "left_ear": faces[[1, 5]].ravel(),
                "right_ear": faces[2], "skin_reference": faces[0], "beard_allow": faces[3],
                "hair_skin_region": faces[[0, 1, 2, 4, 5]].ravel()}

    baker, args, _ = synthetic_scene(monkeypatch, centres, labs, np.ones((3, 6), np.float32), masks)
    diagnostics = {}
    _, report = baker.bake_texels(*args, view_names=NAMES, reject_photo_hair=True, diagnostics=diagnostics)
    shares = diagnostics["hair"]["ear_hair_share"]
    assert shares["left"] == pytest.approx(0.5) and shares["right"] == 0
    assert report["photo_hair_rejected_texels"]["left"] == (320 if limit < 0.5 else 0)


def test_rejection_grows_into_the_anti_aliased_edge_on_ears_only(monkeypatch):
    # Patch 5 (skin-hued hair edge, 0.5 mm from the rejected ear patch 2) is dropped with it; patch 6 (2 cm away) and
    # patch 7, the same edge colour beside rejected hair on the plain face (patch 4), are not.
    centres = np.array([[0, .04, .06], [.08, 0, 0], [-.08, 0, 0], [0, -.06, .05], [.06, .04, .02], [-.0805, 0, 0],
                        [-.06, 0, 0], [.0605, .04, .02]])
    skin, beard, hair, edge = [60., 12., 18.], [25., 1., 2.], [50., 2., 3.], [45., 10., 15.]
    row = [skin, skin, skin, beard, skin, skin, skin, skin]
    labs = np.array([row, row, [skin, skin, hair, beard, hair, edge, skin, edge]])

    def masks(faces):
        return {"face": np.r_[faces[0], faces[3], faces[4], faces[7]], "left_ear": faces[1],
                "right_ear": faces[[2, 5, 6]].ravel(), "skin_reference": faces[0], "beard_allow": faces[3],
                "hair_skin_region": faces[[0, 1, 2, 4, 5, 6, 7]].ravel()}

    baker, args, _ = synthetic_scene(monkeypatch, centres, labs, np.ones((3, 8), np.float32), masks)
    _, report = baker.bake_texels(*args, view_names=NAMES, reject_photo_hair=True)
    assert report["photo_hair_rejected_texels"]["right"] == 3 * 320  # patches 2, 4 and the grown edge 5


@pytest.mark.parametrize("photo_hair", [False, True])
def test_forehead_hair_rejection_still_applies_with_photo_hair_rejection(monkeypatch, photo_hair):
    # Patch 5 is forehead above the eyes: the hairline-dark front sample must never reach the bake.
    centres = np.array([[0, .04, .06], [.08, 0, 0], [-.08, 0, 0], [0, -.06, .05], [.06, .04, .02], [0, .1, .06],
                        [0, 0, .06]])
    skin, beard, hairline = [60., 12., 18.], [25., 1., 2.], [18., 10., 14.]
    row = [skin, skin, skin, beard, skin, hairline, skin]
    labs = np.array([row, row[:5] + [skin, skin], row[:5] + [skin, skin]])
    weights = np.ones((3, 7), np.float32)
    weights[1:, 5] = 0  # only the front photo sees the forehead

    def masks(faces):
        return {"face": np.r_[faces[0], faces[3], faces[4], faces[5]], "left_ear": faces[1], "right_ear": faces[2],
                "eye_region": faces[6], "skin_reference": faces[0], "beard_allow": faces[3],
                "hair_skin_region": faces[[0, 1, 2, 4, 5]].ravel()}

    baker, args, fid = synthetic_scene(monkeypatch, centres, labs, weights, masks)
    texture, report = baker.bake_texels(*args, view_names=NAMES, reject_forehead_hair=True,
                                        reject_photo_hair=photo_hair)
    assert report["forehead_photo_hair_rejected_texels"]["front"] == 320
    assert to_lab(texture[fid == 5])[:, 0].mean() > 55
