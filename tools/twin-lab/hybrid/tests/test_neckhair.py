"""The photo bake must not carry hair or stubble behind the ear and onto the neck sides (synthetic sphere head)."""

import numpy as np
from flamehead.colour import to_lab
from hybridbody.neckhair import region_stats, restrict_photo, restriction_weights, texel_weights
from synth import sphere_template

TONE = np.array([57.8, 8.0, 10.7])


def _head():
    positions, faces, _, _ = sphere_template(radius=0.1, subdivisions=4)
    welded = np.arange(len(positions))
    positions = positions - positions.mean(0)
    x, y, z = positions.T
    face = (z > 0.03) & (y > -0.05)  # the photographed face: the front cap
    ear = (np.abs(x) > 0.085) & (np.abs(y) < 0.02) & (np.abs(z) < 0.02)  # a small patch on each side
    return positions, faces, welded, {"face": face, "ear": ear}


def test_weights_are_zero_on_face_ear_and_front_and_one_behind_ear_on_both_sides():
    positions, faces, welded, groups = _head()
    r = restriction_weights(positions, faces, welded, groups)
    w = r["weight"]
    x, y, z = positions.T
    assert np.all(w[groups["face"] & (z > 0.05)] == 0)  # beard region far from the border is untouched
    assert np.all(w[groups["ear"]] == 0)
    behind = (z < -0.04) & (y < 0.0)
    assert behind.sum() > 20 and w[behind].min() > 0.99
    # in front of the ear's middle, outside the face (the sideburn / temple), nothing changes
    front_side = ~groups["face"] & ~groups["ear"] & (z > 0.03) & (np.abs(x) > 0.05)
    assert front_side.sum() > 0 and w[front_side].max() == 0
    # mirrored left / right behave the same
    mirror = np.array([np.argmin(np.linalg.norm(positions - p * [-1, 1, 1], axis=1)) for p in positions[::7]])
    assert np.abs(w[::7] - w[mirror]).max() < 0.05
    assert r["z_ear_mid"][1] is not None and r["z_ear_mid"][-1] is not None


def test_blend_is_smooth_over_about_13_mm_into_the_beard():
    positions, faces, welded, groups = _head()
    groups = {"face": positions[:, 2] > -0.02, "ear": groups["ear"]}  # a face that reaches back to the core
    w = restriction_weights(positions, faces, welded, groups, blend_m=0.013)["weight"]
    z = positions[:, 2]
    core = ~groups["face"] & ~groups["ear"] & (z < 0.0)
    assert w[core].min() > 0.99
    border = groups["face"] & ~groups["ear"] & (z < 0.0)
    assert border.any() and 0.0 < w[border].max() < 1.0  # the beard side is partially blended, not cut
    order = np.argsort(z[border])
    ramp = w[border][order]
    assert ramp[0] >= ramp[-1] and ramp[-1] < 0.5  # the weight falls away from the core
    assert w[groups["face"] & (z > 0.0)].max() < 1e-9 or w[groups["face"] & (z > 0.0)].max() < 0.01


def test_restrict_photo_replaces_dark_strands_but_keeps_beard():
    positions, faces, welded, groups = _head()
    r = restriction_weights(positions, faces, welded, groups)
    n = 400
    rng = np.random.default_rng(3)
    ty, tx = np.divmod(np.arange(n), 20)
    texture = np.full((20, 20, 3), [190, 140, 120], np.uint8)
    dark = rng.random(n) < 0.4
    texture[ty[dark], tx[dark]] = [40, 30, 25]
    skin = np.full((n, 3), [190, 140, 120], np.uint8)
    behind = np.arange(n) < 200
    weight = np.where(behind, 1.0, 0.0)
    out = restrict_photo(texture, ty, tx, weight, skin)
    sel = behind
    before = region_stats(texture, ty, tx, sel, TONE)
    after = region_stats(out, ty, tx, sel, TONE)
    assert before["dark_texels_l_lt_40"] > 50 and after["dark_texels_l_lt_40"] == 0
    assert after["std_lab"][0] < 0.5 and abs(after["mean_lab"][0] - to_lab(skin[:1])[0, 0]) < 0.6
    keep = ~behind
    np.testing.assert_array_equal(out[ty[keep], tx[keep]], texture[ty[keep], tx[keep]])  # beard unchanged exactly
    assert to_lab(out[ty[keep], tx[keep]]).std() == to_lab(texture[ty[keep], tx[keep]]).std()
    assert r["weight"].shape == (len(positions),)


def test_texel_weights_interpolate_the_vertex_weight():
    vertex = np.array([0.0, 1.0, 0.5])
    tri = np.array([[0, 1, 2], [1, 1, 1]])
    bary = np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]])
    np.testing.assert_allclose(texel_weights(vertex, tri, bary), [0.5, 1.0])
