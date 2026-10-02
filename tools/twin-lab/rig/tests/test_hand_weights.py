"""Weights of a scan whose hands were removed (bodyfix): synthetic grids, no scan data."""

from __future__ import annotations

import numpy as np

from bridges import heal_islands, reassign_hand_weights


def _grid(n: int = 14):
    points = np.array([[x, y] for y in range(n) for x in range(n)])
    faces = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x
            faces += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return points, np.array(faces)


def test_hand_weights_of_the_cap_follow_the_forearm_it_closes(model):
    """A wrist cap weighted to the (misplaced, unconstrained) hand takes the forearm weights around it."""
    n = 14
    points, faces = _grid(n)
    bone = model.bone_index
    hand, lower = bone["hand_l"], bone["lowerarm_l"]
    weights = np.zeros((len(points), len(model.bone_names)))
    weights[:, lower] = 1.0
    cap = points[:, 0] >= n - 4  # the last four columns are the wrist cap, carrying hand weights
    weights[cap, lower], weights[cap, hand] = 0.3, 0.7
    result, count = reassign_hand_weights(weights, faces, [hand])
    assert count == int(cap.sum())
    assert result[:, hand].max() == 0
    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    np.testing.assert_allclose(result[cap, lower], 1.0)


def test_small_distant_patch_is_healed_instead_of_left_floating(model):
    """The last centimetres of a forearm weighted to the thigh bone are re-weighted from the forearm around them."""
    n = 14
    points, faces = _grid(n)
    bone = model.bone_index
    lower, thigh = bone["lowerarm_l"], bone["thigh_l"]
    weights = np.zeros((len(points), len(model.bone_names)))
    weights[:, lower] = 1.0
    patch = (points[:, 0] >= n - 4) & (points[:, 1] < 4)
    weights[patch, lower], weights[patch, thigh] = 0.0, 1.0
    result, count = heal_islands(weights, faces, model.bone_names, list(model.parent), max_faces=200)
    assert count == int(patch.sum())
    assert (result[patch, lower] > 0.9).all() and result[patch, thigh].max() < 0.1


def test_large_regions_with_distant_bones_are_left_alone(model):
    """Half a body weighted to another bone is not a mistake the island healer may 'correct'."""
    n = 14
    points, faces = _grid(n)
    bone = model.bone_index
    weights = np.zeros((len(points), len(model.bone_names)))
    half = points[:, 0] >= n // 2
    weights[~half, bone["lowerarm_l"]] = 1.0
    weights[half, bone["thigh_l"]] = 1.0
    result, count = heal_islands(weights, faces, model.bone_names, list(model.parent), max_faces=50)
    assert count == 0
    np.testing.assert_array_equal(result, weights)
