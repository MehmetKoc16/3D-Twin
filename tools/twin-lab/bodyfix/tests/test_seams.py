"""Closed synthetic surfaces with UV copies and deliberately conflicting correspondences."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rig"))

from mh import MHModel
from rigfit import FitResult
from transfer import SurfaceMap, Transfer
import transfer as transfer_module


def boundary_count(vertices, faces):
    _, inverse = np.unique(np.round(vertices * 1e5).astype(np.int64), axis=0, return_inverse=True)
    f = inverse.ravel()[faces]
    edges = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


@pytest.mark.parametrize("with_fan_inheritance", [False, True])
def test_inverse_and_forward_deformation_keep_caps_closed_and_lips_separate(monkeypatch, with_fan_inheritance):
    import scipy.sparse as sp

    model = MHModel()
    rest = model.shape({}, ground=False)
    fit = FitResult({}, {}, rest, {"upperarm_l": np.array([0., 0., -0.5])},
                    np.zeros(3), rest[:model.nr])
    tetra = np.array([[.20, 1.18, 0], [.20, 1.24, .04], [.20, 1.24, -.04], [.17, 1.24, 0]])
    # Caps on two sides of a 2 mm cut; UV seams give each face its own copies.
    other = tetra.copy()
    other[:, 0] = .402 - other[:, 0]
    face = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    vertices = np.vstack([tetra[face].reshape(-1, 3), other[face].reshape(-1, 3)])
    faces = np.arange(len(vertices)).reshape(-1, 3)
    assert boundary_count(vertices, faces) == 0
    dominant = model.skin_j[np.arange(model.nr), np.argmax(model.skin_w, axis=1)]
    arm = int(np.flatnonzero(dominant == model.bone_index["upperarm_l"])[0])
    torso = int(np.flatnonzero(dominant == model.bone_index["spine_02"])[0])
    owner = np.where(np.arange(len(vertices)) % 2, arm, torso)
    mapping = SurfaceMap(np.repeat(owner[:, None], 3, axis=1),
                         np.tile([1., 0, 0], (len(vertices), 1)), np.zeros(len(vertices)))
    monkeypatch.setattr(transfer_module, "forward_map", lambda *args: mapping)
    inherit = sp.eye(len(vertices), format="lil") if with_fan_inheritance else None
    if inherit is not None:
        # Fan UV copies inherit different edges' transforms, the second source
        # of affine disagreement. Their geometric copies must still agree.
        for i in range(0, len(vertices), 3):
            inherit[i, :] = 0
            inherit[i, (i + 1) % len(vertices)] = 1
        inherit = inherit.tocsr()
    deform = Transfer(model, fit, vertices, faces, smooth_iterations=2, inherit=inherit)
    assert boundary_count(deform.rest_scan, faces) == 0
    target = model.shape({}, {"measure/measure-upperarm-length": .3}, ground=False)
    unposed, posed = deform.deform(target)
    assert boundary_count(unposed, faces) == boundary_count(posed, faces) == 0
    # The close lips retain distinct seam groups rather than being repaired by
    # a large spatial tolerance that would join arm and torso again.
    assert not np.array_equal(deform.scan[0], deform.scan[12])
    np.testing.assert_allclose(deform.scan[12, 0] - deform.scan[0, 0], .002)
