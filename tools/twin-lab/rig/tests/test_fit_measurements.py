"""Regression: the fitted body's exported measurements match the ground truth of the non-personal stand-in.

Bug it guards: the point-to-nearest-point fit cannot see height, so the macro/modifier solve drifted to a body ~9 cm
shorter than the scan (the user's 178 cm scan became a 169 cm body). rigfit.py now ties the body height to the scan.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import trimesh

from rigfit import HAIR_M, Fitter
from twin_export import canonicalize_fit, measure_body


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "big-head"])
def fitted(request, model, tmp_path_factory):
    big_head = request.param
    import make_standin

    out = tmp_path_factory.mktemp("standin")
    old = sys.argv
    sys.argv = ["make_standin.py", str(out)]
    try:
        make_standin.main()
    finally:
        sys.argv = old
    truth = json.load(open(out / "truth.json"))
    mesh = trimesh.load(str(out / "mesh.glb"), force="mesh")
    if big_head:  # image-to-3D scans often have an oversized head / voluminous hair: stretch the top 22 cm by 35 %
        v = np.array(mesh.vertices)
        base = v[:, 1].max() - 0.22
        up = v[:, 1] > base
        v[up, 1] = base + (v[up, 1] - base) * 1.35
        mesh = trimesh.Trimesh(v, mesh.faces, process=False)
    pts, fid = trimesh.sample.sample_surface(mesh, 40000, seed=3)
    res = Fitter(model, pts, mesh.face_normals[fid]).run()
    canonicalize_fit(model, res)
    return truth, mesh, res, big_head


def test_fitted_measurements_match_ground_truth(model, fitted):
    truth, mesh, res, big_head = fitted
    gt = measure_body(model, model.shape(truth["macro"], truth["mods"]))
    got = measure_body(model, res.rest_positions)
    scan_h_cm = (mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min()) * 100
    # the fitted body is as tall as the scan minus the hair allowance (the stand-in has no hair)
    if not big_head:
        assert abs(got["height"] - gt["height"]) < 1.5, (got["height"], gt["height"])
    assert got["height"] > scan_h_cm - HAIR_M * 100 - 1.5
    for k in ("chest", "waist", "hip", "thigh", "inseam", "armLength"):
        assert abs(got[k] - gt[k]) < 3.5, (k, got[k], gt[k])
    assert np.isfinite(list(got.values())).all()
