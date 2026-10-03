"""Synthetic shapes only: the hybrid marker lets the rig reuse an exactly solved native MakeHuman body."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from bodyfix_solution import hybrid_fit, read_solution
from mh import BODY_DIR
from twin_export import measure_body


def solution(model):
    macro = {"gender": 0.6, "muscle": 0.45, "weight": 0.5, "height": 0.55}
    mods = {"measure/measure-neck-circ": 0.1}
    raw = measure_body(model, model.shape(macro, mods, ground=False))
    return {"version": 1, "fittedMacros": macro, "fittedModifiers": mods, "achievedRawCm": raw,
            "achievedCm": dict(raw), "targetsCm": {"height": raw["height"]},
            "residualsCm": {"height": 0.0}, "clothingAllowanceCm": {}, "measurementBasis": "synthetic"}


def marker(cut):
    digest = hashlib.sha256((Path(BODY_DIR) / "manifest.json").read_bytes()).hexdigest()
    return {"version": 1, "frame": "MakeHuman-grounded-A-pose", "bodyManifestSha256": digest, "cutHeightM": cut}


def scene(model, data):
    return model.shape(data["fittedMacros"], data["fittedModifiers"], ground=True)[: model.nr]


def test_absent_marker_keeps_the_scan_fit(model):
    data = read_solution({"dtBodyfix": solution(model)}, model)
    assert hybrid_fit({"dtBodyfix": data}, model, data, scene(model, data)) is None


def test_exact_native_body_gives_an_identity_pose_fit_even_with_a_deformed_head(model):
    data = read_solution({"dtBodyfix": solution(model)}, model)
    vertices = scene(model, data).copy()
    cut = float(vertices[:, 1].max() - 0.25)
    vertices[vertices[:, 1] > cut] += [0.004, 0.002, -0.003]  # a deformed head above the cut
    fit = hybrid_fit({"dtHybrid": marker(cut)}, model, data, vertices)
    assert fit.stats["pose_source"] == "verified-native-A-pose"
    assert fit.stats["native_body_max_deviation_mm"] < 0.02
    assert all(np.allclose(v, 0) for v in fit.pose_rotvec.values())
    np.testing.assert_allclose(fit.root_t, [0, -fit.rest_positions[: model.nr, 1].min(), 0])
    assert fit.macro == data["fittedMacros"] and fit.mods == data["fittedModifiers"]


def test_a_displaced_body_vertex_below_the_cut_is_rejected(model):
    data = read_solution({"dtBodyfix": solution(model)}, model)
    vertices = scene(model, data).copy()
    cut = float(vertices[:, 1].max() - 0.25)
    vertices[np.flatnonzero(vertices[:, 1] < cut - 0.1)[0]] += [0.001, 0, 0]
    with pytest.raises(ValueError, match="exact solved MakeHuman"):
        hybrid_fit({"dtHybrid": marker(cut)}, model, data, vertices)


def test_marker_validation(model):
    data = read_solution({"dtBodyfix": solution(model)}, model)
    vertices = scene(model, data)
    cut = float(vertices[:, 1].max() - 0.25)
    with pytest.raises(ValueError, match="frame"):
        hybrid_fit({"dtHybrid": {**marker(cut), "frame": "other"}}, model, data, vertices)
    with pytest.raises(ValueError, match="manifest"):
        hybrid_fit({"dtHybrid": {**marker(cut), "bodyManifestSha256": "0" * 64}}, model, data, vertices)
    with pytest.raises(ValueError, match="dtBodyfix"):
        hybrid_fit({"dtHybrid": marker(cut)}, model, None, vertices)
    with pytest.raises(ValueError, match="finite neck cut"):
        hybrid_fit({"dtHybrid": marker(float("nan"))}, model, data, vertices)
