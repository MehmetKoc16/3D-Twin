import numpy as np
from flamehead.colour import from_lab
from hybridbody.hairaudit import bake_hair_audit, compare_bakes


def test_audit_distinguishes_raw_rejected_evidence_from_same_support_repaired_bake():
    source = np.array([[60., 12., 18.], [30., 1., 2.], [55., 10., 15.], [25., 1., 2.]])
    repaired = source.copy()
    repaired[1] = repaired[0]
    texture = np.rint(from_lab(repaired).reshape(2, 2, 3) * 255).astype(np.uint8)
    ears = {"left": np.array([True, False, False, False]), "right": np.array([False, True, False, False])}
    diagnostics = {
        "y": np.array([0, 0, 1, 1]), "x": np.array([0, 1, 0, 1]), "points": np.zeros((4, 3)),
        "hair": {"models": {}, "rejected_texels": {"right": 1}, "ear_fallback": {"mirrored": 1}},
        "ear_members": ears, "beard_allow": np.array([False, False, False, True]),
        "temple": np.array([False, False, True, False]),
        "source_labs": dict.fromkeys(("left", "right"), source),
        "source_weights": dict.fromkeys(("left", "right"), np.ones(4)),
        "accepted_weights": dict.fromkeys(("left", "right"), np.array([1., 0., 1., 1.])),
    }
    # An unused face vertex beyond the raster's max id must not break diagnostics.
    masks = {"face": np.array([6, 7, 8, 9, 10, 11, 20])}
    report = bake_hair_audit(texture, diagnostics, masks, np.arange(12).reshape(4, 3))
    ear = report["projection_skin_check"]["right"]["ear"]
    assert ear["raw_photo"]["skin_fraction"] == 0
    assert ear["accepted_photo"]["texels"] == 0
    assert ear["repaired_bake_same_support"]["skin_fraction"] == 1
    assert ear["repaired_bake_same_support"]["texels"] == ear["raw_photo"]["texels"]
    assert report["beard"]["dark_fraction"] == 1


def test_final_atlas_comparison_uses_fixed_support_and_reports_beard_mean_and_std():
    source = np.array([[60., 12., 18.], [30., 1., 2.], [55., 10., 15.], [25., 1., 2.]])
    before = np.rint(from_lab(source).reshape(2, 2, 3) * 255).astype(np.uint8)
    after = before.copy()
    after[0, 1] = before[0, 0]
    geometry = {"y": np.array([0, 0, 1, 1]), "x": np.array([0, 1, 0, 1]),
                "points": np.array([[.08, 0, 0], [-.08, 0, 0], [.06, .04, 0], [0, -.05, .05]]),
                "left_ear": np.array([True, False, False, False]),
                "right_ear": np.array([False, True, False, False]),
                "face_region": np.array([False, False, True, True]),
                "temple": np.array([False, False, True, False]),
                "beard": np.array([False, False, False, True]),
                "left_support": np.ones(4, bool), "right_support": np.ones(4, bool)}
    report = compare_bakes(before, after, geometry)
    ear = report["projection_skin_check"]["right"]["ear"]
    assert ear["before"]["texels"] == ear["after"]["texels"] == 1
    assert ear["before"]["dark_fraction"] == 1 and ear["after"]["dark_fraction"] == 0
    assert report["beard"]["mean_delta_lab"] == report["beard"]["std_delta_lab"] == [0., 0., 0.]
