import numpy as np
import pytest
from hybridbody.headfit import (
    build_template,
    fit_head,
    flame_labels,
    landmark_similarity,
    region_weights,
    seam_vertices,
)
from hybridbody.register import Params
from synth import face_map_for, flame_like, sphere_template

PARAMS = Params(iterations=14, stiffness_start=40, stiffness_end=0.5)


@pytest.fixture(scope="module")
def setup():
    positions, faces, uv, neck = sphere_template(radius=0.1, subdivisions=3)
    face_map = face_map_for(positions, faces)
    template = build_template(positions, faces, face_map, neck)
    flame, truth = flame_like(positions, faces, face_map)
    return positions, faces, template, flame, truth, neck


def test_template_anchors_everything_at_or_below_the_neck_loop(setup):
    positions, _, template, _, _, neck = setup
    assert template.anchor_y == pytest.approx(positions[neck, 1].max() + 5e-4)
    assert (template.base[~template.free, 1] <= template.anchor_y).all()
    assert template.free.sum() > 0.5 * len(template.base)
    # x-mirror partners of the symmetric sphere exist for (almost) every welded vertex
    assert (template.mirror >= 0).mean() > 0.95
    assert template.landmark_bary.shape == (40, len(template.base))
    np.testing.assert_allclose(np.asarray(template.landmark_bary.sum(1)).ravel(), 1)


def test_similarity_recovers_scale_and_rotation(setup):
    _, _, template, flame, truth, _ = setup
    sim = landmark_similarity(flame, template, template.positions)
    assert sim["scale"] == pytest.approx(truth["scale"], rel=0.05)
    assert sim["inliers"].sum() >= 24
    assert sim["initial_inlier_rms_mm"] < 12
    np.testing.assert_allclose(sim["rotation"], truth["rotation"], atol=0.05)


def test_similarity_rejects_an_implausible_scale(setup):
    _, _, template, flame, _, _ = setup
    flame_bad = type(flame)(
        flame.neutral * 3.0,
        flame.faces,
        flame.masks,
        flame.landmark_ids,
        flame.landmark_points * 3.0,
        flame.fit_dir,
        flame.assets_dir,
    )
    with pytest.raises(ValueError, match="plausible"):
        landmark_similarity(flame_bad, template, template.positions)


def test_region_weights_gate_cavities_and_the_neck(setup):
    _, _, template, flame, _, _ = setup
    sim = landmark_similarity(flame, template, template.positions)
    from flamehead.geometry import transform

    aligned = transform(flame.neutral, sim["scale"], sim["rotation"], sim["translation"])
    label, weight = region_weights(template, aligned, flame_labels(flame))
    assert (weight[~template.free] == 0).all()  # nothing at or below the anchor is ever matched
    assert weight.max() == pytest.approx(1.0) and set(label) <= {"face", "scalp", "neck", "other", "none", "ear", "eye"}


def test_fit_head_reaches_the_flame_surface_and_preserves_the_neck(setup):
    _, _, template, flame, _, _ = setup
    neck = {}

    def measure(moved):
        ring = template.base[:, 1] <= template.anchor_y
        neck["moved"] = np.abs(moved[ring] - template.base[ring]).max()
        return {"girth_cm": 1.0}

    fit = fit_head(template, flame, params=PARAMS, neck_measure=measure)
    assert neck["moved"] == 0.0
    assert fit.report["displacement_mm"]["fixed_below_anchor_max"] == 0.0
    face = fit.report["residual_to_flame_surface"]["face"]
    assert face["mean_mm"] < 2.0 and face["max_mm"] < 8.0
    assert fit.report["triangle_quality"]["flipped_triangles"] == 0
    assert fit.report["landmarks"]["after_mean_mm"] < fit.report["landmarks"]["before_mean_mm"]
    # correspondences: closest FLAME faces and barycentrics exist for every free vertex
    ids = np.flatnonzero(template.free)
    assert (fit.face_ids[ids] >= 0).all() and np.allclose(fit.face_bary[ids].sum(1), 1)
    assert np.isfinite(fit.snap_distance[ids]).all()
    assert fit.report["similarity_scale"] == pytest.approx(0.95, rel=0.06)


def test_seam_vertices_are_shared_between_two_islands():
    positions, faces, uv, neck = sphere_template(subdivisions=2)
    template = build_template(positions, faces, face_map_for(positions, faces), neck)
    # A single island: the "torso" is the head island itself, so every vertex is on its own seam.
    seam = seam_vertices(template, template.head_island)
    assert len(seam) == len(template.base)
