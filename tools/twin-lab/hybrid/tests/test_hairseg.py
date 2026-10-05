"""Hair segmentation of the aligned bust and the mesh-graph helpers (synthetic head, no private data)."""

import numpy as np
import pytest
from flamehead.colour import to_lab
from synth_bust import HAIR_RGB, SKIN_RGB, build_kit, hair_region

from hybridbody import hairseg
from hybridbody.hairseg import SegmentParams, VertexGraph
from hybridbody.hy3d import weld_vertices

# the synthetic head has ~5 mm vertex spacing (the real bust 1 mm): fewer rings everywhere
SMALL = SegmentParams.with_overrides(
    {"smooth_rings": 1, "open_rings": 1, "close_rings": 1, "hole_vertices": 40, "smooth_iterations": 3, "ear_margin": 6}
)


def grid(n=12):
    """An n x n vertex grid of 2 n^2 triangles and its vertex positions in the plane z = 0."""
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    positions = np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], axis=1).astype(float)
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            faces += [[a, b, d], [a, d, c]]
    return positions, np.array(faces)


def test_graph_dilate_erode_and_boundary_on_a_grid():
    positions, faces = grid()
    graph = VertexGraph.from_faces(faces, len(positions))
    mask = (positions[:, 0] >= 4) & (positions[:, 0] <= 7) & (positions[:, 1] >= 4) & (positions[:, 1] <= 7)
    bigger = graph.dilate(mask, 1)
    assert bigger.sum() > mask.sum() and (bigger | mask).sum() == bigger.sum()
    assert graph.erode(mask, 1).sum() < mask.sum() and not (graph.erode(mask, 1) & ~mask).any()
    assert graph.boundary(mask).sum() == 12 and not (graph.boundary(mask) & ~mask).any()  # the ring of a 4 x 4 block
    smooth = graph.smooth(mask.astype(float), 2)
    assert smooth[mask].mean() > smooth[~mask].mean() and smooth.max() <= 1.0


def test_graph_components_largest_and_hole_filling():
    positions, faces = grid(16)
    graph = VertexGraph.from_faces(faces, len(positions))
    big = (positions[:, 0] <= 8) & (positions[:, 1] <= 8)
    island = (positions[:, 0] >= 13) & (positions[:, 1] >= 13)
    labels, count = graph.components(big | island)
    assert count == 2 and set(np.unique(labels)) == {-1, 0, 1}
    assert graph.largest(big | island).sum() == big.sum()
    hole = big & ~((positions[:, 0] >= 3) & (positions[:, 0] <= 4) & (positions[:, 1] >= 3) & (positions[:, 1] <= 4))
    assert graph.fill_holes(hole, 10).sum() == big.sum()  # a small hole is filled
    assert graph.fill_holes(hole, 2).sum() == hole.sum()  # a bigger one than allowed stays
    assert graph.components(np.zeros(len(positions), bool))[1] == 0


def test_floor_curve_is_generous_in_front_and_follows_the_ear_and_the_nape(kit):
    frame = kit.frame
    params = SegmentParams()
    for side in (1, -1):
        curve = hairseg.floor_curve(frame, side, params)
        front, temple, behind_ear, back = curve(0.0), curve(50.0), curve(120.0), curve(180.0)
        ear_top = frame.ears[side].y_top - frame.eye_y
        assert front == pytest.approx(params.floor_front * 1e-3) and temple < front
        assert behind_ear < ear_top + 0.005 and curve(90.0) <= ear_top  # the sideburn ends at the ear
        nape = frame.crease_y - frame.eye_y + params.nape_above_crease * 1e-3
        assert back == pytest.approx(nape, abs=1e-6)
    points = np.array([[0.0, kit.eye_y + 0.2, frame.zc + 0.1], [0.0, kit.eye_y + 0.2, frame.zc - 0.1]])
    assert (hairseg.floor_height(points, frame, params) < points[:, 1]).all()
    assert hairseg.azimuth_of(points, frame).tolist() == pytest.approx([0.0, 180.0])


def test_segment_params_overrides_are_validated():
    assert SegmentParams.with_overrides({"hair_lightness": 40, "open_rings": 2}).open_rings == 2
    with pytest.raises(ValueError, match="Unknown hair segmentation parameter"):
        SegmentParams.with_overrides({"nope": 1})
    with pytest.raises(ValueError, match="non-negative"):
        SegmentParams.with_overrides({"hair_lightness": -1})
    assert set(SegmentParams().to_dict()) >= {"hair_lightness", "colour_azimuth", "min_vertices"}


@pytest.fixture(scope="module")
def kit():
    return build_kit()


@pytest.fixture(scope="module")
def synthetic(kit):
    """The synthetic head welded, with hair and skin colours: (positions, lab, graph, truth flag)."""
    truth_split = hair_region(kit.positions, kit.frame)
    colours = np.where(truth_split[:, None], np.array(HAIR_RGB), np.array(SKIN_RGB)).astype(np.float32)
    normals = np.tile([0.0, 0.0, 1.0], (len(kit.positions), 1))
    positions, _, welded_colours, faces, _ = weld_vertices(kit.positions, normals, colours, kit.faces)
    lab = to_lab(welded_colours / 255.0).astype(np.float32)
    truth = hair_region(positions, kit.frame)
    return positions, lab, VertexGraph.from_faces(faces, len(positions)), truth


def test_segment_hair_finds_the_dark_top_and_back_and_leaves_skin_and_ears(kit, synthetic):
    positions, lab, graph, truth = synthetic
    result = hairseg.segment_hair(positions, lab, graph, kit.frame, SMALL)
    mask = result.mask
    precision, recall = (mask & truth).sum() / mask.sum(), (mask & truth).sum() / truth.sum()
    # everything found is hair; what is missed lies under the floor curve, in the ear margin or in the clean-up
    assert precision > 0.95 and recall > 0.6 and result.report["vertices"] == int(mask.sum())
    assert (
        mask & (lab[:, 0] > 55) & (np.abs(np.degrees(np.arctan2(positions[:, 0], positions[:, 2] - kit.frame.zc))) < 90)
    ).sum() < 0.1 * mask.sum()
    # no hair within the ear margin of the twin's ears
    from scipy.spatial import cKDTree

    near_ear = cKDTree(kit.frame.ear_points).query(positions)[0] < SMALL.ear_margin * 1e-3
    assert not (mask & near_ear).any()
    # the result is one connected piece without spurs
    assert graph.components(mask)[1] == 1 and result.candidate.sum() >= mask.sum() * 0.9


def test_segment_hair_colour_threshold_and_redness_decide_where_hair_meets_skin(kit, synthetic):
    positions, lab, graph, truth = synthetic
    base = hairseg.segment_hair(positions, lab, graph, kit.frame, SMALL).mask
    # a dark but RED vertex band (an ear canal, a sideburn shadow) is skin, not hair
    red = lab.copy()
    band = base & (positions[:, 1] > kit.eye_y + 0.04) & (positions[:, 2] > kit.frame.zc + 0.03)
    red[band, 1] = 18.0
    redder = hairseg.segment_hair(positions, red, graph, kit.frame, SMALL).mask
    assert redder.sum() < base.sum() and not (redder & band).any()
    lighter = lab.copy()
    lighter[base, 0] = 55.0  # everything light: the colour rule empties the front and sides, the back still counts
    pale = hairseg.segment_hair(positions, lighter, graph, kit.frame, SMALL).mask
    assert pale.sum() < base.sum()


def test_pale_artefacts_are_flagged_only_behind_the_colour_azimuth_and_away_from_the_edge(kit, synthetic):
    positions, lab, graph, truth = synthetic
    mask = hairseg.segment_hair(positions, lab, graph, kit.frame, SMALL).mask
    az = np.abs(hairseg.azimuth_of(positions, kit.frame))
    assert not hairseg.pale_artefacts(
        positions, lab, graph, mask, kit.frame, SMALL
    ).any()  # nothing pale in the clean one
    patchy = lab.copy()
    back = mask & (az > 150)
    patchy[back, 0] = 60.0  # a light patch the generator painted on the unseen back
    flagged = hairseg.pale_artefacts(positions, patchy, graph, mask, kit.frame, SMALL)
    assert flagged.any() and not (flagged & ~mask).any() and (az[flagged] > SMALL.colour_azimuth).all()
    front = mask & (az < 60)
    patchy_front = lab.copy()
    patchy_front[front, 0] = 60.0
    assert not hairseg.pale_artefacts(positions, patchy_front, graph, mask, kit.frame, SMALL).any()
