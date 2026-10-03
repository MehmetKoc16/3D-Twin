"""Hand removal and displacement-field smoothing. Only the CC0 MakeHuman stand-in (conftest/test_bodyfix fixtures)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rig"))

from bodyfix import run
from glb import Document
from glbio import _accessor
from handcut import _components, _weld, hand_weights, remove_hands
from solver import measurements
from transfer import SurfaceMap, smoothed_field
from twin_export import CLOTHING_ALLOWANCE_CM

from test_bodyfix import model, standin  # noqa: F401  (module-scoped fixtures)


def _edge_array(faces: np.ndarray) -> np.ndarray:
    return np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]), axis=1), axis=0)


def _components_of(vertices: np.ndarray, faces: np.ndarray) -> int:
    welded, count = _weld(vertices)
    labels, _ = _components(welded, count, faces)
    return len(np.unique(labels[welded[faces[:, 0]]]))


def _boundary_edges(vertices: np.ndarray, faces: np.ndarray) -> int:
    welded, _ = _weld(vertices)
    w = welded[faces]
    edges = np.sort(np.vstack([w[:, [0, 1]], w[:, [1, 2]], w[:, [0, 2]]]), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def _edge_ratios(before: np.ndarray, after: np.ndarray, faces: np.ndarray, minimum: float = 3e-3):
    edges = _edge_array(faces)
    old = np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)
    new = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1)
    keep = old > minimum
    return edges[keep], new[keep] / old[keep]


def test_field_diffusion_turns_a_jump_into_a_gradient():
    """Neighbouring scan vertices mapped to different body parts must not receive a displacement jump."""
    n = 41
    grid = np.array([[x * 0.005, y * 0.005, 0.0] for y in range(n) for x in range(n)])
    faces = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x
            faces += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    faces = np.array(faces)
    owner = (grid[:, 0] > 0.1).astype(np.int64)  # left half -> body vertex 0, right half -> body vertex 1
    forward = SurfaceMap(np.stack([owner, owner, owner], axis=1), np.tile([1.0, 0, 0], (len(grid), 1)),
                         np.zeros(len(grid)))
    values = np.array([[0.0, 0, 0], [0.02, 0, 0]])  # a 2 cm jump between neighbours
    raw = forward.sample(values)
    field = smoothed_field(grid, faces, forward, 2, iterations=60)
    smooth = field @ values
    edges = _edge_array(faces)
    length = np.linalg.norm(grid[edges[:, 0]] - grid[edges[:, 1]], axis=1)

    def strain(d: np.ndarray) -> float:
        return float((np.linalg.norm(d[edges[:, 0]] - d[edges[:, 1]], axis=1) / length).max())

    assert strain(raw) > 3.5
    assert strain(smooth) < 0.5  # no edge stretches by more than 1.5x
    np.testing.assert_allclose(np.asarray(field.sum(axis=1)).ravel(), 1.0, atol=1e-9)


def test_arm_length_change_does_not_tear_the_shoulder(standin, model):  # noqa: F811
    """Longer, thinner arms and wider shoulders: no edge near the shoulder / sleeve stretches more than 2x."""
    folder, source, _, positions = standin
    known = measurements(model, positions)
    tape = folder / "arm.json"
    tape.write_text(json.dumps({
        "armLengthCm": known["armLength"] + 4 - CLOTHING_ALLOWANCE_CM["armLength"],
        "upperArmCm": known["upperArm"] - 3 - CLOTHING_ALLOWANCE_CM["upperArm"],
        "shoulderCm": known["shoulder"] + 1.5 - CLOTHING_ALLOWANCE_CM["shoulder"]}))
    out = folder / "arm/bodyfixed.glb"
    report = run(source, tape, out, remove_scan_hands=False)
    assert abs(report["residualsCm"]["armLength"]) < 1.0
    before = Document(source)
    edges, ratio = _edge_ratios(before.vertices, Document(out).vertices, before.faces)
    mid = (before.vertices[edges[:, 0]] + before.vertices[edges[:, 1]]) / 2
    top = before.vertices[:, 1].max()
    zone = (np.abs(mid[:, 0]) > 0.08) & (mid[:, 1] > top - 0.55) & (mid[:, 1] < top - 0.12)
    assert zone.sum() > 500
    assert ratio[zone].max() < 2.0, ratio[zone].max()


def _fused_scan(folder: Path, source: Path, model):  # noqa: F811
    """The stand-in with each hand bridged to the thigh by needle triangles plus a loose fragment on each hand."""
    document = Document(source)
    vertices, faces = document.vertices, document.faces
    hand = np.nonzero(hand_weights(model) > 0.9)[0]
    extra_vertices, extra_faces, extra_rows = [], [], []
    count = len(vertices)
    for side in (1, -1):
        mine = hand[vertices[hand, 0] * side > 0]
        mine = mine[np.argsort(np.abs(vertices[mine, 0]))[:6]]  # the hand vertices closest to the body
        thigh = np.nonzero((vertices[:, 0] * side > 0.05) & (np.abs(vertices[:, 0]) < np.abs(vertices[mine[0], 0]))
                           & (np.abs(vertices[:, 1] - vertices[mine[0], 1]) < 0.15))[0]
        for i in range(len(mine) - 1):
            nearest = thigh[np.argmin(np.linalg.norm(vertices[thigh] - vertices[mine[i]], axis=1))]
            faces = np.vstack([faces, [mine[i], mine[i + 1], nearest]])
        # a tiny tetrahedron hanging off one hand vertex: a fragment as soon as the hand is cut away
        base = vertices[mine[0]]
        extra_vertices += [base + [0.01 * side, 0.01, 0], base + [0.01 * side, 0, 0.01], base + [0.02 * side, 0.01, 0.01]]
        extra_rows += [mine[0]] * 3
        a, b, c = count, count + 1, count + 2
        extra_faces += [[mine[0], a, b], [mine[0], b, c], [mine[0], c, a], [a, c, b]]
        count += 3
    rows = np.concatenate([np.arange(len(vertices)), extra_rows])
    transfer = sp.coo_matrix((np.ones(len(rows)), (np.arange(len(rows)), rows)),
                             shape=(len(rows), len(vertices))).tocsr()
    path = folder / "fused.glb"
    document.write(path, np.vstack([vertices, extra_vertices]), np.vstack([faces, extra_faces]), transfer)
    return path


def test_remove_hands_cuts_fused_fists_and_leaves_no_fragments(standin, model):  # noqa: F811
    folder, source, _, _ = standin
    scan = Document(_fused_scan(folder, source, model))
    mask = np.zeros(len(scan.vertices), dtype=bool)
    mask[:model.nr] = hand_weights(model) > 0.5
    edit = remove_hands(scan.vertices, scan.faces, mask)
    assert edit.stats["deletedTriangles"] > 500
    assert edit.stats["fragmentTriangles"] >= 2  # the loose remains of both tetrahedra
    assert edit.stats["closedLoops"] >= 2  # both wrist openings are closed again
    assert edit.stats["openLoops"] == 0
    assert _components_of(edit.vertices, edit.faces) <= _components_of(scan.vertices, scan.faces)
    # no bridge needle survives; the cut surface is closed (no new boundary compared with the untouched body)
    edges = _edge_array(edit.faces)
    assert np.linalg.norm(edit.vertices[edges[:, 0]] - edit.vertices[edges[:, 1]], axis=1).max() < 0.12
    original = Document(source)
    assert _boundary_edges(edit.vertices, edit.faces) <= _boundary_edges(original.vertices, original.faces)
    # no hand-weighted vertex is left (rows with a single source are the kept vertices)
    transfer = edit.transfer.tocsr()
    single = np.nonzero(np.diff(transfer.indptr) == 1)[0]
    assert not mask[transfer.indices[transfer.indptr[single]]].any()


def test_remove_hands_refuses_to_delete_everything():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        remove_hands(vertices, np.array([[0, 1, 2]]), np.ones(3, dtype=bool))


def test_bodyfix_pipeline_removes_scan_hands(standin, model):  # noqa: F811
    folder, source, _, positions = standin
    fused = _fused_scan(folder, source, model)
    known = measurements(model, positions)
    tape = folder / "hands.json"
    tape.write_text(json.dumps({"armLengthCm": known["armLength"] + 4 - CLOTHING_ALLOWANCE_CM["armLength"],
                                "waistCm": known["waist"] - 2 - CLOTHING_ALLOWANCE_CM["waist"]}))
    out = folder / "hands/bodyfixed.glb"
    report = run(fused, tape, out)
    assert report["scanHands"]["removed"] and report["scanHands"]["closedLoops"] >= 2
    scan, result = Document(fused), Document(out)
    extras = result.json["asset"]["extras"]
    assert extras["dtScanHandsRemoved"] is True
    assert extras["dtBodyfix"]["fittedMacros"] == report["targetMacros"]
    assert extras["dtBodyfix"]["fittedModifiers"] == report["targetModifiers"]
    assert extras["dtBodyfix"]["achievedCm"] == report["afterCm"]
    assert len(result.vertices) < len(scan.vertices)
    edges = _edge_array(result.faces)
    # lengthened arms would stretch any surviving hand-to-thigh bridge far beyond this
    assert np.linalg.norm(result.vertices[edges[:, 0]] - result.vertices[edges[:, 1]], axis=1).max() < 0.15
    assert _components_of(result.vertices, result.faces) <= _components_of(scan.vertices, scan.faces)
    uv = _accessor(result.json, result.binary, result.parts[0][0]["attributes"]["TEXCOORD_0"])
    assert len(uv) == len(result.vertices) and np.isfinite(result.vertices).all()
    assert abs(report["residualsCm"]["armLength"]) < 1.5 and abs(report["residualsCm"]["waist"]) < 1.5
    assert not report["unreachable"]
