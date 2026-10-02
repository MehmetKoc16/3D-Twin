"""Remove the scan's own hands and tidy what is left.

The web app draws MakeHuman hands over the scan and hides the scan's hand triangles by skin weight. A scan whose fists
are fused to the thighs leaves fist fragments behind as soon as the body changes (the re-fit no longer calls them hand).
The robust fix is to delete the hands from the mesh before anything else uses it:

1. scan vertices whose nearest fitted-MakeHuman surface point is dominated by the hand / finger bones are "hand";
2. every triangle that touches one is deleted (this also cuts the fist off the thigh);
3. pieces that were disconnected by the cut and are small are deleted as well (orphan fragments);
4. every NEW boundary loop (wrist opening, scar where the fist was fused to the leg) is closed by a flat fan.

The result is expressed as a sparse vertex-attribute transfer matrix so UVs / colours of the kept vertices survive and
the closing fans copy the UVs of the edge they close (no smeared texture across atlas seams).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from mh import MHModel

HAND_THRESHOLD = 0.5
MAX_HOLE_PERIMETER_M = 1.2
FRAGMENT_FRACTION = 0.15


@dataclass
class MeshEdit:
    vertices: np.ndarray  # (m, 3)
    faces: np.ndarray  # (k, 3)
    transfer: sp.csr_matrix  # (m, n_old): new attributes = transfer @ old attributes
    inherit: sp.csr_matrix  # (m, m): kept vertices keep themselves, fan apexes average the edge they close
    stats: dict = field(default_factory=dict)


def hand_weights(model: MHModel) -> np.ndarray:
    """Per MakeHuman render vertex: skin weight carried by hand and finger bones (both sides)."""
    bones: set[int] = set()
    for side in ("l", "r"):
        root = model.bone_index[f"hand_{side}"]
        bones.add(root)
        for i, parent in enumerate(model.parent):  # parents precede children
            if parent in bones:
                bones.add(i)
    return np.where(np.isin(model.skin_j, list(bones)), model.skin_w, 0).sum(axis=1)


def hand_mask(model: MHModel, vertices: np.ndarray, barycentric: np.ndarray, triangles: np.ndarray,
              threshold: float = HAND_THRESHOLD) -> np.ndarray:
    """Scan vertices whose closest fitted-MakeHuman surface point is dominated by the hands.

    `triangles`/`barycentric` is the scan-to-MakeHuman surface map (`Transfer.forward`). The fitted hand may sit a few
    centimetres away from a fist the fit could not resolve; the surface map has no distance gate on purpose.
    """
    weight = (barycentric * hand_weights(model)[triangles]).sum(axis=1)
    return weight > threshold


def _weld(vertices: np.ndarray) -> tuple[np.ndarray, int]:
    key = np.round(vertices * 1e5).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    return inverse, int(inverse.max()) + 1


def _components(welded: np.ndarray, count: int, faces: np.ndarray) -> tuple[np.ndarray, int]:
    w = welded[faces]
    edges = np.vstack([w[:, [0, 1]], w[:, [1, 2]], w[:, [0, 2]]])
    graph = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    count_, labels = connected_components(graph, directed=False)
    return labels, int(count_)


def _edge_counts(welded_faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in welded_faces:
        for p, q in ((a, b), (b, c), (c, a)):
            counts[(min(p, q), max(p, q))] += 1
    return counts


def _new_boundary_loops(faces: np.ndarray, welded: np.ndarray,
                        before: dict[tuple[int, int], int]) -> list[list[tuple[int, int]]]:
    """Closed loops of directed raw-vertex edges that became boundary through the cut."""
    after = _edge_counts(welded[faces])
    nxt: dict[int, list[tuple[int, int, int]]] = defaultdict(list)  # welded a -> (welded b, raw a, raw b)
    for tri in faces:
        for i in range(3):
            ra, rb = int(tri[i]), int(tri[(i + 1) % 3])
            wa, wb = int(welded[ra]), int(welded[rb])
            key = (min(wa, wb), max(wa, wb))
            if wa != wb and after[key] == 1 and before.get(key, 0) >= 2:
                nxt[wa].append((wb, ra, rb))
    loops = []
    used: set[tuple[int, int]] = set()
    for start in list(nxt):
        for first in nxt[start]:
            if (start, first[0]) in used:
                continue
            loop, current, edge = [], start, first
            while True:
                used.add((current, edge[0]))
                loop.append((edge[1], edge[2]))
                current = edge[0]
                if current == start:
                    loops.append(loop)
                    break
                options = [e for e in nxt.get(current, []) if (current, e[0]) not in used]
                if not options:
                    break  # open chain: leave it, never invent geometry from a broken loop
                edge = options[0]
    return loops


def remove_hands(vertices: np.ndarray, faces: np.ndarray, hand: np.ndarray, *,
                 max_hole_perimeter: float = MAX_HOLE_PERIMETER_M) -> MeshEdit:
    """Delete hand triangles, orphan fragments, then close the new openings. See the module docstring."""
    n = len(vertices)
    welded, count = _weld(vertices)
    # A welded vertex is "hand" if any of its duplicates is (UV seams duplicate vertices at one position).
    hand_welded = np.zeros(count, dtype=bool)
    np.logical_or.at(hand_welded, welded, hand)
    before = _edge_counts(welded[faces])
    drop = hand_welded[welded[faces]].any(axis=1)
    kept = faces[~drop]
    stats = {"handVertices": int(hand_welded.sum()), "deletedTriangles": int(drop.sum()),
             "fragmentTriangles": 0, "closedLoops": 0, "openLoops": 0, "addedTriangles": 0}
    if len(kept) == 0:
        raise ValueError("hand removal would delete the whole scan; the hand fit is unusable")

    # Orphans: components cut loose by the deletion (touching it) that are small relative to the body.
    label, ncomp = _components(welded, count, kept)
    face_label = label[welded[kept[:, 0]]]
    sizes = np.bincount(face_label, minlength=ncomp)
    main = int(np.argmax(sizes))
    cut_vertices = np.zeros(count, dtype=bool)
    cut_vertices[welded[faces[drop]].ravel()] = True
    touched = np.zeros(len(sizes), dtype=bool)
    touched[label[np.nonzero(cut_vertices)[0]]] = True
    small = touched & (sizes < FRAGMENT_FRACTION * sizes[main])
    small[main] = False
    orphan = small[face_label]
    stats["fragmentTriangles"] = int(orphan.sum())
    kept = kept[~orphan]

    used = np.zeros(n, dtype=bool)
    used[kept.ravel()] = True
    old_index = np.nonzero(used)[0]
    remap = np.full(n, -1, dtype=np.int64)
    remap[old_index] = np.arange(len(old_index))
    new_vertices = [vertices[old_index]]
    rows, cols, vals = [np.arange(len(old_index))], [old_index], [np.ones(len(old_index))]
    own = [(np.arange(len(old_index)), np.arange(len(old_index)), np.ones(len(old_index)))]
    new_faces = [remap[kept]]
    total = len(old_index)

    # Fans close every new loop; the fan apex is copied per triangle so each keeps the UV island of its edge.
    for loop in _new_boundary_loops(kept, welded, before):
        points = vertices[[a for a, _ in loop]]
        perimeter = float(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).sum())
        if perimeter > max_hole_perimeter:
            stats["openLoops"] += 1
            continue
        centre = points.mean(axis=0)
        k = len(loop)
        apex = np.arange(total, total + k)
        ra = np.array([a for a, _ in loop])
        rb = np.array([b for _, b in loop])
        new_vertices.append(np.tile(centre, (k, 1)))
        rows += [apex, apex]
        cols += [remap[ra], remap[rb]]
        vals += [np.full(k, 0.5), np.full(k, 0.5)]
        new_faces.append(np.stack([remap[rb], remap[ra], apex], axis=1))
        own += [(apex, remap[ra], np.full(k, 0.5)), (apex, remap[rb], np.full(k, 0.5))]
        total += k
        stats["closedLoops"] += 1
        stats["addedTriangles"] += k
    # cols[0] holds old indices already; later blocks hold compact ids, which old_index maps back to old attributes.
    transfer = sp.coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows),
          np.concatenate([old_index] + [old_index[c] for c in cols[1:]]))),
        shape=(total, n)).tocsr()
    inherit = sp.coo_matrix((np.concatenate([o[2] for o in own]),
                             (np.concatenate([o[0] for o in own]), np.concatenate([o[1] for o in own]))),
                            shape=(total, total)).tocsr()
    return MeshEdit(np.vstack(new_vertices), np.vstack(new_faces), transfer, inherit, stats)
