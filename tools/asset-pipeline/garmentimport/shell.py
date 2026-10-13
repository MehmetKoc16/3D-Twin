"""Welded face-graph tools: boundary diagnostics, shell clean-up and opening smoothing.

Garments are stored with UV seams split (several render vertices at one position). Everything here works on the
position-welded topology (10 micrometre grid), so a seam is never mistaken for an open edge, and every vertex
modification is applied to all copies of a position at once.

Why this module exists: MakeHuman clothes model hems, collars and cuffs as several stacked, nearly coincident layers
(rolled bands within about a millimetre of the wall). Per-triangle inner-wall detection (`geometry.outer_shell`) decides
those layers face by face and leaves an interleaved patchwork: hexagonal holes through which another layer shows, and
dozens of tiny boundary loops instead of one clean loop per opening. The passes here make the decision per smooth
patch and per connected component instead, so each opening ends in one clean boundary loop.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

WELD_DIGITS = 5  # 10 micrometres; seam copies are bit-identical, layers of a hem are >= 0.1 mm apart


def weld_ids(positions, digits=WELD_DIGITS):
    """Index of the welded position for every vertex."""
    key = np.round(np.asarray(positions, float)*10**digits).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def face_normals_areas(positions, faces):
    tri = np.asarray(positions, float)[faces]
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    return cross/np.maximum(length, 1e-30)[:, None], length/2


class EdgeTable:
    """Unique welded edges with the faces that use them (a face uses an edge once)."""

    def __init__(self, positions, faces):
        self.faces = np.asarray(faces)
        self.welded = weld_ids(positions)[self.faces]
        half = np.concatenate([self.welded[:, [0, 1]], self.welded[:, [1, 2]], self.welded[:, [2, 0]]])
        self.edges, inverse, self.count = np.unique(np.sort(half, axis=1), axis=0, return_inverse=True,
                                                   return_counts=True)
        self.inverse = inverse.reshape(-1)
        self.face_of_half = np.tile(np.arange(len(self.faces)), 3)
        self.face_of_half.setflags(write=False)
        order = np.argsort(self.inverse, kind="stable")
        self._sorted_edge = self.inverse[order]
        self._sorted_face = self.face_of_half[order]

    def manifold_pairs(self):
        """(face a, face b, edge id) for every edge shared by exactly two faces."""
        two = np.flatnonzero(self.count == 2)
        start = np.searchsorted(self._sorted_edge, two)
        return self._sorted_face[start], self._sorted_face[start+1], two

    def boundary_edges(self):
        return np.flatnonzero(self.count == 1)

    def face_of_edge(self, edge_ids):
        """First face using each of the given edges."""
        return self._sorted_face[np.searchsorted(self._sorted_edge, edge_ids)]


def boundary_loops(positions, faces):
    """Welded boundary edges grouped into connected loops: list of arrays of welded vertex ids, plus the welded points."""
    table = EdgeTable(positions, faces)
    edges = table.edges[table.boundary_edges()]
    count = int(table.welded.max())+1 if len(table.welded) else 0
    points = np.zeros((count, 3))
    points[table.welded.reshape(-1)] = np.asarray(positions, float)[table.faces.reshape(-1)]
    if not len(edges):
        return [], points, table
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    _, label = connected_components(graph, directed=False)
    vertices = np.unique(edges)
    return [vertices[label[vertices] == k] for k in np.unique(label[vertices])], points, table


def mesh_summary(positions, faces, sharp_deg=40.0):
    """Topology numbers on the welded mesh: boundary edges/loops, non-manifold edges and sharp dihedral edges."""
    faces = np.asarray(faces)
    table = EdgeTable(positions, faces)
    loops, _, _ = boundary_loops(positions, faces)
    a, b, _ = table.manifold_pairs()
    normals, _ = face_normals_areas(positions, faces)
    angle = np.degrees(np.arccos(np.clip((normals[a]*normals[b]).sum(axis=1), -1, 1)))
    return {"weldedVertices": int(table.welded.max())+1 if len(faces) else 0, "triangles": int(len(faces)),
            "boundaryEdges": int((table.count == 1).sum()), "boundaryLoops": len(loops),
            "boundaryLoopSizes": sorted((len(l) for l in loops), reverse=True)[:12],
            "nonManifoldEdges": int((table.count > 2).sum()),
            "sharpEdgePercent": float((angle > sharp_deg).mean()*100) if len(angle) else 0.0, "sharpEdgeDegrees": sharp_deg}


def _patches(positions, faces, table, crease_deg):
    """Faces grouped into smooth patches: connected through manifold edges whose dihedral angle is below `crease_deg`."""
    normals, _ = face_normals_areas(positions, faces)
    a, b, _ = table.manifold_pairs()
    angle = np.degrees(np.arccos(np.clip((normals[a]*normals[b]).sum(axis=1), -1, 1)))
    smooth = angle < crease_deg
    graph = coo_matrix((np.ones(int(smooth.sum())), (a[smooth], b[smooth])), shape=(len(faces),)*2)
    return connected_components(graph, directed=False)


def regularize_removal(positions, faces, removed, crease_deg=25.0, confident=0.8, min_area=1.5e-4, mixed_area=2.5e-3):
    """Make the inner-layer decision per smooth patch instead of per triangle.

    `removed` is the per-triangle estimate (True = inner layer). A patch (faces joined across edges flatter than
    `crease_deg`) whose area is at least `min_area` and whose estimate is at least `confident` one-sided gets that label
    as a whole. Any other patch (rim strips, slivers, stitching, seam flaps) takes the label of its neighbours,
    weighted by shared edge length; a tie removes it. Only a patch of at least `mixed_area` whose estimate is mixed keeps
    its per-triangle labels (a smoothly rounded rim can join inner and outer wall in one patch, and one label would then
    lose a whole wall).
    """
    positions, faces, removed = np.asarray(positions, float), np.asarray(faces), np.asarray(removed, bool)
    table = EdgeTable(positions, faces)
    count, patch = _patches(positions, faces, table, crease_deg)
    _, area = face_normals_areas(positions, faces)
    patch_area = np.bincount(patch, weights=area, minlength=count)
    fraction = np.bincount(patch, weights=area*removed, minlength=count)/np.maximum(patch_area, 1e-30)
    label = np.full(count, -1)
    large = patch_area >= min_area
    sure = large & ((fraction >= confident) | (fraction <= 1-confident))
    label[sure] = (fraction[sure] > .5).astype(int)
    keep_own = (patch_area >= mixed_area) & ~sure  # big mixed patch: keep the triangle labels
    label[keep_own] = -2
    # Shared edge length between patches (all manifold edges that join different patches).
    a, b, edge = table.manifold_pairs()
    points = np.zeros((int(table.welded.max())+1, 3))
    points[table.welded.reshape(-1)] = positions[table.faces.reshape(-1)]
    length = np.linalg.norm(points[table.edges[edge, 0]]-points[table.edges[edge, 1]], axis=1)
    pa, pb = patch[a], patch[b]
    different = pa != pb
    neighbours = {}
    for x, y, l in zip(pa[different], pb[different], length[different]):
        neighbours.setdefault(int(x), {}).setdefault(int(y), 0.0)
        neighbours.setdefault(int(y), {}).setdefault(int(x), 0.0)
        neighbours[int(x)][int(y)] += l
        neighbours[int(y)][int(x)] += l

    def tally(p):
        removed_length = sum(l for q, l in neighbours.get(p, {}).items() if label[q] == 1)
        kept_length = sum(l for q, l in neighbours.get(p, {}).items() if label[q] in (0, -2))
        return removed_length, kept_length
    undecided = {int(p) for p in np.flatnonzero(label == -1)}
    while undecided:
        best, best_score = None, -1.0
        for p in sorted(undecided):
            r, k = tally(p)
            total = sum(neighbours.get(p, {}).values())
            score = (r+k)/max(total, 1e-12)
            if r+k > 0 and (score > best_score or (score == best_score and patch_area[p] > patch_area[best])):
                best, best_score = p, score
        if best is None:  # no decided neighbour anywhere: fall back to the patch's own estimate
            for p in undecided:
                label[p] = int(fraction[p] > .5)
            break
        r, k = tally(best)
        label[best] = 1 if r >= k else 0
        undecided.discard(best)
    result = removed.copy()
    decided = label[patch] >= 0
    result[decided] = label[patch][decided] == 1
    return result, {"patches": int(count), "decidedPatches": int((label >= 0).sum()),
                    "trianglesChanged": int((result != removed).sum())}


def restore_bridges(positions, faces, removed, crease_deg=25.0, enclosed=0.8, max_width=0.006, passes=3):
    """Put back thin removed patches that are mostly enclosed by kept triangles.

    A removed patch narrower than `max_width` (twice its area over its perimeter) whose edges border kept triangles for at
    least `enclosed` of its perimeter is a hairline slit in the kept wall (a lapped side seam, a seam allowance flap that
    was paired with its neighbour), not an opening: removing it only leaves a crack. Wide patches are inner layers
    behind a hem or collar and stay removed. Repeats `passes` times because restoring one patch changes its
    neighbours' share.
    """
    positions, faces, removed = np.asarray(positions, float), np.asarray(faces), np.asarray(removed, bool).copy()
    table = EdgeTable(positions, faces)
    count, patch = _patches(positions, faces, table, crease_deg)
    a, b, edge = table.manifold_pairs()
    points = np.zeros((int(table.welded.max())+1, 3))
    points[table.welded.reshape(-1)] = positions[table.faces.reshape(-1)]
    length = np.linalg.norm(points[table.edges[edge, 0]]-points[table.edges[edge, 1]], axis=1)
    open_edges = table.boundary_edges()
    open_length = np.linalg.norm(points[table.edges[open_edges, 0]]-points[table.edges[open_edges, 1]], axis=1)
    open_patch = patch[table.face_of_edge(open_edges)]
    between = patch[a] != patch[b]
    restored = 0
    for _ in range(passes):
        _, area = face_normals_areas(positions, faces)
        patch_area = np.bincount(patch, weights=area, minlength=count)
        gone = np.bincount(patch, weights=area*removed, minlength=count)/np.maximum(patch_area, 1e-30) > .5
        perimeter = np.zeros(count)
        kept_perimeter = np.zeros(count)
        for x, y in ((a, b), (b, a)):
            use = between & gone[patch[x]]
            np.add.at(perimeter, patch[x][use], length[use])
            np.add.at(kept_perimeter, patch[x][use & ~gone[patch[y]]], length[use & ~gone[patch[y]]])
        np.add.at(perimeter, open_patch[gone[open_patch]], open_length[gone[open_patch]])
        width = 2*patch_area/np.maximum(perimeter, 1e-30)
        back = gone & (perimeter > 0) & (kept_perimeter >= enclosed*perimeter) & (width <= max_width)
        if not back.any():
            break
        removed[back[patch]] = False
        restored += int(back.sum())
    return removed, {"restoredPatches": restored, "enclosedFraction": enclosed, "maxWidthMm": max_width*1000}


def kept_components(positions, faces):
    """Connected components of a (kept) triangle set through shared welded manifold edges."""
    table = EdgeTable(positions, faces)
    a, b, _ = table.manifold_pairs()
    return connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(faces),)*2), directed=False)


def stacked_components(positions, faces, reach=0.006, parallel=0.7, cover=0.6):
    """Components that duplicate a larger component: most of their area lies within `reach` of it along its normal.

    A rolled hem or collar band modelled as a second surface hugging the wall adds a duplicate layer whose own boundary
    never coincides with the wall's, giving ragged double openings. Returns (per-triangle drop mask, report).
    Coverage uses +-normal segments of length `reach` against near-parallel triangles of another, strictly larger
    component (area). Everything uses the welded topology.
    """
    import trimesh
    positions, faces = np.asarray(positions, float), np.asarray(faces)
    count, component = kept_components(positions, faces)
    normals, area = face_normals_areas(positions, faces)
    component_area = np.bincount(component, weights=area, minlength=count)
    triangles = positions[faces]
    centers = triangles.mean(axis=1)
    tree = trimesh.Trimesh(positions, faces, process=False).triangles_tree
    covered = np.zeros(len(faces), bool)
    for i in range(len(faces)):
        for sign in (1.0, -1.0):
            direction = sign*normals[i]
            origin = centers[i]+direction*2e-5
            end = centers[i]+direction*reach
            candidates = np.fromiter(tree.intersection(np.r_[np.minimum(origin, end)-1e-7,
                                                              np.maximum(origin, end)+1e-7]), int)
            candidates = candidates[component[candidates] != component[i]]
            if not len(candidates):
                continue
            candidates = candidates[np.abs((normals[candidates]*normals[i]).sum(axis=1)) > parallel]
            if not len(candidates):
                continue
            denominator = (normals[candidates]*direction).sum(axis=1)
            usable = np.abs(denominator) > 1e-9
            candidates, denominator = candidates[usable], denominator[usable]
            if not len(candidates):
                continue
            distance = ((triangles[candidates, 0]-origin)*normals[candidates]).sum(axis=1)/denominator
            near = (distance >= 0) & (distance <= reach)
            if not near.any():
                continue
            candidates, distance = candidates[near], distance[near]
            bary = trimesh.triangles.points_to_barycentric(triangles[candidates], origin+direction*distance[:, None])
            hit = (bary >= -1e-6).all(axis=1)
            if hit.any() and (component_area[component[candidates[hit]]] > component_area[component[i]]).any():
                covered[i] = True
                break
    share = np.bincount(component, weights=area*covered, minlength=count)/np.maximum(component_area, 1e-30)
    drop = share >= cover
    return drop[component], {"components": int(count), "droppedComponents": int(drop.sum()),
                             "droppedTriangles": int(drop[component].sum()),
                             "droppedAreaCm2": [round(float(component_area[k]*1e4), 2) for k in np.flatnonzero(drop)],
                             "reachMm": reach*1000, "coverFraction": cover}


def sliver_components(positions, faces, max_width=0.006, max_share=0.01):
    """Thin leftover strips: components that are narrow (twice area over perimeter <= `max_width`) and tiny.

    Stitching strips and seam-allowance fragments that survive the shell selection as separate small components add a
    long slit-like boundary loop of their own. A closed component (a button) has no boundary and is never selected;
    `max_share` is the largest share of the total kept area. Returns (per-triangle drop mask, report).
    """
    positions, faces = np.asarray(positions, float), np.asarray(faces)
    count, component = kept_components(positions, faces)
    table = EdgeTable(positions, faces)
    _, area = face_normals_areas(positions, faces)
    component_area = np.bincount(component, weights=area, minlength=count)
    points = np.zeros((int(table.welded.max())+1, 3))
    points[table.welded.reshape(-1)] = positions[table.faces.reshape(-1)]
    length = np.linalg.norm(points[table.edges[:, 0]]-points[table.edges[:, 1]], axis=1)
    half_component = component[table.face_of_half]
    low = np.full(len(table.edges), count)
    high = np.full(len(table.edges), -1)
    np.minimum.at(low, table.inverse, half_component)
    np.maximum.at(high, table.inverse, half_component)
    perimeter = np.zeros(count)
    boundary = table.count == 1
    np.add.at(perimeter, low[boundary], length[boundary])
    crossing = (low != high) & ~boundary
    np.add.at(perimeter, low[crossing], length[crossing])
    np.add.at(perimeter, high[crossing], length[crossing])
    width = 2*component_area/np.maximum(perimeter, 1e-30)
    drop = (perimeter > 0) & (width <= max_width) & (component_area <= max_share*component_area.sum())
    return drop[component], {"droppedComponents": int(drop.sum()), "droppedTriangles": int(drop[component].sum()),
                             "droppedWidthMm": [round(float(width[k]*1000), 2) for k in np.flatnonzero(drop)],
                             "maxWidthMm": max_width*1000, "maxAreaShare": max_share}


def smooth_openings(positions, faces, iterations=2, strength=0.5, min_loop=8, max_move=0.005):
    """Lightly relax the open boundary loops of the welded mesh (hem, cuffs, neck) against jagged edges.

    Each boundary vertex moves `strength` of the way to the mean of its two loop neighbours, `iterations` times, and every
    UV copy of a position moves identically, so seams stay closed. Only simple loops (every vertex has two boundary
    neighbours) of at least `min_loop` vertices are touched; others are left alone. No vertex moves more than `max_move`
    metres in total, so a corner of a cuff is rounded, not eaten. Returns (positions, report).
    """
    positions = np.asarray(positions, float).copy()
    weld = weld_ids(positions)
    welded = weld[np.asarray(faces)]
    count = int(weld.max())+1
    half = np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]])
    unique, number = np.unique(np.sort(half, axis=1), axis=0, return_counts=True)
    edges = unique[number == 1]
    if not len(edges):
        return positions, {"smoothedLoops": 0, "maxMoveMm": 0.0}
    degree = np.bincount(edges.reshape(-1), minlength=count)
    points = np.zeros((count, 3))
    points[weld] = positions
    _, label = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count)),
                                    directed=False)
    on_loop = np.unique(edges)
    movable = np.zeros(count, bool)
    loops = 0
    for k in np.unique(label[on_loop]):
        members = on_loop[label[on_loop] == k]
        if len(members) >= min_loop and (degree[members] == 2).all():
            movable[members] = True
            loops += 1
    if not movable.any():
        return positions, {"smoothedLoops": 0, "maxMoveMm": 0.0}
    original = points.copy()
    for _ in range(iterations):
        total = np.zeros_like(points)
        np.add.at(total, edges[:, 0], points[edges[:, 1]])
        np.add.at(total, edges[:, 1], points[edges[:, 0]])
        points[movable] += strength*(total[movable]/2-points[movable])
    moved = points-original
    size = np.linalg.norm(moved, axis=1)
    moved *= np.minimum(1.0, max_move/np.maximum(size, 1e-12))[:, None]
    positions += moved[weld]
    return positions, {"smoothedLoops": loops, "maxMoveMm": float(np.linalg.norm(moved, axis=1).max()*1000),
                       "iterations": iterations, "strength": strength, "maxMoveLimitMm": max_move*1000}
