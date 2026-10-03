"""Similarity alignment and topological patch surgery in welded position space."""

import warnings

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def symmetry_map(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    mirrored = vertices.copy()
    mirrored[:, 0] *= -1
    distance, ids = cKDTree(vertices).query(mirrored)
    return ids, {
        "max_distance_mm": float(distance.max() * 1000),
        "involution_ratio": float(np.mean(ids[ids] == np.arange(len(ids)))),
    }


def similarity(source: np.ndarray, target: np.ndarray, band=(0.75, 1.35)):
    a, b = source.mean(0), target.mean(0)
    x, y = source - a, target - b
    u, s, vt = np.linalg.svd(x.T @ y / len(x))
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag(sign) @ u.T
    raw = np.sum(s * sign) / np.mean(np.sum(x * x, axis=1))
    scale = float(np.clip(raw, *band))
    return scale, r, b - scale * (r @ a), raw != scale


def transform(v, s, r, t):
    return s * (v @ r.T) + t


def closest_surface(vertices, faces, points, candidates=24):
    """Bounded nearest-triangle search without rtree (triangle vertices plus centroids).

    This is approximate for arbitrary extremely long triangles; candidates are refined
    by exact point-to-triangle distances and this limitation is recorded in the README.
    """
    tri = vertices[faces]
    tree = cKDTree(np.concatenate((tri.mean(1), tri[:, 0], tri[:, 1], tri[:, 2])))
    out, ids = [], []
    for start in range(0, len(points), 4096):
        p = points[start : start + 4096]
        _, k = tree.query(p, k=min(candidates, len(tri) * 4))
        k = np.asarray(k).reshape(len(p), -1) % len(tri)
        cp = trimesh.triangles.closest_point(tri[k].reshape(-1, 3, 3), np.repeat(p, k.shape[1], axis=0)).reshape(
            len(p), -1, 3
        )
        dist = np.sum((cp - p[:, None]) ** 2, axis=2)
        best = dist.argmin(1)
        out.append(cp[np.arange(len(p)), best])
        ids.append(k[np.arange(len(p)), best])
    return np.concatenate(out), np.concatenate(ids)


def align(neutral, face_ids, scan_v, scan_f, source_lm, target_lm):
    if len(source_lm) < 8:
        raise ValueError("At least eight scan landmarks are required; refusing a guessed transplant")
    s, r, t, clamped = similarity(source_lm, target_lm)
    all_source_lm, all_target_lm = source_lm.copy(), target_lm.copy()
    initial = np.linalg.norm(transform(source_lm, s, r, t) - target_lm, axis=1)
    # Exclude landmark outliers, then constrain ICP with repeated stable landmarks.
    keep = initial <= np.quantile(initial, 0.85)
    source_lm, target_lm = source_lm[keep], target_lm[keep]
    face_ids = np.asarray(face_ids)[::2]
    head_f = scan_f[(scan_v[scan_f, 1] > scan_v[:, 1].max() - 0.34).all(1)]
    for _ in range(12):
        posed = transform(neutral[face_ids], s, r, t)
        cp, _ = closest_surface(scan_v, head_f, posed)
        distance = np.linalg.norm(cp - posed, axis=1)
        good = distance <= min(0.035, np.quantile(distance, 0.70))
        repeat = max(1, int(np.sum(good) / max(len(source_lm), 1)))
        a = np.concatenate((neutral[face_ids][good], np.repeat(source_lm, repeat, axis=0)))
        b = np.concatenate((cp[good], np.repeat(target_lm, repeat, axis=0)))
        ns, nr, nt, hit = similarity(a, b)
        clamped |= hit
        delta = np.linalg.norm(transform(source_lm, ns, nr, nt) - transform(source_lm, s, r, t), axis=1).mean()
        s, r, t = ns, nr, nt
        if delta < 0.00002:
            break
    residual = np.linalg.norm(transform(source_lm, s, r, t) - target_lm, axis=1)
    all_residual = np.linalg.norm(transform(all_source_lm, s, r, t) - all_target_lm, axis=1)
    posed = transform(neutral[face_ids], s, r, t)
    cp, _ = closest_surface(scan_v, head_f, posed)
    distance = np.linalg.norm(cp - posed, axis=1)
    good = distance <= min(0.035, np.quantile(distance, 0.70))
    if clamped:
        warnings.warn("FLAME similarity scale hit the [0.75, 1.35] clamp", stacklevel=2)
    return transform(neutral, s, r, t), {
        "scale": s,
        "scale_clamped": bool(clamped),
        "landmark_count": len(all_source_lm),
        "landmark_rejected": int((~keep).sum()),
        "landmark_initial_rms_mm": float(np.sqrt(np.mean(initial**2)) * 1000),
        "landmark_rms_mm": float(np.sqrt(np.mean(all_residual**2)) * 1000),
        "landmark_trimmed_rms_mm": float(np.sqrt(np.mean(residual**2)) * 1000),
        "landmark_max_mm": float(all_residual.max() * 1000),
        "icp_trimmed_rms_mm": float(np.sqrt(np.mean(distance[good] ** 2)) * 1000),
        "matrix": (np.column_stack((s * r, t))).tolist(),
    }


def edge_table(faces):
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    undirected = np.sort(directed, axis=1)
    _, first, count = np.unique(undirected, axis=0, return_index=True, return_counts=True)
    return directed[first], count


def boundary_loops(faces):
    edges, count = edge_table(faces)
    boundary = edges[count == 1]
    if len(boundary) == 0:
        return []
    outgoing = {}
    incoming = {}
    for a, b in boundary:
        if int(a) in outgoing or int(b) in incoming:
            raise ValueError("Patch boundary branches or has inconsistent winding")
        outgoing[int(a)], incoming[int(b)] = int(b), int(a)
    loops = []
    while outgoing:
        start = next(iter(outgoing))
        loop, current = [], start
        while True:
            loop.append(current)
            if current not in outgoing:
                raise ValueError("Open patch boundary chain")
            current = outgoing.pop(current)
            if current == start:
                break
        loops.append(np.array(loop, np.int64))
    return loops


def largest_component(faces):
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, inverse = np.unique(edges, axis=0, return_inverse=True)
    owners = np.tile(np.arange(len(faces)), 3)
    order = np.argsort(inverse)
    same = inverse[order][1:] == inverse[order][:-1]
    a, b = owners[order][:-1][same], owners[order][1:][same]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(faces), len(faces))).tocsr()
    _, labels = connected_components(graph, directed=False)
    return labels == np.bincount(labels).argmax()


def bridge_rings(vertices, outer, inner):
    """Zipper two consistently oriented boundary rings; every ring edge is used once."""
    # Existing face boundary orientations are opposite across a stitch. Reverse
    # the inner walk for geometric matching; output winding opposes both originals.
    inner = inner[::-1]
    i, j = np.unravel_index(
        np.linalg.norm(vertices[outer][:, None] - vertices[inner][None], axis=2).argmin(), (len(outer), len(inner))
    )
    outer, inner = np.roll(outer, -i), np.roll(inner, -j)
    triangles = []
    i = j = 0

    def progress(ring):
        length = np.linalg.norm(vertices[ring] - vertices[np.roll(ring, -1)], axis=1)
        return np.cumsum(length) / max(length.sum(), 1e-12)

    outer_progress, inner_progress = progress(outer), progress(inner)
    while i < len(outer) or j < len(inner):
        a, b = outer[i % len(outer)], inner[j % len(inner)]
        da = outer_progress[i] if i < len(outer) else np.inf
        db = inner_progress[j] if j < len(inner) else np.inf
        if da <= db:
            triangles.append([outer[(i + 1) % len(outer)], a, b])
            i += 1
        else:
            triangles.append([a, b, inner[(j + 1) % len(inner)]])
            j += 1
    return np.asarray(triangles, np.int64)


def cap_ring(vertices, faces, loop):
    """Close a small internal FLAME opening, without inventing a mouth cavity."""
    center = len(vertices)
    vertices = np.vstack((vertices, vertices[loop].mean(0)))
    cap = np.column_stack((np.roll(loop, -1), loop, np.full(len(loop), center)))
    return vertices, np.vstack((faces, cap)), center


def blend_with_foldover_control(unblended, patch, field):
    """Smoothly relax only a neighbourhood of triangles that would turn inside out."""
    strength = np.ones(len(unblended))
    before = unblended[patch]
    n0 = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
    for _ in range(24):
        vertices = unblended + strength[:, None] * field
        after = vertices[patch]
        n1 = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
        bad = (np.einsum("ij,ij->i", n0, n1) < 0) | (np.linalg.norm(n1, axis=1) < 2e-12)
        if not bad.any():
            return vertices, strength
        # Ears can be much thinner than the scan's ear proxy. Relax locally around
        # foldovers instead of collapsing them onto that proxy or moving scan hair.
        distance = cKDTree(unblended[np.unique(patch[bad])]).query(unblended)[0]
        ramp = np.clip(distance / 0.008, 0, 1)
        strength *= 0.5 + 0.5 * ramp * ramp * (3 - 2 * ramp)
    raise ValueError(f"Geometric blend still folded {int(bad.sum())} FLAME triangles after local relaxation")


def replace_face(scan_v, scan_f, flame_v, flame_f, region):
    selected = region[flame_f].all(1)
    if not selected.any():
        raise ValueError("Empty FLAME face patch")
    patch = flame_f[selected]
    loops = boundary_loops(patch)
    if not loops:
        raise ValueError("FLAME patch needs an outer boundary")
    # Longest spatial perimeter is the external facial border, not eye/mouth holes.
    loops.sort(key=lambda ring: np.linalg.norm(flame_v[ring] - flame_v[np.roll(ring, 1)], axis=1).sum(), reverse=True)
    outer_flame = loops[0]
    cap_centers = []
    for loop in loops[1:]:
        flame_v, patch, center = cap_ring(flame_v, patch, loop)
        cap_centers.append((center, loop))
    centroids = scan_v[scan_f].mean(1)
    head = centroids[:, 1] > scan_v[:, 1].max() - 0.34
    cp, nearest = closest_surface(flame_v[: len(region)], flame_f, centroids[head])
    remove = np.zeros(len(scan_f), bool)
    remove[head] = selected[nearest] & (np.linalg.norm(cp - centroids[head], axis=1) < 0.045)
    if remove.sum() < 20:
        raise ValueError("Insufficient overlapping scan face triangles")
    ids = np.flatnonzero(remove)
    # Grow no geometry outside the nearest-surface selection; discard small islands.
    remove[ids[~largest_component(scan_f[ids])]] = False
    # Retained enclosed islands would create secondary stitch loops: remove only
    # components wholly enclosed in the selected region, leaving the body component.
    kept_ids = np.flatnonzero(~remove)
    main = largest_component(scan_f[kept_ids])
    for fi in kept_ids[~main]:
        if head[fi]:
            remove[fi] = True
    scan_loops = boundary_loops(scan_f[~remove])
    original_loops = boundary_loops(scan_f)
    old_edges = {
        tuple(sorted((int(a), int(b)))) for loop in original_loops for a, b in zip(loop, np.roll(loop, -1), strict=True)
    }
    new_loops = [
        loop
        for loop in scan_loops
        if any(tuple(sorted((int(a), int(b)))) not in old_edges for a, b in zip(loop, np.roll(loop, -1), strict=True))
    ]
    if len(new_loops) != 1:
        raise ValueError(f"Scan removal created {len(new_loops)} stitch loops; refusing ambiguous surgery")
    outer_scan = new_loops[0]
    # A geometric blend band moves only the inserted head. The scan remains exact.
    ring = scan_v[outer_scan]
    a, b = ring, np.roll(ring, -1, axis=0)
    p = flame_v[outer_flame]
    direction = b - a
    fraction = np.clip(
        np.einsum("ijk,jk->ij", p[:, None] - a[None], direction) / np.maximum((direction**2).sum(1), 1e-20), 0, 1
    )
    projected = a[None] + fraction[:, :, None] * direction[None]
    distances = ((projected - p[:, None]) ** 2).sum(2)
    nearest = distances.argmin(1)
    target = projected[np.arange(len(p)), nearest]
    inward = flame_v[np.unique(patch)].mean(0) - p
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-12)
    displacement = (target - p) * 0.98 + inward * 0.00035
    # Leave a 0.25mm bridging strip rather than duplicate/degenerate boundary vertices.
    distance, nearest_ring = cKDTree(p).query(flame_v)
    weight = np.clip(1 - distance / 0.018, 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    unblended = flame_v.copy()
    field = weight[:, None] * displacement[nearest_ring]
    flame_v, strength = blend_with_foldover_control(unblended, patch, field)
    v = np.vstack((scan_v, flame_v))
    offset = len(scan_v)
    patch = patch + offset
    bridge = bridge_rings(v, outer_scan, outer_flame + offset)
    faces = np.vstack((scan_f[~remove], patch, bridge))
    edges, counts = edge_table(faces)
    if np.any(counts > 2) or np.sum(counts == 1) != sum(len(x) for x in original_loops):
        bad = edges[counts > 2]
        patch_bad = int(np.sum(edge_table(patch)[1] > 2))
        bridge_bad = int(np.sum(edge_table(bridge)[1] > 2))
        raise ValueError(
            f"Stitch boundary/nonmanifold counts: before={sum(len(x) for x in original_loops)}, "
            f"after={int(np.sum(counts == 1))}, nonmanifold={int(np.sum(counts > 2))}, "
            f"patch={patch_bad}, bridge={bridge_bad}, bad_edge_ids={bad.tolist()}"
        )
    # Shared vertex IDs make the topological boundary gap exactly zero. Also report
    # the geometric width of the connecting strip, which is a separate quality metric.
    strip_width = np.linalg.norm(flame_v[outer_flame] - target, axis=1)
    report = {
        "removed_triangles": int(remove.sum()),
        "added_triangles": int(len(patch) + len(bridge)),
        "bridge_triangles": len(bridge),
        "stitch_gap_max_mm": 0.0,
        "bridge_width_max_mm": float(strip_width.max() * 1000),
        "boundary_edges_before": sum(len(x) for x in original_loops),
        "boundary_edges_after": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "capped_flame_openings": len(cap_centers),
        "flame_flipped_triangles": 0,
        "blend_strength_min": float(strength[np.unique(patch - offset)].min()),
        "blend_max_displacement_mm": float(
            np.linalg.norm(flame_v[np.unique(patch - offset)] - unblended[np.unique(patch - offset)], axis=1).max()
            * 1000
        ),
    }
    return v, faces, remove, cap_centers, outer_scan, outer_flame, report
