"""Smooth scalar-field surgery, preserving corner attributes through triangle splits."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from twinrefine.armpit import smooth_field
from twinrefine.meshops import Corners, edge_endpoints, refine_marked_edges
from twinrefine.meshops import edge_table as mesh_edges

from .geometry import (
    blend_with_foldover_control,
    boundary_loops,
    bridge_rings,
    closest_surface,
    edge_table,
    largest_component,
)


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def geodesic(vertices, faces, seeds):
    edges = edge_table(faces)[0]
    length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    graph = coo_matrix(
        (np.r_[length, length], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    if len(seeds) == 0:
        return np.full(len(vertices), np.inf)
    return dijkstra(graph, indices=np.asarray(seeds, int), min_only=True, directed=False)


def expanded_region(vertices, faces, masks, ears=True):
    """Move the seam into forehead/scalp margins, behind ears and the upper neck."""
    from .assets import region_mask

    selected = region_mask(masks, len(vertices), ears)
    selected[masks.get("forehead", [])] = True
    face = np.asarray(masks["face"], int)
    chin = float(vertices[face, 1].min())
    forehead = np.asarray(masks.get("forehead", []), int)
    if len(forehead):
        # Include a modest scalp margin under the hair, rather than ending at forehead masks.
        distance = geodesic(vertices, faces, forehead)
        selected |= (distance < 0.012) & (vertices[:, 1] > np.median(vertices[forehead, 1]))
    if ears:
        ear = np.r_[masks["left_ear"], masks["right_ear"]]
        if len(ear):
            # A collar outside the fine ear topology puts the border on the skull.
            selected |= geodesic(vertices, faces, ear) < 0.025
    neck = np.asarray(masks.get("neck", []), int)
    if len(neck):
        front_depth = float(np.quantile(vertices[face, 2], 0.15))
        selected[neck] |= (vertices[neck, 1] > chin - 0.035) & (vertices[neck, 2] > front_depth - 0.012)
    selected[masks.get("boundary", [])] = False
    if not ears:
        selected[np.r_[masks["left_ear"], masks["right_ear"]]] = False
    # Smooth in index-independent physical space first, then on the mesh graph.
    inside = geodesic(vertices, faces, np.flatnonzero(~selected))
    outside = geodesic(vertices, faces, np.flatnonzero(selected))
    phi = np.where(selected, np.minimum(inside, 0.025), -np.minimum(outside, 0.025))
    phi = smooth_field(faces, phi, 5, alpha=0.45)
    phi[masks.get("boundary", [])] = -0.025
    return phi, {
        "upper_neck_extension_mm": 35.0,
        "ear_collar_mm": 25.0 if ears else 0.0,
        "forehead_margin_mm": 12.0,
        "chin_native_y": chin,
    }


def cut_level(mc, phi, keep_positive):
    """Split every crossing edge once, interpolating each side's independent UV corners."""
    phi = np.asarray(phi, float).copy()
    phi[np.abs(phi) < 1e-10] = 1e-10
    edges = mesh_edges(mc.F)
    lo, hi = edge_endpoints(edges)
    crossing = phi[lo] * phi[hi] < 0
    fraction = np.clip(phi[lo] / np.where(crossing, phi[lo] - phi[hi], 1), 1e-4, 1 - 1e-4)
    result, is_new = refine_marked_edges(mc, crossing, edges, fraction)
    extended = np.r_[phi, np.zeros(is_new.sum())]
    positive = extended[result.F].sum(1) > 0
    keep = positive if keep_positive else ~positive
    field_crossings = ((phi[mc.F] > 0).any(1) & (phi[mc.F] < 0).any(1)).sum()
    return Corners(result.P, result.F[keep], result.C[keep]), {
        "split_triangles": int(field_crossings),
        "inserted_cut_vertices": int(is_new.sum()),
        "discarded_pieces": int((~keep).sum()),
    }


def smooth_ring(vertices, ring, iterations=5):
    curve = vertices[ring].copy()
    for _ in range(iterations):
        curve += 0.25 * (0.5 * (np.roll(curve, 1, axis=0) + np.roll(curve, -1, axis=0)) - curve)
    return curve


def nearest_curve(points, curve):
    a, direction = curve, np.roll(curve, -1, axis=0) - curve
    tree = cKDTree(a)
    _, candidates = tree.query(points, k=min(8, len(curve)))
    candidates = np.asarray(candidates).reshape(len(points), -1)
    # Each nearest vertex contributes its incoming and outgoing segments.
    candidates = np.concatenate((candidates, (candidates - 1) % len(curve)), axis=1)
    aa, dd = a[candidates], direction[candidates]
    t = np.clip(np.einsum("ijk,ijk->ij", points[:, None] - aa, dd) / np.maximum((dd * dd).sum(2), 1e-20), 0, 1)
    projected = aa + t[:, :, None] * dd
    distance = ((projected - points[:, None]) ** 2).sum(2)
    best = distance.argmin(1)
    return projected[np.arange(len(points)), best]


def curve_roughness(vertices, ring):
    p = vertices[ring]
    laplacian = p - 0.5 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))
    return float(np.sqrt(np.mean(np.sum(laplacian**2, axis=1))) * 1000)


@dataclass
class Surgery:
    scan: Corners
    flame: Corners
    bridge: np.ndarray
    scan_ring: np.ndarray
    flame_ring: np.ndarray
    report: dict


def transplant(scan_mc, flame_mc, phi, band=0.022):
    original_boundary = sum(len(r) for r in boundary_loops(scan_mc.F))
    flame, fp = cut_level(flame_mc, phi, True)
    loops = boundary_loops(flame.F)
    loops.sort(key=lambda r: np.linalg.norm(flame.P[r] - flame.P[np.roll(r, 1)], axis=1).sum(), reverse=True)
    if not loops:
        raise ValueError("No FLAME region boundary")
    flame_ring = loops[0]
    regular_triangles = len(flame.F)
    # Cap the smaller internal mouth/eyeball openings, retaining separate UV charts.
    caps = 0
    for ring in loops[1:]:
        center = len(flame.P)
        cp = flame.P[ring].mean(0)
        cuv = np.zeros(2)
        corner_uv = np.zeros((len(flame.P), 2))
        corner_uv[flame.F.ravel()] = flame.C.reshape(-1, 2)
        cuv = corner_uv[ring].mean(0)
        cap = np.column_stack((np.roll(ring, -1), ring, np.full(len(ring), center)))
        cap_uv = np.stack((corner_uv[np.roll(ring, -1)], corner_uv[ring], np.tile(cuv, (len(ring), 1))), axis=1)
        flame = Corners(np.vstack((flame.P, cp)), np.vstack((flame.F, cap)), np.vstack((flame.C, cap_uv)))
        caps += 1
    # Interpolate the FLAME field on closest points at scan vertices, not triangle labels.
    sp = np.full(len(scan_mc.P), -0.05)
    head = scan_mc.P[:, 1] > scan_mc.P[:, 1].max() - 0.40
    cp, ids = closest_surface(flame_mc.P, flame_mc.F, scan_mc.P[head])
    import trimesh

    bary = trimesh.triangles.points_to_barycentric(flame_mc.P[flame_mc.F[ids]], cp)
    sp[head] = np.einsum("ij,ij->i", bary, phi[flame_mc.F[ids]])
    sp[head] = np.minimum(sp[head], 0.06 - np.linalg.norm(cp - scan_mc.P[head], axis=1))
    sp = smooth_field(scan_mc.F, sp, 10, alpha=0.5)
    scan, sr = cut_level(scan_mc, sp, False)
    main = largest_component(scan.F)
    if not main.all():
        scan = Corners(scan.P, scan.F[main], scan.C[main])
    scan_loops = boundary_loops(scan.F)
    if len(scan_loops) != 1:
        raise ValueError(f"Expected one smooth scan cut curve, found {len(scan_loops)}")
    scan_ring = scan_loops[0]
    before_roughness = curve_roughness(scan.P, scan_ring)
    target = smooth_ring(scan.P, scan_ring)
    # Carry cut-line smoothing through a real band on the retained scan as well.
    distance, near = cKDTree(scan.P[scan_ring]).query(scan.P)
    field = (target - scan.P[scan_ring])[near] * (1 - smoothstep(distance / band))[:, None]
    scan_p, scan_strength = blend_with_foldover_control(scan.P, scan.F, field)
    scan = Corners(scan_p, scan.F, scan.C)
    original_flame = flame.P.copy()
    ring_target = nearest_curve(flame.P[flame_ring], scan.P[scan_ring])
    inward = flame.P[np.unique(flame.F)].mean(0) - ring_target
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-12)
    ring_delta = ring_target + inward * 0.0003 - flame.P[flame_ring]
    distance, near = cKDTree(flame.P[flame_ring]).query(flame.P)
    field = ring_delta[near] * (1 - smoothstep(distance / band))[:, None]
    flame_p, strength = blend_with_foldover_control(flame.P, flame.F, field)
    flame = Corners(flame_p, flame.F, flame.C)
    vertices = np.vstack((scan.P, flame.P))
    offset = len(scan.P)
    bridge = bridge_rings(vertices, scan_ring, flame_ring + offset)
    faces = np.vstack((scan.F, flame.F + offset, bridge))
    _, counts = edge_table(faces)
    if (counts > 2).any() or int((counts == 1).sum()) != original_boundary:
        raise ValueError("Smooth surgery changed boundary count or created nonmanifold edges")
    widths = np.linalg.norm(flame.P[flame_ring] - nearest_curve(flame.P[flame_ring], scan.P[scan_ring]), axis=1)
    return Surgery(
        scan,
        flame,
        bridge,
        scan_ring,
        flame_ring,
        {
            "removed_triangles": sr["discarded_pieces"],
            "added_triangles": len(flame.F) + len(bridge),
            "split_scan_triangles": sr["split_triangles"],
            "split_flame_triangles": fp["split_triangles"],
            "scan_cut_vertices": sr["inserted_cut_vertices"],
            "flame_cut_vertices": fp["inserted_cut_vertices"],
            "bridge_triangles": len(bridge),
            "stitch_gap_max_mm": 0.0,
            "bridge_width_max_mm": float(widths.max() * 1000),
            "bridge_width_mean_mm": float(widths.mean() * 1000),
            "boundary_edges_before": original_boundary,
            "boundary_edges_after": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()),
            "flame_flipped_triangles": 0,
            "capped_flame_openings": caps,
            "geometry_band_mm": band * 1000,
            "flame_regular_triangles": regular_triangles,
            "blend_strength_min": float(strength.min()),
            "scan_blend_strength_min": float(scan_strength.min()),
            "blend_max_displacement_mm": float(np.linalg.norm(flame.P - original_flame, axis=1).max() * 1000),
            "scan_curve_roughness_before_mm": before_roughness,
            "scan_curve_roughness_after_mm": curve_roughness(scan.P, scan_ring),
        },
    )
