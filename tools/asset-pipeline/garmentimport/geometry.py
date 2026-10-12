"""Exact closest-triangle bindings and conservative garment coverage."""
from __future__ import annotations

import numpy as np
import trimesh


def surface(positions, faces, *, weld=False):
    mesh = trimesh.Trimesh(positions, faces, process=False)
    if weld:
        mesh.merge_vertices(digits_vertex=7)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    return mesh


def nearest(mesh, points, batch=512):
    """Bound memory on large garments; return exact closest points, distances, face IDs."""
    result = [trimesh.proximity.closest_point(mesh, p) for p in np.array_split(points, max(1, (len(points)+batch-1)//batch))]
    return tuple(np.concatenate([r[i] for r in result]) for i in range(3))


def binding(points, body, faces):
    mesh = surface(body, faces)
    closest, _, tid = nearest(mesh, points)
    indices = faces[tid]
    weights = trimesh.triangles.points_to_barycentric(body[indices], closest)
    weights = np.maximum(weights, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    # The app stores WORLD-axis vectors, not a scalar in a rotating normal frame.
    offsets = points - (body[indices] * weights[..., None]).sum(axis=1)
    return indices, weights, offsets


def reconstruct(bind, body, refs=None):
    indices, weights, offsets = bind
    scales = np.ones(3) if refs is None else np.array([
        abs(body[refs[k][0], a] - body[refs[k][1], a]) / refs[k][2] for a, k in enumerate("xyz")])
    return (body[indices] * weights[..., None]).sum(axis=1) + offsets * scales


def scale_refs(body):
    refs = {}
    for axis, key in enumerate("xyz"):
        a, b = int(body[:, axis].argmin()), int(body[:, axis].argmax())
        refs[key] = [a, b, float(body[b, axis] - body[a, axis])]
    return refs


def inside(mesh, points):
    """Ray-parity containment on the complete welded body, independent of garment binding."""
    from trimesh.ray.ray_util import contains_points
    # Nearly axial rays keep R-tree candidate boxes narrow, avoiding diagonal rays' huge body-wide boxes.
    # The slight tilt avoids exact alignment with axis-parallel seams; use both forward/backward ray parity.
    return np.concatenate([contains_points(mesh.ray,p,check_direction=[1e-6,2e-6,1])
                           for p in np.array_split(points, max(1, (len(points)+511)//512))])


def fit(points, faces, body, body_faces, clearance):
    """Smooth only the displacement field; exterior garment shape is never Laplacian-smoothed."""
    mesh = surface(body, body_faces, weld=True)
    if not mesh.is_watertight:
        raise ValueError("Reference body must be closed after welding for containment validation")
    original = points.copy()
    moved = np.zeros(len(points), bool)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    # Weld positional copies in the displacement pass so UV seams cannot open.
    _, copies = np.unique(np.round(original, 9), axis=0, return_inverse=True)
    for iteration in range(8):
        closest, distance, tid = nearest(mesh, points)
        contained = inside(mesh, points)
        active = contained | (distance < clearance - 1e-7)
        if not active.any():
            break
        moved |= active
        direction = points - closest
        direction[contained] *= -1
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        zero = distance < 1e-9
        direction[zero] = mesh.face_normals[tid[zero]]
        points[active] = closest[active] + direction[active] * (clearance + 1e-6)
        if iteration < 3:
            delta = points - original
            sums, count = np.zeros_like(points), np.zeros(len(points))
            for pairs in (edges, edges[:, ::-1]):
                a, b = pairs.T
                np.add.at(sums, a, delta[b]); np.add.at(count, a, 1)
            delta[moved] = .75 * delta[moved] + .25 * (sums / np.maximum(count[:, None], 1))[moved]
            group = np.zeros((copies.max()+1, 3))
            np.add.at(group, copies, delta)
            group /= np.bincount(copies)[:, None]
            points = original + group[copies]
    # Closely opposing surfaces (e.g. the armpit) can make nearest-surface projection oscillate.
    # Search a bounded outward escape only for unresolved vertices, keeping successful exterior points fixed.
    escaped = 0
    for factor in (2,4,8,16,32):
        closest,distance,tid = nearest(mesh,points)
        contained = inside(mesh,points)
        active = contained | (distance<clearance-1e-7)
        if not active.any():
            break
        ids = np.flatnonzero(active)
        directions = np.concatenate([mesh.face_normals[tid[ids]][:,None,:],
                                     np.broadcast_to(np.vstack([np.eye(3),-np.eye(3)]),(len(ids),6,3))],axis=1)
        candidates = closest[ids,None,:]+directions*(clearance*factor+1e-6)
        outside = (~inside(mesh,candidates.reshape(-1,3))).reshape(len(ids),7)
        _,gap,_ = nearest(mesh,candidates.reshape(-1,3))
        acceptable = outside & (gap.reshape(len(ids),7)>=clearance-1e-7)
        displacement = np.linalg.norm(candidates-points[ids,None,:],axis=2)
        displacement[~acceptable] = np.inf
        choices = displacement.argmin(axis=1)
        good = acceptable.any(axis=1)
        points[ids[good]] = candidates[np.flatnonzero(good),choices[good]]
        moved[ids[good]] = True
        escaped += int(good.sum())
    residual = inside(mesh, points)
    _,gap,_ = nearest(mesh,points)
    return points, {"movedVertices": int(moved.sum()), "remainingInsidePercent": float(residual.mean()*100),
                    "minClearanceMm":float(gap.min()*1000),"escapeVertices":escaped,
                    "maxDisplacementMm": float(np.linalg.norm(points-original, axis=1).max()*1000)}


def delete_mask(body, body_faces, garment, garment_faces, min_depth=.002, max_depth=.02):
    """Runtime footprint rule: all corners covered, normal prism <=2 cm, interior barycentrics.

    Upstream MHCLO delete lists are authored manually, not inferred by a MakeHuman algorithm.
    Boundary-adjacent corners are retained, and all UV copies receive the same result.
    """
    mesh = surface(garment, garment_faces)
    closest, distance, tid = nearest(mesh, body)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[tid], closest)
    body_mesh = surface(body, body_faces)
    normals = mesh.face_normals[tid].copy()
    flip = (normals * body_mesh.vertex_normals).sum(axis=1) < 0
    normals[flip] *= -1
    depth = ((closest - body) * normals).sum(axis=1)
    covered = (depth >= min_depth) & (distance <= max_depth) & (bary.min(axis=1) > .015)
    # Erode one garment boundary band. Hem/neck/cuff holes must not expose deleted body.
    edges = np.sort(mesh.edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary):
        from scipy.spatial import cKDTree
        near = cKDTree(mesh.vertices[np.unique(boundary)]).query(closest)[0]
        covered &= near > .006
    _, group = np.unique(np.round(body, 7), axis=0, return_inverse=True)
    all_covered = np.ones(group.max()+1, bool)
    np.logical_and.at(all_covered, group, covered)
    return np.flatnonzero(all_covered[group])


def skin_weights(bind, joints, weights):
    """Exactly the app's barycentric influence blend, top four, stable bone-index ties."""
    indices, bary, _ = bind
    out_i = np.zeros((len(indices), 4), np.uint16)
    out_w = np.zeros((len(indices), 4))
    for v, (tri, blend) in enumerate(zip(indices, bary)):
        totals = {}
        for j, coefficient in zip(tri, blend):
            for bone, weight in zip(joints[j], weights[j]):
                totals[int(bone)] = totals.get(int(bone), 0.) + coefficient * weight
        top = sorted(totals.items(), key=lambda x: (-x[1], x[0]))[:4]
        for k, (bone, weight) in enumerate(top):
            out_i[v, k], out_w[v, k] = bone, weight
        out_w[v] /= out_w[v].sum()
    return out_i, out_w


def outer_shell(parts, body, body_faces, thickness=.008):
    """Remove close opposing inner faces by inward ray pairing, using body distance for shell order.

    Restrict the removal to paired faces, never to every back-facing garment face: open flaps must survive.
    Face winding must describe a solid's outward surface. Ambiguous/inconsistent winding needs manual selection.
    """
    from .mesh_io import Part, clean
    counts = [len(p.faces) for p in parts]
    starts = np.cumsum([0]+[len(p.positions) for p in parts[:-1]])
    mesh = surface(np.concatenate([p.positions for p in parts]),
                   np.concatenate([p.faces+s for p, s in zip(parts, starts)]))
    centers = mesh.triangles_center
    normals = mesh.face_normals
    body_mesh = surface(body, body_faces, weld=True)
    _, gaps, _ = nearest(body_mesh, centers)
    gaps[inside(body_mesh, centers)] *= -1
    removed = np.zeros(len(centers), bool)
    # Only an 8 mm segment is relevant. Unbounded rays through a dense shoe can create millions of
    # irrelevant candidates even in small batches. Query segment AABBs, then intersect local planes/triangles.
    tree = mesh.triangles_tree
    for start in range(0,len(centers),256):
        ray_ids, triangle_ids = [], []
        for ray in range(start,min(start+256,len(centers))):
            origin = centers[ray]-normals[ray]*1e-5
            end = centers[ray]-normals[ray]*thickness
            candidates = np.array(list(tree.intersection(np.concatenate([
                np.minimum(origin,end)-1e-7,np.maximum(origin,end)+1e-7]))),int)
            opposing = (normals[candidates]*normals[ray]).sum(axis=1)<-.8
            candidates = candidates[opposing & (gaps[ray]<gaps[candidates]-1e-5)]
            ray_ids.extend([ray]*len(candidates)); triangle_ids.extend(candidates)
        if not ray_ids:
            continue
        rays, triangles = np.array(ray_ids,int),np.array(triangle_ids,int)
        origins = centers[rays]-normals[rays]*1e-5
        directions = -normals[rays]
        target_normals = normals[triangles]
        distance = ((mesh.triangles[triangles,0]-origins)*target_normals).sum(axis=1)/(
            directions*target_normals).sum(axis=1)
        valid = (distance>=0)&(distance<=thickness-1e-5)
        locations = origins[valid]+directions[valid]*distance[valid,None]
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[triangles[valid]],locations)
        removed[rays[valid][(bary>=-1e-7).all(axis=1)]] = True
    output = []
    start = 0
    for p, count in zip(parts, counts):
        keep = ~removed[start:start+count]
        if keep.any():
            output.append(clean(Part(p.positions.copy(), p.faces[keep], p.uv.copy(), p.material)))
        start += count
    if not output:
        raise ValueError("Outer-shell selection removed the complete garment")
    return output, {"removedInnerTriangles": int(removed.sum()), "pairingThicknessMm": thickness*1000,
        "method": "inward rays, opposing normals, farther from body",
        "warnings": ["Inspect shell selection at linings, seams and flaps; inconsistent winding needs manual source cleanup."]}


def rigid_shoe_parts(points, body, sole_height=.035):
    """Review-only rigid sole/heel attachments; the shipped loader cannot consume overrides yet."""
    records = []
    for side, sign in (("l", 1), ("r", -1)):
        selected = points[:, 0]*sign > 0
        if not selected.any():
            continue
        low = points[selected, 1].min()
        # An explicit sole height is reviewable and avoids treating the upper as a rigid hull.
        ids = np.flatnonzero(selected & (points[:, 1] <= low+sole_height))
        foot, ball = body.names.index(f"foot_{side}"), body.names.index(f"ball_{side}")
        bones = {b["name"]: b for b in body.rig["bones"]}
        foot_head = body.positions[bones[f"foot_{side}"]["head"]["vert"]]
        ball_head = body.positions[bones[f"ball_{side}"]["head"]["vert"]]
        # Keep each heel/sole segment rigid; split at the anatomical ball joint.
        for bone, head, mask in ((foot, foot_head, points[ids, 2] < ball_head[2]),
                                 (ball, ball_head, points[ids, 2] >= ball_head[2])):
            vertices = ids[mask]
            if len(vertices):
                records.append({"bone": body.names[bone], "vertices": vertices.tolist(),
                                "localPositionsM": (points[vertices]-head).tolist()})
    return {"version": 1, "requiresRuntimeChange": True, "soleHeightMm": sole_height*1000,
            "attachments": records,
            "warnings": ["Manual review of the sole/heel boundary is required. Current app derives all skin weights from barycentrics and ignores this sidecar.",
                         "Foot and ball segments are individually rigid; the transition is not a cloth simulation."]}
