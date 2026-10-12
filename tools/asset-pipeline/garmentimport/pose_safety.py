"""Pose-aware import adjustments that retain the existing 36-byte binding contract."""
from __future__ import annotations

import numpy as np

from .geometry import bone_head, nearest, prism_coverage, skin_weights, surface

POSES = (("A", "a-pose"), ("T", "t-pose"), ("relaxed", "relaxed"), ("walk", "walk"))


def joint(body, name, end="head"):
    return bone_head(body, name, end)


def region_blends(points, body):
    """Compact smoothstep fields around the shoulder underside, elbow and torso sides."""
    armpit, elbow, side = (np.zeros(len(points)) for _ in range(3))
    def field(center, radius):
        t = np.clip(1-np.linalg.norm(points-center, axis=1)/radius, 0, 1)
        return t*t*(3-2*t)
    for suffix, sign in (("l", 1), ("r", -1)):
        shoulder = joint(body, f"upperarm_{suffix}")
        # The collar/shoulder cap is not the armpit. Fade the field out above
        # the shoulder underside so clearance does not inflate overlapping collars.
        below = np.clip((shoulder[1]+.005-points[:, 1])/.05, 0, 1)
        below = below*below*(3-2*below)
        armpit = np.maximum(armpit, field(shoulder+[0, -.07, 0], .17)*below)
        elbow = np.maximum(elbow, field(joint(body, f"lowerarm_{suffix}"), .14))
        center = shoulder+[-sign*.025, -.23, 0]
        side = np.maximum(side, field(center, .20))
    return armpit, elbow, side


def body_regions(body):
    fields = np.column_stack(region_blends(body.positions[:body.render_count], body))
    labels = np.asarray(["armpits", "elbows", "side seams"])[fields.argmax(axis=1)]
    labels[fields.max(axis=1) < .04] = "other"
    return labels


def protected_vertices(body):
    """Never delete skin influenced by hands/fingers, neck or head, including UV copies."""
    protected_bones = np.array([n.startswith(("hand", "neck", "head", "thumb", "index", "middle", "ring", "pinky"))
                                for n in body.names])
    protected = (protected_bones[body.joints] & (body.weights > .001)).any(axis=1)
    protected |= body.positions[:body.render_count, 1] >= joint(body, "neck_01")[1]-.025
    _, group = np.unique(np.round(body.positions[:body.render_count], 7), axis=0, return_inverse=True)
    union = np.zeros(group.max()+1, bool)
    np.logical_or.at(union, group, protected)
    return union[group]


def regional_clearance(points, body, baseline):
    a, e, _ = region_blends(points, body)
    return baseline+.004*a+.003*e


def stable_binding(points, faces, bind, body, refs):
    """Select nearby, less deforming real body triangles; fold rest differences into offsets.

    The contract cannot store multiple triangle bindings. Score local candidates by
    skin-weight agreement with a garment-neighbourhood average and minimum posed
    triangle area; no fictitious body triangle or unsupported skin override is emitted.
    """
    from scipy.spatial import cKDTree
    from .body import skin
    a, _, s = region_blends(points, body)
    active = np.maximum(a, s) > .08
    indices, bary, offsets = (v.copy() for v in bind)
    if not active.any():
        return (indices, bary, offsets), {"reboundVertices": 0}
    ji, jw = skin_weights(bind, body.joints, body.weights)
    dense = np.zeros((len(points), len(body.names)))
    np.add.at(dense, (np.arange(len(points))[:, None], ji), jw)
    _, copies = np.unique(np.round(points, 7), axis=0, return_inverse=True)
    group = np.zeros((copies.max()+1, len(body.names)))
    np.add.at(group, copies, dense)
    group /= np.bincount(copies)[:, None]
    dense = group[copies]
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    for _ in range(3):
        sums, count = dense.copy(), np.ones(len(points))
        for u, v in (edges.T, edges[:, ::-1].T):
            np.add.at(sums, u, dense[v]); np.add.at(count, u, 1)
        dense = .5*dense+.5*sums/count[:, None]
    bp, bf = body.positions[:body.render_count], body.faces
    triangles = bp[bf]
    rest_area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    quality = np.ones(len(bf))
    for _, pose in POSES:
        posed = skin(bp, body.joints, body.weights, body.transforms(body.pose(pose)))[bf]
        area = np.linalg.norm(np.cross(posed[:, 1]-posed[:, 0], posed[:, 2]-posed[:, 0]), axis=1)
        quality = np.minimum(quality, area/np.maximum(rest_area, 1e-12))
    tree = cKDTree(triangles.mean(axis=1))
    # Choose the same candidate for positional UV copies.
    changed, maximum_distance = 0, 0.
    scales = np.asarray([abs(body.positions[refs[k][0], i]-body.positions[refs[k][1], i])/refs[k][2]
                         for i, k in enumerate("xyz")])
    import trimesh
    for group_id in np.unique(copies[active]):
        ids = np.flatnonzero(copies == group_id)
        p = points[ids[0]]
        _, candidates = tree.query(p, k=min(32, len(bf)))
        candidates = np.atleast_1d(candidates)
        closest = trimesh.triangles.closest_point(triangles[candidates], np.tile(p, (len(candidates), 1)))
        weights = trimesh.triangles.points_to_barycentric(triangles[candidates], closest)
        weights = np.maximum(weights, 0); weights /= weights.sum(axis=1)[:, None]
        candidate_bind = bf[candidates], weights, np.zeros_like(closest)
        ci, cw = skin_weights(candidate_bind, body.joints, body.weights)
        influences = np.zeros((len(candidates), len(body.names)))
        np.add.at(influences, (np.arange(len(candidates))[:, None], ci), cw)
        distance = np.linalg.norm(closest-p, axis=1)
        score = np.linalg.norm(influences-dense[ids].mean(axis=0), axis=1)+.25*(1-quality[candidates])+distance*4
        score[distance > distance.min()+.012] = np.inf
        choice = score.argmin()
        indices[ids] = bf[candidates[choice]]; bary[ids] = weights[choice]
        offsets[ids] = (points[ids]-(bp[indices[ids]]*bary[ids, :, None]).sum(axis=1))/scales
        changed += len(ids); maximum_distance = max(maximum_distance, distance[choice])
    return (indices, bary, offsets), {"reboundVertices": changed, "maxAnchorDistanceMm": maximum_distance*1000,
                                     "method": "local skin-neighbourhood and posed-area triangle selection"}


def pose_delete_mask(body, points, faces, bind, seed):
    from .body import skin
    from .visibility import triangle_pairs
    bp = body.positions[:body.render_count]
    covered = np.zeros(body.render_count, bool)
    covered[np.asarray(seed, int)] = True
    ji, jw = skin_weights(bind, body.joints, body.weights)
    mesh = surface(points, faces, weld=True)
    edge, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    boundary = np.unique(edge[counts == 1])
    closest, rest_distance, _ = nearest(mesh, bp)
    # Protect open necklines/cuffs/hems in rest pose, even if contact occurs in another pose.
    eligible = rest_distance < .065
    if len(boundary):
        from scipy.spatial import cKDTree
        samples = []
        for u, v in edge[counts == 1]:
            steps = max(2, int(np.ceil(np.linalg.norm(mesh.vertices[u]-mesh.vertices[v])/.002))+1)
            t = np.linspace(0, 1, steps)[:, None]
            samples.append(mesh.vertices[u]*(1-t)+mesh.vertices[v]*t)
        eligible &= cKDTree(np.vstack(samples)).query(closest)[0] > .009
    pose_counts = {}
    for key, pose in POSES:
        transforms = body.transforms(body.pose(pose))
        pb = skin(bp, body.joints, body.weights, transforms)
        pg = skin(points, ji, jw, transforms)
        coverage = prism_coverage(pb, pg, faces, band=.035, slack=.06) & eligible
        pairs = triangle_pairs(pb, body.faces, pg, faces)
        if len(pairs):
            corners = np.unique(body.faces[pairs[:, 0]])
            # Cover entire intersecting triangles, rather than leaving a visible
            # corner wedge. Anatomical protection still wins after this union.
            coverage[corners[rest_distance[corners] < .065]] = True
        covered |= coverage
        pose_counts[key] = int(coverage.sum())
    covered &= ~protected_vertices(body)
    _, group = np.unique(np.round(bp, 7), axis=0, return_inverse=True)
    union = np.zeros(group.max()+1, bool)
    np.logical_or.at(union, group, covered)
    deleted = np.flatnonzero(union[group])
    return deleted, {"poseCoverageVertices": pose_counts, "deletedVertices": len(deleted),
                     "marginMm": 35, "openingBandMm": 8, "protectedVertices": int(protected_vertices(body).sum())}
