"""Cut mesh 'bridges' between body parts that are not neighbours in the skeleton.

Scans of people whose hands touch their thighs or whose arms press against the torso are fused there. Skinning a fused
bridge stretches it into a thin sheet as soon as the parts move apart (T-pose: spikes from the hands to the hips).
Skeleton-distant influences are dropped before unposing. The rig CLI preserves
surface triangles: refined scans already have separate, closed arm/torso lips,
and deleting triangles based on noisy weights would reopen them. The legacy
triangle-dropping / rigid-copy modes remain available for explicit experiments.
"""

from __future__ import annotations

from collections import deque

import numpy as np

TRUNK = ("Root", "pelvis", "spine_01", "spine_02", "spine_03")


def bone_distance_matrix(names: list[str], parents: list[int]) -> np.ndarray:
    n = len(names)
    node = {i: (0 if names[i] in TRUNK else i + 1) for i in range(n)}
    adj: dict[int, set[int]] = {}
    for i in range(n):
        p = parents[i]
        if p >= 0 and node[i] != node[p]:
            adj.setdefault(node[i], set()).add(node[p])
            adj.setdefault(node[p], set()).add(node[i])
    dist = np.full((n, n), 99, dtype=np.int32)
    nodes = sorted(set(node.values()))
    nd = {}
    for s in nodes:
        d = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        nd[s] = d
    for i in range(n):
        for j in range(n):
            dist[i, j] = nd[node[i]].get(node[j], 99)
    return dist


def snap_incompatible(joints, weights, names, parents, max_dist=2):
    """Drop influences on bones that are not skeleton neighbours of the vertex's dominant bone (smoothing blends
    e.g. hand and thigh weights where a fist touches the leg; those vertices would stretch when the parts separate)."""
    dist = bone_distance_matrix(names, parents)
    dom = joints[np.arange(len(joints)), np.argmax(weights, axis=1)].astype(np.int64)
    keep = dist[dom[:, None], joints.astype(np.int64)] <= max_dist
    w = np.where(keep, weights, 0.0).astype(np.float32)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return w / s, int((~keep & (weights > 1e-6)).any(axis=1).sum())


def cut_bridges(indices, joints, weights, names, parents, max_dist=2, mode="drop"):
    """Returns (new_indices, src_vertex (m,), new_joints (m,4), new_weights (m,4), stats).

    src_vertex maps every output vertex to an input vertex (to copy positions/uv/normals); the first n entries are
    the identity.
    """
    n = len(joints)
    dist = bone_distance_matrix(names, parents)
    dom = joints[np.arange(n), np.argmax(weights, axis=1)].astype(np.int64)
    tri = indices.astype(np.int64).reshape(-1, 3)
    d = dom[tri]
    bad = (
        (dist[d[:, 0], d[:, 1]] > max_dist) | (dist[d[:, 1], d[:, 2]] > max_dist) | (dist[d[:, 0], d[:, 2]] > max_dist)
    )
    if mode == "preserve":
        # A pre-cut scan already has caps separating the limbs. Incompatible
        # transferred weights are repaired before unposing; deleting its faces
        # here would reopen those caps. Preserve topology, including UV seams.
        stats = {"bad_triangles": int(bad.sum()), "triangles": int(len(tri)),
                 "added_vertices": 0, "preserved_triangles": int(bad.sum())}
        return tri.astype(np.uint32), np.arange(n), joints, weights.astype(np.float32), stats
    if mode == "drop":
        # after unposing the two parts are pulled apart, so the bridge triangles are the stretched needles: remove them
        keep = ~bad
        stats = {"bad_triangles": int(bad.sum()), "triangles": int(len(tri)), "added_vertices": 0}
        return tri[keep].astype(np.uint32), np.arange(n), joints, weights.astype(np.float32), stats
    src = list(range(n))
    nj = [joints[i] for i in range(n)]
    nw = [weights[i] for i in range(n)]
    copies: dict[tuple[int, int], int] = {}
    out = tri.copy()
    for t in np.nonzero(bad)[0]:
        a, b, c = tri[t]
        da = [dom[a], dom[b], dom[c]]
        # label = the dominant bone that is closest (tree distance) to the other two vertices' bones
        cost = [sum(dist[x, y] for y in da) for x in da]
        label = da[int(np.argmin(cost))]
        ok = [dist[x, label] <= max_dist for x in da]
        members = [v for v, o in zip((a, b, c), ok) if o]
        for k, (v, o) in enumerate(zip((a, b, c), ok)):
            if o:
                continue
            key = (int(v), int(label))
            if key not in copies:
                copies[key] = len(src)
                src.append(int(v))
                # weights of the triangle's part: mean of the in-part vertices' weights (dense over bones then top4)
                dense: dict[int, float] = {}
                for m_ in members:
                    for j_, w_ in zip(joints[m_], weights[m_]):
                        dense[int(j_)] = dense.get(int(j_), 0.0) + float(w_) / len(members)
                top = sorted(dense.items(), key=lambda kv: -kv[1])[:4]
                jj = np.zeros(4, dtype=joints.dtype)
                ww = np.zeros(4, dtype=np.float32)
                for q, (j_, w_) in enumerate(top):
                    jj[q], ww[q] = j_, w_
                ww /= max(ww.sum(), 1e-9)
                nj.append(jj)
                nw.append(ww)
            out[t, k] = copies[key]
    stats = {"bad_triangles": int(bad.sum()), "triangles": int(len(tri)), "added_vertices": len(src) - n}
    return out.astype(np.uint32), np.array(src), np.array(nj), np.array(nw, dtype=np.float32), stats


def inpaint_weights(W, faces, unknown):
    """Harmonic inpainting of the weight rows `unknown` (bool mask) from the others, over the surface graph of `faces`.

    Islands that touch no known vertex keep their previous row. Returns (new W, number of vertices that were solved).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    W = np.array(W, dtype=np.float64)
    n = len(W)
    if not unknown.any() or unknown.all():
        return W, 0
    tri = np.asarray(faces, dtype=np.int64)
    edges = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
    adjacency = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adjacency = adjacency + adjacency.T
    laplacian = (sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency).tocsr()
    u = np.nonzero(unknown)[0]
    f = np.nonzero(~unknown)[0]
    # (L_UU + eps I) X_U = -L_UF X_F ; eps keeps islands that touch no known vertex solvable (their rows end up ~0)
    system = (laplacian[u][:, u] + 1e-6 * sp.eye(len(u))).tocsc()
    solved = np.maximum(np.asarray(spsolve(system, -(laplacian[u][:, f] @ W[f]))).reshape(len(u), -1), 0)
    ok = solved.sum(axis=1) > 1e-6
    W[u[ok]] = solved[ok]
    return W, int(ok.sum())


def reassign_hand_weights(W, faces, hand_cols, threshold=0.2):
    """Scan hands were removed (bodyfix): no scan surface may follow a hand bone any more.

    The fitted MakeHuman hand is unconstrained (there is no fist left to fit): it may sit on a thigh, a hip or beside the
    wrist, and anything weighted to it would fly off when unposed or animated. Vertices with a hand-family weight above
    `threshold` (the wrist cap, the forearm end, jeans under the misplaced hand) are re-weighted by harmonic inpainting
    over the scan surface from the vertices that carry no hand weight: the cap follows the forearm it closes, jeans
    follow the thigh (the surface itself, not the euclidean distance, decides: a fist beside a hip stays a forearm).
    Smaller hand weights are dropped. W is (n, bones), faces index its rows; returns (W, vertices re-weighted).
    """
    W = np.array(W, dtype=np.float64)
    hand = W[:, hand_cols].sum(axis=1)
    unknown = hand > threshold
    W[np.ix_(np.nonzero((hand > 0) & ~unknown)[0], hand_cols)] = 0
    W[np.ix_(np.nonzero(unknown)[0], hand_cols)] = 0
    W, count = inpaint_weights(W, faces, unknown)
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
    return W, count


def heal_islands(W, faces, names, parents, max_dist=2, max_faces=2000):
    """Small surface patches whose dominant bone is skeleton-distant from everything around them are mis-weighted.

    With the scan hands gone, the weight transfer can still hand the last centimetres of a forearm that lies against the
    thigh to the thigh bone. `cut_bridges` would then drop the connecting triangles and leave a floating patch. A patch
    (connected through compatible triangles, fewer than `max_faces`) is instead re-weighted from the surface around it.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(W)
    dist = bone_distance_matrix(names, parents)
    dom = np.argmax(W, axis=1)
    tri = np.asarray(faces, dtype=np.int64)
    d = dom[tri]
    bad = (dist[d[:, 0], d[:, 1]] > max_dist) | (dist[d[:, 1], d[:, 2]] > max_dist) | (dist[d[:, 0], d[:, 2]] > max_dist)
    good = tri[~bad]
    edges = np.vstack([good[:, [0, 1]], good[:, [1, 2]], good[:, [0, 2]]])
    _, label = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)),
                                    directed=False)
    sizes = np.bincount(label[good[:, 0]], minlength=label.max() + 1)
    island = (sizes[label] > 0) & (sizes[label] < max_faces)
    touched = np.zeros(n, dtype=bool)  # only patches that really border an incompatible neighbour
    touched[tri[bad].ravel()] = True
    keep = np.zeros(label.max() + 1, dtype=bool)
    keep[np.unique(label[touched & island])] = True
    unknown = island & keep[label]
    W, count = inpaint_weights(W, faces, unknown)
    return W, count
