"""Cut mesh 'bridges' between body parts that are not neighbours in the skeleton.

Scans of people whose hands touch their thighs or whose arms press against the torso are fused there. Skinning a fused
bridge stretches it into a thin sheet as soon as the parts move apart (T-pose: spikes from the hands to the hips).
Every triangle whose vertices are dominated by bones that are more than `max_dist` apart in the (trunk-merged)
skeleton tree is made rigid with its majority part: its off-part vertices are replaced by copies that carry the weights
of the triangle's part. The surface stays closed in the rest pose; a hairline crack opens only when the parts move.
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
