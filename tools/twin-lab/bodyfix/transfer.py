"""Surface correspondence and smooth, detail-preserving body displacement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mh import MHModel
from rigfit import FitResult
from solver import definitions


@dataclass
class SurfaceMap:
    vertices: np.ndarray
    barycentric: np.ndarray
    distance: np.ndarray

    def sample(self, values: np.ndarray) -> np.ndarray:
        return np.einsum("nk,nkj->nj", self.barycentric, values[self.vertices])


def topology_map(vertices: np.ndarray, faces: np.ndarray) -> SurfaceMap:
    """Exact surface anchors when a scan retains MakeHuman's vertex topology."""
    incidence = trimesh.Trimesh(vertices, faces, process=False).vertex_faces[:, 0]
    triangles = faces[incidence]
    bary = (triangles == np.arange(len(vertices))[:, None]).astype(float)
    return SurfaceMap(triangles, bary, np.zeros(len(vertices)))


def surface_map(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray,
                normals: np.ndarray | None = None) -> SurfaceMap:
    """Closest triangle among faces incident to 12 nearby vertices, normal gated.

    Uses vertex incidence rather than centroid proximity so large/skinny triangles
    remain candidates. No optional rtree dependency is needed; queries are batched.
    """
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    tree = cKDTree(vertices)
    incidence = mesh.vertex_faces
    chosen_faces, bary, distances = [], [], []
    for start in range(0, len(points), 512):
        p = points[start:start + 512]
        _, nearby = tree.query(p, k=min(12, len(vertices)))
        if nearby.ndim == 1:
            nearby = nearby[:, None]
        candidates = incidence[nearby].reshape(len(p), -1)
        valid = candidates >= 0
        candidates = np.maximum(candidates, 0)
        triangles = vertices[faces[candidates]].reshape(-1, 3, 3)
        query = np.repeat(p, candidates.shape[1], axis=0)
        closest = trimesh.triangles.closest_point(triangles, query)
        distance = np.linalg.norm(closest - query, axis=1).reshape(candidates.shape)
        score = distance.copy()
        if normals is not None:
            dots = np.einsum("nkj,nj->nk", mesh.face_normals[candidates],
                             normals[start:start + len(p)])
            score += 0.12 * np.clip((0.6 - dots) / 0.6, 0, None)
        score[~valid] = np.inf
        pick = np.argmin(score, axis=1)
        row = np.arange(len(p))
        selected = candidates[row, pick]
        q = closest.reshape(len(p), -1, 3)[row, pick]
        weights = trimesh.triangles.points_to_barycentric(vertices[faces[selected]], q)
        weights = np.maximum(np.nan_to_num(weights), 0)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        chosen_faces.append(faces[selected])
        bary.append(weights)
        distances.append(distance[row, pick])
    return SurfaceMap(np.concatenate(chosen_faces), np.concatenate(bary),
                      np.concatenate(distances))


def smooth_operator(vertices: np.ndarray, faces: np.ndarray) -> sp.csr_matrix:
    """One neighbour averaging step, welding UV seams without changing topology."""
    _, inverse = np.unique(np.round(vertices, 7), axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    weld = sp.coo_matrix((np.ones(len(vertices)), (inverse, np.arange(len(vertices)))),
                         shape=(count, len(vertices))).tocsr()
    average = sp.diags(1 / np.asarray(weld.sum(axis=1)).ravel()) @ weld
    edges = inverse[trimesh.Trimesh(vertices, faces, process=False).edges_unique]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    adjacency = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                              shape=(count, count)).tocsr()
    adjacency = adjacency + adjacency.T + sp.eye(count)
    neighbour = sp.diags(1 / np.asarray(adjacency.sum(axis=1)).ravel()) @ adjacency
    expand = sp.coo_matrix((np.ones(len(vertices)), (np.arange(len(vertices)), inverse)),
                           shape=(len(vertices), count)).tocsr()
    return (expand @ (0.75 * sp.eye(count) + 0.25 * neighbour) @ average).tocsr()


class Transfer:
    """Fixed correspondence, usable inside a measurement refinement solve."""

    def __init__(self, model: MHModel, fit: FitResult, scan: np.ndarray,
                 faces: np.ndarray) -> None:
        self.model, self.fit, self.scan = model, fit, scan
        mesh = trimesh.Trimesh(scan, faces, process=False)
        matching_topology = len(scan) == model.nr and np.array_equal(faces, model.faces)
        self.forward = (topology_map(fit.posed_vertices, model.faces) if matching_topology
                        else surface_map(fit.posed_vertices, model.faces, scan, mesh.vertex_normals))
        heads = model.rest_heads(fit.rest_positions)
        rotations = {key: Rotation.from_rotvec(value).as_matrix()
                     for key, value in fit.pose_rotvec.items()}
        rots, posed = model.fk(heads, rotations, fit.root_t)
        template_a, template_b = model.blended_affine(model.skin_j, model.skin_w,
                                                     heads, rots, posed)
        self.affine = np.stack([self.forward.sample(template_a[:, :, k]) for k in range(3)], axis=2)
        self.offset = self.forward.sample(template_b)
        self.rest_scan = np.linalg.solve(self.affine, (scan - self.offset)[..., None])[..., 0]
        self.reverse = (topology_map(self.rest_scan, faces) if matching_topology else
                        surface_map(self.rest_scan, faces, fit.rest_positions[:model.nr],
                                    trimesh.Trimesh(fit.rest_positions[:model.nr], model.faces,
                                                    process=False).vertex_normals))
        self.smoothing = smooth_operator(fit.rest_positions[:model.nr], model.faces)
        self.rigid = []
        for name in ("hand_l", "hand_r", "foot_l", "foot_r"):
            bone = model.bone_index[name]
            descendants = {bone}
            for i, parent in enumerate(model.parent):
                if parent in descendants:
                    descendants.add(i)
            weight = np.where(np.isin(model.skin_j, list(descendants)), model.skin_w, 0).sum(axis=1)
            self.rigid.append((bone, weight))
        self.neck_top = heads[model.bone_index["head"], 1]
        t = np.clip((fit.rest_positions[:model.nr, 1] - (self.neck_top - 0.05)) / 0.05, 0, 1)
        self.head_blend = (t * t * (3 - 2 * t))[:, None]
        self.head_bone = model.bone_index["head"]
        self.foot_verts = next(d["verts"] for d in definitions() if d["id"] == "footLength")
        # Keep above-neck scan details rigid even when they lie off the template.
        t = np.clip((self.rest_scan[:, 1] - (self.neck_top - 0.05)) / 0.05, 0, 1)
        self.scan_head_blend = (t * t * (3 - 2 * t))[:, None]

    def deform(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = target[:self.model.nr] - self.fit.rest_positions[:self.model.nr]
        for _ in range(2):
            delta = self.smoothing @ delta
        joint_delta = self.model.rest_heads(target) - self.model.rest_heads(self.fit.rest_positions)
        for bone, weight in self.rigid:
            # Hands translate; feet also scale along their depth as a single shoe
            # block. This permits EU size targets without distorting toe details.
            rigid_delta = np.broadcast_to(joint_delta[bone], delta.shape).copy()
            if self.model.bone_names[bone].startswith("foot_"):
                a, b = self.foot_verts[:2]
                old_length = abs(self.fit.rest_positions[a, 2] - self.fit.rest_positions[b, 2])
                new_length = abs(target[a, 2] - target[b, 2])
                center = self.model.rest_heads(self.fit.rest_positions)[bone, 2]
                rigid_delta[:, 2] += ((self.fit.rest_positions[:self.model.nr, 2] - center)
                                      * (new_length / max(old_length, 1e-9) - 1))
            blend = (0.85 * weight**2)[:, None]
            delta = delta * (1 - blend) + rigid_delta * blend
        head_shift = joint_delta[self.head_bone]
        delta = delta * (1 - self.head_blend) + head_shift * self.head_blend
        scan_delta = self.forward.sample(delta)
        scan_delta = scan_delta * (1 - self.scan_head_blend) + head_shift * self.scan_head_blend
        rest = self.rest_scan + scan_delta
        posed = self.scan + np.einsum("nij,nj->ni", self.affine, scan_delta)
        # The head has one translation, with no skin-blend stretching or rotation.
        head_rotation = self.affine[np.argmax(self.scan_head_blend[:, 0])]
        rigid_head_delta = head_rotation @ head_shift
        posed = posed * (1 - self.scan_head_blend) + (self.scan + rigid_head_delta) * self.scan_head_blend
        return rest, posed

    def landmarks(self, rest: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.vstack([self.reverse.sample(rest), target[self.model.nr:]])
