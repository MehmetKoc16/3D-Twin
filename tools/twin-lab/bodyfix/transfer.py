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


FIELD_ITERATIONS = 60
FIELD_PRUNE = 2e-4
REGION_GROWTH = 12
CAP_COUPLING = 30.0


def forward_map(model: MHModel, fit: FitResult, scan: np.ndarray, faces: np.ndarray) -> SurfaceMap:
    """Scan vertex -> triangle of the fitted, posed MakeHuman surface."""
    if len(scan) == model.nr and np.array_equal(faces, model.faces):
        return topology_map(fit.posed_vertices, model.faces)
    mesh = trimesh.Trimesh(scan, faces, process=False)
    return surface_map(fit.posed_vertices, model.faces, scan, mesh.vertex_normals)


ARM_BONES = ("upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky")


def arm_weights(model: MHModel) -> np.ndarray:
    """Per MakeHuman render vertex: skin weight on the arm, hand and finger bones (the clavicle and neck are excluded)."""
    arm = [i for i, name in enumerate(model.bone_names) if name.split("_")[0] in ARM_BONES]
    return np.where(np.isin(model.skin_j, arm), model.skin_w, 0).sum(axis=1)


def smoothed_field(scan: np.ndarray, faces: np.ndarray, forward: SurfaceMap, count: int,
                   iterations: int = FIELD_ITERATIONS, region: np.ndarray | None = None,
                   inherit: sp.spmatrix | None = None) -> sp.csr_matrix:
    """(n_scan x count) operator: sample a per-MakeHuman-vertex field, then diffuse it over the SCAN surface.

    The nearest-surface correspondence is noisy where the scan departs from the body (loose sleeves, the lips of the
    armpit caps, fused parts): neighbouring scan vertices can land on different body parts and receive very different
    displacements, which tears the sleeve into flaps. Diffusing the sampled field along the scan's own edges (welded by
    position, so UV seams do not cut it) turns such jumps into gradients of the length of the diffusion, and spreads a
    limb length change along the limb instead of letting it jump at a seam. Cut surfaces (armpit gap) are not
    connected, so arm and torso keep their own fields. The armpit cap is connected to both lips by short rungs; edges
    that jump from arm-dominated to body-dominated vertices (the rungs) are coupled `CAP_COUPLING` times harder, so both
    lips receive one displacement and the cap is carried along instead of stretched into a flap. The product is
    precomputed once, so evaluating a deformation stays one sparse mat-vec.

    `region` (per body vertex, 0..1) confines the diffusion to the arms and shoulders, where the scan's sleeves and cap
    lips need it; everywhere else the plain sampled field is kept so that torso, neck, legs and feet reach their tape
    targets exactly as before (diffusion would attenuate thin parts such as the neck). The region is itself grown a few
    centimetres over the scan so sleeve vertices mapped to the torso are still smoothed, and fades out smoothly.
    """
    n = len(scan)
    key = np.round(scan * 1e5).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    welded = int(inverse.max()) + 1
    tri = inverse[faces]
    edges = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    weight = np.ones(len(edges))
    if region is not None:
        arm_scan = np.zeros(welded)
        count_ = np.bincount(inverse, minlength=welded)
        np.add.at(arm_scan, inverse, (forward.barycentric * region[forward.vertices]).sum(axis=1))
        arm_scan /= np.maximum(count_, 1)
        weight += CAP_COUPLING * (np.abs(arm_scan[edges[:, 0]] - arm_scan[edges[:, 1]]) > 0.5)
    adjacency = sp.coo_matrix((weight, (edges[:, 0], edges[:, 1])), shape=(welded, welded)).tocsr()
    adjacency = adjacency + adjacency.T
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)
    step = (0.5 * sp.eye(welded) + 0.5 * sp.diags(1 / degree) @ adjacency).tocsr()
    rows = np.repeat(np.arange(n), forward.vertices.shape[1])
    sample = sp.coo_matrix((forward.barycentric.ravel(), (rows, forward.vertices.ravel())),
                           shape=(n, count)).tocsr()
    if inherit is not None:  # closing fans follow the edge they close instead of a (meaningless) body triangle
        sample = (inherit @ sample).tocsr()
    pool = sp.coo_matrix((np.ones(n), (inverse, np.arange(n))), shape=(welded, n)).tocsr()
    pool_mean = (sp.diags(1 / np.asarray(pool.sum(axis=1)).ravel()) @ pool).tocsr()
    field = (pool_mean @ sample).tocsr()
    for _ in range(iterations):
        field = (step @ field).tocsr()
        field.data[field.data < FIELD_PRUNE] = 0
        field.eliminate_zeros()
    sums = np.maximum(np.asarray(field.sum(axis=1)).ravel(), 1e-12)
    field = sp.diags(1 / sums) @ field
    if region is not None:
        seed = np.asarray(pool @ (forward.barycentric * region[forward.vertices]).sum(axis=1)).ravel()
        seed = (seed / np.maximum(np.asarray(pool.sum(axis=1)).ravel(), 1e-12) > 0.1).astype(float)
        for _ in range(REGION_GROWTH):
            seed = step @ seed
        blend = np.clip(seed * 3, 0, 1)
        field = (sp.diags(blend) @ field + sp.diags(1 - blend) @ (pool_mean @ sample)).tocsr()
    expand = sp.coo_matrix((np.ones(n), (np.arange(n), inverse)), shape=(n, welded)).tocsr()
    return (expand @ field).tocsr()


class Transfer:
    """Fixed correspondence, usable inside a measurement refinement solve."""

    def __init__(self, model: MHModel, fit: FitResult, scan: np.ndarray,
                 faces: np.ndarray, *, smooth_iterations: int = FIELD_ITERATIONS,
                 inherit: sp.spmatrix | None = None) -> None:
        self.model, self.fit, self.scan = model, fit, scan
        matching_topology = len(scan) == model.nr and np.array_equal(faces, model.faces)
        self.forward = forward_map(model, fit, scan, faces)
        heads = model.rest_heads(fit.rest_positions)
        rotations = {key: Rotation.from_rotvec(value).as_matrix()
                     for key, value in fit.pose_rotvec.items()}
        rots, posed = model.fk(heads, rotations, fit.root_t)
        template_a, template_b = model.blended_affine(model.skin_j, model.skin_w,
                                                     heads, rots, posed)
        self.field = smoothed_field(scan, faces, self.forward, model.nr, smooth_iterations, arm_weights(model),
                                    inherit)
        self.affine = np.stack([self.forward.sample(template_a[:, :, k]) for k in range(3)], axis=2)
        self.offset = self.forward.sample(template_b)
        if inherit is not None:
            # Closing fans (hand removal) have no body counterpart of their own: they follow the edge they close,
            # in the displacement field (see smoothed_field) and in the local skinning transform alike.
            self.affine = np.asarray(inherit @ self.affine.reshape(len(scan), 9)).reshape(len(scan), 3, 3)
            self.offset = np.asarray(inherit @ self.offset)
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

    def align(self, target: np.ndarray) -> np.ndarray:
        """Target body expressed relative to the fitted body's pelvis.

        A shaped MakeHuman body is not anchored anywhere (a different height or leg length moves the whole body). That
        rigid part of the displacement is meaningless on the scan, and harmful: the posed displacement is the rest
        displacement rotated by each part's local skinning transform, so a global shift of a few decimetres turns
        into a different shift for every limb that is posed away from the template (arms 20-35 degrees out):
        shoulders shear, sleeves flap, hands drift. The result is grounded afterwards, so only the shape matters.
        """
        pelvis = self.model.bone_index["pelvis"]
        shift = (self.model.rest_heads(target)[pelvis]
                 - self.model.rest_heads(self.fit.rest_positions)[pelvis])
        return target - shift

    def deform(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = self.align(target)
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
        scan_delta = self.field @ delta
        scan_delta = scan_delta * (1 - self.scan_head_blend) + head_shift * self.scan_head_blend
        rest = self.rest_scan + scan_delta
        posed = self.scan + np.einsum("nij,nj->ni", self.affine, scan_delta)
        # The head has one translation, with no skin-blend stretching or rotation.
        head_rotation = self.affine[np.argmax(self.scan_head_blend[:, 0])]
        rigid_head_delta = head_rotation @ head_shift
        posed = posed * (1 - self.scan_head_blend) + (self.scan + rigid_head_delta) * self.scan_head_blend
        return rest, posed

    def landmarks(self, rest: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.vstack([self.reverse.sample(rest), self.align(target)[self.model.nr:]])
