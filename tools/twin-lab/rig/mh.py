"""MakeHuman body model (base mesh, morph targets, rig, skin weights) in numpy.

Mirrors packages/avatar-core (macro tents, modifiers, joint points) so the twin lab can fit our parametric body
to a scan without going through Node.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

from glbio import _accessor, _split

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BODY_DIR = os.path.join(REPO, "apps", "web", "public", "assets", "body")
POSE_DIR = os.path.join(REPO, "apps", "web", "public", "assets", "poses")

# Modifiers used when fitting a scan (skip the fine face features: the scan mesh is only used for rigging).
FIT_MOD_PREFIXES = ("measure/", "torso/", "hip/", "neck/", "stomach/", "buttocks/", "pelvis/", "breast/", "armslegs/")
FIT_MOD_EXTRA = ("head/head-scale-depth", "head/head-scale-horiz", "head/head-scale-vert", "head/head-fat")


@dataclass
class Bone:
    name: str
    parent: int  # index or -1
    head_vert: int
    tail_vert: int


class MHModel:
    def __init__(self) -> None:
        self.manifest = json.load(open(os.path.join(BODY_DIR, "manifest.json"), encoding="utf8"))
        rig = json.load(open(os.path.join(BODY_DIR, "rig.json"), encoding="utf8"))
        self.nr = self.manifest["renderVertexCount"]  # render vertices
        joint_pts = np.array([j["position"] for j in self.manifest["jointPoints"]], dtype=np.float64)
        js, binary = _split(open(os.path.join(BODY_DIR, "base.glb"), "rb").read())
        prim = js["meshes"][0]["primitives"][0]
        pos = _accessor(js, binary, prim["attributes"]["POSITION"]).astype(np.float64)
        self.faces = _accessor(js, binary, prim["indices"]).astype(np.int64).reshape(-1, 3)
        self.uv = _accessor(js, binary, prim["attributes"]["TEXCOORD_0"])
        self.skin_j = _accessor(js, binary, prim["attributes"]["JOINTS_0"]).astype(np.int64)
        self.skin_w = _accessor(js, binary, prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
        assert len(pos) == self.nr
        # raw base, ground offset already applied in the glb (see ARCHITECTURE "Units and ground")
        self.base = np.vstack([pos, joint_pts])  # (N,3) combined vertex space
        names = [b["name"] for b in rig["bones"]]
        self.bone_names = names
        idx = {n: i for i, n in enumerate(names)}
        self.bone_index = idx
        self.bones = [
            Bone(b["name"], idx[b["parent"]] if b["parent"] else -1, b["head"]["vert"], b["tail"]["vert"])
            for b in rig["bones"]
        ]
        self.parent = np.array([b.parent for b in self.bones])
        self.head_verts = np.array([b.head_vert for b in self.bones])
        self._build_morphs(open(os.path.join(BODY_DIR, "morphs.bin"), "rb").read())

    # ---------------------------------------------------------------- morphs
    def _build_morphs(self, blob: bytes) -> None:
        targets = self.manifest["targets"]
        n = self.base.shape[0]
        rows, cols, vals = [], [], []
        for ti, t in enumerate(targets):
            arr = np.frombuffer(blob, dtype=np.dtype([("i", "<u4"), ("d", "<f4", 3)]), count=t["count"], offset=t["byteOffset"])
            ii = arr["i"].astype(np.int64)
            for k in range(3):
                rows.append(ii * 3 + k)
                cols.append(np.full(len(ii), ti))
                vals.append(arr["d"][:, k].astype(np.float64))
        self.D = sp.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * 3, len(targets))
        )
        self.target_index = {t["id"]: i for i, t in enumerate(targets)}
        self.modifiers = self.manifest["modifiers"]
        self.mod_index = {m["id"]: i for i, m in enumerate(self.modifiers)}
        self.macro_vars = {v["id"]: v for v in self.manifest["macroVariables"]}

    def macro_target_weights(self, macro: dict[str, float]) -> np.ndarray:
        w = np.zeros(len(self.manifest["targets"]))
        bucket_w: dict[str, dict[str, float]] = {}
        for vid, v in self.macro_vars.items():
            val = float(np.clip(macro.get(vid, v["default"]), v["min"], v["max"]))
            bs = sorted(v["buckets"], key=lambda b: b["at"])
            out = {b["name"]: 0.0 for b in bs}
            if len(bs) == 1 or val <= bs[0]["at"]:
                out[bs[0]["name"]] = 1.0
            elif val >= bs[-1]["at"]:
                out[bs[-1]["name"]] = 1.0
            else:
                i = 0
                while val >= bs[i + 1]["at"]:
                    i += 1
                t = (val - bs[i]["at"]) / (bs[i + 1]["at"] - bs[i]["at"])
                out[bs[i]["name"]] = 1 - t
                out[bs[i + 1]["name"]] = t
            bucket_w[vid] = out
        for ti, t in enumerate(self.manifest["targets"]):
            cond = t.get("macroConditions")
            if not cond:
                continue
            p = 1.0
            for c in cond:
                p *= bucket_w[c["variable"]][c["bucket"]]
            w[ti] += p
        return w

    def modifier_target_weights(self, mods: dict[str, float]) -> np.ndarray:
        w = np.zeros(len(self.manifest["targets"]))
        for m in self.modifiers:
            v = float(np.clip(mods.get(m["id"], m["default"]), m["min"], m["max"]))
            if v > 0 and "incrTarget" in m:
                w[self.target_index[m["incrTarget"]]] += v
            elif v < 0 and "decrTarget" in m:
                w[self.target_index[m["decrTarget"]]] -= v
        return w

    def shape(self, macro: dict[str, float], mods: dict[str, float] | None = None, ground: bool = True) -> np.ndarray:
        w = self.macro_target_weights(macro) + self.modifier_target_weights(mods or {})
        pos = self.base + (self.D @ w).reshape(-1, 3)
        if ground:
            pos = pos - np.array([0.0, pos[: self.nr, 1].min(), 0.0])
        return pos

    def fit_modifier_columns(self) -> list[tuple[str, str, int]]:
        """(modifier id, 'incr'|'decr', target index) for every modifier used when fitting a scan."""
        out = []
        for m in self.modifiers:
            mid = m["id"]
            if not (mid.startswith(FIT_MOD_PREFIXES) or mid in FIT_MOD_EXTRA):
                continue
            for kind in ("incr", "decr"):
                key = f"{kind}Target"
                if key in m and (kind == "incr" or m["min"] < 0):
                    out.append((mid, kind, self.target_index[m[key]]))
        return out

    # ----------------------------------------------------------------- skeleton / skinning
    def rest_heads(self, positions: np.ndarray) -> np.ndarray:
        return positions[self.head_verts]

    def fk(self, heads: np.ndarray, local_rot: dict[str, np.ndarray] | np.ndarray, root_t: np.ndarray | None = None):
        """World rotations (B,3,3) and posed head positions (B,3). local_rot: name->rotation-matrix dict."""
        nb = len(self.bones)
        rots = np.tile(np.eye(3), (nb, 1, 1))
        posed = np.zeros((nb, 3))
        for i, b in enumerate(self.bones):
            q = np.eye(3)
            if isinstance(local_rot, dict):
                q = local_rot.get(b.name, np.eye(3))
            else:
                q = local_rot[i]
            if b.parent < 0:
                rots[i] = q
                posed[i] = heads[i] + (root_t if root_t is not None else 0.0)
            else:
                p = b.parent
                rots[i] = rots[p] @ q
                posed[i] = posed[p] + rots[p] @ (heads[i] - heads[p])
        return rots, posed

    def skin_matrices(self, heads, rots, posed):
        """Per-bone affine (R, t): v' = R (v - head) + posed_head."""
        return rots, posed - np.einsum("bij,bj->bi", rots, heads)

    def lbs(self, verts, joints4, weights4, heads, rots, posed):
        """Linear blend skinning of `verts` with 4-influence weights."""
        R, t = self.skin_matrices(heads, rots, posed)
        out = np.zeros_like(verts)
        for k in range(4):
            j = joints4[:, k]
            w = weights4[:, k : k + 1]
            out += w * (np.einsum("nij,nj->ni", R[j], verts) + t[j])
        return out

    def blended_affine(self, joints4, weights4, heads, rots, posed):
        """Per-vertex blended (A (n,3,3), b (n,3)) with v' = A v + b."""
        R, t = self.skin_matrices(heads, rots, posed)
        A = np.zeros((len(joints4), 3, 3))
        b = np.zeros((len(joints4), 3))
        for k in range(4):
            j = joints4[:, k]
            w = weights4[:, k]
            A += w[:, None, None] * R[j]
            b += w[:, None] * t[j]
        return A, b


def load_pose(pid: str) -> dict[str, np.ndarray]:
    d = json.load(open(os.path.join(POSE_DIR, f"{pid}.json"), encoding="utf8"))
    return {k: Rotation.from_quat(v).as_matrix() for k, v in d["bones"].items()}
