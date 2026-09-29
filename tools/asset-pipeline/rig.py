"""MPFB2 `game_engine` rig preset (CC0 JSON data) -> joint points + rig.json bones.

Every CUBE joint reference of the preset becomes a "joint point" (centroid of the joint helper cube). Bones are
world-aligned (identity rest rotation, roll 0); the rest pose is MakeHuman's relaxed A-pose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import DM_TO_M
from mh_obj import BaseMesh


def load_rig(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _refs(rig: dict):
    for name in sorted(rig):
        b = rig[name]
        for end in ("head", "tail"):
            yield name, end, b[end]


def joint_point_names(rig: dict) -> list[str]:
    """Distinct helper cubes used by the rig, sorted by name (stable order = joint point order)."""
    return sorted({r["cube_name"] for _, _, r in _refs(rig) if r["strategy"] == "CUBE"})


def bone_order(rig: dict) -> list[str]:
    """Parents before children: root first, then depth-first with siblings sorted by name."""
    children: dict[str, list[str]] = {}
    roots = []
    for name in sorted(rig):
        parent = rig[name]["parent"]
        if parent:
            children.setdefault(parent, []).append(name)
        else:
            roots.append(name)
    if len(roots) != 1:
        raise ValueError(f"expected a single root bone, got {roots}")
    out: list[str] = []

    def visit(n: str) -> None:
        out.append(n)
        for c in children.get(n, []):
            visit(c)

    visit(roots[0])
    return out


def cube_vertices(mesh: BaseMesh, names: list[str]) -> np.ndarray:
    arr = np.array([mesh.groups[n] for n in names], dtype=np.int64)
    if arr.shape[1] != 8:
        raise ValueError("joint cubes must have 8 vertices")
    return arr


def joint_positions_m(mesh: BaseMesh, cubes: np.ndarray, offset_y: float) -> np.ndarray:
    """(J, 3) joint point positions in meters in the raw (rest) base mesh, feet-grounded via offset_y."""
    pts = mesh.verts[cubes].mean(axis=1) * DM_TO_M
    pts[:, 1] += offset_y
    return pts


def build_bones(
    rig: dict,
    mesh: BaseMesh,
    names: list[str],
    render_count: int,
    offset_y: float,
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (rig.json bones, head positions, tail positions) in bone order; positions in meters."""
    jindex = {n: j for j, n in enumerate(names)}
    cubes = cube_vertices(mesh, names)
    pts = joint_positions_m(mesh, cubes, offset_y)

    def ref(r: dict):
        if any(abs(float(x)) > 0 for x in r.get("offset", [0, 0, 0])):
            raise ValueError("non-zero joint offsets are not supported")
        if r["strategy"] == "CUBE":
            j = jindex[r["cube_name"]]
            return {"strategy": "VERTEX", "vert": render_count + j}, pts[j]
        if r["strategy"] == "MEAN":
            return None, None  # resolved below (only the Root tail: a 2-vertex mean inside the ground cube)
        raise ValueError(f"unsupported strategy {r['strategy']}")

    order = bone_order(rig)
    first_child = {}
    for n in order:
        parent = rig[n]["parent"]
        if parent and parent not in first_child:
            first_child[parent] = n

    bones, heads, tails = [], {}, {}
    for name in order:
        b = rig[name]
        h, hp = ref(b["head"])
        if h is None:
            raise ValueError(f"{name}: MEAN head is not supported")
        t, tp = ref(b["tail"])
        if t is None:
            # MPFB2 defines the Root tail as a tiny offset inside the ground cube. It would not follow the leg-height
            # morphs (those move the ground joint), so the Root bone instead spans ground -> first child (pelvis).
            child = rig[first_child[name]]["head"]
            if child["strategy"] != "CUBE":
                raise ValueError("unsupported Root tail")
            t, tp = ref(child)
        bones.append({"name": name, "parent": b["parent"] or None, "head": h, "tail": t, "roll": 0})
        heads[name], tails[name] = hp, tp
    return bones, heads, tails
