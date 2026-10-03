"""Numerical closedness and T-pose armpit checks, without decoding or viewing images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from glbio import _accessor, _node_matrix, _split, read_glb
from mh import MHModel, load_pose


def armpit_zone(points: np.ndarray) -> np.ndarray:
    """The same rest-frame region as refine's webbing_stats: y 1.10..1.38 m, |x| >= 0.10 m."""
    return (points[:, 1] >= 1.10) & (points[:, 1] <= 1.38) & (np.abs(points[:, 0]) >= .10)


def boundary_stats(vertices: np.ndarray, faces: np.ndarray, tolerance: float) -> dict:
    _, inverse = np.unique(np.round(vertices / tolerance).astype(np.int64), axis=0, return_inverse=True)
    f = inverse.ravel()[faces]
    valid = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[valid]
    edges = np.sort(np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1), axis=2)
    _, inverse_edge, counts = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True, return_counts=True)
    boundary = (counts[inverse_edge].reshape(-1, 3) == 1).any(axis=1)
    zone = armpit_zone(vertices[faces[valid]].mean(axis=1))
    return {"boundaryEdges": int((counts == 1).sum()), "boundaryFaces": int(boundary.sum()),
            "armpitBoundaryFaces": int((boundary & zone).sum())}


def inspect(path: Path) -> dict:
    scene = read_glb(str(path))
    if len(scene.prims) != 1:
        raise ValueError("mesh QA requires a single primitive")
    p = scene.prims[0]
    v, f = p.positions.astype(np.float64), p.indices.astype(np.int64)
    result = {"file": path.name, "vertices": len(v), "faces": len(f),
              "boundary10um": boundary_stats(v, f, 1e-5),
              "boundary1_5mm": boundary_stats(v, f, .0015),
              "assetExtras": sorted(scene.extras)}
    js, binary = _split(path.read_bytes())
    if js.get("skins"):
        model = MHModel()
        skin = js["skins"][0]
        if [js["nodes"][i].get("name") for i in skin["joints"]] != model.bone_names:
            raise ValueError("skin joint order must match the MakeHuman rig")
        parents = {c: i for i, node in enumerate(js["nodes"]) for c in node.get("children", [])}

        def world(index: int) -> np.ndarray:
            local = _node_matrix(js["nodes"][index])
            return world(parents[index]) @ local if index in parents else local

        heads = np.array([world(i)[:3, 3] for i in skin["joints"]])
        attrs = js["meshes"][0]["primitives"][0]["attributes"]
        joints = _accessor(js, binary, attrs["JOINTS_0"]).astype(np.int64)
        weights = _accessor(js, binary, attrs["WEIGHTS_0"]).astype(np.float64)
        rots, posed_heads = model.fk(heads, load_pose("t-pose"))
        posed = model.lbs(v, joints, weights, heads, rots, posed_heads)
        _, first, inverse = np.unique(np.round(v * 1e5).astype(np.int64), axis=0,
                                      return_index=True, return_inverse=True)
        wf = inverse.ravel()[f]
        edges = np.unique(np.sort(np.vstack([wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]]), axis=1), axis=0)
        rest, animated = v[first], posed[first]
        l0 = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
        l1 = np.linalg.norm(animated[edges[:, 0]] - animated[edges[:, 1]], axis=1)
        zone = armpit_zone(rest[edges].mean(axis=1)) & (l0 > 1e-4)
        stretch = l1[zone] / l0[zone]
        result["tPoseArmpits"] = {"edges": int(zone.sum()), "over2": int((stretch > 2).sum()),
                                    "over3": int((stretch > 3).sum()),
                                    "p99": float(np.percentile(stretch, 99)) if len(stretch) else 0,
                                    "max": float(stretch.max()) if len(stretch) else 0}
        result["tPoseBoundary10um"] = boundary_stats(posed, f, 1e-5)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    personal = Path(__file__).resolve().parents[3] / "user-data"
    if args.out and args.mesh.resolve().is_relative_to(personal) and not args.out.resolve().is_relative_to(personal):
        raise ValueError("Personal diagnostics must stay under user-data/")
    result = inspect(args.mesh)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(encoded, encoding="utf8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
