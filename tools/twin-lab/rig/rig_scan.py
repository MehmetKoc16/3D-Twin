"""Rig a scan mesh with our own 53-bone MakeHuman rig (fit + weight transfer + unpose).

python rig_scan.py <scan.glb> <out_dir> [--keep-pose] [--smooth N]

Writes <out_dir>/rigged.glb, rig_report.json, fitted_template.glb (debug: posed template) and the web-app package
twin.json + mh2twin.bin (see twin_export.py). The GLB has our bone names, identity rest rotations (node translation
only) and is in the MakeHuman A-pose rest frame, so apps/web/public/assets/poses/*.json apply unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh

from glbio import GlbScene, Prim, _split, read_glb, write_skinned_glb, write_static_glb
from mh import MHModel
from rigfit import Fitter, log, top4, transfer_weights
from twin_export import canonicalize_fit, write_twin_package
from bodyfix_solution import read_solution
from surface import cap_weights, constant_uv_faces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("out")
    ap.add_argument("--keep-pose", action="store_true", help="do not unpose the scan to the template rest pose")
    ap.add_argument("--fingers", choices=["keep", "merge"], default="keep", help="merge: fold finger weights into the hand bones (use when the scan has fused/blob fingers)")
    ap.add_argument("--reuse-fit", action="store_true", help="load <out>/fit.pkl instead of refitting")
    ap.add_argument("--cut-bridges", action="store_true", help="repair non-neighbouring skin influences while preserving the scan surface")
    ap.add_argument("--smooth", type=int, default=6)
    ap.add_argument("--samples", type=int, default=120000)
    ap.add_argument("--weights", choices=["transfer", "geodesic"], default="transfer", help="skin weights: MakeHuman transfer (default) or geometry-only geodesic voxel binding (baseline)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene = read_glb(args.scan)
    counts = [len(p.positions) for p in scene.prims]
    verts = np.vstack([p.positions for p in scene.prims]).astype(np.float64)
    offs = np.cumsum([0, *counts])
    faces = np.vstack([p.indices.astype(np.int64) + o for p, o in zip(scene.prims, offs)])
    log(f"scan: {len(verts)} verts, {len(faces)} tris, bbox {verts.min(0).round(3)} .. {verts.max(0).round(3)}")

    # welded copy for geometry work (UV seams / duplicate verts must not split the surface)
    key = np.round(verts * 1e5).astype(np.int64)
    _u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    uverts = verts[first]
    ufaces = inv[faces]
    ufaces = ufaces[(ufaces[:, 0] != ufaces[:, 1]) & (ufaces[:, 1] != ufaces[:, 2]) & (ufaces[:, 0] != ufaces[:, 2])]
    wm = trimesh.Trimesh(uverts, ufaces, process=False)
    flipped = False
    if wm.volume < 0 and wm.is_watertight:
        log("mesh is inside-out (negative volume), flipping normals for fitting")
        wm.invert()
        flipped = True
    pts, fid = trimesh.sample.sample_surface(wm, args.samples, seed=3)
    nrm = wm.face_normals[fid]

    model = MHModel()
    # bodyfix marks scans whose hands it cut off (the app draws MakeHuman hands) in the GLB's asset extras
    extras = _split(Path(args.scan).read_bytes())[0].get("asset", {}).get("extras", {})
    scan_hands_removed = bool(extras.get("dtScanHandsRemoved"))
    bodyfix = read_solution(extras, model)
    pkl = os.path.join(args.out, "fit.pkl")
    if bodyfix is not None:
        log("bodyfix solution found: keeping shape, fitting pose and translation to the corrected scan")
        # A cached unconstrained fit must never override the embedded solved body.
        res = Fitter(model, pts, nrm, fixed_shape=(bodyfix["fittedMacros"], bodyfix["fittedModifiers"])).run()
        with open(pkl, "wb") as fh:
            pickle.dump(res, fh)
    elif args.reuse_fit and os.path.exists(pkl):
        with open(pkl, "rb") as fh:
            res = pickle.load(fh)
    else:
        res = Fitter(model, pts, nrm).run()
        with open(pkl, "wb") as fh:
            pickle.dump(res, fh)
    # the browser rebuilds the body from the NET modifier values: continue with exactly that body (twin_export.py)
    canonicalize_fit(model, res, quantize=bodyfix is None)
    log(f"canonical fit: shape deviation from the raw fit {res.stats['canonical_shape_dev_mm']}")

    heads = model.rest_heads(res.rest_positions)
    rots, posed = model.fk(heads, {b: _mat(v) for b, v in res.pose_rotvec.items()}, res.root_t)
    W, tstats = transfer_weights(model, res.posed_vertices, uverts, wm.faces, smooth_iters=args.smooth)
    if args.fingers == "merge":
        for side in ("l", "r"):
            hand = model.bone_index[f"hand_{side}"]
            for name, bi in model.bone_index.items():
                if name.endswith("_" + side) and name.split("_")[0] in ("index", "middle", "ring", "pinky", "thumb"):
                    W[:, hand] += W[:, bi]
                    W[:, bi] = 0.0
    if scan_hands_removed:
        from bridges import heal_islands, reassign_hand_weights

        cols = [bi for name, bi in model.bone_index.items()
                if name.endswith(("_l", "_r")) and name.split("_")[0] in ("hand", "thumb", "index", "middle", "ring", "pinky")]
        W, nhand = reassign_hand_weights(W, wm.faces, cols)
        log(f"scan hands were removed (bodyfix): {nhand} hand-weighted vertices took their neighbours' weights")
        W, nisland = heal_islands(W, wm.faces, model.bone_names, list(model.parent))
        log(f"healed {nisland} vertices of small patches weighted to skeleton-distant bones")
    caps = np.concatenate([constant_uv_faces(p.indices, p.uv) for p in scene.prims])
    W, capstats = cap_weights(W, faces, caps, inv)
    log(f"cap weights: {capstats}")
    jn, jw = top4(W)  # per unique vertex
    if args.cut_bridges:
        # drop influences of skeleton-distant bones (hand/thigh blends at contact) BEFORE unposing: an inverse blend of
        # bones with very different transforms is nearly singular and throws vertices far away
        from bridges import snap_incompatible

        jw, nsnap = snap_incompatible(jn, jw, model.bone_names, list(model.parent))
        log(f"snapped {nsnap} vertices with skeleton-distant influences")
    log(f"weight transfer: nn distance median {tstats['nn_dist_median_cm']:.2f} cm, p99 {tstats['nn_dist_p99_cm']:.2f} cm")

    if args.keep_pose:
        out_heads = posed
        rest_uverts = uverts
    else:
        A, b = model.blended_affine(jn.astype(np.int64), jw.astype(np.float64), heads, rots, posed)
        rest_uverts = np.linalg.solve(A, (uverts - b)[..., None])[..., 0]
        out_heads = heads
    # ground: lowest vertex of the unposed mesh on y = 0
    shift = np.array([0.0, rest_uverts[:, 1].min(), 0.0])
    rest_uverts = rest_uverts - shift
    out_heads = out_heads - shift
    tails = res.rest_positions[[b.tail_vert for b in model.bones]] - shift
    if args.weights == "geodesic":
        from geoskin import geodesic_weights

        if args.keep_pose:
            raise SystemExit("--weights geodesic needs the unposed mesh")
        log("geodesic voxel binding (baseline)")
        jn, jw = top4(geodesic_weights(rest_uverts, wm.faces, out_heads, tails))
    # centre x/z: template x-symmetry (pelvis head at x = 0) is kept, so only the y shift is applied
    rest_verts = rest_uverts[inv]

    un = trimesh.Trimesh(rest_uverts, wm.faces, process=False).vertex_normals * (-1.0 if flipped else 1.0)  # welded normals: no seam creases
    prims_out, sj, sw = [], [], []
    for p, o, c in zip(scene.prims, offs[:-1], counts):
        v = rest_verts[o : o + c]
        prims_out.append(replace(p, positions=v.astype(np.float32), normals=un[inv[o : o + c]].astype(np.float32)))
        sj.append(jn[inv[o : o + c]])
        sw.append(jw[inv[o : o + c]])
    cut = []
    if args.cut_bridges:
        from bridges import cut_bridges

        for i, p in enumerate(prims_out):
            idx, srcv, cj, cw, cstats = cut_bridges(p.indices, sj[i], sw[i], model.bone_names,
                                                  list(model.parent), mode="preserve")
            log(f"bridge cut {p.name}: {cstats}")
            prims_out[i] = replace(p, positions=p.positions[srcv], indices=idx,
                                   normals=p.normals[srcv] if p.normals is not None else None,
                                   uv=p.uv[srcv] if p.uv is not None else None)
            sj[i], sw[i] = cj, cw
            cut.append(cstats)
    joints = [
        {"name": bn, "parent": model.bone_names[bp.parent] if bp.parent >= 0 else None, "head": out_heads[i]}
        for i, (bn, bp) in enumerate(zip(model.bone_names, model.bones))
    ]
    out_scene = replace(scene, prims=prims_out)
    write_skinned_glb(os.path.join(args.out, "rigged.glb"), out_scene, joints, sj, sw)
    if args.keep_pose:
        log("--keep-pose: the mesh is not in the rest frame, twin.json / mh2twin.bin are not written")
    elif len(prims_out) != 1:
        raise SystemExit("twin.json export needs a single-primitive mesh (the app binds one SkinnedMesh)")
    else:
        opts = {"fingers": args.fingers, "cutBridges": args.cut_bridges, "smooth": args.smooth, "weights": args.weights}
        twin = write_twin_package(
            args.out,
            model,
            res,
            prims_out[0].positions.astype(np.float64),
            prims_out[0].normals.astype(np.float64) if prims_out[0].normals is not None else None,
            float(shift[1]),
            {"scan": os.path.basename(args.scan), "options": opts},
            bodyfix=bodyfix,
        )
        log(
            f"twin.json: mapping {twin['fit']['mappingCm']} cm; body lowest vertex at y = "
            f"{twin['fit']['bodyLowestYM'] * 100:.2f} cm (twin frame); measurements {twin['measurementsCm']}"
        )

    dbg = Prim(res.posed_vertices.astype(np.float32), model.faces.astype(np.uint32), None, None, 0, "fitted_template")
    write_static_glb(os.path.join(args.out, "fitted_template.glb"), GlbScene([dbg]))
    report = {
        "scan": os.path.basename(args.scan),
        "vertices": int(len(verts)),
        "fit": res.stats,
        "macro": res.macro,
        "mods": {k: round(v, 3) for k, v in res.mods.items() if abs(v) > 0.02},
        "transfer": tstats,
        "caps": capstats,
        "bridges": cut,
        "unposed": not args.keep_pose,
        "height_m": float(rest_verts[:, 1].max()),
    }
    json.dump(report, open(os.path.join(args.out, "rig_report.json"), "w"), indent=1)
    np.save(os.path.join(args.out, "rest_heads.npy"), out_heads)
    log(f"done -> {os.path.join(args.out, 'rigged.glb')}")


def _mat(rv: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(rv).as_matrix()


if __name__ == "__main__":
    main()
