"""Asset pipeline entry point:  python tools/asset-pipeline/build.py [--out DIR] [--no-fetch] [--check]

Raw MakeHuman + MPFB2 CC0 data (pinned commits, .cache/) -> base.glb, morphs.bin, manifest.json, rig.json,
measures.json, face-map.json in apps/web/public/assets/body/.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import DEFAULT_OUT_DIR, OUTPUT_FILES, RIG_NAME, SOURCES, VENV_DIR  # noqa: E402

REQUIRED_MODULES = ("numpy", "pygltflib")


def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def ensure_environment() -> None:
    """Use the local .venv interpreter when deps are missing here, else print the setup command."""
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        return
    venv_py = _venv_python()
    if venv_py.exists() and Path(sys.executable).resolve() != venv_py.resolve():
        print(f"[build] missing {', '.join(missing)} in this interpreter; re-running with {venv_py}")
        sys.exit(subprocess.call([str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]]))
    py = "py -3.12" if sys.platform == "win32" else "python3"
    activate = r".venv\Scripts\python.exe" if sys.platform == "win32" else ".venv/bin/python"
    print(
        f"[build] missing Python packages: {', '.join(missing)}\n"
        "Create the pipeline virtualenv once (from the repository root):\n"
        f"  {py} -m venv tools/asset-pipeline/.venv\n"
        f"  tools/asset-pipeline/{activate} -m pip install -r tools/asset-pipeline/requirements.txt\n"
        "then re-run `npm run assets:build`.",
        file=sys.stderr,
    )
    sys.exit(2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_all(out_dir: Path, verbose: bool = True) -> dict:
    import numpy as np

    import face_map as face_map_mod
    import gltf_writer
    import macro
    import measures as measures_mod
    import mh_obj
    import rig as rig_mod
    import targets as targets_mod
    import weights as weights_mod
    import writers
    from config import CACHE_DIR, DM_TO_M, MH_DATA, MPFB_DATA

    log = print if verbose else (lambda *a, **k: None)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = mh_obj.load_base_mesh(MH_DATA / "3dobjs" / "base.obj")
    R = mesh.render_count
    rig_dir = MPFB_DATA / "rigs" / "standard"
    rig = rig_mod.load_rig(rig_dir / f"rig.{RIG_NAME}.json")
    joint_names = rig_mod.joint_point_names(rig)
    cubes = rig_mod.cube_vertices(mesh, joint_names)
    J = len(joint_names)
    log(f"[build] render vertices {R}, joint points {J}")

    # --- targets -> morphs.bin -------------------------------------------------------------------------------
    catalog = targets_mod.build_catalog()
    packer = targets_mod.TargetPacker(mesh, cubes)
    blobs: list[bytes] = []
    target_entries: list[dict] = []
    kept_ids: set[str] = set()
    offset = 0
    for spec in catalog.targets:
        entries = packer.pack(spec)
        if len(entries) == 0:
            continue  # empty upstream target (e.g. average/average) -> no effect, not shipped
        blobs.append(entries.tobytes())
        target_entries.append(writers.target_entry(spec.id, spec.group, offset, len(entries), spec.macro_conditions))
        kept_ids.add(spec.id)
        offset += len(entries) * targets_mod.ENTRY_DTYPE.itemsize
    ids = [t["id"] for t in target_entries]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate target ids")
    modifiers = []
    for m in catalog.modifiers:
        if m.decr and m.decr not in kept_ids:
            m.decr = None
        if m.incr and m.incr not in kept_ids:
            m.incr = None
        if not (m.decr or m.incr):
            raise RuntimeError(f"modifier {m.id} has no targets")
        modifiers.append(writers.modifier_entry(m))
    morphs_bytes = b"".join(blobs)
    log(f"[build] {len(target_entries)} targets, {len(modifiers)} modifiers, morphs.bin {len(morphs_bytes)} bytes")

    # --- ground offset: neutral body (macro defaults) gets min y = 0 ------------------------------------------
    base_raw = np.concatenate([mesh.verts[mesh.render_mh], mesh.verts[cubes].mean(axis=1)]) * DM_TO_M
    prov_stub = {"source": "", "details": {}}
    probe = writers.build_manifest(
        render_count=R, joint_names=joint_names, joint_positions=base_raw[R:], macro_variables=macro.MACRO_VARIABLES,
        targets=target_entries, modifiers=modifiers, provenance=prov_stub,
    )
    neutral = macro.MorphSet(probe, morphs_bytes, base_raw).positions()
    offset_y = round(float(-neutral[:R, 1].min()), 6)
    base = base_raw.copy()
    base[:, 1] += offset_y
    neutral[:, 1] += offset_y
    log(f"[build] ground offset y = {offset_y:.6f} m (neutral body min y -> 0)")

    # --- rig + skin -----------------------------------------------------------------------------------------
    bones, heads, tails = rig_mod.build_bones(rig, mesh, joint_names, R, offset_y)
    bone_names = [b["name"] for b in bones]
    joints, wts = weights_mod.load_weights(rig_dir / f"weights.{RIG_NAME}.json", bone_names, mesh)

    # --- measures ---------------------------------------------------------------------------------------------
    nip_idx, nip_delta = targets_mod.parse_target(targets_mod.TARGET_DIR / "breast" / "nipple-point-incr.target")
    ctx = measures_mod.Context(mesh, neutral, joint_names, bone_names, joints, wts)
    ctx.nipple_y = measures_mod.nipple_height(mesh, neutral, nip_idx, nip_delta)
    measure_defs, landmarks = measures_mod.derive(ctx)
    modifier_ids = {m["id"] for m in modifiers}
    for md in measure_defs:
        for d in md["drivers"]:
            if d not in modifier_ids:
                raise RuntimeError(f"measure {md['id']} driver {d} is not an exported modifier")

    # --- base.glb ---------------------------------------------------------------------------------------------
    pos_render = base[:R]
    normals = mh_obj.vertex_normals(pos_render, mesh.tris, mesh.render_mh)
    gltf_writer.write_glb(
        out_dir / "base.glb", pos_render, normals, mesh.render_uv, joints, wts, mesh.tris,
        bone_names, [b["parent"] for b in bones], np.stack([heads[n] for n in bone_names]),
    )

    # --- JSON + bin -------------------------------------------------------------------------------------------
    (out_dir / "morphs.bin").write_bytes(morphs_bytes)
    src = SOURCES
    provenance = {
        "source": (
            f"MakeHuman base mesh and targets (makehumancommunity/makehuman@{src['mh']['sha']}), "
            f"MPFB2 game_engine rig and weights (makehumancommunity/mpfb2@{src['mpfb2']['sha']}); CC0-1.0 data"
        ),
        "details": {
            "makehuman": {"repo": "makehumancommunity/makehuman", "sha": src["mh"]["sha"]},
            "mpfb2": {"repo": "makehumancommunity/mpfb2", "sha": src["mpfb2"]["sha"]},
            "rig": RIG_NAME,
            "groundOffsetY": offset_y,
            "restPose": "MakeHuman A-pose (world-aligned bones, identity rest rotation)",
            "vertexIndexSpace": "render vertices of base.glb followed by jointPoints",
        },
    }
    manifest = writers.build_manifest(
        render_count=R, joint_names=joint_names, joint_positions=base[R:], macro_variables=macro.MACRO_VARIABLES,
        targets=target_entries, modifiers=modifiers, provenance=provenance,
    )
    writers.write_json(out_dir / "manifest.json", manifest)
    writers.write_json(out_dir / "rig.json", {"version": 1, "bones": bones})
    writers.write_json(out_dir / "measures.json", {"version": 1, "measures": measure_defs})

    # --- face-map.json ----------------------------------------------------------------------------------------
    morphset = macro.MorphSet(manifest, morphs_bytes, base)
    face = face_map_mod.build_face_map(mesh, morphset.positions(), morphset, manifest)
    writers.write_json(out_dir / "face-map.json", face.data)
    if verbose:
        rep = face.report
        log(
            f"[build] face-map: {rep['bound']} landmarks bound, max distance {rep['max_distance_mm']:.1f} mm "
            f"(xy {rep['max_xy_distance_mm']:.1f} mm), single UV island {rep['seams']['single_island']}, "
            f"flipped faces {len(rep['flipped_faces'])}, fitModifiers {len(rep['fit_modifiers'])}"
        )
        face_map_mod.write_debug_images(CACHE_DIR / "debug" / "face_map_uv.png", mesh, face)

    hashes = {f: sha256(out_dir / f) for f in OUTPUT_FILES}
    if verbose:
        total = 0
        for f in OUTPUT_FILES:
            size = (out_dir / f).stat().st_size
            total += size
            print(f"[build] {f:15s} {size:>10,d} bytes  sha256 {hashes[f][:16]}")
        print(f"[build] total {total:,d} bytes ({total / 1e6:.2f} MB)")
    return hashes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    ap.add_argument("--no-fetch", action="store_true", help="do not touch the network; use .cache as is")
    ap.add_argument("--check", action="store_true",
                    help="build into a temp dir and verify the files in --out are byte-identical")
    args = ap.parse_args()

    ensure_environment()
    import fetch

    if args.no_fetch:
        fetch.ensure_present()
    else:
        fetch.fetch_all()

    if args.check:
        with tempfile.TemporaryDirectory(prefix="dt-assets-") as tmp:
            fresh = build_all(Path(tmp), verbose=False)
        bad = []
        for f in OUTPUT_FILES:
            existing = args.out / f
            if not existing.exists() or sha256(existing) != fresh[f]:
                bad.append(f)
        if bad:
            print(f"[check] MISMATCH in {args.out}: {', '.join(bad)} (run `npm run assets:build`)", file=sys.stderr)
            return 1
        print(f"[check] OK: {args.out} matches a fresh build")
        return 0

    build_all(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
