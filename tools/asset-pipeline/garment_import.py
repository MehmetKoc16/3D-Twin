"""Import a licensed static garment into the current wardrobe contract, for offline review."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Catalogue licences the app accepts for importer garments (GarmentLicense minus the runtime-only `user`).
LICENSES = ("CC0-1.0","CC-BY-3.0","CC-BY-4.0")


def require_venv():
    if Path(sys.prefix).resolve() != (HERE/".venv").resolve():
        raise RuntimeError("Use tools/asset-pipeline/.venv/Scripts/python.exe; this tool never selects another Python")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path,help="GLB/glTF, OBJ or authored MHCLO; FBX must be converted first")
    p.add_argument("--mhclo",type=Path,help="Authored MHCLO for an OBJ source (otherwise auto-detect its companion)")
    p.add_argument("--ignore-mhclo",action="store_true",help="Explicitly refit an OBJ instead of using its companion")
    p.add_argument("--id",required=True)
    p.add_argument("--category",choices=("top","bottom","shoes"),required=True)
    p.add_argument("--kind",choices=("tshirt","longsleeve","sweatshirt","pants","jeans","shorts","sneakers","shoes","boots"),required=True)
    p.add_argument("--license",choices=LICENSES,required=True,
                   help="Exact licence (the app's GarmentLicense catalogue values); CC-BY versions are never relabelled. "
                        "Declarations do not replace the lead's source/licence review")
    p.add_argument("--attribution",required=True,help="Title, author, source URL, exact licence, and changes")
    p.add_argument("--source-url",default="")
    p.add_argument("--node",action="append",help="GLB node name to include (repeatable; isolate clothing from a full character)")
    p.add_argument("--exclude-material",action="append",help="GLB material name to exclude, e.g. Skin (repeatable)")
    p.add_argument("--skin-pose",choices=("evaluated","bind"),default="evaluated",
                   help="Evaluate default skin pose, or use the mesh's authored bind pose with node transforms")
    p.add_argument("--label-en"); p.add_argument("--label-tr")
    p.add_argument("--layer",type=int,default=1)
    p.add_argument("--out",type=Path,default=HERE/".cache/garment-import-out")
    p.add_argument("--body",type=Path,default=ROOT/"apps/web/public/assets/body")
    p.add_argument("--pose",choices=("auto","a","t"),default="auto")
    p.add_argument("--source-body-height",type=float,help="Authoring body's full height in source coordinate units")
    p.add_argument("--transform",type=Path,help="Small JSON sidecar; default <source>.import.json when present")
    p.add_argument("--max-tris",type=int,default=30000)
    p.add_argument("--subdivide",type=int,choices=(0,1,2,3),default=0,
                   help="Linear midpoint subdivision before fitting; useful for coarse facets crossing the body")
    p.add_argument("--texture-max",type=int,default=2048)
    p.add_argument("--clearance-mm",type=float,help="Default top 3, bottom 3, shoes 4; permitted range 2..4")
    p.add_argument("--layer-gap-mm",type=float,default=3,help="Additional clearance per outer layer (default 3 mm)")
    p.add_argument("--shell",choices=("outer","keep"),default="outer",help="Strip close opposing inner walls, or keep a reviewed single shell")
    p.add_argument("--rigid-sole-height-mm",type=float,default=35,help="Shoe sole/heel region above its lowest point; written to <id>.rigid.json (app rigid-sole format v1)")
    p.add_argument("--mirror-pair",action="store_true",help="Mirror an aligned single left shoe across X=0 to create a pair")
    p.add_argument("--no-delete-mask",action="store_true")
    p.add_argument("--validate",action="store_true",help="Run the actual avatar-core solver/grading and four posed containment checks")
    p.add_argument("--compare-original",type=Path,help="Original app .bind.bin; uses its sibling index.json entry")
    return p


def import_garment(args):
    import re
    import numpy as np
    import garments
    from garmentimport.body import align,load_body
    from garmentimport.body import skin
    from garmentimport.geometry import RIGID_REVIEW_NOTES,binding,delete_mask,fit,reconstruct,scale_refs,outer_shell,rigid_shoe_parts,skin_weights
    from garmentimport.mesh_io import atlas,clean,decimate,load_parts,write_glb,refine
    from garmentimport.pose_safety import POSES,pose_delete_mask,regional_clearance,stable_binding
    from garmentimport.untangle import untangle
    from garmentimport.validation import read_binding,validate
    from garmentimport.visibility import triangle_pairs

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}",args.id):
        raise ValueError("id must contain only lowercase ASCII letters, digits, underscores or hyphens (1..80 characters)")
    kinds = {"top":{"tshirt","longsleeve","sweatshirt"},"bottom":{"pants","jeans","shorts"},"shoes":{"sneakers","shoes","boots"}}
    if args.kind not in kinds[args.category]:
        raise ValueError("kind does not belong to category")
    if args.mirror_pair and args.category!="shoes":
        raise ValueError("mirror-pair is only supported for shoes")
    if not args.attribution.strip():
        raise ValueError("A nonempty attribution is required for every source")
    if args.max_tris<4 or not 16<=args.texture_max<=2048 or args.layer<0:
        raise ValueError("Invalid triangle/texture/layer limits")
    clearance = args.clearance_mm if args.clearance_mm is not None else {"top":3,"bottom":3,"shoes":4}[args.category]
    if not np.isfinite(clearance) or not 2<=clearance<=4:
        raise ValueError("clearance-mm must be between 2 and 4")
    if not np.isfinite(args.layer_gap_mm) or not 0<=args.layer_gap_mm<=10:
        raise ValueError("layer-gap-mm must be between 0 and 10")
    if not np.isfinite(args.rigid_sole_height_mm) or not 0<args.rigid_sole_height_mm<=150:
        raise ValueError("rigid-sole-height-mm must be between 0 and 150")
    clearance += max(0,args.layer-1)*args.layer_gap_mm
    output = args.out.resolve()
    if output.is_relative_to((ROOT/"apps").resolve()) or output.is_relative_to((ROOT/"packages").resolve()):
        raise ValueError("Importer outputs must stay outside shipped app/package assets; use the review cache")
    body = load_body(args.body.resolve())
    print(f"garment-import: {args.id}: loading source",file=sys.stderr,flush=True)
    native_path = args.mhclo or (args.source if args.source.suffix.lower()==".mhclo" else None)
    if args.ignore_mhclo and native_path:
        raise ValueError("ignore-mhclo conflicts with an explicit MHCLO source")
    if not native_path and args.source.suffix.lower()==".obj" and not args.ignore_mhclo:
        import mhclo
        matches = [p for p in args.source.parent.glob("*.mhclo") if
                   mhclo.parse_mhclo(p.read_text(encoding="utf-8",errors="replace")).header.get("obj_file")==args.source.name]
        if len(matches)>1:
            raise ValueError("Several MHCLO companions match; select one with --mhclo")
        native_path = matches[0] if matches else None
    native_bind = native_points = native_refs = None
    native_report = None
    native_delete = None
    if native_path:
        from garmentimport.native import load_native
        if args.transform or args.source.with_suffix(".import.json").exists() or args.pose=="t" or args.source_body_height or args.node or args.exclude_material or args.skin_pose!="evaluated":
            raise ValueError("Authored MHCLO is already in hm08 rest pose; transforms, reposing and source selection are incompatible")
        parts,native_bind,native_points,native_refs,native_report,native_delete = load_native(native_path.resolve(),body)
        if args.source.suffix.lower()==".obj" and native_report["objFile"]!=args.source.name:
            raise ValueError("MHCLO obj_file does not match the requested OBJ")
    else:
        parts = load_parts(args.source.resolve(),args.node,args.exclude_material,skin_pose=args.skin_pose)
    parts = [clean(p) for p in parts]
    material_warnings = sorted({warning for p in parts for warning in p.material.get("warnings",[])})
    input_count = {"vertices":sum(len(p.positions) for p in parts),"triangles":sum(len(p.faces) for p in parts),"parts":len(parts)}
    # Simplify before texture baking but after scale alignment, for meter-space tolerances.
    original_positions = np.concatenate([p.positions for p in parts])
    transform_path = args.transform or args.source.with_suffix(".import.json")
    sidecar = json.loads(transform_path.read_text(encoding="utf-8")) if transform_path.exists() else {}
    if args.transform and not args.transform.exists():
        raise ValueError(f"Transform sidecar not found: {args.transform}")
    if native_path:
        aligned = original_positions
        alignment = {"method":"authored MHCLO on runtime neutral hm08", "scale":.1, "pose":"a", "warnings":[]}
    else:
        aligned,alignment = align(original_positions,body,args.category,args.pose,sidecar,args.source_body_height)
        alignment["sourceSkinPose"] = args.skin_pose
    start = 0
    for p in parts:
        p.positions = aligned[start:start+len(p.positions)]; start += len(p.positions)
    if args.mirror_pair:
        if native_path:
            raise ValueError("mirror-pair is incompatible with native MHCLO; author a pair first")
        if any(p.positions[:,0].min()<=0 for p in parts):
            raise ValueError("mirror-pair requires a single shoe aligned wholly on the left (+X) foot")
        from garmentimport.mesh_io import Part
        import copy
        mirrored = []
        for p in parts:
            points = p.positions.copy(); points[:,0] *= -1
            mirrored.append(Part(points,p.faces[:,::-1].copy(),p.uv.copy(),copy.deepcopy(p.material)))
        parts += mirrored
        alignment["mirroredPair"] = True
    shell_report = {"removedInnerTriangles":0,"method":"manual keep"}
    print(f"garment-import: {args.id}: shell selection and simplification",file=sys.stderr,flush=True)
    if args.shell=="outer":
        parts,shell_report = outer_shell(parts,body.positions[:body.render_count],body.faces)
    parts = decimate(parts,args.max_tris)
    parts = refine(parts,args.subdivide,args.max_tris)
    pos,faces,uv,channels,alpha_mode,base_color = atlas(parts,args.texture_max)
    before_fit = pos.copy()
    print(f"garment-import: {args.id}: clearance fitting ({len(pos)} vertices)",file=sys.stderr,flush=True)
    clearances = regional_clearance(pos,body,clearance/1000) if args.category=="top" else clearance/1000
    pos,fit_report = fit(pos,faces,body.positions[:body.render_count],body.faces,clearances)
    refs = native_refs or scale_refs(body.positions[:body.render_count])
    if native_bind is not None:
        from scipy.spatial import cKDTree
        distance,mapping = cKDTree(native_points).query(before_fit)
        bind = tuple(a[mapping].copy() for a in native_bind)
        changed = distance>1e-7
        if changed.any():
            replacement = binding(before_fit[changed],body.positions[:body.render_count],body.faces)
            bind[0][changed],bind[1][changed] = replacement[:2]
        scales = np.array([abs(body.positions[refs[k][0],i]-body.positions[refs[k][1],i])/refs[k][2] for i,k in enumerate("xyz")])
        bind[2][:] = (pos-(body.positions[bind[0]]*bind[1][...,None]).sum(axis=1))/scales
        native_report["rowsReboundAfterSimplification"] = int(changed.sum())
        native_report["authoredMappingMaxMm"] = float(distance.max()*1000)
    else:
        bind = binding(pos,body.positions[:body.render_count],body.faces)
    stability_report = None
    untangle_report = None
    if args.category=="top":
        bind,stability_report = stable_binding(pos,faces,bind,body,refs)
        print(f"garment-import: {args.id}: bounded four-pose shell separation",file=sys.stderr,flush=True)
        separated,untangle_report = untangle(pos,faces,bind,body)
        separated,recovery = fit(separated,faces,body.positions[:body.render_count],body.faces,clearances)
        gi,gw = skin_weights(bind,body.joints,body.weights)
        final_pairs = {key:len(triangle_pairs(skin(separated,gi,gw,body.transforms(body.pose(pose))),faces))
                       for key,pose in POSES}
        accepted = all(final_pairs[key]<=untangle_report["beforePairs"][key] for key,_ in POSES)
        untangle_report.update({"afterClearancePairs":final_pairs,"accepted":accepted,"clearanceRecovery":recovery})
        if accepted:
            pos = separated
            scales = np.array([abs(body.positions[refs[k][0],i]-body.positions[refs[k][1],i])/refs[k][2]
                               for i,k in enumerate("xyz")])
            bind[2][:] = (pos-(body.positions[bind[0]]*bind[1][...,None]).sum(axis=1))/scales
            fit_report = {**recovery,"maxDisplacementMm":float(np.linalg.norm(pos-before_fit,axis=1).max()*1000),
                          "initialFitting":fit_report}
    packed = garments.pack_binding(*bind)
    # Test the serialized float32 records, not only the float64 construction.
    from garmentimport.validation import DTYPE
    records = np.frombuffer(packed,dtype=DTYPE)
    stored = records["i"].astype(int),records["w"].astype(float),records["o"].astype(float)
    error = np.linalg.norm(reconstruct(stored,body.positions,refs)-pos.astype(np.float32),axis=1)*1000
    if error.max()>=1:
        raise ValueError(f"Serialized binding round-trip error {error.max():.6f} mm exceeds 1 mm")
    if fit_report["remainingInsidePercent"]>0:
        raise ValueError(f"Clearance fitting left {fit_report['remainingInsidePercent']:.3f}% vertices inside; inspect alignment/source mesh")
    if fit_report["minClearanceMm"] < clearance-1e-3:
        raise ValueError(f"Clearance fitting reached only {fit_report['minClearanceMm']:.3f} mm; inspect opposing source surfaces")
    if fit_report["minRegionalClearanceResidualMm"] < -1e-3:
        raise ValueError("Clearance fitting did not satisfy regional minimum offsets")
    defs = {d["id"]:d for d in body.measures["measures"]}
    arm = defs["armLength"]["verts"]
    lm = {"bust_y":float(body.positions[defs["chest"]["verts"],1].mean()),
          "waist_y":float(body.positions[defs["waist"]["verts"],1].mean()),
          "hip_y":float(body.positions[defs["hip"]["verts"],1].mean()),
          "crotch":defs["inseam"]["vert"],"acromion_l":arm[0],"elbow_v":arm[1],"wrist_v":arm[2]}
    ref = garments.BodyRef(body.positions,body.render_count,np.arange(body.render_count),lm,defs)
    native,ease,notes = garments.measure_template(args.id,args.category,pos,faces,ref)
    if not native or not all(np.isfinite(v) and v>0 for v in native.values()):
        raise ValueError("Imported geometry has invalid native measurements; inspect its category/alignment")
    entry = {"id":args.id,"kind":args.kind,"category":args.category,
             "label":{"tr":args.label_tr or args.id,"en":args.label_en or args.id},"license":args.license,
             "attribution":args.attribution.strip()+" Changes: aligned, proxy reposed where needed, clearance fitted, triangle bound, UV islands simplified where needed, PBR maps baked/downscaled, converted to glTF.",
             "source":f"{args.source_url or args.source.name} sha256:{hashlib.sha256(args.source.read_bytes()).hexdigest()}",
             "mesh":f"{args.id}.glb","binding":f"{args.id}.bind.bin","scaleRefs":refs,
             "nativeMeasures":native,"defaultEase":ease,"layer":args.layer,"baseColor":base_color}
    delete_report = None
    if args.no_delete_mask:
        delete = np.zeros(0,int)
    else:
        seed = native_delete if native_delete is not None else delete_mask(body.positions[:body.render_count],body.faces,pos,faces)
        delete,delete_report = pose_delete_mask(body,pos,faces,stored,seed)
    if len(delete):
        entry["deleteVerts"] = f"{args.id}.delete.bin"
    output.mkdir(parents=True,exist_ok=True)
    index_path = output/"index.json"
    catalog = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"version":1,"garments":[]}
    if catalog.get("version")!=1 or not isinstance(catalog.get("garments"),list):
        raise ValueError("Existing output index is not a version 1 garment catalogue")
    # Keep the previous accepted files intact if validation or export fails.
    import uuid
    stage = output/f".{args.id}-{uuid.uuid4().hex}.pending"
    stage.mkdir()
    write_glb(stage/entry["mesh"],pos,faces,uv,channels,alpha_mode)
    (stage/entry["binding"]).write_bytes(packed)
    delete_path = stage/f"{args.id}.delete.bin"
    if len(delete):
        delete_path.write_bytes(np.asarray(delete,dtype="<u4").tobytes())
    report = {"id":args.id,"license":args.license,"source":entry["source"],"input":input_count,
              "output":{"vertices":len(pos),"triangles":len(faces),"deletedBodyVertices":len(delete),
                        "textureSizes":{k:list(v.shape[1::-1]) for k,v in channels.items()}},
              "alignment":alignment,"fit":fit_report,"measureNotes":notes,"shell":shell_report,
              "clearanceMm":clearance,"nativeBinding":native_report,"subdivisionLevels":args.subdivide,
              "materialWarnings":material_warnings,
              "poseSafety":{"binding":stability_report,"deleteMask":delete_report,"shellSeparation":untangle_report,
                            "regionalClearanceMaxMm":float(np.max(clearances)*1000)},
              "roundTrip":{"maxMm":float(error.max()),"meanMm":float(error.mean()),"passed":True}}
    rigid = None
    if args.category=="shoes":
        rigid = rigid_shoe_parts(pos,body,args.rigid_sole_height_mm/1000)
        (stage/f"{args.id}.rigid.json").write_text(json.dumps(rigid,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        report["rigidParts"] = {"file":f"{args.id}.rigid.json","format":"v1","soleHeightMm":rigid["soleHeightMm"],
                                "vertices":sum(len(a["vertices"]) for a in rigid["attachments"]),
                                "bones":[a["bone"] for a in rigid["attachments"]],"warnings":RIGID_REVIEW_NOTES}
    offsets = np.linalg.norm(pos-(body.positions[bind[0]]*bind[1][...,None]).sum(axis=1),axis=1)
    report["looseParts"] = {"offsetOver50mmVertices":int((offsets>.05).sum()),
        "warnings":["Loose fabric uses body-triangle skin transfer; review hood/flaps in poses. Authored helper deformation needs runtime helper or skin override support."] if (offsets>.05).any() else []}
    compare = None
    if args.compare_original:
        from scipy.spatial import cKDTree
        original_index = json.loads(args.compare_original.with_name("index.json").read_text(encoding="utf-8"))
        original_entry = next(e for e in original_index["garments"] if e["binding"]==args.compare_original.name)
        original_bind = read_binding(args.compare_original)
        original_positions = reconstruct(original_bind,body.positions,original_entry.get("scaleRefs"))
        distance,mapping = cKDTree(original_positions).query(before_fit)
        # Cleanup can merge identical (position,UV) copies; compare corresponding positions, not buffer lengths.
        compare = tuple(a[mapping] for a in original_bind),original_entry
        report["comparisonBeforeFit"] = {"meanMm":float(distance.mean()*1000),"maxMm":float(distance.max()*1000),
            "mapping":"nearest original reconstructed vertex after cleanup/decimation",
            "originalVertexCount":len(original_positions),"importedVertexCount":len(before_fit)}
    if args.validate or compare:
        print(f"garment-import: {args.id}: four poses and three avatar-core shapes",file=sys.stderr,flush=True)
        report["validation"] = validate(body,pos,faces,stored,entry,stage,stage/entry["binding"],compare,rigid,delete=delete)
    report_path = stage/f"{args.id}.report.json"
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if "validation" in report and not report["validation"]["passed"]:
        raise ValueError(f"Validation failed; inspect {report_path}; catalogue entry was not accepted")
    # Reload immediately before publication so sequential imports append to the current catalogue.
    if index_path.exists():
        catalog = json.loads(index_path.read_text(encoding="utf-8"))
    catalog["garments"] = [e for e in catalog["garments"] if e["id"]!=args.id]+[entry]
    temp_index = index_path.with_suffix(".json.tmp")
    temp_index.write_text(json.dumps(catalog,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    runtime_request = stage/f"{args.id}.runtime-request.json"
    if runtime_request.exists():
        request = json.loads(runtime_request.read_text(encoding="utf-8"))
        request["bindingPath"] = str(output/entry["binding"])
        runtime_request.write_text(json.dumps(request),encoding="utf-8")
    for artifact in stage.iterdir():
        artifact.replace(output/artifact.name)
    stage.rmdir()
    stale_delete = output/f"{args.id}.delete.bin"
    if not len(delete) and stale_delete.exists():
        stale_delete.unlink()
    temp_index.replace(index_path)
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        require_venv()
        report = import_garment(args)
    except (RuntimeError,ValueError,OSError,ImportError) as error:
        print(f"garment-import: {error}",file=sys.stderr)
        return 1
    print(json.dumps(report,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
