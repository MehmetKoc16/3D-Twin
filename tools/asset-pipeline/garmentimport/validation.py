"""Measure binding reconstruction, actual runtime grading, and posed-body containment."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .body import skin
from .geometry import inside, reconstruct, skin_weights, surface


DTYPE = np.dtype([("i","<u4",3),("w","<f4",3),("o","<f4",3)])


def read_binding(path):
    raw = path.read_bytes()
    if len(raw)%36:
        raise ValueError("Invalid binding byte length")
    rec = np.frombuffer(raw,dtype=DTYPE)
    return rec["i"].astype(int),rec["w"].astype(float),rec["o"].astype(float)


def penetration(points,body_positions,faces,garment_faces=None):
    mesh = surface(body_positions,faces,weld=True)
    if not mesh.is_watertight:
        return {"insidePercent":None,"reason":"Posed body is not watertight"}
    contained = inside(mesh,points)
    result = {"insidePercent":float(contained.mean()*100),"insideVertices":int(contained.sum()),
              "testedVertices":len(points)}
    if garment_faces is not None:
        centers = points[garment_faces].mean(axis=1)
        centroid_inside = inside(mesh,centers)
        result["triangleCentroidInsidePercent"] = float(centroid_inside.mean()*100)
        result["insideTriangleCentroids"] = int(centroid_inside.sum())
        result["testedTriangleCentroids"] = len(centers)
    return result


def validate(body,points,faces,bind,template,output,binding_path,compare=None,rigid=None):
    reconstructed = reconstruct(bind,body.positions,template.get("scaleRefs"))
    error = np.linalg.norm(reconstructed-points.astype(np.float32),axis=1)*1000
    report = {"roundTrip":{"maxMm":float(error.max()),"meanMm":float(error.mean()),"passed":bool(error.max()<1)},
              "poses":{},"shapes":{},"limitations":[
                  "Containment is vertex ray parity on the complete body, before app collision correction or body deletion.",
                  "Posed bodies may self-intersect; the metric measures ray parity, not a physical cloth simulation.",
                  "Vertex and triangle-centroid samples can miss edge/interior collisions; clearance fitting constrains vertices only.",
                  "Grading uses actual avatar-core rings; app-specific length grading and layering are not included."]}
    garment_joints,garment_weights = skin_weights(bind,body.joints,body.weights)
    rigid_joints,rigid_weights = garment_joints.copy(),garment_weights.copy()
    if rigid:
        for attachment in rigid["attachments"]:
            ids = np.array(attachment["vertices"],int)
            rigid_joints[ids] = body.names.index(attachment["bone"])
            rigid_weights[ids] = [1,0,0,0]
        report["rigidOverridePoses"] = {}
        report["limitations"].append("Rigid sole/heel metrics use the review-only sidecar; the current app ignores it.")
    for key,pose_name in (("A","a-pose"),("T","t-pose"),("relaxed","relaxed"),("walk","walk")):
        transforms = body.transforms(body.pose(pose_name))
        posed_body = skin(body.positions[:body.render_count],body.joints,body.weights,transforms)
        posed_garment = skin(reconstructed,garment_joints,garment_weights,transforms)
        report["poses"][key] = penetration(posed_garment,posed_body,body.faces,faces)
        if rigid:
            report["rigidOverridePoses"][key] = penetration(
                skin(reconstructed,rigid_joints,rigid_weights,transforms),posed_body,body.faces,faces)
    node = shutil.which("node")
    if not node:
        raise ValueError("Validation requires Node and the repo's npm dependencies to run avatar-core")
    request_path = output/f"{template['id']}.runtime-request.json"
    result_path = output/f"{template['id']}.runtime-result.json"
    request = {"bodyDir":str(body.directory.resolve()),"base":body.morphs.base.tolist(),"faces":body.faces.tolist(),
               "bindingPath":str(binding_path.resolve()),"template":template}
    request_path.write_text(json.dumps(request),encoding="utf-8")
    run = subprocess.run([node,str(Path(__file__).with_name("runtime.mjs")),str(request_path),str(result_path)],
                         capture_output=True,text=True,timeout=180)
    if run.returncode:
        raise ValueError(f"avatar-core validation failed: {run.stderr[-2500:]}")
    results = json.loads(result_path.read_text(encoding="utf-8"))
    edges = np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    rest_lengths = np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1)
    for name,result in results.items():
        graded = np.array(result["graded"]).reshape(-1,3)
        bound = np.array(result["bound"]).reshape(-1,3)
        shape = np.array(result["body"]).reshape(-1,3)
        length = np.linalg.norm(graded[edges[:,0]]-graded[edges[:,1]],axis=1)
        ratio = length/np.maximum(rest_lengths,1e-8)
        stable = rest_lengths>=.001
        stable_max = float(ratio[stable].max()) if stable.any() else float(ratio.max())
        extent = float(np.ptp(graded,axis=0).max())
        finite = bool(np.isfinite(graded).all() and result["heatmapFinite"])
        max_growth = float((length-rest_lengths).max())
        runtime_error = float(np.linalg.norm(reconstruct(bind,shape,template.get("scaleRefs"))-bound,axis=1).max()*1000)
        passed = finite and extent<5 and float(np.quantile(ratio,.99))<10 and max_growth<.1 and runtime_error<.001
        shape_report = {"passed":passed,"finite":finite,"maxExtentM":extent,
                        "edgeStretchP99":float(np.quantile(ratio,.99)),"edgeStretchMax":float(ratio.max()),
                        "edgeStretchMaxOver1mm":stable_max,"maxEdgeLengthM":float(length.max()),
                        "maxEdgeGrowthMm":max_growth*1000,
                        "warnings":["Local edge stretch exceeds 10x; inspect grading near armpits/cuffs."] if stable_max>=10 else [],
                        "achievedCm":result["achievedCm"],"unreachable":result["unreachable"],
                        "fitReport":result["fitReport"],"heatmapMinMm":result["heatmapMinMm"],
                        "heatmapMaxMm":result["heatmapMaxMm"],"poses":{}}
        for key,pose_name in (("A","a-pose"),("T","t-pose"),("relaxed","relaxed"),("walk","walk")):
            transforms = body.transforms(body.pose(pose_name),shape)
            posed_body = skin(shape[:body.render_count],body.joints,body.weights,transforms)
            posed_garment = skin(graded,garment_joints,garment_weights,transforms)
            shape_report["poses"][key] = penetration(posed_garment,posed_body,body.faces,faces)
        # The Python binding evaluator must agree with actual bindGarment too.
        shape_report["runtimeBindingMaxErrorMm"] = runtime_error
        report["shapes"][name] = shape_report
    if compare is not None:
        original_bind,original_template = compare
        if len(original_bind[0])!=len(bind[0]):
            report["comparison"] = {"comparable":False,"reason":"Cleanup or decimation changed the render vertex count"}
        else:
            original = reconstruct(original_bind,body.positions,original_template.get("scaleRefs"))
            base_difference = np.linalg.norm(original-reconstructed,axis=1)*1000
            report["comparison"] = {"comparable":True,
                "sameTriangleRecordPercent":float(np.all(original_bind[0]==bind[0],axis=1).mean()*100),
                "baseMeanDifferenceMm":float(base_difference.mean()),"baseMaxDifferenceMm":float(base_difference.max()),
                "deformedMeanDifferenceMm":{}}
            for name,result in results.items():
                shape = np.array(result["body"]).reshape(-1,3)
                d = reconstruct(original_bind,shape,original_template.get("scaleRefs"))-reconstruct(bind,shape,template.get("scaleRefs"))
                report["comparison"]["deformedMeanDifferenceMm"][name] = float(np.linalg.norm(d,axis=1).mean()*1000)
    report["acceptance"] = {"roundTripMaxMm":1,"maxExtentM":5,"edgeStretchP99Max":10,
                            "maxEdgeGrowthMm":100,"localEdgeStretchIsDiagnostic":True,"posePenetrationIsDiagnostic":True}
    report["passed"] = report["roundTrip"]["passed"] and all(s["passed"] for s in report["shapes"].values())
    return report
