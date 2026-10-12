"""MakeHuman runtime neutral frame, body rig, proxy reposing and guarded similarity alignment."""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pygltflib
from scipy.spatial.transform import Rotation

from macro import MorphSet
from .mesh_io import accessor
from .geometry import binding, nearest, skin_weights, surface


@dataclass
class Body:
    positions: np.ndarray
    faces: np.ndarray
    joints: np.ndarray
    weights: np.ndarray
    names: list[str]
    rig: dict
    measures: dict
    morphs: MorphSet
    directory: Path

    @property
    def render_count(self):
        return len(self.joints)

    def transforms(self, pose, positions=None):
        p = self.positions if positions is None else positions
        bones = {b["name"]:b for b in self.rig["bones"]}
        worlds = {}
        def joint(ref):
            if ref["strategy"]=="VERTEX":
                return p[ref["vert"]]
            if ref["strategy"]=="MEAN":
                return p[ref["verts"]].mean(axis=0)
            return np.array(ref["position"])
        def world(name):
            if name in worlds:
                return worlds[name]
            b = bones[name]
            head = joint(b["head"])
            local = np.eye(4)
            local[:3,:3] = Rotation.from_quat(pose.get(name,[0,0,0,1])).as_matrix()
            local[:3,3] = head-(joint(bones[b["parent"]]["head"]) if b["parent"] else 0)
            worlds[name] = (world(b["parent"]) if b["parent"] else np.eye(4)) @ local
            return worlds[name]
        result = []
        for name in self.names:
            inverse = np.eye(4); inverse[:3,3] = -joint(bones[name]["head"])
            result.append(world(name) @ inverse)
        return np.array(result)

    def pose(self, name):
        return json.loads((self.directory.parent/"poses"/f"{name}.json").read_text(encoding="utf-8"))["bones"]

    def landmarks(self, positions=None):
        p = self.positions if positions is None else positions
        bones = {b["name"]:b for b in self.rig["bones"]}
        return {key:p[bones[bone][end]["vert"]] for key,bone,end in (
            ("shoulder_l","upperarm_l","head"),("shoulder_r","upperarm_r","head"),
            ("hip_l","thigh_l","head"),("hip_r","thigh_r","head"),
            ("ankle_l","calf_l","tail"),("ankle_r","calf_r","tail"))}


def load_body(directory):
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    g = pygltflib.GLTF2().load(str(directory/"base.glb")); prim = g.meshes[0].primitives[0]
    render = accessor(g,prim.attributes.POSITION,directory/"base.glb")
    base = np.concatenate([render,[j["position"] for j in manifest["jointPoints"]]])
    model = MorphSet(manifest,(directory/"morphs.bin").read_bytes(),base)
    skin = g.skins[0]
    return Body(model.positions(),accessor(g,prim.indices,directory/"base.glb").reshape(-1,3).astype(int),
                accessor(g,prim.attributes.JOINTS_0,directory/"base.glb").astype(int),
                accessor(g,prim.attributes.WEIGHTS_0,directory/"base.glb").astype(float),
                [g.nodes[j].name for j in skin.joints],
                json.loads((directory/"rig.json").read_text(encoding="utf-8")),
                json.loads((directory/"measures.json").read_text(encoding="utf-8")),model,directory)


def skin(points,joints,weights,transforms,inverse=False):
    blended = (transforms[joints]*weights[:,:,None,None]).sum(axis=1)
    if inverse:
        # Invert the blended affine transformation, not a blend of inverse bone matrices.
        condition = np.linalg.cond(blended[:,:3,:3])
        if np.max(condition)>1e6:
            raise ValueError("Proxy skinning is singular; supply an evaluated rest-pose source mesh")
        blended = np.linalg.inv(blended)
    homogeneous = np.column_stack([points,np.ones(len(points))])
    return np.einsum("nij,nj->ni",blended,homogeneous)[:,:3]


def similarity(source,target):
    if len(source)<3 or np.linalg.matrix_rank(source-source.mean(axis=0))<2:
        raise ValueError("Landmark alignment needs three non-collinear pairs")
    a,b = source-source.mean(axis=0),target-target.mean(axis=0)
    u,s,vt = np.linalg.svd(a.T@b)
    sign = np.ones(3); sign[-1] = np.sign(np.linalg.det(vt.T@u.T))
    rotation = vt.T@np.diag(sign)@u.T
    scale = (s*sign).sum()/(a*a).sum()
    return scale,rotation,target.mean(axis=0)-scale*source.mean(axis=0)@rotation.T


def align(points,body,category,pose="auto",sidecar=None,source_body_height=None,icp_iterations=8):
    sidecar = sidecar or {}
    unknown = set(sidecar)-{"matrix","pose","sourceBodyHeightM","landmarks","scale","rotationDeg","translationM","icp"}
    if unknown:
        raise ValueError(f"Unknown transform sidecar keys: {sorted(unknown)}")
    original = points.copy()
    points = points.copy()
    warnings = []
    if "matrix" in sidecar:
        matrix = np.array(sidecar["matrix"],float)
        if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1]) or np.linalg.det(matrix[:3,:3])<=0:
            raise ValueError("matrix must be a finite orientation-preserving row-major affine 4x4")
        points = points@matrix[:3,:3].T+matrix[:3,3]
        scale = float(np.cbrt(np.linalg.det(matrix[:3,:3])))
        method = "manual matrix"
    else:
        rotation = Rotation.from_euler("xyz",sidecar.get("rotationDeg",[0,0,0]),degrees=True).as_matrix()
        points = points@rotation.T
        height = float(np.ptp(body.positions[:body.render_count,1]))
        source_height = source_body_height or sidecar.get("sourceBodyHeightM")
        if source_height is not None:
            if not np.isfinite(source_height) or source_height<=0:
                raise ValueError("Source body height must be positive and in meters in the source coordinate units")
            scale = height/source_height
            method = "source body height"
        else:
            # Unit inference, not a claim to recover human height from a cropped mesh.
            extent = np.ptp(points,axis=0)
            expected = {"top":.65,"bottom":.95,"shoes":.30}[category]
            dimension = extent[2] if category=="shoes" else max(extent[1],extent[0]*.55)
            units = np.array([.001,.01,.1,1,10,100,1000])
            scale = float(units[np.abs(np.log(np.maximum(dimension*units,1e-12)/expected)).argmin()])
            method = "category extent unit inference"
            warnings.append("Cropped garments do not determine body height. Confirm scale with sourceBodyHeightM or landmarks.")
        scale *= sidecar.get("scale",1.)
        if not np.isfinite(scale) or scale<=0:
            raise ValueError("scale must be finite and positive")
        points *= scale
        # Keep already grounded models; otherwise place category on the body's anatomical region.
        centroid = (points.min(axis=0)+points.max(axis=0))/2
        if not (points[:,1].min()>=-.05 and points[:,1].max()<=height*1.15 and abs(centroid[0])<.15):
            top = {"top":height*.88,"bottom":height*.62,"shoes":height*.12}[category]
            points += [-centroid[0],top-points[:,1].max(),-centroid[2]]
        points += np.asarray(sidecar.get("translationM",[0,0,0]))
    requested = sidecar.get("pose",pose)
    if requested not in ("auto","a","t"):
        raise ValueError("pose must be auto, a or t")
    candidate_scores = {}
    sample = points[::max(1,len(points)//1800)]
    for candidate in ("a","t"):
        posed = body.positions[:body.render_count] if candidate=="a" else skin(body.positions[:body.render_count],body.joints,body.weights,body.transforms(body.pose("t-pose")))
        _,distance,_ = nearest(surface(posed,body.faces),sample)
        candidate_scores[candidate] = float(np.quantile(distance,.75))
    detected = min(candidate_scores,key=candidate_scores.get) if requested=="auto" else requested
    if requested=="auto" and abs(candidate_scores["a"]-candidate_scores["t"])<.01:
        warnings.append("Pose is ambiguous (often sleeveless tops/bottoms); use --pose a/t or a sidecar.")
    transforms = body.transforms(body.pose("t-pose")) if detected=="t" else body.transforms({})
    posed_body = skin(body.positions[:body.render_count],body.joints,body.weights,transforms)
    landmarks = sidecar.get("landmarks")
    if landmarks:
        targets = body.landmarks()
        if detected=="t":
            # Transform each landmark with its owning bone; shared joint refs must not be overwritten.
            owners = {"shoulder_l":"upperarm_l","shoulder_r":"upperarm_r","hip_l":"thigh_l", "hip_r":"thigh_r",
                      "ankle_l":"calf_l","ankle_r":"calf_r"}
            targets = {name:skin(value[None,:],np.array([[body.names.index(owners[name])]]),np.ones((1,1)),transforms)[0]
                       for name,value in targets.items()}
        names = sorted(landmarks)
        if set(names)-set(targets):
            raise ValueError("Landmarks must be shoulder_l/r, hip_l/r, ankle_l/r")
        source = np.array([landmarks[n] for n in names],float)
        s,r,t = similarity(source,np.array([targets[n] for n in names]))
        points = s*original@r.T+t
        scale = float(s)
        method += " + landmarks"
    if sidecar.get("icp",True) and "matrix" not in sidecar:
        mesh = surface(posed_body,body.faces)
        for _ in range(icp_iterations):
            subset = points[::max(1,len(points)//1500)]
            closest,distance,_ = nearest(mesh,subset)
            keep = distance<=np.quantile(distance,.7)
            translation = np.median(closest[keep]-subset[keep],axis=0)
            translation = np.clip(translation,-.005,.005)
            if np.linalg.norm(translation)<1e-5:
                break
            points += translation
        method += " + trimmed translation ICP"
    if detected=="t":
        bind = binding(points,posed_body,body.faces)
        joints,weights = skin_weights(bind,body.joints,body.weights)
        points = skin(points,joints,weights,transforms,inverse=True)
    return points,{"method":method,"scale":scale,"pose":detected,"poseScoresM":candidate_scores,"warnings":warnings}
