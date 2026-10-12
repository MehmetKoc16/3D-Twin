"""Synthetic importer checks plus an optional shipped CC0 T-shirt integration test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
pytest.importorskip("rtree")
pytest.importorskip("fast_simplification")

import garments
from garment_import import import_garment, parser
from garmentimport.body import align, load_body, similarity, skin
from garmentimport.geometry import binding, delete_mask, fit, inside, reconstruct, skin_weights, surface
from garmentimport.mesh_io import Part, atlas, clean, decimate, load_parts, write_glb, refine
from garmentimport.validation import DTYPE, read_binding, penetration

PIPELINE = Path(__file__).resolve().parents[1]
BODY = PIPELINE.parents[1]/"apps/web/public/assets/body"
WARDROBE = BODY.parent/"garments"


def material(color=(1,1,1,1)):
    return {"base":None,"mr":None,"normal":None,"occlusion":None,"emissive":None,
            "factor":list(color),"roughness":.7,"metallic":.2,"emissiveFactor":[0,0,0],
            "alphaMode":"OPAQUE","alphaCutoff":.5,"normalScale":1.,"occlusionStrength":1.}


def test_binding_binary_and_nearest_triangle():
    body = np.array([[0.,0,0],[4,0,0],[0,4,0],[.3,.3,-.1],[.4,.3,-.1],[.3,.4,-.1]])
    faces = np.array([[0,1,2],[3,4,5]])
    points = np.array([[.7,.7,.002],[.2,.3,.04],[2.9,.3,.2]])
    bind = binding(points,body,faces)
    assert np.all(bind[0] == faces[0])
    assert np.all(bind[1]>=0) and np.allclose(bind[1].sum(axis=1),1)
    rec = np.frombuffer(garments.pack_binding(*bind),dtype=DTYPE)
    assert DTYPE.itemsize==36
    stored = rec["i"].astype(int),rec["w"].astype(float),rec["o"].astype(float)
    assert np.linalg.norm(reconstruct(stored,body)-points,axis=1).max()<1e-6


def test_fit_inside_clearance_and_preserve_exterior():
    body = trimesh.creation.icosphere(subdivisions=2,radius=.3)
    garment = trimesh.creation.icosphere(subdivisions=2,radius=.31)
    points = garment.vertices.copy()
    points[points[:,2]>.15] *= .9
    exterior = np.linalg.norm(points,axis=1)>.305
    fitted,report = fit(points,garment.faces,body.vertices,body.faces,.003)
    assert report["movedVertices"]>0
    assert not inside(body,fitted).any()
    assert body.nearest.on_surface(fitted)[1].min()>=.003-1e-6
    assert np.allclose(fitted[exterior],points[exterior])


def test_uv_copies_stay_closed_after_fit():
    body = trimesh.creation.icosphere(subdivisions=1,radius=.3)
    points = np.vstack([body.vertices,body.vertices])*.99
    faces = np.vstack([body.faces,body.faces+len(body.vertices)])
    fitted,_ = fit(points,faces,body.vertices,body.faces,.002)
    assert np.allclose(fitted[:len(body.vertices)],fitted[len(body.vertices):],atol=1e-8)


def test_skinning_top_four_and_inverse_blended_transform():
    indices = np.array([[0,1,2]])
    weights = np.array([[.2,.3,.5]])
    bind = indices,weights,np.zeros((1,3))
    bi = np.tile(np.arange(4),(3,1))
    bw = np.tile([.4,.3,.2,.1],(3,1))
    gi,gw = skin_weights(bind,bi,bw)
    assert np.array_equal(gi,[[0,1,2,3]]) and np.allclose(gw,bw[:1])
    transforms = np.tile(np.eye(4),(4,1,1)); transforms[:,0,3] = [.1,.2,.3,.4]
    points = np.array([[1.,2,3]])
    posed = skin(points,gi,gw,transforms)
    assert np.allclose(skin(posed,gi,gw,transforms,inverse=True),points)


def test_similarity_landmarks():
    source = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    target = 2*source@rotation.T+[.1,2.,.3]
    s,r,t = similarity(source,target)
    assert np.allclose(s*source@r.T+t,target)
    with pytest.raises(ValueError,match="non-collinear"):
        similarity(source[:2],target[:2])


def test_cleanup_keeps_uv_seams():
    positions = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,0],[0,0,0]])
    uv = np.array([[0.,0],[1,0],[0,1],[1,1],[0,0]])
    part = clean(Part(positions,np.array([[0,1,2],[3,2,1],[4,1,2],[0,0,1]]),uv,material()))
    assert len(part.positions)==4 and len(part.faces)==2
    assert np.count_nonzero(np.all(part.positions==[0,0,0],axis=1))==2


def test_decimation_transfers_uv_per_island():
    mesh = trimesh.creation.icosphere(subdivisions=4)
    # Continuous synthetic UV coordinate field, no seams.
    uv = (mesh.vertices[:,[0,2]]+1)/2
    parts = decimate([Part(mesh.vertices,mesh.faces,uv,material())],800)
    assert sum(len(p.faces) for p in parts)<=800
    assert all(np.isfinite(p.uv).all() and p.uv.min()>=0 and p.uv.max()<=1 for p in parts)


def test_linear_refinement_preserves_surface_uv_and_budget():
    p = Part(np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]),
             np.array([[0.,0],[1,0],[0,1]]),material())
    output = refine([p],2,16)[0]
    assert len(output.faces)==16
    assert np.allclose(output.uv,output.positions[:,:2])
    assert np.all(output.positions[:,2]==0)
    with pytest.raises(ValueError,match="Subdivision exceeds"):
        refine([output],1,16)


def test_pbr_atlas_multi_material_one_primitive(tmp_path):
    pos = np.array([[0.,0,0],[1,0,0],[0,1,0]])
    faces = np.array([[0,1,2]]); uv=np.array([[0.,0],[1,0],[0,1]])
    parts = [Part(pos,faces,uv,material((1,0,0,1))),Part(pos+[0,0,1],faces,uv,material((0,0,1,1)))]
    positions,triangles,uvs,maps,mode,_ = atlas(parts,64)
    target=tmp_path/"multi.glb"; write_glb(target,positions,triangles,uvs,maps,mode)
    loaded=load_parts(target)
    assert len(loaded)==1 and len(loaded[0].positions)==6
    assert all(max(tex["image"].size)<=64 for key,tex in loaded[0].material.items() if isinstance(tex,dict))
    assert np.allclose(loaded[0].material["factor"],[1,1,1,1])
    arr=np.asarray(loaded[0].material["base"]["image"])
    assert arr[...,0].max()==255 and arr[...,2].max()==255
    mr=np.asarray(loaded[0].material["mr"]["image"])
    assert mr[...,1].max()==round(.7*255) and mr[...,2].max()==round(.2*255)


def test_obj_quads_and_uvs(tmp_path):
    obj=tmp_path/"quad.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n",encoding="utf-8")
    parts=load_parts(obj)
    assert sum(len(p.faces) for p in parts)==2
    assert set(map(tuple,parts[0].uv))=={(0.,0.),(1.,0.),(1.,1.),(0.,1.)}


def test_delete_mask_does_not_hide_beyond_open_hem():
    garment=trimesh.creation.box(extents=[.6,.6,.6])
    body=trimesh.creation.box(extents=[.58,.58,.58])
    mask=delete_mask(body.vertices,body.faces,garment.vertices,garment.faces)
    assert len(mask)<=len(body.vertices)
    # Far-away points cannot be hidden; this also checks the depth cap.
    assert not len(delete_mask(body.vertices+[0,2,0],body.faces,garment.vertices,garment.faces))


def test_penetration_samples_triangle_interiors_too():
    body = trimesh.creation.icosphere(subdivisions=1,radius=.3)
    points = np.array([[-.5,-.5,0],[.5,-.5,0],[0,1.,0]])
    result = penetration(points,body.vertices,body.faces,np.array([[0,1,2]]))
    assert result["insidePercent"]==0
    assert result["triangleCentroidInsidePercent"]==100


def test_skinned_source_bakes_default_node_pose(tmp_path):
    import pygltflib as gl
    from garmentimport.mesh_io import accessor
    pos=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    faces=np.array([[0,1,2]]); uv=np.array([[0.,0],[1,0],[0,1]])
    positions,triangles,uvs,maps,mode,_=atlas([Part(pos,faces,uv,material())],16)
    target=tmp_path/"skin.glb"; write_glb(target,positions,triangles,uvs,maps,mode)
    g=gl.GLTF2().load(str(target)); blob=bytearray(g.binary_blob())
    def add(arr,dtype,kind,component):
        data=np.ascontiguousarray(arr,dtype=dtype)
        blob.extend(b"\0"*((-len(blob))%4))
        vi=len(g.bufferViews); g.bufferViews.append(gl.BufferView(buffer=0,byteOffset=len(blob),byteLength=data.nbytes))
        blob.extend(data.tobytes()); ai=len(g.accessors)
        g.accessors.append(gl.Accessor(bufferView=vi,componentType=component,type=kind,count=len(data)))
        return ai
    g.meshes[0].primitives[0].attributes.JOINTS_0=add(np.zeros((3,4)),"<u2","VEC4",5123)
    g.meshes[0].primitives[0].attributes.WEIGHTS_0=add(np.tile([1,0,0,0],(3,1)),"<f4","VEC4",5126)
    inverse=add(np.eye(4).reshape(1,16),"<f4","MAT4",5126)
    g.nodes.append(gl.Node(name="proxy",translation=[.3,.4,.5]))
    g.nodes[0].skin=0; g.skins=[gl.Skin(joints=[1],inverseBindMatrices=inverse)]
    g.scenes[0].nodes.append(1); g.buffers[0].byteLength=len(blob); g.set_binary_blob(bytes(blob)); g.save_binary(str(target))
    parts=load_parts(target)
    assert np.allclose(parts[0].positions,pos+[.3,.4,.5])
    bind_parts=load_parts(target,skin_pose="bind")
    assert np.allclose(bind_parts[0].positions,pos)


@pytest.mark.parametrize("licence",["CC0-1.0","CC-BY-3.0"])
def test_synthetic_torso_cli(tmp_path,live_body,licence):
    # Synthetic permissive sleeveless shell with continuous UVs and deliberate duplication at the seam.
    from PIL import Image
    rings,segments=20,64
    a=np.linspace(0,2*np.pi,segments+1)
    points=np.array([[.22*np.cos(t),y,.18*np.sin(t)] for y in np.linspace(.9,1.42,rings) for t in a])
    uv=np.array([[i/segments,j/(rings-1)] for j in range(rings) for i in range(segments+1)])
    faces=[]
    for j in range(rings-1):
        for i in range(segments):
            k=j*(segments+1)+i
            faces.extend([[k,k+1,k+segments+2],[k,k+segments+2,k+segments+1]])
    m=material((.4,.6,.9,1)); m["base"]={"image":Image.new("RGBA",(32,32),(255,255,255,255)),"wrapS":10497,"wrapT":10497}
    pos,tris,uv,maps,mode,_=atlas([Part(points,np.array(faces),uv,m)],32)
    source=tmp_path/"synthetic.glb"; write_glb(source,pos,tris,uv,maps,mode)
    identity=tmp_path/"identity.json"; identity.write_text(json.dumps({"matrix":np.eye(4).tolist(),"pose":"a"}),encoding="utf-8")
    args=parser().parse_args([str(source),"--id","synthetic","--category","top","--kind","tshirt",
        "--license",licence,"--attribution",f"Synthetic test shell, {licence}","--transform",str(identity),
        "--out",str(tmp_path/"out"),"--texture-max","32"])
    report=import_garment(args)
    assert report["roundTrip"]["maxMm"]<1 and report["fit"]["remainingInsidePercent"]==0
    assert report["license"]==licence
    assert json.loads((args.out/"index.json").read_text(encoding="utf-8"))["garments"][0]["license"]==licence
    assert not list(args.out.glob("*.pending"))


@pytest.fixture(scope="module")
def live_body():
    if not (BODY/"base.glb").exists() or (BODY/"base.glb").read_bytes()[:4]!=b"glTF":
        pytest.skip("Permissive MakeHuman body assets not built / LFS not pulled")
    return load_body(BODY)


def test_real_body_pose_and_repose(live_body):
    body=live_body
    transforms=body.transforms(body.pose("t-pose"))
    indices=np.flatnonzero(body.positions[:body.render_count,0]>.3)[::8]
    original=body.positions[indices]
    posed=skin(original,body.joints[indices],body.weights[indices],transforms)
    gi,gw=skin_weights(binding(posed,skin(body.positions[:body.render_count],body.joints,body.weights,transforms),body.faces),body.joints,body.weights)
    recovered=skin(posed,gi,gw,transforms,inverse=True)
    assert np.linalg.norm(recovered-original,axis=1).max()<1e-5


def test_align_source_height_and_override(live_body):
    points=live_body.positions[:live_body.render_count]
    shifted=points*100+[20,30,40]
    matrix=np.eye(4); matrix[:3,:3]*=.01; matrix[:3,3]=[-.2,-.3,-.4]
    output,report=align(shifted,live_body,"top","a",{"matrix":matrix.tolist()})
    assert np.allclose(output,points)
    assert report["pose"]=="a"
    output,report=align(points*100,live_body,"top","a",{"icp":False},float(np.ptp(points[:,1])*100))
    assert np.allclose(output,points)
    assert report["scale"]==pytest.approx(.01)


@pytest.mark.parametrize("licence",["CC0-1.0","CC-BY-3.0","CC-BY-4.0"])
def test_cli_accepts_the_app_garment_licences(licence):
    args=parser().parse_args(["x.obj","--id","x","--category","top","--kind","tshirt","--license",licence,"--attribution","Example"])
    assert args.license==licence


@pytest.mark.parametrize("bad",["CC-BY-NC-4.0","CC-BY-SA-4.0","AGPL-3.0","CC-BY-2.0","CC-BY","user"])
def test_cli_rejects_incompatible_licence(bad):
    with pytest.raises(SystemExit):
        parser().parse_args(["x.obj","--id","x","--category","top","--kind","tshirt","--license",bad,"--attribution","Example"])


def test_cli_rejects_shipped_output(tmp_path):
    args=parser().parse_args(["x.obj","--id","x","--category","top","--kind","tshirt","--license","CC0-1.0","--attribution","Example","--out",str(WARDROBE)])
    with pytest.raises(ValueError,match="shipped"):
        import_garment(args)


def test_native_binding_preserves_authored_body_and_reconstructs_helpers():
    from mhclo import parse_mhclo
    from garmentimport.native import convert_records
    body = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    full = np.vstack([body,[.2,.2,.2]])
    cl = parse_mhclo("basemesh hm08\nx_scale 0 1 10\ny_scale 0 2 10\nz_scale 0 3 10\nverts 0\n0 1 2 -.1 .5 .6 .02 .03 .04\n4\n")
    points, bind, refs, report = convert_records(cl,full,np.arange(4),body,
        np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]]))
    assert np.array_equal(bind[0][0],[0,1,2])
    assert np.all(bind[1]>=0) and np.allclose(bind[1].sum(axis=1),1)
    assert np.allclose(reconstruct(bind,body,refs),points)
    assert report["authoredBodyRows"]==1 and report["helperReboundRows"]==1
    assert report["conversionMaxMm"]<1e-8


def test_native_binding_rejects_legacy_shear_and_invalid_indices():
    from mhclo import parse_mhclo
    from garmentimport.native import convert_records
    body = np.eye(3)
    cl = parse_mhclo("verts 0\n0\n")
    with pytest.raises(ValueError,match="scale references"):
        convert_records(cl,body,np.arange(3),body,np.array([[0,1,2]]))
    cl.indices[0,0]=8
    with pytest.raises(ValueError,match="indices"):
        convert_records(cl,body,np.arange(3),body,np.array([[0,1,2]]))


def test_outer_shell_removes_inner_wall_and_preserves_open_fabric():
    from garmentimport.geometry import outer_shell
    body = trimesh.creation.icosphere(subdivisions=2,radius=.29)
    outer = trimesh.creation.icosphere(subdivisions=2,radius=.31)
    inner = trimesh.creation.icosphere(subdivisions=2,radius=.306)
    inner.faces = inner.faces[:,::-1]
    uv = np.zeros((len(outer.vertices),2))
    parts = [Part(outer.vertices,outer.faces,uv,material()),Part(inner.vertices,inner.faces,uv,material())]
    result, report = outer_shell(parts,body.vertices,body.faces)
    assert report["removedInnerTriangles"] == len(inner.faces)
    assert sum(len(p.faces) for p in result) == len(outer.faces)
    sheet = Part(np.array([[-.1,0,.4],[.1,0,.4],[0,.2,.4]]),np.array([[0,1,2]]),np.zeros((3,2)),material())
    result,report = outer_shell([sheet],body.vertices,body.faces)
    assert report["removedInnerTriangles"]==0 and len(result[0].faces)==1


RIGID_POINTS=np.array([[.18,.005,.01],[.19,.006,.02],[.18,.005,.19],[.19,.006,.20],[.18,.09,.1],
                       [-.18,.005,.01],[-.19,.006,.02],[-.18,.005,.19],[-.19,.006,.20]])


def test_rigid_shoe_attachment_preserves_segment_distances(live_body):
    from garmentimport.geometry import rigid_shoe_parts
    body = live_body
    result = rigid_shoe_parts(RIGID_POINTS[:5],body)
    ids = sorted(v for a in result["attachments"] for v in a["vertices"])
    assert ids==[0,1,2,3]
    transforms = body.transforms(body.pose("walk"))
    for a in result["attachments"]:
        vertices=np.array(a["vertices"])
        ji=np.full((len(vertices),1),body.names.index(a["bone"]))
        posed=skin(RIGID_POINTS[vertices],ji,np.ones((len(vertices),1)),transforms)
        assert np.allclose(np.linalg.norm(posed[1:]-posed[0],axis=1),
                           np.linalg.norm(RIGID_POINTS[vertices][1:]-RIGID_POINTS[vertices][0],axis=1))


def test_rigid_sidecar_is_exactly_the_app_v1_format(live_body):
    """Mirrors `parseGarmentRigidAttachments` in packages/avatar-core/src/garment/rigid.ts."""
    from garmentimport.geometry import bone_head, rigid_shoe_parts
    body = live_body
    result = json.loads(json.dumps(rigid_shoe_parts(RIGID_POINTS,body,.035),allow_nan=False))
    assert list(result)==["version","soleHeightMm","attachments"] and result["version"]==1
    assert result["soleHeightMm"]==pytest.approx(35) and result["soleHeightMm"]>0
    assert {a["bone"] for a in result["attachments"]}=={"foot_l","ball_l","foot_r","ball_r"}
    seen=[]
    float32=RIGID_POINTS.astype(np.float32).astype(float)
    for a in result["attachments"]:
        assert list(a)==["bone","vertices","localPositionsM"]
        assert a["vertices"] and len(a["vertices"])==len(a["localPositionsM"])
        assert all(type(v) is int and 0<=v<len(RIGID_POINTS) for v in a["vertices"])
        assert all(len(p)==3 and all(isinstance(n,float) for n in p) for p in a["localPositionsM"])
        # local = exported (float32) neutral position - neutral head, so every vertex of a bone agrees on the head
        heads=float32[a["vertices"]]-np.array(a["localPositionsM"])
        assert np.allclose(heads,bone_head(body,a["bone"]),atol=1e-6)
        assert np.ptp(heads,axis=0).max()<1e-4
        seen += a["vertices"]
    assert len(seen)==len(set(seen))==8


def test_rigid_sidecar_is_accepted_by_avatar_core(tmp_path,live_body):
    import shutil
    from garmentimport.geometry import rigid_shoe_parts
    from garmentimport.validation import check_rigid_runtime
    node=shutil.which("node")
    if not node or not (PIPELINE.parents[1]/"packages/avatar-core/src/garment/rigid.ts").exists():
        pytest.skip("Node / avatar-core sources unavailable")
    rigid=rigid_shoe_parts(RIGID_POINTS,live_body)
    path=tmp_path/"shoe.rigid.json"; path.write_text(json.dumps(rigid),encoding="utf-8")
    faces=np.array([[0,1,2],[1,2,3],[2,3,4],[5,6,7],[6,7,8]])
    result=check_rigid_runtime(rigid,path,RIGID_POINTS,faces,live_body,node)
    assert result["ok"] and result["rigidVertices"]==8 and result["neutralRebuildMaxErrorM"]<1e-5
    assert sorted(a["bone"] for a in result["attachments"])==sorted(a["bone"] for a in rigid["attachments"])


def test_real_tshirt_reimport(tmp_path,live_body):
    if not (WARDROBE/"tshirt.glb").exists() or (WARDROBE/"tshirt.glb").read_bytes()[:4]!=b"glTF":
        pytest.skip("CC0 T-shirt asset not built / LFS not pulled")
    identity=tmp_path/"identity.json"
    identity.write_text(json.dumps({"matrix":np.eye(4).tolist(),"pose":"a"}),encoding="utf-8")
    args=parser().parse_args([str(WARDROBE/"tshirt.glb"),"--id","reimport","--category","top","--kind","tshirt",
        "--license","CC0-1.0","--attribution","CC0 T-shirt by MRT (MargaretToigo), MakeHuman community assets",
        "--transform",str(identity),"--out",str(tmp_path/"out"),"--texture-max","64",
        "--validate","--compare-original",str(WARDROBE/"tshirt.bind.bin")])
    report=import_garment(args)
    assert report["roundTrip"]["maxMm"]<1
    assert report["comparisonBeforeFit"]["maxMm"]<1
    assert report["validation"]["passed"]
    assert set(report["validation"]["poses"])=={"A","T","relaxed","walk"}
    assert all(shape["runtimeBindingMaxErrorMm"]<.001 for shape in report["validation"]["shapes"].values())
    entry=json.loads((args.out/"index.json").read_text(encoding="utf-8"))["garments"][0]
    assert len(read_binding(args.out/entry["binding"])[0])==report["output"]["vertices"]


def test_real_native_tshirt_keeps_authored_bindings(live_body):
    from garmentimport.native import load_native
    source = PIPELINE/".cache/garments/tshirt/toigo_basic_tucked_t-shirt.mhclo"
    if not source.exists() or not (WARDROBE/"tshirt.bind.bin").exists():
        pytest.skip("Permissive MakeHuman native T-shirt is not cached")
    parts,bind,points,refs,report,delete = load_native(source,live_body)
    original = read_binding(WARDROBE/"tshirt.bind.bin")
    assert report["helperReboundRows"]==0
    assert np.array_equal(bind[0],original[0])
    assert np.allclose(bind[1],original[1],atol=1e-7)
    assert np.linalg.norm(reconstruct(bind,live_body.positions,refs)-points,axis=1).max()<1e-7
    assert len(parts[0].positions)==len(bind[0])
    entry = next(e for e in json.loads((WARDROBE/"index.json").read_text())["garments"] if e["id"]=="tshirt")
    original_delete = np.frombuffer((WARDROBE/entry["deleteVerts"]).read_bytes(),dtype="<u4") if entry.get("deleteVerts") else np.zeros(0,int)
    assert np.array_equal(delete,original_delete)
