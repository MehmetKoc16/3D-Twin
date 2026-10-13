"""Static GLB/OBJ loading and one-material PBR atlas export for the existing app loader."""
from __future__ import annotations

import base64
import copy
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygltflib as gl
from PIL import Image
import trimesh

from .geometry import nearest, surface


@dataclass
class Part:
    positions: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    material: dict


def accessor(g, index, path):
    a = g.accessors[index]
    if a.sparse is not None or a.bufferView is None:
        raise ValueError("Sparse glTF accessors must be expanded by the source converter")
    view = g.bufferViews[a.bufferView]
    buffer = g.buffers[view.buffer]
    blob = g.binary_blob() if buffer.uri is None else (
        base64.b64decode(buffer.uri.split(",", 1)[1]) if buffer.uri.startswith("data:") else (path.parent / buffer.uri).read_bytes())
    dtype = {5120:"i1", 5121:"u1", 5122:"<i2", 5123:"<u2", 5125:"<u4", 5126:"<f4"}[a.componentType]
    size = {"SCALAR":1, "VEC2":2, "VEC3":3, "VEC4":4, "MAT4":16}[a.type]
    stride = view.byteStride or np.dtype(dtype).itemsize*size
    result = np.ndarray((a.count, size), dtype=dtype, buffer=blob,
                        offset=(view.byteOffset or 0)+(a.byteOffset or 0), strides=(stride, np.dtype(dtype).itemsize)).copy()
    if a.normalized and a.componentType != 5126:
        result = np.maximum(result.astype(float)/np.iinfo(dtype).max, -1)
    return result


def texture(g, info, path):
    if info is None:
        return None
    if info.texCoord not in (None, 0) or info.extensions:
        raise ValueError("Bake alternate UV sets / texture transforms in the source converter first")
    tex = g.textures[info.index]
    if tex.source is None:
        raise ValueError("Compressed textures must be converted to PNG/JPEG first")
    im = g.images[tex.source]
    if im.bufferView is not None:
        v = g.bufferViews[im.bufferView]
        blob = g.binary_blob()[v.byteOffset or 0:(v.byteOffset or 0)+v.byteLength]
    elif im.uri.startswith("data:"):
        blob = base64.b64decode(im.uri.split(",", 1)[1])
    else:
        blob = (path.parent/im.uri).read_bytes()
    sampler = g.samplers[tex.sampler] if tex.sampler is not None else gl.Sampler()
    return {"image":Image.open(io.BytesIO(blob)).convert("RGBA"), "wrapS":sampler.wrapS or 10497,
            "wrapT":sampler.wrapT or 10497}


def gltf_material(g, index, path):
    m = g.materials[index] if index is not None else gl.Material()
    if m.extensions:
        raise ValueError("Bake nonstandard PBR material extensions in the source converter first")
    p = m.pbrMetallicRoughness or gl.PbrMetallicRoughness()
    return {"base":texture(g,p.baseColorTexture,path), "mr":texture(g,p.metallicRoughnessTexture,path),
            "normal":texture(g,m.normalTexture,path), "occlusion":texture(g,m.occlusionTexture,path),
            "emissive":texture(g,m.emissiveTexture,path), "factor":p.baseColorFactor or [1,1,1,1],
            "roughness":p.roughnessFactor if p.roughnessFactor is not None else 1.,
            "metallic":p.metallicFactor if p.metallicFactor is not None else 1.,
            "emissiveFactor":m.emissiveFactor or [0,0,0], "alphaMode":m.alphaMode or "OPAQUE",
            "alphaCutoff":m.alphaCutoff if m.alphaCutoff is not None else .5,
            "normalScale":m.normalTexture.scale if m.normalTexture and m.normalTexture.scale is not None else 1.,
            "occlusionStrength":m.occlusionTexture.strength if m.occlusionTexture and m.occlusionTexture.strength is not None else 1.}


def load_parts(path, node_names=None, exclude_material_names=None, mhmat_path=None, skin_pose="evaluated"):
    if skin_pose not in ("evaluated","bind"):
        raise ValueError("skin_pose must be evaluated or bind")
    if path.suffix.lower() in (".glb", ".gltf"):
        from scipy.spatial.transform import Rotation
        g = gl.GLTF2().load(str(path))
        if g.extensionsRequired:
            raise ValueError(f"Convert required glTF extensions first: {g.extensionsRequired}")
        parts = []
        parents = {child:i for i,n in enumerate(g.nodes) for child in n.children or []}
        worlds = {}
        def node_world(index):
            if index in worlds:
                return worlds[index]
            node = g.nodes[index]
            if node.matrix:
                local = np.array(node.matrix).reshape(4,4).T
            else:
                local = np.eye(4)
                local[:3,:3] = Rotation.from_quat(node.rotation or [0,0,0,1]).as_matrix() @ np.diag(node.scale or [1,1,1])
                local[:3,3] = node.translation or [0,0,0]
            worlds[index] = (node_world(parents[index]) if index in parents else np.eye(4)) @ local
            return worlds[index]
        def visit(index):
            node = g.nodes[index]
            world = node_world(index)
            if node.mesh is not None and (not node_names or node.name in node_names):
                for p in g.meshes[node.mesh].primitives:
                    if p.material is not None and g.materials[p.material].name in (exclude_material_names or []):
                        continue
                    if p.targets:
                        raise ValueError("Apply garment morph targets before import")
                    if p.mode not in (None,4):
                        raise ValueError("Export glTF triangle primitives before import")
                    pos = accessor(g,p.attributes.POSITION,path).astype(float)
                    if node.skin is not None and skin_pose=="evaluated":
                        skin = g.skins[node.skin]
                        inverse = accessor(g,skin.inverseBindMatrices,path).reshape(-1,4,4).transpose(0,2,1) if skin.inverseBindMatrices is not None else np.tile(np.eye(4),(len(skin.joints),1,1))
                        transforms = np.array([node_world(j) for j in skin.joints]) @ inverse
                        if p.attributes.JOINTS_0 is None or p.attributes.WEIGHTS_0 is None:
                            raise ValueError("Skinned GLB has no JOINTS_0/WEIGHTS_0")
                        blended = np.zeros((len(pos),4,4))
                        sums = np.zeros(len(pos))
                        for group in (0,1):
                            ja = getattr(p.attributes,f"JOINTS_{group}",None)
                            wa = getattr(p.attributes,f"WEIGHTS_{group}",None)
                            if ja is not None and wa is not None:
                                joints = accessor(g,ja,path).astype(int)
                                weights = accessor(g,wa,path).astype(float)
                                blended += (transforms[joints]*weights[:,:,None,None]).sum(axis=1)
                                sums += weights.sum(axis=1)
                        if np.any(sums<=0):
                            raise ValueError("Skinned source has unweighted vertices")
                        blended /= sums[:,None,None]
                        pos = np.einsum("nij,nj->ni",blended,np.column_stack([pos,np.ones(len(pos))]))[:,:3]
                    else:
                        pos = pos @ world[:3,:3].T + world[:3,3]
                    faces = accessor(g,p.indices,path).reshape(-1,3) if p.indices is not None else np.arange(len(pos)).reshape(-1,3)
                    if np.linalg.det(world[:3,:3]) < 0:
                        faces = faces[:,::-1]
                    uv = accessor(g,p.attributes.TEXCOORD_0,path) if p.attributes.TEXCOORD_0 is not None else np.zeros((len(pos),2))
                    material = gltf_material(g,p.material,path)
                    if p.attributes.TEXCOORD_0 is None and any(material[k] for k in ("base","mr","normal","occlusion","emissive")):
                        raise ValueError("Textured mesh has no UVs")
                    if p.attributes.COLOR_0 is not None:
                        raise ValueError("Bake vertex colours into the base-color texture first")
                    parts.append(Part(pos,faces.astype(np.int64),uv.astype(float),material))
            for child in node.children or []:
                visit(child)
        for root in g.scenes[g.scene or 0].nodes:
            visit(root)
    elif path.suffix.lower() == ".obj":
        scene = trimesh.load_scene(path, process=False, maintain_order=False)
        # MakeHuman OBJ companions use MHMAT instead of MTL; reuse our data-only parser.
        mh_material = None
        material_path = mhmat_path or path.with_suffix(".mhmat")
        if material_path.exists():
            import mhclo
            mh_material = mhclo.parse_mhmat(material_path.read_text(encoding="utf-8",errors="replace"))
        parts = []
        for name in sorted(scene.graph.nodes_geometry):
            transform, key = scene.graph[name]
            mesh = scene.geometry[key]
            pos = trimesh.transform_points(mesh.vertices,transform)
            mat = getattr(mesh.visual,"material",None)
            pbr = mat.to_pbr() if mat is not None and hasattr(mat,"to_pbr") else mat
            m = {"factor":np.asarray(getattr(pbr,"baseColorFactor",None) if getattr(pbr,"baseColorFactor",None) is not None else [255]*4)/255.,
                 "roughness":pbr.roughnessFactor if getattr(pbr,"roughnessFactor",None) is not None else .85,
                 "metallic":pbr.metallicFactor if getattr(pbr,"metallicFactor",None) is not None else 0.,
                 "emissiveFactor":list(pbr.emissiveFactor) if getattr(pbr,"emissiveFactor",None) is not None else [0,0,0],"alphaMode":"OPAQUE","alphaCutoff":.5,
                 "normalScale":1.,"occlusionStrength":1.}
            for key, field in (("base","baseColorTexture"),("mr","metallicRoughnessTexture"),("normal","normalTexture"),
                               ("occlusion","occlusionTexture"),("emissive","emissiveTexture")):
                image = getattr(pbr,field,None)
                m[key] = {"image":image.convert("RGBA"),"wrapS":10497,"wrapT":10497} if image else None
            if mh_material is not None:
                m["factor"] = [*[float(v) for v in mh_material.get("diffuseColor","1 1 1").split()[:3]],float(mh_material.get("opacity","1"))]
                for key,slot in (("base","diffuseTexture"),("normal","normalmapTexture")):
                    if slot in mh_material:
                        tex_path = material_path.parent/mh_material[slot]
                        if not tex_path.exists():
                            raise ValueError(f"MHMAT texture is missing: {tex_path.name}")
                        m[key] = {"image":Image.open(tex_path).convert("RGBA"),"wrapS":10497,"wrapT":10497}
                if m["factor"][3]<1:
                    m["alphaMode"] = "BLEND"
                if "specularTexture" in mh_material:
                    m["warnings"] = ["Legacy MHMAT specularTexture is not a metallic/roughness map; convert it to PBR before shipping."]
            uv = getattr(mesh.visual,"uv",None)
            uv = np.array(uv, float) if uv is not None else np.zeros((len(pos),2))
            uv[:,1] = 1-uv[:,1]  # OBJ bottom-left -> glTF top-left.
            faces = mesh.faces.copy()
            if np.linalg.det(transform[:3,:3]) < 0:
                faces = faces[:,::-1]
            parts.append(Part(pos,faces,uv,m))
    else:
        raise ValueError("Use GLB, glTF or OBJ; convert FBX to a static GLB in your authoring tool")
    if not parts:
        raise ValueError("No triangle garment meshes found")
    return parts


def clean(part):
    if not np.isfinite(part.positions).all() or not np.isfinite(part.uv).all():
        raise ValueError("Nonfinite garment positions or UVs")
    if not len(part.faces):
        raise ValueError("Garment has no triangles")
    if part.faces.min() < 0 or part.faces.max() >= len(part.positions):
        raise ValueError("Garment triangle index outside vertex buffer")
    # Keep seams: merge only identical position + UV pairs, preserving first occurrence order.
    _, first, remap = np.unique(np.round(np.column_stack([part.positions,part.uv]),10),axis=0,return_index=True,return_inverse=True)
    order = np.argsort(first)
    reverse = np.empty(len(order),int); reverse[order] = np.arange(len(order))
    first = first[order]
    part.positions,part.uv,part.faces = part.positions[first],part.uv[first],reverse[remap[part.faces]]
    mesh = surface(part.positions,part.faces)
    part.faces = part.faces[mesh.nondegenerate_faces()]
    _, first = np.unique(np.sort(part.faces,axis=1),axis=0,return_index=True)
    part.faces = part.faces[np.sort(first)]
    keep = np.unique(part.faces)
    remap = np.zeros(len(part.positions),int); remap[keep] = np.arange(len(keep))
    part.positions,part.uv,part.faces = part.positions[keep],part.uv[keep],remap[part.faces]
    if not len(part.faces):
        raise ValueError("Garment has no nondegenerate triangles")
    return part


def decimate(parts, budget):
    """Quadric decimation per UV-connected island, then barycentric UV transfer within that island."""
    import fast_simplification
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    total = sum(len(p.faces) for p in parts)
    if total <= budget:
        return parts
    output = []
    for p in parts:
        edges = np.concatenate([p.faces[:,[0,1]],p.faces[:,[1,2]],p.faces[:,[2,0]]])
        graph = coo_matrix((np.ones(len(edges)),edges.T),shape=(len(p.positions),len(p.positions))).tocsr()
        _, labels = connected_components(graph,directed=False)
        for label in np.unique(labels):
            faces = p.faces[labels[p.faces[:,0]] == label]
            keep = np.unique(faces)
            remap = np.zeros(len(p.positions),int); remap[keep] = np.arange(len(keep))
            vertices, uv, faces = p.positions[keep], p.uv[keep], remap[faces]
            target = max(1, int(len(faces)*budget/total))
            if len(faces) > max(target,4):
                source = surface(vertices,faces)
                vertices, faces = fast_simplification.simplify(vertices,faces,target_count=target)
                closest,_,tid = nearest(source,vertices)
                bary = trimesh.triangles.points_to_barycentric(source.triangles[tid],closest)
                uv = (uv[source.faces[tid]]*bary[...,None]).sum(axis=1)
            output.append(clean(Part(vertices,faces,uv,copy.deepcopy(p.material))))
    # Preserve material/island identities for atlas packing; refuse an unmet cap rather than destroy ornaments.
    if sum(len(p.faces) for p in output) > budget*1.1:
        raise ValueError("Too many independent UV islands for this triangle budget; simplify the source or increase --max-tris")
    return output


def refine(parts, levels, budget):
    """Linear midpoint subdivision: sample coarse triangle interiors without smoothing exterior fabric."""
    if sum(len(p.faces) for p in parts)*4**levels > budget:
        raise ValueError("Subdivision exceeds max-tris; reduce levels or explicitly increase the review budget")
    for p in parts:
        for _ in range(levels):
            p.positions,p.faces,attributes = trimesh.remesh.subdivide(
                p.positions,p.faces,vertex_attributes={"uv":p.uv})
            p.uv = attributes["uv"]
    return parts


def _sample(tex, u, v, default):
    if tex is None:
        return np.broadcast_to(default, (*u.shape,4)).copy().astype(float)
    def wrap(x, mode):
        if mode == 33071:
            return np.clip(x,0,1)
        if mode == 33648:
            return 1-np.abs(np.mod(x,2)-1)
        return np.mod(x,1)
    arr = np.asarray(tex["image"])
    x = np.minimum((wrap(u,tex["wrapS"])*arr.shape[1]).astype(int),arr.shape[1]-1)
    y = np.minimum((wrap(v,tex["wrapT"])*arr.shape[0]).astype(int),arr.shape[0]-1)
    return arr[y,x].astype(float)/255


def uv_coverage(uv, faces, width, height):
    """Rasterize texel centers covered by UV triangles, excluding unused atlas islands."""
    mask = np.zeros((height, width), bool)
    pixels = uv * [width, height] - .5
    for triangle in pixels[faces]:
        lo = np.maximum(np.ceil(triangle.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.floor(triangle.max(axis=0)).astype(int), [width-1, height-1])
        if np.any(hi < lo):
            continue
        a, b = triangle[1]-triangle[0], triangle[2]-triangle[0]
        det = a[0]*b[1]-a[1]*b[0]
        if abs(det) < 1e-12:
            continue
        x, y = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        x, y = x-triangle[0,0], y-triangle[0,1]
        u, v = (x*b[1]-y*b[0])/det, (a[0]*y-a[1]*x)/det
        mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1] |= (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)
    return mask


def dilate_padding(image, coverage):
    """Extend the nearest covered texel into every unused texel, for mip-safe gutters."""
    from scipy.ndimage import distance_transform_edt
    if not coverage.any():
        raise ValueError("UV triangles cover no texture texel centers; increase max-texture")
    indices = distance_transform_edt(~coverage, return_distances=False, return_indices=True)
    return image[indices[0], indices[1]]


def atlas(parts, max_size=1024):
    # Group islands with the same material by equality of serialized scalar values and image content.
    import hashlib
    groups = {}
    for p in parts:
        m = p.material
        key = tuple((k,v["image"].size,hashlib.sha256(v["image"].tobytes()).hexdigest(),v["wrapS"],v["wrapT"]) if isinstance(v,dict)
                    else (k,str(v)) for k,v in sorted(m.items()))
        groups.setdefault(key,[]).append(p)
    side = math.ceil(math.sqrt(len(groups)))
    source_size = max([16]+[max(value["image"].size) for p in parts for value in p.material.values()
                            if isinstance(value,dict) and "image" in value])
    tile = min(max_size//side,source_size)
    if tile < 16:
        raise ValueError("Too many materials for the texture atlas")
    size = tile*side
    channels = {k:np.zeros((size,size,4),np.uint8) for k in ("base","mr","normal","occlusion","emissive")}
    positions,uvs,faces = [],[],[]
    covered = np.zeros((size,size),bool)
    n = 0
    for i, group in enumerate(groups.values()):
        m = group[0].material
        source_uv = np.concatenate([p.uv for p in group])
        lo, hi = source_uv.min(axis=0),source_uv.max(axis=0)
        span = np.maximum(hi-lo,1e-6)
        # Two pixels of replicated gutter, including the UV domain endpoints.
        coords = np.clip((np.arange(tile)-2)/(tile-5),0,1)
        u,v = np.meshgrid(lo[0]+coords*span[0],lo[1]+coords*span[1])
        defaults = {"base":[1,1,1,1],"mr":[1,1,1,1],"normal":[.5,.5,1,1],"occlusion":[1,1,1,1],"emissive":[1,1,1,1]}
        row,col = divmod(i,side)
        group_uv = (source_uv-lo)/span*(tile-5)+2.5
        group_faces = []
        offset = 0
        for p in group:
            group_faces.append(p.faces+offset)
            offset += len(p.positions)
        coverage = uv_coverage(group_uv/tile,np.concatenate(group_faces),tile,tile)
        if not coverage.any() and (not any(m[k] is not None for k in channels) or
                                   np.ptp(source_uv,axis=0).max()<1e-9):
            # Constant UVs sample a single texel, including when an untextured
            # source is re-imported from our constant-texture glTF export.
            coverage[2,2] = True
        covered[row*tile:(row+1)*tile,col*tile:(col+1)*tile] = coverage
        for key in channels:
            arr = _sample(m[key],u,v,defaults[key])
            if key in ("base","emissive"):
                rgb = arr[...,:3]
                linear = np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                linear *= np.asarray(m["factor"][:3] if key=="base" else m["emissiveFactor"])
                arr[...,:3] = np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055)
                if key == "base":
                    arr[...,3] *= m["factor"][3]
                    if m["alphaMode"] == "OPAQUE":
                        arr[...,3] = 1
                    elif m["alphaMode"] == "MASK":
                        arr[...,3] = (arr[...,3] >= m["alphaCutoff"]).astype(float)
            elif key == "mr":
                arr[...,1] *= m["roughness"]; arr[...,2] *= m["metallic"]
            elif key == "normal":
                normal = arr[...,:3]*2-1
                normal[...,:2] *= m["normalScale"]
                normal /= np.maximum(np.linalg.norm(normal,axis=2,keepdims=True),1e-9)
                arr[...,:3] = (normal+1)/2
            elif key == "occlusion":
                arr[...,0] = 1+m["occlusionStrength"]*(arr[...,0]-1)
            baked = np.round(np.clip(arr,0,1)*255).astype(np.uint8)
            channels[key][row*tile:(row+1)*tile,col*tile:(col+1)*tile] = dilate_padding(baked,coverage)
        for p in group:
            mapped = (p.uv-lo)/span*(tile-5)+2
            mapped += [col*tile+.5,row*tile+.5]
            positions.append(p.positions); uvs.append(mapped/size); faces.append(p.faces+n); n += len(p.positions)
    mode = "BLEND" if any(p.material["alphaMode"]=="BLEND" for p in parts) else "MASK" if any(p.material["alphaMode"]=="MASK" for p in parts) else "OPAQUE"
    base = channels["base"][covered,:3].mean(axis=0)/255
    # Fill unused material-grid cells from the nearest occupied cell as well.
    occupied = np.zeros_like(covered)
    for i in range(len(groups)):
        row,col = divmod(i,side)
        occupied[row*tile:(row+1)*tile,col*tile:(col+1)*tile] = True
    if not occupied.all():
        channels = {key:dilate_padding(value,occupied) for key,value in channels.items()}
    for key in ("normal","occlusion","emissive"):
        used = any(p.material[key] is not None for p in parts)
        if key=="emissive":
            used |= any(np.any(p.material["emissiveFactor"]) for p in parts)
        if not used:
            del channels[key]
    return np.concatenate(positions),np.concatenate(faces),np.concatenate(uvs),channels,mode,"#"+"".join(f"{round(c*255):02x}" for c in base)


def write_glb(path, positions, faces, uv, channels, alpha_mode):
    g = gl.GLTF2(asset=gl.Asset(version="2.0",generator="Dijital Ikiz garment importer"))
    blob = bytearray()
    def view(data,target=None):
        blob.extend(b"\0"*((-len(blob))%4))
        index = len(g.bufferViews)
        g.bufferViews.append(gl.BufferView(buffer=0,byteOffset=len(blob),byteLength=len(data),target=target))
        blob.extend(data)
        return index
    def acc(arr,dtype,kind,component,target):
        arr = np.ascontiguousarray(arr,dtype=dtype)
        index = len(g.accessors)
        bounds = {"min":arr.min(axis=0).tolist(),"max":arr.max(axis=0).tolist()} if kind=="VEC3" else {}
        g.accessors.append(gl.Accessor(bufferView=view(arr.tobytes(),target),componentType=component,count=len(arr),type=kind,**bounds))
        return index
    normals = surface(positions,faces,weld=True)
    # Smooth normals across UV copies without changing positions or vertex ordering.
    from scipy.spatial import cKDTree
    _, mapping = cKDTree(normals.vertices).query(positions)
    attrs = gl.Attributes(POSITION=acc(positions,"<f4","VEC3",5126,34962),
                          NORMAL=acc(normals.vertex_normals[mapping],"<f4","VEC3",5126,34962),
                          TEXCOORD_0=acc(uv,"<f4","VEC2",5126,34962))
    if "normal" in channels:
        triangles=positions[faces]; tex_uv=uv[faces]
        e1,e2=triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]
        d1,d2=tex_uv[:,1]-tex_uv[:,0],tex_uv[:,2]-tex_uv[:,0]
        det=d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0]
        inv=np.zeros(len(det)); good=np.abs(det)>1e-12; inv[good]=1/det[good]
        tangent=(e1*d2[:,1,None]-e2*d1[:,1,None])*inv[:,None]
        bitangent=(e2*d1[:,0,None]-e1*d2[:,0,None])*inv[:,None]
        t,b=np.zeros_like(positions),np.zeros_like(positions)
        for corner in range(3):
            np.add.at(t,faces[:,corner],tangent); np.add.at(b,faces[:,corner],bitangent)
        n=normals.vertex_normals[mapping]
        t-=n*(t*n).sum(axis=1,keepdims=True)
        length=np.linalg.norm(t,axis=1)
        fallback=length<1e-9
        axis=np.zeros_like(t); axis[:,0]=1; axis[np.abs(n[:,0])>.9]=[0,1,0]
        t[fallback]=np.cross(axis[fallback],n[fallback])
        t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-12)
        sign=np.where((np.cross(n,t)*b).sum(axis=1)<0,-1.,1.)
        attrs.TANGENT=acc(np.column_stack([t,sign]),"<f4","VEC4",5126,34962)
    indices = acc(faces.reshape(-1),"<u4","SCALAR",5125,34963)
    tex = {}
    g.samplers = [gl.Sampler(magFilter=9729,minFilter=9987,wrapS=33071,wrapT=33071)]
    images = {}
    for key, image in channels.items():
        data = io.BytesIO(); Image.fromarray(image).save(data,format="PNG",optimize=True)
        payload = data.getvalue()
        if payload in images:
            tex[key] = images[payload]
            continue
        tex[key] = len(g.textures)
        images[payload] = tex[key]
        g.images.append(gl.Image(bufferView=view(payload),mimeType="image/png"))
        g.textures.append(gl.Texture(source=len(g.images)-1,sampler=0))
    g.materials = [gl.Material(pbrMetallicRoughness=gl.PbrMetallicRoughness(
        baseColorFactor=[1,1,1,1],baseColorTexture=gl.TextureInfo(index=tex["base"]),
        metallicFactor=1,roughnessFactor=1,metallicRoughnessTexture=gl.TextureInfo(index=tex["mr"])),
        normalTexture=gl.NormalMaterialTexture(index=tex["normal"]) if "normal" in tex else None,
        occlusionTexture=gl.OcclusionTextureInfo(index=tex["occlusion"]) if "occlusion" in tex else None,
        emissiveTexture=gl.TextureInfo(index=tex["emissive"]) if "emissive" in tex else None,
        emissiveFactor=[1,1,1] if "emissive" in tex else [0,0,0],
        alphaMode=alpha_mode,alphaCutoff=.5 if alpha_mode=="MASK" else None,doubleSided=True)]
    g.meshes = [gl.Mesh(primitives=[gl.Primitive(attributes=attrs,indices=indices,material=0)])]
    g.nodes = [gl.Node(mesh=0,name=path.stem)]; g.scenes = [gl.Scene(nodes=[0])]; g.scene=0
    blob.extend(b"\0"*((-len(blob))%4)); g.buffers = [gl.Buffer(byteLength=len(blob))]
    g.set_binary_blob(bytes(blob)); g.save_binary(str(path))
