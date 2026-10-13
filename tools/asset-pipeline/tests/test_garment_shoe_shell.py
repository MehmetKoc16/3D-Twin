"""Synthetic shoe catalogue and reflection regressions."""
import json

import numpy as np
import pygltflib as gl
import trimesh

from garment_import import import_garment, parser
from garmentimport.mesh_io import Part, atlas, load_parts, write_glb
from test_garment_textures import part


def test_glb_reflection_keeps_outward_winding(tmp_path):
    from PIL import Image
    p = part(Image.new('RGBA',(16,16),'white'))
    points,faces,uv,maps,mode,_ = atlas([p],16)
    path = tmp_path/'mirror.glb'
    write_glb(path,points,faces,uv,maps,mode)
    g = gl.GLTF2().load(str(path))
    g.nodes[0].scale = [-1,1,1]
    g.save_binary(str(path))
    mirrored = load_parts(path)[0]
    normal = np.cross(mirrored.positions[mirrored.faces[0,1]]-mirrored.positions[mirrored.faces[0,0]],
                      mirrored.positions[mirrored.faces[0,2]]-mirrored.positions[mirrored.faces[0,0]])
    assert normal[2]>0


def test_shoe_import_catalogue_links_generated_rigid_sidecar(tmp_path):
    from PIL import Image
    boxes = trimesh.util.concatenate([
        trimesh.creation.box(extents=[.13,.12,.28],transform=trimesh.transformations.translation_matrix([x,.035,.09]))
        for x in (.188,-.188)])
    material = part(Image.new('RGBA',(16,16),'white')).material
    material['base'] = None
    points,faces,uv,maps,mode,_ = atlas([Part(boxes.vertices,boxes.faces,np.zeros((len(boxes.vertices),2)),material)],16)
    source = tmp_path/'shoes.glb'
    write_glb(source,points,faces,uv,maps,mode)
    transform = tmp_path/'identity.json'
    transform.write_text(json.dumps({'pose':'a','matrix':np.eye(4).tolist()}))
    args = parser().parse_args([str(source),'--id','synthetic-shoes','--category','shoes','--kind','sneakers',
                               '--license','CC0-1.0','--attribution','Synthetic CC0 test shoes',
                               '--transform',str(transform),'--out',str(tmp_path/'out'),'--shell','keep',
                               '--no-delete-mask'])
    import_garment(args)
    entry = json.loads((args.out/'index.json').read_text())['garments'][0]
    assert entry['rigidAttachments']=='synthetic-shoes.rigid.json'
    rigid = json.loads((args.out/entry['rigidAttachments']).read_text())
    assert rigid['version']==1 and len(rigid['attachments'])==4
