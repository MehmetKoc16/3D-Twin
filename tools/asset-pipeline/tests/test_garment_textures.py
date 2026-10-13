"""Synthetic regression checks for shipped texture quality and payload size."""
import numpy as np
import pygltflib as gl
from PIL import Image

from garment_import import parser
from garmentimport.mesh_io import Part, atlas, dilate_padding, uv_coverage, write_glb


def part(image):
    material = {'base':{'image':image,'wrapS':33071,'wrapT':33071},'mr':None,'normal':None,
                'occlusion':None,'emissive':None,'factor':[1,1,1,1],'roughness':.7,'metallic':0,
                'emissiveFactor':[0,0,0],'alphaMode':'OPAQUE','alphaCutoff':.5,
                'normalScale':1,'occlusionStrength':1}
    return Part(np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]),
                np.array([[0.,0],[1,0],[0,1]]),material)


def test_base_color_ignores_uncovered_atlas_pixels():
    pixels = np.full((64,64,4),[255,100,0,255],dtype=np.uint8)
    y,x = np.indices((64,64))
    pixels[x+y <= 64] = [40,80,120,255]
    pos,faces,uv,channels,mode,color = atlas([part(Image.fromarray(pixels))],64)
    coverage = uv_coverage(uv,faces,64,64)
    mean = channels['base'][coverage,:3].mean(axis=0)
    assert color == '#'+''.join(f'{round(c):02x}' for c in mean)
    assert np.linalg.norm(mean-[40,80,120]) < 4


def test_padding_extends_nearest_edge_for_all_channels():
    image = np.zeros((8,8,4),np.uint8)
    coverage = np.zeros((8,8),bool)
    coverage[2:6,2:6] = True
    image[2:6,2:4] = [10,30,50,90]
    image[2:6,4:6] = [200,220,240,255]
    padded = dilate_padding(image,coverage)
    assert np.array_equal(padded[coverage],image[coverage])
    assert (padded[:,0]==[10,30,50,90]).all()
    assert (padded[:,-1]==[200,220,240,255]).all()


def test_atlas_padding_removes_colored_unused_uv_islands():
    pixels = np.full((64,64,4),[255,100,0,255],np.uint8)
    y,x = np.indices((64,64))
    pixels[x+y <= 65] = [40,80,120,255]
    _,faces,uv,channels,_,_ = atlas([part(Image.fromarray(pixels))],64)
    coverage = uv_coverage(uv,faces,64,64)
    assert (channels['base'][~coverage,:3] == [40,80,120]).all()


def test_identical_channel_images_share_glb_image_and_texture(tmp_path):
    p = part(Image.new('RGBA',(16,16),'white'))
    pos,faces,uv,channels,mode,_ = atlas([p],16)
    channels['occlusion'] = channels['base'].copy()
    path = tmp_path/'dedup.glb'
    write_glb(path,pos,faces,uv,channels,mode)
    g = gl.GLTF2().load(str(path))
    assert len(g.images)==2 and len(g.textures)==2
    assert g.materials[0].occlusionTexture.index == g.materials[0].pbrMetallicRoughness.baseColorTexture.index


def test_web_texture_default_and_explicit_2048():
    required = ['source.glb','--id','test','--category','shoes','--kind','sneakers',
                '--license','CC0-1.0','--attribution','Synthetic']
    assert parser().parse_args(required).texture_max==1024
    assert parser().parse_args(required+['--max-texture','2048']).texture_max==2048
    for limit in (1024,2048):
        p = part(Image.new('RGBA',(4096,2048),'white'))
        *_,channels,mode,color = atlas([p],limit)
        assert all(max(image.shape[:2])==limit for image in channels.values())


def test_untextured_source_without_uvs_keeps_constant_material():
    p = part(Image.new('RGBA',(16,16),'white'))
    p.material['base'] = None
    p.material['factor'] = [1,0,0,1]
    p.uv[:] = 0
    _,_,_,channels,_,color = atlas([p])
    assert color=='#ff0000'
    assert (channels['base']==[255,0,0,255]).all()


def test_constant_uv_samples_one_textured_texel():
    p = part(Image.new('RGBA',(16,16),(20,40,60,255)))
    p.uv[:] = .5
    _,_,_,channels,_,color = atlas([p],16)
    assert color=='#14283c'
    assert (channels['base']==[20,40,60,255]).all()
