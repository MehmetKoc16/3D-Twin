"""Close-up shoe inspection using the importer's orthographic preview renderer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .body import load_body, skin
from .geometry import prism_coverage, skin_weights
from .mesh_io import load_parts
from .validation import read_binding
from .visibility import hidden_triangles, triangle_pairs


def review(directory, identifier, prefix="fixed"):
    from parts_debug import Scene
    root = Path(__file__).resolve().parents[3]
    body = load_body(root/"apps/web/public/assets/body")
    entry = next(e for e in json.loads((directory/"index.json").read_text())['garments'] if e['id']==identifier)
    part = load_parts(directory/entry['mesh'])[0]
    bind = read_binding(directory/entry['binding'])
    joints, weights = skin_weights(bind,body.joints,body.weights)
    deleted = np.fromfile(directory/entry['deleteVerts'],dtype='<u4') if entry.get('deleteVerts') else []
    effective = np.union1d(deleted,np.flatnonzero(prism_coverage(body.positions[:body.render_count],part.positions,part.faces)))
    hidden = hidden_triangles(body.faces,body.render_count,effective)
    rigid_path = directory/entry['rigidAttachments'] if entry.get('rigidAttachments') else None
    if rigid_path is not None:
        for a in json.loads(rigid_path.read_text())['attachments']:
            ids = np.array(a['vertices'],int)
            joints[ids] = body.names.index(a['bone'])
            weights[ids] = [1,0,0,0]
    panels, metrics = [], {}
    for pose in ('A','walk'):
        transforms = body.transforms(body.pose('a-pose' if pose=='A' else 'walk'))
        bp = skin(body.positions[:body.render_count],body.joints,body.weights,transforms)
        gp = skin(part.positions,joints,weights,transforms)
        pairs = triangle_pairs(gp,part.faces)
        metrics[pose] = {'pairs':len(pairs),'affectedTriangles':len(np.unique(pairs)),
                         'triangleCount':len(part.faces),'percent':len(np.unique(pairs))/len(part.faces)*100}
        for foot in ('both','left','right'):
            face_ids = np.ones(len(part.faces),bool) if foot=='both' else (
                (part.positions[part.faces,0]>0).all(axis=1) if foot=='left' else (part.positions[part.faces,0]<0).all(axis=1))
            for view,angle in (('front',0),('three-quarter',45 if foot!='right' else -45),('side',90 if foot!='right' else -90)):
                theta = np.deg2rad(angle)
                rotation = np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])
                points = gp@rotation.T
                region = points[np.unique(part.faces[face_ids])]
                lo,hi = region.min(axis=0)-[.012,.015,.012],region.max(axis=0)+[.012,.065,.012]
                texture = np.array(part.material['base']['image'])
                canvas = Scene([{'pos':bp@rotation.T,'tris':body.faces[~hidden],'color':[195,154,123]}],
                               'front',(lo,hi),800).render([{'pos':points,'tris':part.faces[face_ids],
                                'uv':part.uv,'tex':texture,'color':[255,255,255]}])
                image = Image.fromarray(canvas)
                path = directory/f'{prefix}-{pose}-{foot}-{view}.png'
                image.save(path)
                panel = Image.new('RGB',(410,330),'white')
                image.thumbnail((400,300))
                panel.paste(image,(0,25))
                ImageDraw.Draw(panel).text((3,3),f'{prefix} {pose} {foot} {view}',fill='black')
                panels.append(panel)
        print(pose,metrics[pose],flush=True)
    contact = Image.new('RGB',(410*3,330*6),'white')
    for i,panel in enumerate(panels):
        contact.paste(panel,((i%3)*410,(i//3)*330))
    contact.save(directory/f'{prefix}-closeups.png')
    metrics['limitations'] = 'CPU textured face shading; no normal map, mipmap or app 10 mm rigid blend simulation.'
    (directory/f'{prefix}-closeups.json').write_text(json.dumps(metrics,indent=2)+'\n')


if __name__=='__main__':
    from garment_import import require_venv
    require_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--id',default='khronos-materials-variants-shoe')
    parser.add_argument('--prefix',default='fixed')
    args = parser.parse_args()
    review(args.directory,args.id,args.prefix)
