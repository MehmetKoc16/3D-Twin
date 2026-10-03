"""Write private previews for the lead; this module never displays images."""

import cv2
import numpy as np
from headrecon.previews import _label
from PIL import Image
from twinrefine.render import render
from twinrefine.scan import welded_vertex_normals
from twintex.camera import OrthoCamera


def write_previews(folder, scan, photos, cameras):
    folder.mkdir(parents=True, exist_ok=True)
    normals = welded_vertex_normals(scan.verts, scan.faces)
    head = scan.verts[:, 1] > scan.verts[:, 1].max() - 0.30
    tiles, paths = [], []
    for name, azimuth in (("front", 0), ("right", 270), ("left", 90), ("three_quarter", 35)):
        cam = OrthoCamera.azimuth(name, azimuth).fit_bounds(scan.verts[head], 560, 560, margin=0.06)
        rgb = render(scan.verts, scan.faces, normals, cam, 560, 560, scan.uv, scan.atlas, ss=1)
        path = folder / f"{name}.png"
        Image.fromarray(rgb).save(path)
        paths.append(str(path))
        view = "front" if name in {"front", "three_quarter"} else "right"
        y0, y1, x0, x1 = cameras[view].crop.astype(int)
        photo = photos[view]
        crop = photo[max(y0, 0) : min(y1, photo.shape[0]), max(x0, 0) : min(x1, photo.shape[1])]
        crop = cv2.resize(crop, (560, 560), interpolation=cv2.INTER_AREA)
        if name == "left":
            crop = crop[:, ::-1]
        tiles.append(
            np.concatenate(
                (_label(crop, f"{view} crop" + (" mirrored" if name == "left" else "")), _label(rgb, name)), axis=1
            )
        )
    contact = folder / "contact.png"
    Image.fromarray(np.concatenate(tiles, axis=0)).save(contact)
    return {"views": paths, "contact": str(contact)}
