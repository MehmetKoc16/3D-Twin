import numpy as np
from synth import CENTER, VIEW_YAW, H, W, ellipsoid, make_cam, photo_from, twin_head
from twinrefine.scan import welded_vertex_normals
from twintex.raster import rasterize_uv

from headrecon.bake import BakeParams, bake_head
from headrecon.calibrate import Calib


def _texels(v, f, uv, n, S):
    tri, bary = rasterize_uv(uv * S, f, S, S)
    ty, tx = np.nonzero(tri >= 0)
    lam = bary[ty, tx]
    P = np.einsum("ij,ijk->ik", lam, v[f[tri[ty, tx]]])
    N = np.einsum("ij,ijk->ik", lam, n[f[tri[ty, tx]]])
    return ty, tx, P, N


def test_front_photo_colours_land_on_front_facing_texels():
    v, f, uv = ellipsoid((0.09, 0.115, 0.10), 3, soup=True)
    n = welded_vertex_normals(v, f)
    th = twin_head()
    atlas = np.zeros((256, 256, 3), np.uint8)
    atlas[...] = (255, 0, 255)  # sentinel: the old texture
    photos, calibs = {}, {}
    for name, yaw in VIEW_YAW.items():
        cam = make_cam(yaw, dist=0.5)
        photos[name] = photo_from(v, f, cam, name)
        calibs[name] = Calib(cam, 1.0, H)
    out, rep = bake_head(v, f, uv, n, atlas, th, photos, calibs, BakeParams(gutter=0))
    assert rep.texels > 1000
    ty, tx, P, N = _texels(v, f, uv, n, 256)
    sel = (N[:, 2] > 0.9) & (P[:, 1] > th.chin_y + 0.06)
    assert sel.sum() > 20
    p = make_cam(0.0).project(P[sel])
    expect = np.stack([p[:, 0] / W * 255, p[:, 1] / H * 255, np.full(sel.sum(), 120.0)], axis=1)
    got = out[ty[sel], tx[sel]].astype(float)
    # the photo gradient is plain sRGB, the bake works in linear light and back: allow quantisation / bilinear error
    assert np.abs(got - expect).mean() < 14.0
    head = P[:, 1] > th.chin_y + 0.03
    assert (np.abs(out[ty[head], tx[head]].astype(int) - [255, 0, 255]).sum(axis=1) > 60).mean() > 0.97


def test_unseen_texels_are_filled_not_left_dark():
    v, f, uv = ellipsoid((0.09, 0.115, 0.10), 3, soup=True)
    n = welded_vertex_normals(v, f)
    th = twin_head()
    atlas = np.full((256, 256, 3), 10, np.uint8)  # dark old texture (the "dark band")
    cam = make_cam(0.0)
    photos = {"front": photo_from(v, f, cam, "front", color_fn=np.array([200, 160, 140], np.uint8))}
    calibs = {"front": Calib(cam, 1.0, H)}
    out, _ = bake_head(v, f, uv, n, atlas, th, photos, calibs, BakeParams(gutter=0))
    ty, tx, P, _N = _texels(v, f, uv, n, 256)
    back = (P[:, 2] < CENTER[2] - 0.05) & (P[:, 1] > th.chin_y + 0.05)  # only the front photo exists: the back is unseen
    assert back.sum() > 20
    assert out[ty[back], tx[back]].astype(float).mean() > 100  # filled from the seen texels, not the dark old atlas
