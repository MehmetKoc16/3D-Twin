import numpy as np
from synth import VIEW_YAW, H, W, ellipsoid, make_cam, photo_from, twin_head

from headrecon.calibrate import SHRINK, Calib
from headrecon.reshape import ReshapeParams, reshape_head
from headrecon.viewcam import iou, sample_surface, silhouette


def _setup():
    target, tf, _ = ellipsoid((0.075, 0.115, 0.095), 3, soup=False)  # narrower head than the twin
    v, f, _ = ellipsoid((0.09, 0.115, 0.10), 3, soup=True)
    photos, calibs = {}, {}
    for name, yaw in VIEW_YAW.items():
        cam = make_cam(yaw, dist=0.5)
        photos[name] = photo_from(target, tf, cam, name)
        calibs[name] = Calib(cam, 0.9, H * 0.55)
    return v, f, target, tf, photos, calibs


def test_reshape_narrows_the_head_towards_the_photos():
    v, f, target, tf, photos, calibs = _setup()
    th = twin_head()
    res = reshape_head(v, f, th, photos, calibs, prm=ReshapeParams(rounds=4))
    w0 = np.ptp(v[:, 0])
    w1 = np.ptp(res.verts[:, 0])
    assert w1 < w0 - 0.008  # moved a good part of the 15 mm that the target is narrower
    hist = res.stats["rounds"]
    assert hist[-1]["front"] < 0.6 * hist[0]["front"]
    cam = calibs["front"].cam
    t = silhouette(cam, sample_surface(target, tf), W, H, SHRINK) > 0
    row = int(cam.project(np.array([[0.0, th.chin_y + 0.02, 0.0]]))[0, 1] * SHRINK)
    keep0 = (v[f][:, :, 1] > th.chin_y + 0.02).all(axis=1)
    keep1 = (res.verts[f][:, :, 1] > th.chin_y + 0.02).all(axis=1)
    a = silhouette(cam, sample_surface(v, f[keep0]), W, H, SHRINK) > 0
    b = silhouette(cam, sample_surface(res.verts, f[keep1]), W, H, SHRINK) > 0
    assert iou(b[:row], t[:row]) > iou(a[:row], t[:row])


def test_reshape_leaves_the_neck_alone_and_keeps_seams_closed():
    v, f, target, tf, photos, calibs = _setup()
    th = twin_head()
    th.chin_y = 1.70  # the ellipsoid bottom is at 1.535: its lower part is beyond the falloff band
    res = reshape_head(v, f, th, photos, calibs, prm=ReshapeParams(rounds=2))
    low = v[:, 1] < th.chin_y - 0.11
    assert low.any()
    assert np.abs(res.disp[low]).max() < 1e-9
    # vertices with identical positions (triangle soup) move identically: no cracks
    key = np.round(v * 1e5).astype(np.int64)
    _u, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    ref = np.zeros((inv.max() + 1, 3))
    ref[inv] = res.disp
    assert np.abs(res.disp - ref[inv]).max() < 1e-9
    assert not np.isnan(res.verts).any()
