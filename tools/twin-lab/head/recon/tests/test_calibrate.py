from synth import VIEW_YAW, ellipsoid, make_cam, photo_from, twin_head

from headrecon.calibrate import HeadFitMesh, calibrate_view


def test_calibration_recovers_the_camera():
    v, f, _ = ellipsoid((0.09, 0.115, 0.10), 3, soup=False)
    th = twin_head()
    hm = HeadFitMesh(v, f, th.chin_y)
    for name, yaw_true in (("front", 4.0), ("profile_nose_right", -86.0)):
        cam = make_cam(yaw_true, dist=0.55, tx=15.0, ty=-10.0, pitch=0.0)
        ph = photo_from(v, f, cam, name)
        cal = calibrate_view(ph, VIEW_YAW[name], hm, th.ymax, th.pivot, yaw_range=10.0)
        assert cal.iou > 0.9
        assert abs(cal.cam.dist - 0.55) < 0.12
        assert abs(cal.cam.yaw - yaw_true) < 8.0
