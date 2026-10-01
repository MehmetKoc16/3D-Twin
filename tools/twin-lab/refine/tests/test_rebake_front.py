import cv2
import numpy as np
import pytest
from helpers import flat_face_depth, sheet_scan
from twintex.camera import OrthoCamera
from twintex.colorspace import linear_to_u8, u8_to_linear
from twintex.views import View

from twinrefine import facedepth, frontview, landmarks, rebake


def _grid_source(grid, name="front", scale=1.0):
    """Photo that is the face grid itself: pixel (i, j) = world cell (i, j); colour = smooth function of the pixel."""
    H, W = grid.H, grid.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lin = np.stack([xx / W, yy / H, 0.5 + 0 * xx], axis=-1).astype(np.float32) * 0.8 * scale

    def to_pixel(P):
        px, py = grid.to_px(P[:, 0], P[:, 1])
        return np.stack([px - 0.5, py - 0.5], axis=1)

    return rebake.PhotoSource(name, lin, to_pixel), lin


def test_rebake_paints_face_texels_with_photo_colours():
    mc = sheet_scan(30)
    grid = facedepth.Grid.around(np.array([[-0.08, 1.54], [0.08, 1.71]]), 0.01, 0.0005)
    fd = flat_face_depth(grid, mc, offset=0.0)
    atlas = np.full((128, 128, 3), 77, np.uint8)
    src, lin = _grid_source(grid)
    out, rep = rebake.rebake_face(mc, atlas, fd, [src])
    assert rep.texels > 500 and rep.mean_weight > 0.3
    changed = (out != atlas).any(axis=2)
    assert changed.sum() > 500
    # texel centres in the face centre carry the photo colour at their world position
    res = rebake.face_texels(mc, np.arange(len(mc.F)), 128)
    rows, cols, P, _fid, _lam = res
    centre = np.hypot(P[:, 0], P[:, 1] - 1.62) < 0.03
    s = src.to_pixel(P[centre])
    expect = linear_to_u8(cv2.remap(lin, s[:, 0].reshape(1, -1).astype(np.float32), s[:, 1].reshape(1, -1).astype(np.float32),
                                    cv2.INTER_CUBIC).reshape(-1, 3))
    got = out[rows[centre], cols[centre]]
    assert np.abs(got.astype(int) - expect.astype(int)).mean() < 2.5
    # texels far from the face keep the old colour
    far = np.hypot(P[:, 0], P[:, 1] - 1.62) > 0.075
    assert (out[rows[far], cols[far]] == 77).all()


def test_rebake_portrait_gains_match_the_front_photo():
    mc = sheet_scan(30)
    grid = facedepth.Grid.around(np.array([[-0.08, 1.54], [0.08, 1.71]]), 0.01, 0.0005)
    fd = flat_face_depth(grid, mc, offset=0.0, radius=0.06)
    atlas = np.full((128, 128, 3), 77, np.uint8)
    portrait, _ = _grid_source(grid, "portrait", scale=0.8)  # darker portrait
    front, _ = _grid_source(grid, "front", scale=1.0)
    out, rep = rebake.rebake_face(mc, atlas, fd, [portrait, front])
    assert rep.gains is not None
    np.testing.assert_allclose(rep.gains, 1.25, atol=0.03)  # portrait * 1.25 = front
    out_front, _ = rebake.rebake_face(mc, atlas, fd, [front])
    inner = (out_front != atlas).any(axis=2)
    assert np.abs(out[inner].astype(int) - out_front[inner].astype(int)).mean() < 3.0
    assert rep.sources == ["portrait", "front"]


def _synthetic_front_view(flow_amp=4.0):
    H, W = 300, 200
    cam = OrthoCamera.axis("front")
    cam.scale, cam.tx, cam.ty, cam.width, cam.height = 150.0, 100.0, 250.0, W, H
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    flow = np.stack([flow_amp * np.sin(yy / 40.0), flow_amp * np.cos(xx / 35.0)], axis=-1).astype(np.float32)
    rgb = np.zeros((H, W, 3), np.uint8)
    view = View("front", cam, u8_to_linear(rgb), np.ones((H, W), np.float32), np.ones((H, W), np.float32), flow=flow)
    return frontview.FrontView(rgb, np.ones((H, W), np.float32), view)


def test_frontview_world_photo_mappings_are_inverse():
    fv = _synthetic_front_view()
    rng = np.random.default_rng(0)
    P = np.column_stack([rng.uniform(-0.3, 0.3, 50), rng.uniform(0.2, 1.4, 50), np.zeros(50)])
    s = fv.world_to_photo(P)
    back = fv.photo_to_world_xy(s)
    np.testing.assert_allclose(back, P[:, :2], atol=2e-4)  # < 0.2 mm
    fv0 = _synthetic_front_view(0.0)
    s0 = fv0.world_to_photo(P)
    # without flow it is the plain camera: x_px = scale * x + tx (pixel index = pixel coordinate - 0.5)
    np.testing.assert_allclose(s0[:, 0], 150.0 * P[:, 0] + 100.0 - 0.5, atol=1e-4)


def test_person_head_crop_follows_the_mask():
    H, W = 800, 400
    alpha = np.zeros((H, W), np.float32)
    alpha[100:700, 250:330] = 1.0  # a person standing off-centre
    x0, y0, x1, y1 = landmarks.person_head_crop(alpha, (H, W), 0.2)
    assert 200 <= x0 < 250 and x1 > 300
    assert y0 <= 100 < y1 and (y1 - y0) == pytest.approx(0.2 * 600, abs=2)
    assert landmarks.person_head_crop(None, (H, W), 0.2)[0] >= 0


def test_detect_landmarks_returns_none_without_a_face():
    pytest.importorskip("mediapipe")
    if not landmarks.LANDMARKER_TASK.exists():
        pytest.skip("face_landmarker.task not present")
    blank = np.full((400, 300, 3), 180, np.uint8)
    assert landmarks.detect_landmarks(blank, None, full_frame=True) is None
