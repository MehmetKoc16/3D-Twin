import numpy as np
import pytest
from twintex.camera import OrthoCamera
from twintex.colorspace import u8_to_linear
from twintex.views import View

from twinrefine import facedepth, facefit, frontview, hairmask


@pytest.fixture(scope="module")
def landmarks_world(model, mh_fit):
    fm = facefit.load_face_map()
    fmodel = facefit.FaceModel(model, mh_fit, fm)
    return fm, fmodel.landmark_points(np.zeros(len(fmodel.modifiers)))[:, :2]


def _photo(hair_from_y: float, scale: float = 800.0):
    """Front view of a flat 'skin' photo whose pixels above ``hair_from_y`` (world metres) are dark hair."""
    H, W = 700, 500
    cam = OrthoCamera.axis("front")
    cam.scale, cam.tx, cam.ty, cam.width, cam.height = scale, W / 2, 1.62 * scale + H / 2, W, H
    yy = np.arange(H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    world_y = (cam.ty - (yy + 0.5)) / scale
    rgb = np.where(world_y[..., None] > hair_from_y, 12, 150).astype(np.uint8) * np.ones((1, 1, 3), np.uint8)
    view = View("front", cam, u8_to_linear(rgb), np.ones((H, W), np.float32), np.ones((H, W), np.float32))
    return frontview.FrontView(rgb, np.ones((H, W), np.float32), view)


def test_oval_grows_up_to_the_hairline(landmarks_world):
    fm, L = landmarks_world
    top = L[fm.oval, 1].max()
    fv = _photo(hair_from_y=top + 0.02)
    grid = facedepth.Grid.around(L, 0.05, 0.0005)
    oval = facedepth.oval_mask(grid, L[fm.oval])
    grown, info = hairmask.refine_oval(fv, grid, L, fm, oval)
    gx, gy = grid.centres()
    assert info["grown_px"] > 1000
    assert gy[grown].max() == pytest.approx(top + 0.02, abs=0.004)  # stops at the hairline
    assert gy[grown].max() < top + 0.0215
    assert grown[oval].mean() > 0.95  # nothing of the face is lost


def test_hair_inside_the_oval_is_cut_out(landmarks_world):
    fm, L = landmarks_world
    top = L[fm.oval, 1].max()
    brow = L[list(facefit.MEDIAPIPE_EYEBROWS), 1].max()
    hair_y = brow + 0.5 * (top - brow)  # hair starts above the brows, inside the oval
    fv = _photo(hair_from_y=hair_y)
    grid = facedepth.Grid.around(L, 0.05, 0.0005)
    oval = facedepth.oval_mask(grid, L[fm.oval])
    out, info = hairmask.refine_oval(fv, grid, L, fm, oval)
    gx, gy = grid.centres()
    assert info["hair_px"] > 500
    assert gy[out].max() < hair_y + 0.004
    assert out.sum() < oval.sum()


def test_no_hair_no_change(landmarks_world):
    fm, L = landmarks_world
    fv = _photo(hair_from_y=5.0)  # no hair in the frame
    grid = facedepth.Grid.around(L, 0.05, 0.0005)
    oval = facedepth.oval_mask(grid, L[fm.oval])
    out, info = hairmask.refine_oval(fv, grid, L, fm, oval)
    assert info["hair_px"] == 0
    # a bright forehead above the oval lets it grow, but the oval itself stays inside
    assert out[oval].mean() > 0.95
