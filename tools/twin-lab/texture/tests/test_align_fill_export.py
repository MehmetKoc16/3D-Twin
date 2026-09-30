import json

import cv2
import numpy as np
import trimesh
from PIL import Image
from synthetic import framed_camera, make_capsule, render_view

from twintex import align, matting
from twintex.camera import OrthoCamera
from twintex.delight import remove_shading
from twintex.export import encode_texture, write_glb
from twintex.fill import dilate_fill, mirror_fill, nearest_visible_fill, push_pull


# ------------------------------------------------------------------------------------------ alignment
def test_similarity_fit_recovers_scale_and_translation():
    mesh = make_capsule()
    W, H = 360, 520
    truth = framed_camera(mesh, "front", W, H, margin=0.12)
    truth.scale *= 0.93
    truth.tx += 17.0
    truth.ty -= 9.0
    _, alpha = render_view(mesh, truth, W, H)
    cam0 = OrthoCamera.axis("front")
    cam0.width, cam0.height = W, H
    cam, iou0, iou1 = align.fit_similarity(cam0, mesh.vertices, mesh.faces, alpha)
    assert iou1 > 0.985, iou1
    assert abs(cam.scale / truth.scale - 1) < 0.01
    assert abs(cam.tx - truth.tx) < 3.0 and abs(cam.ty - truth.ty) < 3.0


def test_flow_improves_silhouette_overlap_for_a_slightly_different_body():
    mesh = make_capsule(radius=0.22)
    fat = make_capsule(radius=0.25)  # the "photo" shows a slightly wider body than the mesh
    W, H = 360, 520
    cam_img = framed_camera(fat, "front", W, H, margin=0.1)
    _, alpha = render_view(fat, cam_img, W, H)
    cam0 = OrthoCamera.axis("front")
    cam0.width, cam0.height = W, H
    cam, _, iou_sim = align.fit_similarity(cam0, mesh.vertices, mesh.faces, alpha)
    flow, iou_flow = align.refine_flow(cam, mesh.vertices, mesh.faces, alpha)
    assert flow.shape == (H, W, 2) and np.isfinite(flow).all()
    assert iou_flow > iou_sim + 0.005
    assert np.abs(flow).max() <= 0.035 * H + 1e-3


# ------------------------------------------------------------------------------------------------ fill
def test_push_pull_keeps_known_pixels_and_interpolates_holes():
    H = W = 64
    ramp = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
    img = np.stack([ramp, 1 - ramp, np.full_like(ramp, 0.5)], -1)
    conf = np.ones((H, W), np.float32)
    conf[20:44, 20:44] = 0.0
    hole_img = img.copy()
    hole_img[20:44, 20:44] = 0.0  # garbage inside the hole must be ignored
    out = push_pull(hole_img, conf)
    assert np.allclose(out[conf == 1], img[conf == 1], atol=1e-5)
    err = np.abs(out[20:44, 20:44] - img[20:44, 20:44])
    assert err.max() < 0.2 and err.mean() < 0.06
    # partial confidence blends observation and surroundings
    conf2 = np.full((H, W), 0.5, np.float32)
    out2 = push_pull(img, conf2)
    assert np.isfinite(out2).all()


def test_dilate_fill_grows_the_valid_area_by_the_requested_padding():
    img = np.zeros((32, 32, 3), np.float32)
    img[12:20, 12:20] = [0.2, 0.4, 0.6]
    mask = np.zeros((32, 32), bool)
    mask[12:20, 12:20] = True
    out, m = dilate_fill(img, mask, 3)
    assert m[9:23, 9:23].all() and not m[8, 16]
    assert np.allclose(out[10, 16], [0.2, 0.4, 0.6], atol=1e-5)


def test_mirror_fill_copies_across_the_sagittal_plane():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(4000, 3)).astype(np.float32)
    pts[:, 0] = np.abs(pts[:, 0]) + 0.02  # right half only ...
    left = pts.copy()
    left[:, 0] *= -1  # ... plus its mirrored twin
    pos = np.concatenate([pts, left])
    nrm = np.tile(np.array([[0, 0, 1.0]], np.float32), (len(pos), 1))
    col = np.concatenate([np.tile([[1.0, 0.0, 0.0]], (4000, 1)), np.zeros((4000, 3))]).astype(np.float32)
    conf = np.concatenate([np.ones(4000), np.zeros(4000)]).astype(np.float32)
    conf2, col2 = mirror_fill(pos, nrm, col, conf, x_mid=0.0, max_dist=0.05)
    assert (conf2[4000:] > 0.85).mean() > 0.95
    assert np.allclose(col2[4000:][conf2[4000:] > 0.85], [1, 0, 0])
    assert (conf2[:4000] == conf[:4000]).all()


def test_nearest_visible_fill_prefers_close_and_similarly_oriented_sources():
    # known red patch on the front (n=+z) and a blue patch on the back (n=-z); the queries sit next to each
    rng = np.random.default_rng(1)
    red = np.column_stack([rng.uniform(-.1, .1, 300), rng.uniform(-.1, .1, 300), np.full(300, 0.2)])
    blue = np.column_stack([rng.uniform(-.1, .1, 300), rng.uniform(-.1, .1, 300), np.full(300, -0.2)])
    q_front = np.array([[0.0, 0.0, 0.2], [0.05, 0.0, 0.19]])
    q_back = np.array([[0.0, 0.0, -0.2]])
    pos = np.concatenate([red, blue, q_front, q_back]).astype(np.float32)
    nrm = np.concatenate([np.tile([0, 0, 1.0], (300, 1)), np.tile([0, 0, -1.0], (300, 1)),
                          np.tile([0, 0, 1.0], (2, 1)), np.tile([0, 0, -1.0], (1, 1))]).astype(np.float32)
    col = np.concatenate([np.tile([1.0, 0, 0], (300, 1)), np.tile([0, 0, 1.0], (300, 1)), np.zeros((3, 3))]).astype(np.float32)
    conf = np.concatenate([np.ones(600), np.zeros(3)]).astype(np.float32)
    fill = nearest_visible_fill(pos, nrm, col, conf, smooth_iters=0)
    assert fill[600, 0] > 0.9 and fill[601, 0] > 0.9  # red
    assert fill[602, 2] > 0.9  # blue
    # no known points at all -> finite fallback
    f0 = nearest_visible_fill(pos[:5], nrm[:5], col[:5], np.zeros(5, np.float32))
    assert np.isfinite(f0).all()


# ---------------------------------------------------------------------------------- matting / delighting
def test_matting_separates_a_person_shaped_blob_from_a_gradient_studio_background():
    H, W = 480, 360
    yy, xx = np.mgrid[0:H, 0:W]
    bg = 150 + 40 * (1 - yy / H) + 12 * (xx / W)
    img = np.stack([bg, bg, bg + 5], -1)
    person = ((xx - 180) / 70.0) ** 2 + ((yy - 250) / 190.0) ** 2 < 1.0
    img[person] = np.array([70, 75, 90]) + 25 * np.sin(xx[person] / 9.0)[:, None]
    img = np.clip(img + np.random.default_rng(0).normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    a = matting.matte(img)
    inter = np.logical_and(a > 0.5, person).sum()
    union = np.logical_or(a > 0.5, person).sum()
    assert inter / union > 0.95


def test_delight_flattens_a_lighting_gradient_but_keeps_garment_contrast():
    H, W = 300, 200
    yy, xx = np.mgrid[0:H, 0:W]
    alpha = np.ones((H, W), np.float32)
    base = np.where(yy < 150, 0.5, 0.1).astype(np.float32)  # light shirt / dark trousers
    light = 0.6 + 0.8 * (xx / W)  # strong left-to-right illumination gradient
    img = np.repeat((base * light)[..., None], 3, axis=2).astype(np.float32)
    out = remove_shading(img, alpha, strength=1.0)
    def lr_spread(im):
        top = im[:150, :, 1]
        return np.abs(top[:, -20:].mean() - top[:, :20].mean()) / top.mean()
    assert lr_spread(out) < 0.5 * lr_spread(img)
    assert out[:150, :, 1].mean() / out[150:, :, 1].mean() > 3.0  # garment contrast preserved


# ----------------------------------------------------------------------------------------------- export
def test_glb_export_round_trips_through_trimesh(tmp_path):
    mesh = make_capsule(count=12)
    n = np.asarray(mesh.vertex_normals, np.float32)
    uv = np.column_stack([(mesh.vertices[:, 0] + .5), (mesh.vertices[:, 1] / 1.5)]).astype(np.float32)
    tex = (np.random.default_rng(2).uniform(0, 1, (64, 64, 3)) * 255).astype(np.uint8)
    for fmt, mime in (("png", "image/png"), ("jpeg", "image/jpeg")):
        data, m = encode_texture(tex, fmt)
        assert m == mime
        path = tmp_path / f"t_{fmt}.glb"
        write_glb(path, mesh.vertices, n, uv, mesh.faces, data, m, extras={"k": 1})
        scene = trimesh.load(path)
        geom = next(iter(scene.geometry.values()))
        assert len(geom.faces) == len(mesh.faces) and len(geom.vertices) == len(mesh.vertices)
        assert geom.visual.kind == "texture"
        img = geom.visual.material.baseColorTexture
        assert img.size == (64, 64)
        # trimesh flips v on load (OpenGL convention): the file itself keeps the glTF top-left convention
        assert np.allclose(np.sort(geom.visual.uv[:, 0]), np.sort(uv[:, 0]), atol=1e-6)
        assert np.allclose(np.sort(geom.visual.uv[:, 1]), np.sort(1 - uv[:, 1]), atol=1e-6)
        raw = path.read_bytes()
        jl = int.from_bytes(raw[12:16], "little")
        j = json.loads(raw[20 : 20 + jl])
        assert j["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] == 0.0
        assert j["images"][0]["mimeType"] == mime and j["samplers"][0]["wrapS"] == 33071
    Image.fromarray(tex)  # sanity: the array is a valid image
    assert cv2.__version__
