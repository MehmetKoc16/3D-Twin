import json

import numpy as np
import trimesh
from PIL import Image
from synthetic import framed_camera, make_capsule, render_view

from twintex.pipeline import PipelineConfig, choose_side_camera, run


def _write_views(mesh, folder, names, W=360, H=520):
    folder.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        # a different framing per view exercises the alignment (scale / translation estimation)
        cam = framed_camera(mesh, n, W, H, margin=0.06 + 0.03 * i)
        cam.tx += 5 * i
        rgb, _ = render_view(mesh, cam, W, H, bg=0.62)
        Image.fromarray(rgb).save(folder / f"{n}.png")


def test_pipeline_end_to_end_on_synthetic_capsule(tmp_path):
    mesh = make_capsule()
    mesh_path = tmp_path / "mesh.glb"
    mesh.export(mesh_path)
    views = tmp_path / "views"
    _write_views(mesh, views, ["front", "back", "left", "right"])
    out = tmp_path / "out"
    cfg = PipelineConfig(mesh_path=mesh_path, views_dir=views, out_dir=out, size=512, padding=4, max_faces=0,
                         previews=True, debug_maps=True, texture_format="png")
    rep = run(cfg, log=lambda *_: None)
    assert (out / "textured.glb").exists() and (out / "report.json").exists()
    assert set(rep["views"]) == {"front", "back", "left", "right"}
    for v in rep["views"].values():
        assert v["iou_similarity"] > 0.95
    assert rep["coverage"]["conf_ge_0.5"] > 0.6
    for name in ("front", "threequarter", "side", "back", "head_front", "sheet"):
        assert (out / "previews" / f"{name}.png").exists()
    scene = trimesh.load(out / "textured.glb")
    geom = next(iter(scene.geometry.values()))
    assert geom.visual.kind == "texture" and geom.visual.material.baseColorTexture.size == (512, 512)
    # the front preview shows the analytic colours on the front of the capsule (roughly, it is shaded)
    front = np.array(Image.open(out / "previews" / "front.png")).astype(np.float32) / 255
    centre = front[front.shape[0] // 2, front.shape[1] // 2]
    assert np.isfinite(centre).all()
    assert json.loads((out / "report.json").read_text())["mesh_faces"] == len(mesh.faces)


def test_side_view_orientation_detection():
    body = make_capsule()
    nose = trimesh.creation.icosphere(subdivisions=2, radius=0.09)
    nose.vertices += [0.0, 1.0, 0.27]  # a nose bump towards +Z at head height
    mesh = trimesh.util.concatenate([body, nose])
    W, H = 360, 520
    cam = framed_camera(mesh, "left", W, H)
    _, alpha_left = render_view(mesh, cam, W, H)  # camera on the character's left (+X): nose points to image left
    assert choose_side_camera("left", mesh.vertices, mesh.faces, alpha_left, log=lambda *_: None) == "left"
    assert choose_side_camera("right", mesh.vertices, mesh.faces, alpha_left, log=lambda *_: None) == "left"
    cam_r = framed_camera(mesh, "right", W, H)
    _, alpha_right = render_view(mesh, cam_r, W, H)
    assert choose_side_camera("left", mesh.vertices, mesh.faces, alpha_right, log=lambda *_: None) == "right"
