import numpy as np
import pytest
from glbio import _split, read_glb
from hybridbody.assemble import KIND, assemble, eye_offsets, mesh_validity, sphere_fit, write_glb
from hybridbody.partstex import make_tile
from synth import sphere_template
from test_template import synthetic_part


def test_sphere_fit_recovers_centre_and_radius_from_a_cap():
    rng = np.random.default_rng(2)
    direction = rng.normal(size=(400, 3))
    direction[:, 2] = np.abs(direction[:, 2]) + 0.3  # front cap only
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    points = np.array([0.03, 1.6, 0.13]) + 0.0125 * direction
    centre, radius = sphere_fit(points)
    np.testing.assert_allclose(centre, [0.03, 1.6, 0.13], atol=1e-6)
    assert radius == pytest.approx(0.0125, abs=1e-6)


def test_eye_offsets_align_centre_in_xy_and_front_pole_in_z_with_clamping():
    rng = np.random.default_rng(4)

    def eye(cx, cz, radius=0.0146):
        d = rng.normal(size=(300, 3))
        d[:, 2] = np.abs(d[:, 2]) + 0.2
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return np.array([cx, 1.6, cz]) + radius * d

    bound = np.vstack((eye(0.033, 0.127), eye(-0.033, 0.127)))
    flame = {"left": eye(0.0338, 0.1312, 0.0131) + [0, -0.001, 0], "right": eye(-0.0338, 0.1312, 0.0131)}
    shift, report = eye_offsets(bound, flame)
    left = shift[bound[:, 0] > 0][0]
    assert abs(left[0] - 0.0008) < 3e-4 and left[1] < 0 and left[2] > 0
    moved_pole = (bound[bound[:, 0] > 0] + left)[:, 2].max()
    assert abs(moved_pole - flame["left"][:, 2].max()) < 1e-9
    big = {k: v + [0.02, 0, 0] for k, v in flame.items()}
    clamped, _ = eye_offsets(bound, big, limit=0.004)
    assert np.abs(clamped).max() <= 0.004 + 1e-12 and set(report) == {"left", "right"}


def make_mesh(**kwargs):
    positions, faces, uv, _ = sphere_template(subdivisions=2)
    hair = synthetic_part()
    hair.category = "hair"
    hair_tile = make_tile("hair", np.zeros((16, 16, 3), np.uint8), hair.uv, 16, 16)
    hair_tile.origin = (0, 64)
    brow = synthetic_part()
    brow.category = "eyebrows"
    brow_tile = make_tile("eyebrows", np.zeros((16, 16, 3), np.uint8), brow.uv, 16, 16)
    brow_tile.origin = (20, 64)
    mesh = assemble(
        positions,
        faces,
        uv * [1.0, 64 / 80],
        [
            ("eyebrows", brow, brow.positions + [0.0, 1.65, 0.0], brow_tile),
            ("hair", hair, hair.positions + [0.0, 1.7, 0.0], hair_tile),
        ],
        (64, 80),
        **kwargs,
    )
    return mesh, positions, faces


def test_assemble_concatenates_parts_without_back_copies_by_default():
    mesh, positions, faces = make_mesh()
    assert mesh.parts["hair"]["vertices"][1] - mesh.parts["hair"]["vertices"][0] == 3
    assert len(mesh.faces) == len(faces) + 2
    assert mesh.uv.shape == (len(mesh.positions), 2) and (mesh.kind[len(positions) :] > 0).all()


def test_assemble_can_add_offset_back_faces_for_cards():
    mesh, positions, faces = make_mesh(double_sided=("eyebrows", "eyelashes", "hair"))
    assert mesh.parts["body"]["vertices"] == (0, len(positions))
    assert mesh.parts["hair"]["vertices"][1] - mesh.parts["hair"]["vertices"][0] == 6  # 3 front + 3 back
    assert len(mesh.faces) == len(faces) + 2 * 2
    front = mesh.faces[len(faces) + 2]  # hair front
    back = mesh.faces[len(faces) + 3]
    assert set(front) & set(back) == set()  # distinct vertices: never welded
    gap = np.linalg.norm(mesh.positions[front] - mesh.positions[back[::-1]], axis=1)  # reversed winding
    assert (gap > 5e-5).all() and (gap < 5e-4).all()
    assert (mesh.kind[len(positions) :] > 0).all() and KIND["hair"] in mesh.kind
    assert mesh.uv.shape == (len(mesh.positions), 2) and mesh.uv[len(positions) :, 1].min() >= 64 / 80 - 1e-9


def test_without_drops_a_part_and_renumbers():
    mesh, _, _ = make_mesh()
    bare = mesh.without("hair")
    assert len(bare.positions) == len(mesh.positions) - 3 and bare.faces.max() < len(bare.positions)
    assert (bare.kind != KIND["hair"]).all()


def test_drop_vertices_removes_their_faces_and_orphans():
    positions, faces, uv, _ = sphere_template(subdivisions=2)
    mesh = assemble(positions, faces, uv, [], (64, 64), drop_vertices=np.array([0, 1, 2]))
    assert len(mesh.faces) < len(faces) and mesh.faces.max() == len(mesh.positions) - 1
    assert len(mesh.positions) <= len(positions)


def test_mesh_validity_reports_a_closed_sphere_and_an_opened_one():
    positions, faces, uv, _ = sphere_template(subdivisions=2)
    closed = assemble(positions, faces, uv, [], (64, 64))
    report = mesh_validity(closed)
    assert report["body_boundary_edges"] == 0 and report["body_nonmanifold_edges"] == 0
    assert report["body_inconsistent_winding_edges"] == 0 and report["finite"] and report["uv_in_range"]
    opened = assemble(positions, faces[10:], uv, [], (64, 64))
    assert mesh_validity(opened)["body_boundary_edges"] > 0


def test_write_glb_roundtrip_keeps_one_textured_primitive_and_extras(tmp_path):
    mesh, _, _ = make_mesh()
    atlas = np.full((80, 64, 3), 120, np.uint8)
    extras = {"dtHybrid": {"version": 1}, "dtHasMakeHumanHands": True}
    info = write_glb(tmp_path / "h.glb", mesh, atlas, extras)
    scene = read_glb(str(tmp_path / "h.glb"))
    assert len(scene.prims) == 1 and len(scene.prims[0].positions) == len(mesh.positions)
    assert scene.extras["dtHybrid"] == {"version": 1} and scene.extras["dtHasMakeHumanHands"] is True
    js, _ = _split((tmp_path / "h.glb").read_bytes())
    assert len(js["meshes"]) == 1 and len(js["materials"]) == 1 and js["images"][0]["mimeType"] == "image/jpeg"
    assert info["extras_keys"] == ["dtHasMakeHumanHands", "dtHybrid"]
    np.testing.assert_allclose(scene.prims[0].uv, mesh.uv.astype(np.float32), atol=1e-6)
    assert js["materials"][0].get("alphaMode") == "MASK" and js["materials"][0]["doubleSided"] is True


def test_write_glb_stores_an_rgba_atlas_as_png_with_the_mask_material(tmp_path):
    import io

    from PIL import Image

    mesh, _, _ = make_mesh()
    atlas = np.full((80, 64, 4), 255, np.uint8)
    atlas[70:, :, 3] = 0
    info = write_glb(tmp_path / "h.glb", mesh, atlas, {"dtHybrid": {"version": 1}})
    scene = read_glb(str(tmp_path / "h.glb"))
    js, _ = _split((tmp_path / "h.glb").read_bytes())
    material = js["materials"][0]
    assert material["alphaMode"] == "MASK" and material["alphaCutoff"] == 0.5 and material["doubleSided"] is True
    assert js["images"][0]["mimeType"] == "image/png" and info["image_mime"] == "image/png"
    back = np.asarray(Image.open(io.BytesIO(scene.images[0]["data"])))
    assert back.shape == (80, 64, 4) and (back == atlas).all()


def synthetic_hair(node="dtHair"):
    from hybridbody.hair import HairMesh, hair_colours
    from hybridbody.hairtex import hair_atlas

    atlas, _ = hair_atlas(128, 64, seed=2)
    positions = np.array([[0, 1.7, 0.08], [0.01, 1.7, 0.08], [0, 1.74, 0.09], [0.01, 1.74, 0.09]], np.float64)
    normals = np.tile([0.0, 0.3, 0.95], (4, 1))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    uv = np.array([[0.1, 0.1], [0.2, 0.1], [0.1, 0.9], [0.2, 0.9]])
    faces = np.array([[0, 1, 2], [2, 1, 3]])
    return HairMesh(positions, normals, uv, faces, atlas, hair_colours([42, 30, 24]), 1, node)


def test_write_glb_adds_the_hair_as_a_second_primitive_with_its_own_material_and_atlas(tmp_path):
    import io

    from PIL import Image

    mesh, _, _ = make_mesh()
    atlas = np.full((80, 64, 4), 255, np.uint8)
    hair = synthetic_hair()
    info = write_glb(tmp_path / "h.glb", mesh, atlas, {"dtHybrid": {"version": 1}}, hair=hair)
    scene = read_glb(str(tmp_path / "h.glb"))
    js, blob = _split((tmp_path / "h.glb").read_bytes())
    assert [p.name for p in scene.prims] == ["twin", "dtHair"] and [p.material for p in scene.prims] == [0, 1]
    assert len(scene.prims[1].positions) == 4 and scene.prims[1].normals is not None
    np.testing.assert_allclose(scene.prims[1].uv, hair.uv.astype(np.float32), atol=1e-6)
    assert scene.extras["dtHairNode"] == "dtHair" and scene.extras["dtHybrid"] == {"version": 1}
    assert [n["name"] for n in js["nodes"]] == ["twin", "dtHair"]
    body, hair_material = js["materials"]
    assert body["name"] == "twin" and body["alphaMode"] == "MASK"
    assert (hair_material["name"], hair_material["alphaMode"], hair_material["alphaCutoff"]) == ("dtHair", "MASK", 0.5)
    assert hair_material["doubleSided"] is True
    assert hair_material["extras"]["dtHair"] == {
        "format": "rcov-groot-bvar/1",
        "colorHex": "#2a1e18",
        "rootHex": "#1f1611",
        "tipHex": "#3a2a20",  # the example of the hair contract
        "cardCount": 1,
    }
    texture = js["textures"][hair_material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    assert texture["source"] == 1 and js["images"][1]["mimeType"] == "image/png"
    view = js["bufferViews"][js["images"][1]["bufferView"]]
    back = np.asarray(Image.open(io.BytesIO(blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]])))
    assert (back == hair.atlas).all()  # the strand data atlas is stored losslessly
    assert info["hair"]["triangles"] == 2 and info["hair"]["atlas_size"] == [128, 64]


def test_hair_colours_and_the_default_colour_choice():
    from hybridbody.hair import DEFAULT_HAIR_HEX, hair_colours
    from hybridbody.partstex import DEFAULT_HAIR_SRGB, srgb_hex
    from hybridbody.pipeline import choose_hair_colour

    assert srgb_hex(DEFAULT_HAIR_SRGB) == DEFAULT_HAIR_HEX == "#2a1e18"
    photo = {"from_photo": True, "srgb": [0x49, 0x41, 0x3D]}
    assert choose_hair_colour(photo, None)["method"] == "default"
    assert srgb_hex(choose_hair_colour(photo, None)["srgb"]) == "#2a1e18"  # no flag needed
    assert choose_hair_colour(photo, None)["measured_hex"] == "#49413d"
    assert srgb_hex(choose_hair_colour(photo, "#102030")["srgb"]) == "#102030"
    assert choose_hair_colour(photo, "photo")["method"].startswith("photo: near-grey")
    assert hair_colours([255, 250, 240])["tipHex"] == "#ffffff"  # the tip colour is clipped, never above white
