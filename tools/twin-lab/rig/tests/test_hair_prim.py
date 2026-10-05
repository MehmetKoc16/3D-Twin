"""The separate hair primitive of the hybrid twin (synthetic shapes only): how the rig stage tells it from the body."""

import numpy as np
import pytest

from glbio import GlbScene, Prim, read_glb, write_skinned_glb
from rig_scan import hair_weight_report, split_hair_prim


def prim(name, material):
    positions = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], np.float32)
    return Prim(positions, np.array([[0, 1, 2]], np.uint32), None, None, material, name)


def test_no_marker_means_one_body_primitive_or_none_for_several():
    assert split_hair_prim(GlbScene([prim("twin", 0)])) == (0, None)
    assert split_hair_prim(GlbScene([prim("a", 0), prim("b", 0)])) == (None, None)


def test_the_marker_names_the_hair_primitive_and_leaves_exactly_one_body():
    scene = GlbScene([prim("twin", 0), prim("dtHair", 1)], extras={"dtHairNode": "dtHair"})
    assert split_hair_prim(scene) == (0, 1)
    reordered = GlbScene([prim("dtHair", 1), prim("twin", 0)], extras={"dtHairNode": "dtHair"})
    assert split_hair_prim(reordered) == (1, 0)
    two_bodies = GlbScene([prim("a", 0), prim("b", 0), prim("dtHair", 1)], extras={"dtHairNode": "dtHair"})
    assert split_hair_prim(two_bodies) == (None, 2)


@pytest.mark.parametrize("marker", ["missing", "", 3])
def test_a_marker_that_names_no_single_node_is_an_error(marker):
    scene = GlbScene([prim("twin", 0), prim("dtHair", 1)], extras={"dtHairNode": marker})
    with pytest.raises(ValueError, match="dtHairNode"):
        split_hair_prim(scene)


def test_hair_weight_report_counts_the_head_chain(model):
    head, neck, spine = (model.bone_index[b] for b in ("head", "neck_01", "spine_03"))
    joints = np.array([[head, neck, 0, 0], [head, spine, 0, 0], [neck, 0, 0, 0]], np.uint16)
    weights = np.array([[0.6, 0.4, 0, 0], [0.5, 0.5, 0, 0], [1.0, 0, 0, 0]], np.float32)
    report = hair_weight_report(model, "dtHair", joints, weights)
    assert report["node"] == "dtHair" and report["vertices"] == 3
    assert report["head_chain_weight_min"] == pytest.approx(0.5)
    assert report["head_chain_weight_mean"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert report["fraction_head_chain_ge_0_95"] == pytest.approx(2 / 3)


def test_skinned_writer_keeps_two_meshes_on_one_skin_with_their_own_materials(tmp_path, model):
    joints = [
        {"name": name, "parent": model.bone_names[bone.parent] if bone.parent >= 0 else None, "head": np.zeros(3)}
        for name, bone in zip(model.bone_names, model.bones)
    ]
    materials = [{"name": "twin"}, {"name": "dtHair", "extras": {"dtHair": {"format": "rcov-groot-bvar/1"}}}]
    scene = GlbScene([prim("twin", 0), prim("dtHair", 1)], materials, extras={"dtHairNode": "dtHair"})
    skin_j = [np.zeros((3, 4), np.uint16)] * 2
    skin_w = [np.tile(np.array([1, 0, 0, 0], np.float32), (3, 1))] * 2
    path = tmp_path / "rigged.glb"
    write_skinned_glb(str(path), scene, joints, skin_j, skin_w)
    back = read_glb(str(path))
    assert [p.name for p in back.prims] == ["twin", "dtHair"] and [p.material for p in back.prims] == [0, 1]
    assert back.extras["dtHairNode"] == "dtHair" and back.materials[1]["extras"]["dtHair"]["format"] == "rcov-groot-bvar/1"


def test_skinned_writer_passes_a_shell_hair_material_with_its_colour_and_normal_textures_through(tmp_path, model):
    """``shell/1`` (hair contract addendum v1.1): a colour texture, a normal map and the material extras survive the rig."""
    joints = [
        {"name": name, "parent": model.bone_names[bone.parent] if bone.parent >= 0 else None, "head": np.zeros(3)}
        for name, bone in zip(model.bone_names, model.bones)
    ]
    shell = {
        "name": "dtHair",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.0, "roughnessFactor": 0.62},
        "normalTexture": {"index": 2, "scale": 1.0},
        "alphaMode": "MASK",
        "alphaCutoff": 0.5,
        "doubleSided": True,
        "extras": {"dtHair": {"format": "shell/1", "colorHex": "#2a2d34", "cardCount": 0}},
    }
    images = [
        {"data": b"body", "mimeType": "image/png"},
        {"data": b"colour", "mimeType": "image/png"},
        {"data": b"normal", "mimeType": "image/jpeg"},
    ]
    textures = [{"sampler": 0, "source": i} for i in range(3)]
    scene = GlbScene(
        [prim("twin", 0), prim("dtHair", 1)],
        [{"name": "twin"}, shell],
        textures,
        [{"magFilter": 9729, "minFilter": 9987}],
        images,
        extras={"dtHairNode": "dtHair"},
    )
    skin_j = [np.zeros((3, 4), np.uint16)] * 2
    skin_w = [np.tile(np.array([1, 0, 0, 0], np.float32), (3, 1))] * 2
    path = tmp_path / "rigged.glb"
    write_skinned_glb(str(path), scene, joints, skin_j, skin_w)
    back = read_glb(str(path))
    assert back.materials[1] == shell and back.textures == textures
    assert [(i["data"], i["mimeType"]) for i in back.images] == [(i["data"], i["mimeType"]) for i in images]
    assert split_hair_prim(back) == (0, 1)
