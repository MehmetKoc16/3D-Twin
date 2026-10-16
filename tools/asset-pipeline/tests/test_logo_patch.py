"""Synthetic tests for GLB logo patching tool."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import pygltflib as gl
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from garmentimport.logo_patch import main, patch_glb  # noqa: E402


def create_synthetic_glb(
    path: Path,
    base_image: Image.Image,
    normal_image: Image.Image | None = None,
    base_mime: str = "image/png",
    normal_mime: str = "image/png",
) -> None:
    """Create a minimal valid GLB with 1 triangle mesh and 1-2 textures."""
    g = gl.GLTF2(asset=gl.Asset(version="2.0", generator="test_logo_patch"))
    blob = bytearray()

    def add_view(data: bytes, target: int | None = None) -> int:
        pad = (-len(blob)) % 4
        blob.extend(b"\x00" * pad)
        idx = len(g.bufferViews)
        g.bufferViews.append(gl.BufferView(
            buffer=0,
            byteOffset=len(blob),
            byteLength=len(data),
            target=target,
        ))
        blob.extend(data)
        return idx

    # Positions: 1 triangle
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype="<f4")
    pos_view = add_view(positions.tobytes(), 34962)  # ARRAY_BUFFER
    pos_acc = len(g.accessors)
    g.accessors.append(gl.Accessor(
        bufferView=pos_view,
        componentType=5126,  # FLOAT
        count=3,
        type="VEC3",
        min=positions.min(axis=0).tolist(),
        max=positions.max(axis=0).tolist(),
    ))

    # UVs
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype="<f4")
    uv_view = add_view(uvs.tobytes(), 34962)
    uv_acc = len(g.accessors)
    g.accessors.append(gl.Accessor(
        bufferView=uv_view,
        componentType=5126,
        count=3,
        type="VEC2",
        min=uvs.min(axis=0).tolist(),
        max=uvs.max(axis=0).tolist(),
    ))

    # Indices
    indices = np.array([0, 1, 2], dtype="<u2")
    idx_view = add_view(indices.tobytes(), 34963)  # ELEMENT_ARRAY_BUFFER
    idx_acc = len(g.accessors)
    g.accessors.append(gl.Accessor(
        bufferView=idx_view,
        componentType=5123,  # UNSIGNED_SHORT
        count=3,
        type="SCALAR",
        min=[0],
        max=[2],
    ))

    # Base image bufferView
    buf = io.BytesIO()
    fmt = "JPEG" if "jpeg" in base_mime.lower() or "jpg" in base_mime.lower() else "PNG"
    base_image.save(buf, format=fmt)
    base_bytes = buf.getvalue()
    base_view = add_view(base_bytes)
    base_img_idx = len(g.images)
    g.images.append(gl.Image(bufferView=base_view, mimeType=base_mime))
    base_tex_idx = len(g.textures)
    g.textures.append(gl.Texture(source=base_img_idx))

    mat = gl.Material(pbrMetallicRoughness=gl.PbrMetallicRoughness(
        baseColorTexture=gl.TextureInfo(index=base_tex_idx),
    ))

    # Optional normal image bufferView
    if normal_image is not None:
        norm_buf = io.BytesIO()
        n_fmt = "JPEG" if "jpeg" in normal_mime.lower() or "jpg" in normal_mime.lower() else "PNG"
        normal_image.save(norm_buf, format=n_fmt)
        norm_bytes = norm_buf.getvalue()
        norm_view = add_view(norm_bytes)
        norm_img_idx = len(g.images)
        g.images.append(gl.Image(bufferView=norm_view, mimeType=normal_mime))
        norm_tex_idx = len(g.textures)
        g.textures.append(gl.Texture(source=norm_img_idx))
        mat.normalTexture = gl.NormalMaterialTexture(index=norm_tex_idx)

    g.materials = [mat]
    g.meshes = [gl.Mesh(primitives=[gl.Primitive(
        attributes=gl.Attributes(POSITION=pos_acc, TEXCOORD_0=uv_acc),
        indices=idx_acc,
        material=0,
    )])]
    g.nodes = [gl.Node(mesh=0)]
    g.scenes = [gl.Scene(nodes=[0])]
    g.scene = 0

    pad = (-len(blob)) % 4
    blob.extend(b"\x00" * pad)
    g.buffers = [gl.Buffer(byteLength=len(blob))]
    g.set_binary_blob(bytes(blob))

    path.parent.mkdir(parents=True, exist_ok=True)
    g.save_binary(str(path))


def get_image_from_glb(gltf: gl.GLTF2, image_idx: int) -> Image.Image:
    """Extract PIL image from GLTF2 object."""
    gl_img = gltf.images[image_idx]
    blob = gltf.binary_blob()
    assert blob is not None
    bv = gltf.bufferViews[gl_img.bufferView]
    offset = bv.byteOffset or 0
    length = bv.byteLength
    return Image.open(io.BytesIO(blob[offset : offset + length]))


def get_accessor_data(gltf: gl.GLTF2, acc_idx: int) -> np.ndarray:
    """Read raw accessor data array from GLTF2 binary blob."""
    acc = gltf.accessors[acc_idx]
    bv = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()
    assert blob is not None
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    dtype_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: "<i2",
        5123: "<u2",
        5125: "<u4",
        5126: "<f4",
    }
    count_map = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }
    dtype = dtype_map[acc.componentType]
    components = count_map[acc.type]
    return np.frombuffer(blob, dtype=dtype, count=acc.count * components, offset=offset)


def test_patch_glb_color_and_normal(tmp_path):
    """Test patching both baseColor (red with white logo) and normal map."""
    src = tmp_path / "model_in.glb"
    dst = tmp_path / "model_out.glb"

    # 64x64 baseColor texture: red fabric with white logo circle
    base_img = Image.new("RGBA", (64, 64), (200, 30, 30, 255))
    draw_b = ImageDraw.Draw(base_img)
    # Circle at center [20, 32], radius 8
    draw_b.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(255, 255, 255, 255))

    # 64x64 normal texture: neutral normal (128, 128, 255) with embossed bump (200, 80, 180)
    norm_img = Image.new("RGBA", (64, 64), (128, 128, 255, 255))
    draw_n = ImageDraw.Draw(norm_img)
    draw_n.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(200, 80, 180, 255))

    create_synthetic_glb(src, base_img, norm_img)

    # Patch: source fabric is shifted by +24px in X -> center [44, 32] (which is plain red / flat normal)
    patch = {
        "image": None,
        "center": [20, 32],
        "radius": 8,
        "source_offset": [24, 0],
        "feather": 2,
    }

    patch_glb(src, dst, [patch])
    assert dst.exists()

    res_gltf = gl.GLTF2().load(str(dst))
    patched_base = np.array(get_image_from_glb(res_gltf, 0))
    patched_norm = np.array(get_image_from_glb(res_gltf, 1))

    orig_base_arr = np.array(base_img)
    orig_norm_arr = np.array(norm_img)

    # Coordinates grid
    y_idx, x_idx = np.ogrid[0:64, 0:64]
    dist = np.hypot(x_idx - 20, y_idx - 32)

    # 1. Inside logo region (dist <= 6): should revert to red background fabric
    inner_mask = dist <= 6
    red_expected = np.array([200, 30, 30, 255])
    diff_base_inner = np.abs(patched_base[inner_mask].astype(float) - red_expected).mean()
    assert diff_base_inner < 2.0, f"Expected red background, average diff was {diff_base_inner}"

    # 2. Outside circle (dist > 8): must remain exactly identical byte-for-byte
    outside_mask = dist > 8
    assert np.array_equal(patched_base[outside_mask], orig_base_arr[outside_mask]), "Outside baseColor was altered"

    # 3. Normal map should also be patched inside and identical outside
    norm_expected = np.array([128, 128, 255, 255])
    diff_norm_inner = np.abs(patched_norm[inner_mask].astype(float) - norm_expected).mean()
    assert diff_norm_inner < 2.0, f"Expected flat normal, average diff was {diff_norm_inner}"
    assert np.array_equal(patched_norm[outside_mask], orig_norm_arr[outside_mask]), "Outside normal was altered"


def test_mesh_and_accessor_data_unchanged(tmp_path):
    """Test that geometry, vertex accessors, materials and samplers are preserved byte-for-byte."""
    src = tmp_path / "model_in.glb"
    dst = tmp_path / "model_out.glb"

    base_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
    create_synthetic_glb(src, base_img)

    patch = {
        "image": None,
        "center": [32, 32],
        "radius": 5,
        "source_offset": [15, 0],
        "feather": 1,
    }
    patch_glb(src, dst, [patch])

    g_orig = gl.GLTF2().load(str(src))
    g_patch = gl.GLTF2().load(str(dst))

    assert len(g_orig.accessors) == len(g_patch.accessors)
    for i in range(len(g_orig.accessors)):
        data_orig = get_accessor_data(g_orig, i)
        data_patch = get_accessor_data(g_patch, i)
        assert np.array_equal(data_orig, data_patch), f"Accessor {i} data changed after patch"

    # Check node and mesh primitives match
    assert len(g_orig.meshes) == len(g_patch.meshes)
    assert g_orig.meshes[0].primitives[0].indices == g_patch.meshes[0].primitives[0].indices


def test_source_region_overflow_raises_error(tmp_path):
    """Test that if the source region exceeds the image bounds, an explicit ValueError is raised."""
    src = tmp_path / "model_in.glb"
    dst = tmp_path / "model_out.glb"

    base_img = Image.new("RGBA", (64, 64), (150, 50, 50, 255))
    create_synthetic_glb(src, base_img)

    # Positive X overflow: center 20 + offset 50 + radius 8 = 78 > 64
    overflow_patch_x = {
        "image": None,
        "center": [20, 32],
        "radius": 8,
        "source_offset": [50, 0],
        "feather": 2,
    }
    with pytest.raises(ValueError, match=r"out of bounds|exceed"):
        patch_glb(src, dst, [overflow_patch_x])

    # Negative X overflow: center 20 + offset (-25) - radius 8 = -13 < 0
    overflow_patch_neg = {
        "image": None,
        "center": [20, 32],
        "radius": 8,
        "source_offset": [-25, 0],
        "feather": 2,
    }
    with pytest.raises(ValueError, match=r"out of bounds|exceed"):
        patch_glb(src, dst, [overflow_patch_neg])

    # Positive Y overflow
    overflow_patch_y = {
        "image": None,
        "center": [32, 20],
        "radius": 8,
        "source_offset": [0, 50],
        "feather": 2,
    }
    with pytest.raises(ValueError, match=r"out of bounds|exceed"):
        patch_glb(src, dst, [overflow_patch_y])


def test_jpeg_texture_support(tmp_path):
    """Test that JPEG textures are preserved in format and properly patched."""
    src = tmp_path / "model_jpeg.glb"
    dst = tmp_path / "model_jpeg_out.glb"

    base_img = Image.new("RGB", (64, 64), (180, 40, 40))
    draw = ImageDraw.Draw(base_img)
    draw.ellipse([20 - 6, 32 - 6, 20 + 6, 32 + 6], fill=(255, 255, 255))

    create_synthetic_glb(src, base_img, base_mime="image/jpeg")

    patch = {
        "image": None,
        "center": [20, 32],
        "radius": 6,
        "source_offset": [24, 0],
        "feather": 2,
    }
    patch_glb(src, dst, [patch])

    res_gltf = gl.GLTF2().load(str(dst))
    assert res_gltf.images[0].mimeType == "image/jpeg"

    patched_img = get_image_from_glb(res_gltf, 0)
    arr = np.array(patched_img)
    # Check center has reverted toward red background (with JPEG compression tolerance)
    center_color = arr[32, 20]
    assert abs(int(center_color[0]) - 180) < 15
    assert abs(int(center_color[1]) - 40) < 15


def test_different_image_dimensions_scaling(tmp_path):
    """Test proportional scaling when textures have different resolutions (e.g. 64x64 color and 32x32 normal)."""
    src = tmp_path / "model_diff.glb"
    dst = tmp_path / "model_diff_out.glb"

    # Base color 64x64, logo at [20, 32], radius 8
    base_img = Image.new("RGBA", (64, 64), (200, 30, 30, 255))
    draw_b = ImageDraw.Draw(base_img)
    draw_b.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(255, 255, 255, 255))

    # Normal map 32x32 (half resolution), bump at [10, 16], radius 4
    norm_img = Image.new("RGBA", (32, 32), (128, 128, 255, 255))
    draw_n = ImageDraw.Draw(norm_img)
    draw_n.ellipse([10 - 4, 16 - 4, 10 + 4, 16 + 4], fill=(200, 80, 180, 255))

    create_synthetic_glb(src, base_img, norm_img)

    # Patch defined in 64x64 coordinates
    patch = {
        "image": None,
        "center": [20, 32],
        "radius": 8,
        "source_offset": [24, 0],
        "feather": 2,
    }

    patch_glb(src, dst, [patch])

    res_gltf = gl.GLTF2().load(str(dst))
    patched_base = np.array(get_image_from_glb(res_gltf, 0))
    patched_norm = np.array(get_image_from_glb(res_gltf, 1))

    # Base image: inside radius 6 reverted to red
    y_64, x_64 = np.ogrid[0:64, 0:64]
    dist_64 = np.hypot(x_64 - 20, y_64 - 32)
    assert np.abs(patched_base[dist_64 <= 6].astype(float) - [200, 30, 30, 255]).mean() < 2.0

    # Normal image: scaled coordinate is [10, 16], radius 4. Inside radius 3 reverted to flat normal
    y_32, x_32 = np.ogrid[0:32, 0:32]
    dist_32 = np.hypot(x_32 - 10, y_32 - 16)
    assert np.abs(patched_norm[dist_32 <= 3].astype(float) - [128, 128, 255, 255]).mean() < 2.0

    # Outside in normal map (dist > 4) is unchanged
    orig_norm_arr = np.array(norm_img)
    assert np.array_equal(patched_norm[dist_32 > 4], orig_norm_arr[dist_32 > 4])


def test_cli_execution_and_image_filtering(tmp_path):
    """Test CLI interface and targeting a specific image."""
    src = tmp_path / "cli_in.glb"
    dst = tmp_path / "cli_out.glb"
    patches_file = tmp_path / "patches.json"

    base_img = Image.new("RGBA", (64, 64), (200, 30, 30, 255))
    draw_b = ImageDraw.Draw(base_img)
    draw_b.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(255, 255, 255, 255))

    norm_img = Image.new("RGBA", (64, 64), (128, 128, 255, 255))
    draw_n = ImageDraw.Draw(norm_img)
    draw_n.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(200, 80, 180, 255))

    create_synthetic_glb(src, base_img, norm_img)

    patches = [{
        "image": None,
        "center": [20, 32],
        "radius": 8,
        "source_offset": [24, 0],
        "feather": 2,
    }]
    patches_file.write_text(json.dumps(patches), encoding="utf-8")

    # CLI with --images 0: only patch image 0 (baseColor), normal map stays untouched
    ret = main([
        str(src),
        str(dst),
        "--patches",
        str(patches_file),
        "--images",
        "0",
    ])
    assert ret == 0
    assert dst.exists()

    res_gltf = gl.GLTF2().load(str(dst))
    patched_base = np.array(get_image_from_glb(res_gltf, 0))
    patched_norm = np.array(get_image_from_glb(res_gltf, 1))

    # Base image was patched
    y_idx, x_idx = np.ogrid[0:64, 0:64]
    dist = np.hypot(x_idx - 20, y_idx - 32)
    assert np.abs(patched_base[dist <= 6].astype(float) - [200, 30, 30, 255]).mean() < 2.0

    # Normal map was NOT patched (bump is still there)
    assert np.array_equal(patched_norm, np.array(norm_img))


def test_patch_glb_fill_mode_color_and_normal(tmp_path):
    """Test fill mode on both baseColor and normal map textures."""
    src = tmp_path / "model_fill_in.glb"
    dst = tmp_path / "model_fill_out.glb"

    # 64x64 baseColor texture: red fabric with white logo circle
    base_img = Image.new("RGBA", (64, 64), (200, 30, 30, 255))
    draw_b = ImageDraw.Draw(base_img)
    draw_b.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(255, 255, 255, 255))

    # 64x64 normal texture: neutral normal (128, 128, 255) with embossed bump (200, 80, 180)
    norm_img = Image.new("RGBA", (64, 64), (128, 128, 255, 255))
    draw_n = ImageDraw.Draw(norm_img)
    draw_n.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(200, 80, 180, 255))

    create_synthetic_glb(src, base_img, norm_img)

    # Patch with mode="fill", no source_offset needed
    patch = {
        "image": None,
        "center": [20, 32],
        "radius": 8,
        "mode": "fill",
        "ring": 24,
        "feather": 2,
        "seed": 42,
    }

    patch_glb(src, dst, [patch])
    assert dst.exists()

    res_gltf = gl.GLTF2().load(str(dst))
    patched_base = np.array(get_image_from_glb(res_gltf, 0))
    patched_norm = np.array(get_image_from_glb(res_gltf, 1))

    orig_base_arr = np.array(base_img)
    orig_norm_arr = np.array(norm_img)

    y_idx, x_idx = np.ogrid[0:64, 0:64]
    dist = np.hypot(x_idx - 20, y_idx - 32)
    inner_mask = dist <= 6
    outside_mask = dist > 8
    ring_mask = (dist > 8) & (dist <= 8 + 24)

    # 1. Base color inside circle should be close to ring median (diff < 8)
    ring_base_med = np.median(orig_base_arr[ring_mask, :3], axis=0)
    diff_base_inner = np.abs(patched_base[inner_mask, :3].astype(float) - ring_base_med).mean()
    assert diff_base_inner < 8.0, f"Expected diff < 8.0 from ring median, got {diff_base_inner}"

    # 2. Base color outside circle must remain byte-for-byte identical
    assert np.array_equal(patched_base[outside_mask], orig_base_arr[outside_mask]), "Outside baseColor was altered"

    # 3. Alpha channel must be preserved
    assert np.array_equal(patched_base[..., 3], orig_base_arr[..., 3]), "Alpha channel was altered"

    # 4. Normal map: auto-detected, no noise added (std < 1)
    std_norm_inner = patched_norm[inner_mask, :3].std(axis=0).max()
    assert std_norm_inner < 1.0, f"Expected normal map std < 1.0, got {std_norm_inner}"
    diff_norm_inner = np.abs(patched_norm[inner_mask, :3].astype(float) - [128, 128, 255]).mean()
    assert diff_norm_inner < 1.0, f"Expected neutral normal inside, got diff {diff_norm_inner}"

    # 5. Normal map outside circle must remain byte-for-byte identical
    assert np.array_equal(patched_norm[outside_mask], orig_norm_arr[outside_mask]), "Outside normal was altered"
    assert np.array_equal(patched_norm[..., 3], orig_norm_arr[..., 3]), "Normal alpha channel was altered"


def test_patch_glb_fill_mode_noise_and_seed_reproducibility(tmp_path):
    """Test fill mode with textured fabric adds noise and same seed yields identical results."""
    src = tmp_path / "model_tex_in.glb"
    dst1 = tmp_path / "model_tex_out1.glb"
    dst2 = tmp_path / "model_tex_out2.glb"
    dst3 = tmp_path / "model_tex_out3.glb"

    # Synthetic textured fabric: red base with random noise
    rng_tex = np.random.default_rng(12345)
    noise_arr = rng_tex.integers(-20, 21, size=(64, 64, 3))
    color_plane = np.clip(np.array([180, 50, 50], dtype=int) + noise_arr, 0, 255).astype(np.uint8)
    base_arr = np.dstack([color_plane, np.full((64, 64), 255, dtype=np.uint8)])
    base_img = Image.fromarray(base_arr, mode="RGBA")
    draw = ImageDraw.Draw(base_img)
    draw.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(255, 255, 255, 255))
    # Pillow's rasterized ellipse includes pixels beyond the mathematical radius.
    # Preservation checks must use the actual input, including those logo pixels.
    input_arr = np.array(base_img)

    create_synthetic_glb(src, base_img)

    y_idx, x_idx = np.ogrid[0:64, 0:64]
    dist = np.hypot(x_idx - 20, y_idx - 32)
    inner_mask = dist <= 6
    outside_mask = dist > 8
    ring_mask = (dist > 8) & (dist <= 8 + 24)

    ring_med = np.median(base_arr[ring_mask, :3], axis=0)
    ring_std = np.std(base_arr[ring_mask, :3], axis=0)
    assert ring_std.mean() > 5.0, "Texture must have non-zero variance for noise test"

    patch_seed42 = {
        "image": None,
        "center": [20, 32],
        "radius": 8,
        "mode": "fill",
        "ring": 24,
        "feather": 2,
        "seed": 42,
    }
    patch_seed99 = {
        "image": None,
        "center": [20, 32],
        "radius": 8,
        "mode": "fill",
        "ring": 24,
        "feather": 2,
        "seed": 99,
    }

    # Patch twice with seed 42, once with seed 99
    patch_glb(src, dst1, [patch_seed42])
    patch_glb(src, dst2, [patch_seed42])
    patch_glb(src, dst3, [patch_seed99])

    res1 = gl.GLTF2().load(str(dst1))
    res2 = gl.GLTF2().load(str(dst2))
    res3 = gl.GLTF2().load(str(dst3))

    img1 = np.array(get_image_from_glb(res1, 0))
    img2 = np.array(get_image_from_glb(res2, 0))
    img3 = np.array(get_image_from_glb(res3, 0))

    # 1. Same seed produces byte-for-byte identical output
    assert np.array_equal(img1, img2), "Same seed did not produce identical output"

    # 2. Inside circle is close to ring median (mean color matches ring median)
    diff_inner = np.abs(img1[inner_mask, :3].astype(float).mean(axis=0) - ring_med).max()
    assert diff_inner < 4.0, f"Expected average diff < 4.0, got {diff_inner}"

    # 3. Noise was actually added in textured fill (std inside > 1)
    assert img1[inner_mask, :3].std() > 1.0, "Expected noise inside textured fill"

    # 4. Different seed produces different noise in filled area
    assert not np.array_equal(img1[inner_mask], img3[inner_mask]), "Different seeds produced identical pixels"

    # 5. Outside circle is byte-for-byte identical to original in all cases
    assert np.array_equal(img1[outside_mask], input_arr[outside_mask])
    assert np.array_equal(img3[outside_mask], input_arr[outside_mask])

    # 6. Alpha channel preserved
    assert np.array_equal(img1[..., 3], input_arr[..., 3])
    assert np.array_equal(img3[..., 3], input_arr[..., 3])


def test_patch_glb_fill_mode_explicit_normal(tmp_path):
    """Test that setting normal=True suppresses noise even on non-neutral texture."""
    src = tmp_path / "model_exp_norm_in.glb"
    dst = tmp_path / "model_exp_norm_out.glb"

    # Green texture with texture noise (normally wouldn't trigger auto-detection)
    rng = np.random.default_rng(777)
    noise_arr = rng.integers(-15, 16, size=(64, 64, 3))
    color_plane = np.clip(np.array([40, 190, 40], dtype=int) + noise_arr, 0, 255).astype(np.uint8)
    base_arr = np.dstack([color_plane, np.full((64, 64), 255, dtype=np.uint8)])
    base_img = Image.fromarray(base_arr, mode="RGBA")
    draw = ImageDraw.Draw(base_img)
    draw.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(255, 255, 255, 255))
    # Compare preservation against the rasterized input, not the pre-logo fabric.
    input_arr = np.array(base_img)

    create_synthetic_glb(src, base_img)

    patch = {
        "image": None,
        "center": [20, 32],
        "radius": 8,
        "mode": "fill",
        "normal": True,
        "ring": 20,
        "feather": 2,
    }
    patch_glb(src, dst, [patch])

    res = gl.GLTF2().load(str(dst))
    patched = np.array(get_image_from_glb(res, 0))

    y_idx, x_idx = np.ogrid[0:64, 0:64]
    dist = np.hypot(x_idx - 20, y_idx - 32)
    inner_mask = dist <= 6
    outside_mask = dist > 8

    # When normal=True is explicitly given, no noise is added (std < 1.0)
    assert patched[inner_mask, :3].std(axis=0).max() < 1.0
    assert np.array_equal(patched[outside_mask], input_arr[outside_mask])
    assert np.array_equal(patched[..., 3], input_arr[..., 3])


def test_patch_glb_fill_mode_cli_and_filtering(tmp_path):
    """Test CLI execution and image filtering with mode='fill'."""
    src = tmp_path / "cli_fill_in.glb"
    dst = tmp_path / "cli_fill_out.glb"
    patches_file = tmp_path / "patches_fill.json"

    base_img = Image.new("RGBA", (64, 64), (190, 40, 40, 255))
    draw_b = ImageDraw.Draw(base_img)
    draw_b.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(255, 255, 255, 255))

    norm_img = Image.new("RGBA", (64, 64), (128, 128, 255, 255))
    draw_n = ImageDraw.Draw(norm_img)
    draw_n.ellipse([20 - 8, 32 - 8, 20 + 8, 32 + 8], fill=(200, 80, 180, 255))

    create_synthetic_glb(src, base_img, norm_img)

    patches = [{
        "image": None,
        "center": [20, 32],
        "radius": 8,
        "mode": "fill",
        "feather": 2,
    }]
    patches_file.write_text(json.dumps(patches), encoding="utf-8")

    # CLI with --images 0: only patch image 0 (baseColor), normal map stays untouched
    ret = main([
        str(src),
        str(dst),
        "--patches",
        str(patches_file),
        "--images",
        "0",
    ])
    assert ret == 0
    assert dst.exists()

    res_gltf = gl.GLTF2().load(str(dst))
    patched_base = np.array(get_image_from_glb(res_gltf, 0))
    patched_norm = np.array(get_image_from_glb(res_gltf, 1))

    y_idx, x_idx = np.ogrid[0:64, 0:64]
    dist = np.hypot(x_idx - 20, y_idx - 32)

    # Base image was patched with fill mode
    assert np.abs(patched_base[dist <= 6, :3].astype(float) - [190, 40, 40]).mean() < 8.0
    assert np.array_equal(patched_base[dist > 8], np.array(base_img)[dist > 8])

    # Normal map was NOT patched (bump is still there)
    assert np.array_equal(patched_norm, np.array(norm_img))


def test_copy_mode_requires_source_offset(tmp_path):
    """Test that copy mode raises ValueError if source_offset is omitted, while fill mode succeeds."""
    src = tmp_path / "model_mode_check.glb"
    dst = tmp_path / "model_mode_out.glb"
    base_img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
    create_synthetic_glb(src, base_img)

    # Copy mode without source_offset -> ValueError
    patch_copy_invalid = {
        "center": [20, 20],
        "radius": 5,
        "mode": "copy",
    }
    with pytest.raises(ValueError, match="source_offset"):
        patch_glb(src, dst, [patch_copy_invalid])

    # Fill mode without source_offset -> succeeds
    patch_fill_valid = {
        "center": [20, 20],
        "radius": 5,
        "mode": "fill",
    }
    patch_glb(src, dst, [patch_fill_valid])
    assert dst.exists()

