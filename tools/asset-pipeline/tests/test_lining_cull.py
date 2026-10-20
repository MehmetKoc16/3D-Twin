"""Synthetic tests for the shoe lining show-through cull (garmentimport.lining_cull)."""
from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pygltflib as gl
import pytest
from PIL import Image

PIPELINE = Path(__file__).resolve().parents[1]
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from garmentimport.lining_cull import (  # noqa: E402
    classify_dark,
    cull_lining,
    main,
    proximity_covered,
    ray_covered,
    read_glb_mesh,
    split_pair,
)

DARK_UV = (0.25, 0.5)  # left texture half: dark grey lining
LIGHT_UV = (0.75, 0.5)  # right texture half: light red canvas
Quad = tuple[np.ndarray, bool]  # (4 corners in metres, is_dark)


def make_texture() -> Image.Image:
    """8x8 texture: left half dark grey (60,60,60), right half light red (200,40,40)."""
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:, :4] = (60, 60, 60)
    arr[:, 4:] = (200, 40, 40)
    return Image.fromarray(arr, mode="RGB")


def rect_z(x0: float, x1: float, y0: float, y1: float, z: float) -> np.ndarray:
    """Axis-aligned rectangle in the plane z = const."""
    return np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=np.float64)


def shoe(dark_z: float, light_z: float | None, *, y_low: float = 0.06, x: float = 0.1) -> list[Quad]:
    """One shoe on the +x side: a big light 'heel' anchor 80 mm behind (sets the shoe centroid, lowest vertex at
    y = 0), an optional 10 mm light canvas quad at ``light_z`` and a 10 mm dark quad at ``dark_z``.

    Quad order: anchor (triangles 0, 1), light (2, 3) if present, dark (last two triangles).
    """
    quads: list[Quad] = [(rect_z(x - 0.02, x + 0.02, 0.0, 0.10, -0.08), False)]
    if light_z is not None:
        quads.append((rect_z(x - 0.005, x + 0.005, y_low, y_low + 0.01, light_z), False))
    quads.append((rect_z(x - 0.005, x + 0.005, y_low, y_low + 0.01, dark_z), True))
    return quads


def mirror(quads: list[Quad]) -> list[Quad]:
    """Mirror a shoe across x = 0 (the partner shoe of the pair)."""
    return [(corners * np.array([-1.0, 1.0, 1.0]), is_dark) for corners, is_dark in quads]


def build_glb(path: Path, quads: list[Quad], *, index_type: int = 5123, with_texture: bool = True) -> None:
    """Minimal GLB: one mesh, one primitive, one material; each quad = 4 vertices and 2 triangles."""
    positions = np.concatenate([c for c, _ in quads]).astype("<f4")
    uvs = np.array([DARK_UV if d else LIGHT_UV for _, d in quads for _ in range(4)], dtype="<f4")
    tris = np.array([[4 * q, 4 * q + 1, 4 * q + 2, 4 * q, 4 * q + 2, 4 * q + 3] for q in range(len(quads))])
    indices = tris.reshape(-1).astype("<u2" if index_type == 5123 else "<u4")

    g = gl.GLTF2(asset=gl.Asset(version="2.0", generator="test_lining_cull"))
    blob = bytearray()

    def add_view(data: bytes, target: int | None = None) -> int:
        blob.extend(b"\x00" * ((-len(blob)) % 4))
        g.bufferViews.append(gl.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data), target=target))
        blob.extend(data)
        return len(g.bufferViews) - 1

    def add_accessor(view: int, ctype: int, count: int, kind: str, arr: np.ndarray) -> int:
        flat = arr.reshape(count, -1)
        g.accessors.append(gl.Accessor(
            bufferView=view, componentType=ctype, count=count, type=kind,
            min=flat.min(axis=0).tolist(), max=flat.max(axis=0).tolist(),
        ))
        return len(g.accessors) - 1

    pos_acc = add_accessor(add_view(positions.tobytes(), 34962), 5126, len(positions), "VEC3", positions)
    uv_acc = add_accessor(add_view(uvs.tobytes(), 34962), 5126, len(uvs), "VEC2", uvs)
    idx_acc = add_accessor(add_view(indices.tobytes(), 34963), index_type, len(indices), "SCALAR", indices)

    pbr = gl.PbrMetallicRoughness(baseColorFactor=[1.0, 1.0, 1.0, 1.0])
    if with_texture:
        buf = io.BytesIO()
        make_texture().save(buf, format="PNG")
        g.images.append(gl.Image(bufferView=add_view(buf.getvalue()), mimeType="image/png"))
        g.textures.append(gl.Texture(source=0))
        pbr.baseColorTexture = gl.TextureInfo(index=0)
    g.materials = [gl.Material(pbrMetallicRoughness=pbr)]
    g.meshes = [gl.Mesh(primitives=[gl.Primitive(
        attributes=gl.Attributes(POSITION=pos_acc, TEXCOORD_0=uv_acc), indices=idx_acc, material=0,
    )])]
    g.nodes = [gl.Node(mesh=0)]
    g.scenes = [gl.Scene(nodes=[0])]
    g.scene = 0
    blob.extend(b"\x00" * ((-len(blob)) % 4))
    g.buffers = [gl.Buffer(byteLength=len(blob))]
    g.set_binary_blob(bytes(blob))
    path.parent.mkdir(parents=True, exist_ok=True)
    g.save_binary(str(path))


def culled_set(src: Path, dst: Path) -> set[int]:
    """Indices of triangles that changed between src and dst; asserts each changed triangle is degenerate."""
    before = read_glb_mesh(src).triangles
    after = read_glb_mesh(dst).triangles
    changed = np.flatnonzero((before != after).any(axis=1))
    for t in changed:
        assert after[t, 0] == after[t, 1] == after[t, 2] == before[t, 0]
    return {int(t) for t in changed}


def json_chunk(raw: bytes) -> bytes:
    length = struct.unpack_from("<I", raw, 12)[0]
    return raw[20 : 20 + length]


def test_dark_layer_outside_light_layer_is_culled(tmp_path):
    """WHY: the sf-converse patches are lining lying a few mm OUTSIDE the red canvas; a dark quad 3 mm farther from
    the shoe centre than a parallel light quad is exactly that case and must be hidden (by method B)."""
    src, dst = tmp_path / "in.glb", tmp_path / "out.glb"
    build_glb(src, shoe(dark_z=0.003, light_z=0.0))
    report = cull_lining(src, dst)
    assert culled_set(src, dst) == {4, 5}
    assert report["dark"] == 2
    assert report["culled_proximity"] == 2 and report["culled_ray"] == 0 and report["culled_total"] == 2
    assert report["gap_mm"]["proximity"]["min"] == pytest.approx(3.0, abs=0.01)


def test_normal_lining_inside_canvas_is_kept(tmp_path):
    """WHY: correctly placed lining sits INSIDE the canvas (light layer outside); it is what you see through the
    ankle opening and must never be removed."""
    src, dst = tmp_path / "in.glb", tmp_path / "out.glb"
    build_glb(src, shoe(dark_z=0.0, light_z=0.003))
    report = cull_lining(src, dst)
    assert report["dark"] == 2
    assert report["culled_total"] == 0
    assert culled_set(src, dst) == set()


def test_lone_dark_surface_is_kept(tmp_path):
    """WHY: a dark feature with no light layer behind it (a black sole stripe) is real geometry; deleting all dark
    triangles opened holes in the sole, so colour alone must not cull. Placed above the sole guard on purpose."""
    src, dst = tmp_path / "in.glb", tmp_path / "out.glb"
    build_glb(src, shoe(dark_z=0.003, light_z=None))
    report = cull_lining(src, dst)
    assert report["dark"] == 2
    assert report["culled_total"] == 0


def test_sole_guard_protects_bottom_35mm(tmp_path):
    """WHY: stacked dark-over-light layers also occur in the rigid sole (stripes over white rubber); the 35 mm sole
    guard keeps them. With sole_mm=0 the same pair is culled, proving the guard (not geometry) kept it."""
    src = tmp_path / "in.glb"
    build_glb(src, shoe(dark_z=0.003, light_z=0.0, y_low=0.010))
    guarded = cull_lining(src, tmp_path / "guarded.glb")
    assert guarded["culled_total"] == 0
    assert guarded["protected_sole"] == 2
    unguarded = cull_lining(src, tmp_path / "unguarded.glb", sole_mm=0.0)
    assert unguarded["culled_total"] == 2
    assert culled_set(src, tmp_path / "unguarded.glb") == {4, 5}


def test_only_index_bytes_change(tmp_path):
    """WHY: the app binds every vertex to the body through a separate .bind.bin, so vertices, attributes, JSON and
    file length must stay identical; only culled triangles may change, and only to degenerate (a, a, a)."""
    src, dst = tmp_path / "in.glb", tmp_path / "out.glb"
    build_glb(src, shoe(dark_z=0.003, light_z=0.0), index_type=5125)
    cull_lining(src, dst)
    a, b = src.read_bytes(), dst.read_bytes()
    assert len(a) == len(b)
    assert json_chunk(a) == json_chunk(b)

    mesh = read_glb_mesh(src)
    start = mesh.index_offset
    end = start + mesh.triangles.size * mesh.index_dtype.itemsize
    assert a[:start] == b[:start]
    assert a[end:] == b[end:]
    out = read_glb_mesh(dst)
    assert np.array_equal(mesh.positions, out.positions)
    assert np.array_equal(mesh.uvs, out.uvs)
    assert culled_set(src, dst) == {4, 5}


def test_mirrored_pair_culls_both_shoes(tmp_path):
    """WHY: the shipped asset is a pair mirrored at x = 0; each shoe needs its own centroid for 'outside', so the
    patch on one shoe and its mirror on the other must both be culled."""
    src, dst = tmp_path / "in.glb", tmp_path / "out.glb"
    right = shoe(dark_z=0.003, light_z=0.0)
    build_glb(src, right + mirror(right))
    mesh = read_glb_mesh(src)
    assert len(split_pair(mesh.positions, mesh.triangles)) == 2
    report = cull_lining(src, dst)
    assert culled_set(src, dst) == {4, 5, 10, 11}
    assert report["culled_total"] == 4


def test_ray_pass_catches_gap_beyond_proximity_radius(tmp_path):
    """WHY: each pass alone left patches; a lining 9 mm outside the canvas is beyond the 8 mm proximity radius and
    must be caught by the inward ray pass (10 mm max gap) that runs on what proximity left."""
    src, dst = tmp_path / "in.glb", tmp_path / "out.glb"
    build_glb(src, shoe(dark_z=0.009, light_z=0.0))
    mesh = read_glb_mesh(src)
    dark = classify_dark(mesh.triangles, mesh.uvs, mesh.texture)
    assert not proximity_covered(mesh.positions, mesh.triangles, dark).any()
    assert ray_covered(mesh.positions, mesh.triangles, dark, dark).sum() == 2
    report = cull_lining(src, dst)
    assert report["culled_proximity"] == 0 and report["culled_ray"] == 2
    assert report["gap_mm"]["ray"]["median"] == pytest.approx(9.0, abs=0.01)
    assert culled_set(src, dst) == {4, 5}


def test_cli_writes_report(tmp_path):
    """WHY: the lead runs the CLI on the shipped shoe and reviews the JSON counts before copying the result."""
    src, dst, rpt = tmp_path / "in.glb", tmp_path / "out.glb", tmp_path / "sub" / "report.json"
    right = shoe(dark_z=0.003, light_z=0.0)
    build_glb(src, right + mirror(right))
    assert main([str(src), str(dst), "--report", str(rpt), "--radius-mm", "8", "--min-hits", "1"]) == 0
    report = json.loads(rpt.read_text(encoding="utf-8"))
    assert report["triangles"] == 12
    assert report["dark"] == 4
    assert report["culled_total"] == 4
    assert report["culled_proximity"] + report["culled_ray"] == report["culled_total"]
    assert report["protected_sole"] == 0
    assert dst.exists()


def test_missing_base_color_texture_raises(tmp_path):
    """WHY: the dark/light rule needs the embedded base colour texture; failing loudly beats culling nothing."""
    src = tmp_path / "in.glb"
    build_glb(src, shoe(dark_z=0.003, light_z=0.0), with_texture=False)
    with pytest.raises(ValueError, match="baseColorTexture"):
        read_glb_mesh(src)
