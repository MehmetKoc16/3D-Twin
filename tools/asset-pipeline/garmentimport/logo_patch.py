"""Patch circular logo regions in GLB textures by copying from neighboring fabric or filling with ring statistics."""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pygltflib as gl


def detect_mime(data: bytes, fallback: str = "image/png") -> str:
    """Detect image MIME type from magic header bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return fallback


def find_material_images(gltf: gl.GLTF2) -> list[int]:
    """Find all image indices referenced by materials in the GLB model."""
    images: list[int] = []
    for mat in gltf.materials:
        tex_infos = []
        pbr = getattr(mat, "pbrMetallicRoughness", None)
        if pbr:
            if getattr(pbr, "baseColorTexture", None):
                tex_infos.append(pbr.baseColorTexture)
            if getattr(pbr, "metallicRoughnessTexture", None):
                tex_infos.append(pbr.metallicRoughnessTexture)
        if getattr(mat, "normalTexture", None):
            tex_infos.append(mat.normalTexture)
        if getattr(mat, "occlusionTexture", None):
            tex_infos.append(mat.occlusionTexture)
        if getattr(mat, "emissiveTexture", None):
            tex_infos.append(mat.emissiveTexture)

        for info in tex_infos:
            if info and getattr(info, "index", None) is not None:
                t_idx = info.index
                if 0 <= t_idx < len(gltf.textures):
                    tex = gltf.textures[t_idx]
                    if getattr(tex, "source", None) is not None:
                        src_idx = tex.source
                        if 0 <= src_idx < len(gltf.images) and src_idx not in images:
                            images.append(src_idx)
    if not images:
        return list(range(len(gltf.images)))
    return images


def apply_patch_to_image(
    img: Image.Image,
    center: list[float] | tuple[float, float],
    radius: float,
    source_offset: list[float] | tuple[float, float] | None = None,
    feather: float = 0.0,
    mode: str = "copy",
    ring: float = 24.0,
    seed: int = 0,
    normal: bool | None = None,
) -> Image.Image:
    """Patch circular logo region in img using 'copy' (from offset) or 'fill' (from ring stats)."""
    w, h = img.size
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    f = float(feather)

    if r <= 0:
        return img

    if mode not in ("copy", "fill"):
        raise ValueError(f"Unknown patch mode: {mode!r}, expected 'copy' or 'fill'")

    if mode == "copy":
        if source_offset is None:
            raise ValueError("source_offset is required when mode is 'copy'")
        dx, dy = float(source_offset[0]), float(source_offset[1])
        src_cx = cx + dx
        src_cy = cy + dy

        # Check bounds of the source circle
        if (src_cx - r < 0 or src_cx + r > w or
                src_cy - r < 0 or src_cy + r > h):
            raise ValueError(
                f"Source region out of bounds for image dimensions ({w}, {h}): "
                f"source circle center=({src_cx:.1f}, {src_cy:.1f}), radius={r:.1f} "
                f"extends to [{src_cx - r:.1f}, {src_cy - r:.1f}, {src_cx + r:.1f}, {src_cy + r:.1f}]"
            )

    orig_mode = img.mode
    # Convert palette images to RGBA to avoid palette index blending issues
    work_img = img if orig_mode in ("RGB", "RGBA", "L") else img.convert("RGBA")
    original = np.array(work_img)
    arr = original.copy()
    is_3d = arr.ndim == 3

    # Target bounding box in pixel coordinates
    x0 = max(0, int(math.floor(cx - r)))
    x1 = min(w, int(math.ceil(cx + r)) + 1)
    y0 = max(0, int(math.floor(cy - r)))
    y1 = min(h, int(math.ceil(cy + r)) + 1)

    if x0 >= x1 or y0 >= y1:
        return img

    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(grid_x - cx, grid_y - cy)

    circle_mask = dist <= r
    if not np.any(circle_mask):
        return img

    tgt_box = original[y0:y1, x0:x1].astype(np.float32)

    if mode == "copy":
        src_x = np.round(grid_x + dx).astype(int)
        src_y = np.round(grid_y + dy).astype(int)

        # Verify all sampled source pixels for pixels within the circle are within image bounds
        if (np.any(src_x[circle_mask] < 0) or np.any(src_x[circle_mask] >= w) or
                np.any(src_y[circle_mask] < 0) or np.any(src_y[circle_mask] >= h)):
            raise ValueError(
                f"Source region sampled pixels exceed image dimensions ({w}, {h})"
            )

        src_x_clamped = np.clip(src_x, 0, w - 1)
        src_y_clamped = np.clip(src_y, 0, h - 1)
        src_box = original[src_y_clamped, src_x_clamped].astype(np.float32)

    elif mode == "fill":
        r_outer = r + float(ring)
        rx0 = max(0, int(math.floor(cx - r_outer)))
        rx1 = min(w, int(math.ceil(cx + r_outer)) + 1)
        ry0 = max(0, int(math.floor(cy - r_outer)))
        ry1 = min(h, int(math.ceil(cy + r_outer)) + 1)

        if rx0 >= rx1 or ry0 >= ry1:
            return img

        grid_ry, grid_rx = np.mgrid[ry0:ry1, rx0:rx1]
        ring_dist = np.hypot(grid_rx - cx, grid_ry - cy)
        ring_mask = (ring_dist > r) & (ring_dist <= r_outer)

        ring_box = original[ry0:ry1, rx0:rx1].copy()
        if not np.any(ring_mask):
            ring_mask = (ring_dist >= r) & (ring_dist <= r_outer)
        if not np.any(ring_mask):
            ring_mask = np.ones_like(ring_dist, dtype=bool)

        if is_3d and original.shape[2] == 4:
            alpha_ring_mask = ring_mask & (ring_box[..., 3] > 0)
            if np.any(alpha_ring_mask):
                ring_pixels = ring_box[alpha_ring_mask]
            else:
                ring_pixels = ring_box[ring_mask]
        else:
            ring_pixels = ring_box[ring_mask]

        med = np.median(ring_pixels, axis=0)
        std = np.std(ring_pixels, axis=0)

        # Detect if image is a normal map
        is_normal = False
        if normal is True:
            is_normal = True
        elif normal is not False:
            if is_3d and original.shape[2] >= 3:
                med_r = float(med[0])
                med_g = float(med[1])
                med_b = float(med[2])
                if med_b > 200 and (108 <= med_r <= 148) and (108 <= med_g <= 148):
                    is_normal = True

        rng = np.random.default_rng(seed)
        box_h, box_w = y1 - y0, x1 - x0

        if is_3d:
            num_channels = original.shape[2]
            has_alpha = (num_channels == 4)
            color_channels = 3 if has_alpha else num_channels

            if is_normal:
                noise_std = np.zeros(color_channels, dtype=np.float32)
            else:
                noise_std = (0.6 * std[:color_channels]).astype(np.float32)

            noise = rng.standard_normal((box_h, box_w, color_channels)).astype(np.float32) * noise_std
            src_box = np.empty((box_h, box_w, num_channels), dtype=np.float32)
            src_box[..., :color_channels] = med[:color_channels].astype(np.float32) + noise
            if has_alpha:
                src_box[..., 3] = tgt_box[..., 3]
        else:
            if is_normal:
                noise_std = 0.0
            else:
                noise_std = 0.6 * float(std)
            noise = rng.standard_normal((box_h, box_w)).astype(np.float32) * noise_std
            src_box = (float(med) + noise).astype(np.float32)

    # Calculate linear alpha: 1.0 inside (radius - feather), decreasing to 0.0 at radius
    r_inner = max(0.0, r - f)
    if f <= 0:
        alpha = np.where(dist <= r, 1.0, 0.0)
    else:
        alpha = np.where(
            dist <= r_inner,
            1.0,
            np.where(dist <= r, (r - dist) / f, 0.0)
        )
    alpha = np.clip(alpha, 0.0, 1.0)

    alpha_c = alpha[..., np.newaxis] if is_3d else alpha

    blended = np.round(np.clip(
        alpha_c * src_box + (1.0 - alpha_c) * tgt_box,
        0,
        255
    )).astype(arr.dtype)

    # Only modify pixels where alpha > 0, ensuring untouched regions stay byte-for-byte identical
    arr[y0:y1, x0:x1] = np.where(alpha_c > 0, blended, original[y0:y1, x0:x1])

    res = Image.fromarray(arr, mode=work_img.mode)
    if res.mode != orig_mode and orig_mode != "P":
        res = res.convert(orig_mode)
    return res


def patch_glb(
    src: str | Path,
    dst: str | Path,
    patches: list[dict[str, Any]] | dict[str, Any],
    images: list[int] | None = None,
) -> gl.GLTF2:
    """Patch circular logo regions in GLB textures and repack safely into dst."""
    src_path = Path(src)
    dst_path = Path(dst)

    if isinstance(patches, dict):
        patch_list = [patches]
    else:
        patch_list = list(patches)

    gltf = gl.GLTF2().load(str(src_path))
    old_blob = gltf.binary_blob()
    if old_blob is None:
        raise ValueError(f"Could not read binary blob from GLB: {src_path}")

    default_images = list(images) if images is not None else find_material_images(gltf)

    # Metadata and loaded PIL instances for images
    loaded_images: dict[int, Image.Image] = {}
    image_mimes: dict[int, str] = {}
    image_bvs: dict[int, int] = {}
    modified_images: set[int] = set()

    def load_img(idx: int) -> Image.Image:
        if idx in loaded_images:
            return loaded_images[idx]
        if idx < 0 or idx >= len(gltf.images):
            raise IndexError(f"Image index {idx} out of range (total {len(gltf.images)})")
        gl_img = gltf.images[idx]
        if gl_img.bufferView is not None:
            image_bvs[idx] = gl_img.bufferView
            bv = gltf.bufferViews[gl_img.bufferView]
            offset = bv.byteOffset or 0
            length = bv.byteLength
            raw_bytes = old_blob[offset : offset + length]
        elif gl_img.uri and gl_img.uri.startswith("data:"):
            raw_bytes = base64.b64decode(gl_img.uri.split(",", 1)[1])
        else:
            raise ValueError(f"Image {idx} has no embedded binary data in GLB")

        mime = gl_img.mimeType or detect_mime(raw_bytes)
        image_mimes[idx] = mime
        pil_img = Image.open(io.BytesIO(raw_bytes))
        loaded_images[idx] = pil_img
        return pil_img

    # Preload default images
    for idx in default_images:
        load_img(idx)

    for patch in patch_list:
        patch_img = patch.get("image")
        center = patch["center"]
        radius = float(patch["radius"])
        mode = patch.get("mode", "copy")
        feather = float(patch.get("feather", 0.0))
        ring = float(patch.get("ring", 24.0))
        seed = int(patch.get("seed", 0))
        normal = patch.get("normal", None)

        if mode == "copy":
            if "source_offset" not in patch or patch["source_offset"] is None:
                raise ValueError("Patch with mode 'copy' requires 'source_offset'")
            source_offset = patch["source_offset"]
        elif mode == "fill":
            source_offset = patch.get("source_offset")
        else:
            raise ValueError(f"Unknown patch mode: {mode!r}, expected 'copy' or 'fill'")

        if patch_img is not None:
            target_list = [patch_img]
        else:
            target_list = default_images

        if not target_list:
            continue

        ref_size = patch.get("reference_size")
        if ref_size is None:
            ref_img = load_img(target_list[0])
            ref_w, ref_h = ref_img.size
        else:
            ref_w, ref_h = float(ref_size[0]), float(ref_size[1])

        for img_idx in target_list:
            cur_img = load_img(img_idx)
            cur_w, cur_h = cur_img.size

            if cur_w == ref_w and cur_h == ref_h:
                scaled_center = center
                scaled_radius = radius
                scaled_offset = source_offset
                scaled_feather = feather
                scaled_ring = ring
            else:
                sx = cur_w / ref_w
                sy = cur_h / ref_h
                sr = math.sqrt(sx * sy)
                scaled_center = [center[0] * sx, center[1] * sy]
                scaled_radius = radius * sr
                scaled_offset = [source_offset[0] * sx, source_offset[1] * sy] if source_offset is not None else None
                scaled_feather = feather * sr
                scaled_ring = ring * sr

            patched_img = apply_patch_to_image(
                cur_img,
                center=scaled_center,
                radius=scaled_radius,
                source_offset=scaled_offset,
                feather=scaled_feather,
                mode=mode,
                ring=scaled_ring,
                seed=seed,
                normal=normal,
            )
            loaded_images[img_idx] = patched_img
            modified_images.add(img_idx)

    # Encode modified images back to bytes
    modified_bv_data: dict[int, bytes] = {}
    for img_idx in modified_images:
        pil_img = loaded_images[img_idx]
        mime = image_mimes.get(img_idx, "image/png").lower()
        out_buf = io.BytesIO()

        if "jpeg" in mime or "jpg" in mime:
            save_img = pil_img.convert("RGB") if pil_img.mode in ("RGBA", "LA", "P") else pil_img
            save_img.save(out_buf, format="JPEG", quality=95)
        else:
            pil_img.save(out_buf, format="PNG")

        new_data = out_buf.getvalue()
        bv_idx = image_bvs.get(img_idx)
        if bv_idx is not None and bv_idx not in modified_bv_data:
            modified_bv_data[bv_idx] = new_data
        else:
            new_bv = len(gltf.bufferViews)
            gltf.bufferViews.append(gl.BufferView(buffer=0, byteOffset=0, byteLength=len(new_data)))
            gltf.images[img_idx].bufferView = new_bv
            gltf.images[img_idx].uri = None
            modified_bv_data[new_bv] = new_data

    # Repack all bufferViews sequentially into a new binary blob, 4-byte aligned
    new_blob = bytearray()
    for i, bv in enumerate(gltf.bufferViews):
        pad = (-len(new_blob)) % 4
        new_blob.extend(b"\x00" * pad)

        if i in modified_bv_data:
            data = modified_bv_data[i]
        else:
            offset = bv.byteOffset or 0
            length = bv.byteLength
            data = old_blob[offset : offset + length]

        bv.byteOffset = len(new_blob)
        bv.byteLength = len(data)
        bv.buffer = 0
        new_blob.extend(data)

    pad = (-len(new_blob)) % 4
    new_blob.extend(b"\x00" * pad)

    if not gltf.buffers:
        gltf.buffers = [gl.Buffer(byteLength=len(new_blob))]
    else:
        gltf.buffers[0].byteLength = len(new_blob)
        gltf.buffers[0].uri = None

    gltf.set_binary_blob(bytes(new_blob))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(dst_path))
    return gltf


def parse_images(val: str | None) -> list[int] | None:
    """Parse comma-separated image indices string (e.g. '3,4,5')."""
    if not val:
        return None
    return [int(x.strip()) for x in val.split(",") if x.strip()]


def parser() -> argparse.ArgumentParser:
    """Return command line parser for logo patching tool."""
    p = argparse.ArgumentParser(
        description="Patch circular logo regions in GLB textures by copying or filling with ring statistics."
    )
    p.add_argument("src", type=Path, help="Source GLB file")
    p.add_argument("dst", type=Path, help="Destination GLB file")
    p.add_argument("--patches", type=Path, required=True, help="Path to patches JSON file")
    p.add_argument("--images", type=str, default=None, help="Comma-separated image indices to patch (e.g. 3,4,5)")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for logo_patch."""
    args = parser().parse_args(argv)
    with open(args.patches, "r", encoding="utf-8") as f:
        patches_data = json.load(f)
    images = parse_images(args.images)
    patch_glb(args.src, args.dst, patches_data, images=images)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
