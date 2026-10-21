"""Restore woven fabric from a verified, seam-free donor rectangle in the same UV island.

Small donors are tiled by reflection, so a full disc need not fit beside the defect.
All PBR maps must use the same donor and mask. The binary buffer is repacked from
bufferViews, dropping old images and gaps while preserving accessor payload bytes.
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def restore_canvas(image: Image.Image, center: tuple[float, float],
                   radii: tuple[float, float], donor: tuple[int, int, int, int],
                   feather: float = 0.18, normal: bool = False) -> Image.Image:
    """Reflect a real canvas swatch into an ellipse, feathering only its outer edge."""
    pixels = np.array(image.convert('RGBA'))
    height, width = pixels.shape[:2]
    x0, y0, x1, y1 = donor
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Donor rectangle must lie inside the image')
    if min(radii) <= 0 or not 0 < feather <= 1:
        raise ValueError('Positive radii and feather in (0, 1] required')
    yy, xx = np.mgrid[:height, :width]
    distance = np.hypot((xx-center[0])/radii[0], (yy-center[1])/radii[1])
    alpha = np.clip((1-distance)/feather, 0, 1)[..., None]
    def reflected(coords: np.ndarray, start: int, length: int) -> np.ndarray:
        phase = (coords-start) % (2*length)
        return start+np.minimum(phase, 2*length-1-phase)
    sample = pixels[reflected(yy, y0, y1-y0), reflected(xx, x0, x1-x0)].astype(float)
    result = sample*alpha+pixels*(1-alpha)
    if normal:
        vectors = result[..., :3]/127.5-1
        vectors /= np.maximum(np.linalg.norm(vectors, axis=2, keepdims=True), 1e-12)
        result[..., :3] = (vectors+1)*127.5
    result = np.where(alpha > 0, np.clip(np.rint(result), 0, 255), pixels).astype('uint8')
    return Image.fromarray(result).convert(image.mode)


def replace_embedded_pngs(src: Path, dst: Path, images: dict[int, Image.Image]) -> None:
    """Replace PNGs and compact bufferViews without decoding or rewriting accessor data."""
    raw = src.read_bytes()
    json_length, kind = struct.unpack_from('<II', raw, 12)
    if raw[:4] != b'glTF' or kind != 0x4E4F534A:
        raise ValueError('Expected GLB JSON chunk')
    document = json.loads(raw[20:20+json_length])
    offset = 20+json_length
    bin_length, bin_kind = struct.unpack_from('<II', raw, offset)
    if bin_kind != 0x004E4942 or offset+8+bin_length != len(raw):
        raise ValueError('Expected one trailing BIN chunk')
    old_blob = raw[offset+8:]
    if len(document['buffers']) != 1 or document['buffers'][0].get('uri') is not None:
        raise ValueError('Expected one embedded buffer')
    replacements: dict[int, bytes] = {}
    for index, image in images.items():
        view_index = document['images'][index]['bufferView']
        if any(a.get('bufferView') == view_index for a in document['accessors']):
            raise ValueError('Image bufferView also contains geometry')
        encoded = io.BytesIO()
        image.save(encoded, format='PNG')
        if sum(im.get('bufferView') == view_index for im in document['images']) != 1:
            raise ValueError('Replacement image bufferView must not be shared')
        replacements[view_index] = encoded.getvalue()
        document['images'][index]['mimeType'] = 'image/png'
    blob = bytearray()
    for index, view in enumerate(document['bufferViews']):
        start, length = view.get('byteOffset', 0), view['byteLength']
        if view.get('buffer', 0) != 0 or start < 0 or length < 0 or start+length > len(old_blob):
            raise ValueError('BufferView lies outside the embedded buffer')
        payload = replacements.get(index, old_blob[start:start+length])
        blob.extend(b'\0' * (-len(blob) % 4))
        view['byteOffset'] = len(blob)
        view['byteLength'] = len(payload)
        blob.extend(payload)
    blob.extend(b'\0' * (-len(blob) % 4))
    document['buffers'][0]['byteLength'] = len(blob)
    metadata = json.dumps(document, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    metadata += b' ' * (-len(metadata) % 4)
    total = 12+8+len(metadata)+8+len(blob)
    dst.write_bytes(struct.pack('<4sII', b'glTF', 2, total)
                    +struct.pack('<II', len(metadata), kind)+metadata
                    +struct.pack('<II', len(blob), bin_kind)+blob)
