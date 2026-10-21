"""WHY: restoring fabric must preserve mapped weave, unaffected texels and existing bindings."""
import io
import json
import struct

import numpy as np
from PIL import Image
import pytest

from garmentimport.canvas_patch import restore_canvas, replace_embedded_pngs


def test_real_donor_and_untouched_pixels():
    """WHY: a random fill or a flat normal disc cannot replace actual woven fabric."""
    pixels = np.full((40, 40, 3), 80, dtype='uint8')
    pixels[2:10, 2:10] = np.indices((8, 8))[0][..., None]*20
    result = np.array(restore_canvas(Image.fromarray(pixels), (25, 25), (8, 8), (2, 2, 10, 10)))
    assert result[21:29, 21:29].std() > 10
    assert np.array_equal(result[:12], pixels[:12])
    assert set(result[22:28, 22:28, 0].ravel()) <= set(pixels[2:10, 2:10, 0].ravel())


def test_normal_feather_is_normalized():
    """WHY: feathered tangent normals need unit length without touching the rest of the atlas."""
    pixels = np.full((40, 40, 3), [128, 128, 255], dtype='uint8')
    pixels[2:10, 2:10] = [190, 128, 238]
    result = np.array(restore_canvas(Image.fromarray(pixels), (25, 25), (8, 8), (2, 2, 10, 10), normal=True))
    assert np.allclose(np.linalg.norm(result[21:29, 21:29]/127.5-1, axis=2), 1, atol=.01)
    assert np.array_equal(result[:12], pixels[:12])


def test_compact_preserves_geometry(tmp_path):
    """WHY: moving or rewriting mesh buffers can invalidate a shipped garment's bind files."""
    image = Image.new('RGB', (4, 4), 'red')
    encoded = io.BytesIO()
    image.save(encoded, format='PNG')
    blob = b'geometry1234'+encoded.getvalue()
    blob += b'\0'*(-len(blob)%4)
    doc = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': len(blob)}],
           'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': 12},
                           {'buffer': 0, 'byteOffset': 12, 'byteLength': len(encoded.getvalue())}],
           'accessors': [{'bufferView': 0, 'count': 3, 'componentType': 5125, 'type': 'SCALAR'}],
           'images': [{'bufferView': 1, 'mimeType': 'image/png'}]}
    js = json.dumps(doc).encode()
    js += b' '*(-len(js)%4)
    src, dst = tmp_path/'before.glb', tmp_path/'after.glb'
    src.write_bytes(struct.pack('<4sII', b'glTF', 2, 28+len(js)+len(blob))
                    +struct.pack('<II', len(js), 0x4E4F534A)+js
                    +struct.pack('<II', len(blob), 0x004E4942)+blob)
    replace_embedded_pngs(src, dst, {0: Image.new('RGB', (4, 4), 'blue')})
    raw = dst.read_bytes()
    length = struct.unpack_from('<I', raw, 12)[0]
    after = json.loads(raw[20:20+length])
    assert after['accessors'] == doc['accessors']
    assert after['bufferViews'][0] == doc['bufferViews'][0]
    assert raw[28+length:28+length+12] == b'geometry1234'
    view = after['bufferViews'][1]
    assert Image.open(io.BytesIO(raw[28+length+view['byteOffset']:])).getpixel((0, 0)) == (0, 0, 255)


def test_compact_drops_old_images_and_gaps_preserving_all_accessor_bytes(tmp_path):
    """WHY: repeated image repairs must not retain old PNGs or alter bound geometry when offsets move."""
    encoded = io.BytesIO()
    noise = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype='uint8')
    Image.fromarray(noise).save(encoded, format='PNG')
    old_png = encoded.getvalue()
    blob = bytearray(old_png)
    blob.extend(b'\0'*(-len(blob)%4))
    blob.extend(b'orphaned image bytes'*300)
    blob.extend(b'\0'*(-len(blob)%4))
    views = [{'buffer': 0, 'byteOffset': 0, 'byteLength': len(old_png)}]
    accessors, expected = [], []
    for components, kind, dtype in [(3, 'VEC3', '<f4'), (3, 'VEC3', '<f4'),
                                     (2, 'VEC2', '<f4'), (1, 'SCALAR', '<u4')]:
        # Include accessor offsets and strided records, not just contiguous arrays.
        payload = b'HEAD'+np.arange(components*3, dtype=dtype).tobytes()+b'TAIL'
        views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(payload),
                      'byteStride': components*4})
        accessors.append({'bufferView': len(views)-1, 'byteOffset': 4, 'count': 3,
                          'componentType': 5126 if dtype == '<f4' else 5125, 'type': kind})
        expected.append(payload)
        blob.extend(payload)
    doc = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': len(blob)}],
           'bufferViews': views, 'accessors': accessors,
           'images': [{'bufferView': 0, 'mimeType': 'image/png'}]}
    js = json.dumps(doc).encode()
    js += b' '*(-len(js)%4)
    src, dst = tmp_path/'before.glb', tmp_path/'after.glb'
    src.write_bytes(struct.pack('<4sII', b'glTF', 2, 28+len(js)+len(blob))
                    +struct.pack('<II', len(js), 0x4E4F534A)+js
                    +struct.pack('<II', len(blob), 0x004E4942)+blob)
    replace_embedded_pngs(src, dst, {0: Image.new('RGB', (4, 4), 'blue')})
    raw = dst.read_bytes()
    length = struct.unpack_from('<I', raw, 12)[0]
    after = json.loads(raw[20:20+length])
    compact_blob = raw[28+length:]
    assert dst.stat().st_size < len(old_png)
    assert after['accessors'] == accessors
    assert after['buffers'][0]['byteLength'] == len(compact_blob)
    for accessor, payload in zip(after['accessors'], expected):
        view = after['bufferViews'][accessor['bufferView']]
        assert view['byteOffset'] % 4 == 0
        assert compact_blob[view['byteOffset']:view['byteOffset']+view['byteLength']] == payload
    referenced_size = sum((v['byteLength']+3)//4*4 for v in after['bufferViews'])
    assert len(compact_blob) == referenced_size
    first_output = raw
    replace_embedded_pngs(dst, dst, {0: Image.new('RGB', (4, 4), 'blue')})
    assert dst.read_bytes() == first_output


def test_invalid_donor_rejected():
    """WHY: an out-of-bounds donor can copy an atlas seam instead of fabric."""
    with pytest.raises(ValueError, match='Donor'):
        restore_canvas(Image.new('RGB', (8, 8)), (4, 4), (2, 2), (7, 7, 10, 10))
