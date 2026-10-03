"""Fit -> texture -> align -> transplant -> single-material private GLB."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from headrecon.reshape import STABLE_LM
from headrecon.twinhead import detect_twin_landmarks
from PIL import Image
from twinrefine.scan import Scan, load_scan, welded_vertex_normals
from twintex.raster import rasterize_uv

from . import REPO
from .assets import landmarks, load_masks, read_mesh, region_mask
from .camera import load_cameras
from .geometry import align, edge_table, replace_face, symmetry_map
from .glb import save
from .texture import bake, pack_atlases, sample, uv_atlas


def private_path(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO / "user-data"):
        raise ValueError("FLAME outputs and previews must stay under user-data/")


def harmonize_seam(scan, welded_v, original_faces, removed, ring, flame_texture, flame_uv, flame_v, flame_ring):
    """Adjust only retained skin texels within 18mm of the stitch, never hair/body."""
    from scipy.spatial import cKDTree

    atlas = scan.atlas.copy()
    ids = np.flatnonzero(~removed)
    distance = cKDTree(welded_v[ring]).query(welded_v)[0]
    nearby = ids[(distance[original_faces[ids]] < 0.018).any(1)]
    if len(nearby) == 0:
        return atlas, {"modified_scan_texels": 0}
    # Surface-space raster mapping uses original UV seam indices.
    fid, bary = rasterize_uv(
        scan.uv * [atlas.shape[1], atlas.shape[0]], scan.faces[nearby], atlas.shape[1], atlas.shape[0]
    )
    y, x = np.nonzero(fid >= 0)
    tri = original_faces[nearby[fid[y, x]]]
    points = np.einsum("ij,ijk->ik", bary[y, x], welded_v[tri])
    dist, closest = cKDTree(flame_v[flame_ring]).query(points)
    target = sample(flame_texture, flame_uv[flame_ring[closest]] * [flame_texture.shape[1], flame_texture.shape[0]])
    source = atlas[y, x].astype(float)
    # Skin compatibility gate protects dark hair and clothes beside the facial border.
    skin = (source.mean(1) > 45) & (source.mean(1) < 235) & (source[:, 0] > source[:, 2] * 1.03)
    skin &= np.linalg.norm(source - target, axis=1) < 100
    overlap = skin & (dist < 0.018)
    gain = (
        np.clip(np.median(target[overlap], axis=0) / np.maximum(np.median(source[overlap], axis=0), 1), 0.75, 1.35)
        if overlap.any()
        else np.ones(3)
    )
    weight = np.clip(1 - dist / 0.018, 0, 1) * skin
    corrected = source * (1 + weight[:, None] * (gain - 1))
    atlas[y, x] = np.clip(np.rint(corrected), 0, 255).astype(np.uint8)
    return atlas, {"modified_scan_texels": int((weight > 0).sum()), "gain": gain.tolist()}


def split_uv(vertices, faces, corner_uv):
    """Keep topology welded while duplicating only glTF UV seams."""
    keys = np.column_stack((faces.ravel(), np.round(corner_uv.reshape(-1, 2), 8)))
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return vertices[faces.ravel()[first]], inverse.reshape(-1, 3), corner_uv.reshape(-1, 2)[first].astype(np.float32)


def extra_charts(atlas, faces, vertex_uv, tile_origin, tile_size):
    """Bake cap/bridge triangles separately so UV interpolation cannot cross atlases."""
    if len(faces) == 0:
        return np.empty((0, 3, 2), np.float32)
    side = int(np.ceil(np.sqrt(len(faces))))
    cell = tile_size / side
    if cell < 4:
        raise ValueError("Too many bridge triangles for padding tile")
    base = np.column_stack((np.arange(len(faces)) % side, np.arange(len(faces)) // side)) * cell
    corners = np.stack((base + 1, base + [cell - 1, 1], base + [1, cell - 1]), axis=1)
    fid, b = rasterize_uv(corners.reshape(-1, 2), np.arange(len(faces) * 3).reshape(-1, 3), tile_size, tile_size)
    y, x = np.nonzero(fid >= 0)
    colors = sample(atlas, vertex_uv[faces].reshape(-1, 2) * atlas.shape[0]).reshape(-1, 3, 3)
    tile = np.zeros((tile_size, tile_size, 3), np.uint8)
    tile[y, x] = np.clip(np.einsum("ij,ijk->ik", b[y, x], colors[fid[y, x]]), 1, 255).astype(np.uint8)
    from scipy import ndimage

    nearest = ndimage.distance_transform_edt(fid < 0, return_distances=False, return_indices=True)
    tile[fid < 0] = tile[nearest[0][fid < 0], nearest[1][fid < 0]]
    ox, oy = tile_origin
    atlas[oy : oy + tile_size, ox : ox + tile_size] = tile
    return ((corners + [ox, oy]) / atlas.shape[0]).astype(np.float32)


def run(
    inp: Path,
    fit: Path,
    photos_dir: Path,
    assets: Path,
    out: Path,
    *,
    include_ears=False,
    texture_size=2048,
    preview_dir=None,
    previews=True,
    detector=None,
):
    started = time.perf_counter()
    private_path(out)
    preview_dir = preview_dir or out.parent / "previews"
    if previews:
        private_path(preview_dir)
    scan = load_scan(inp)
    if scan.atlas is None:
        raise ValueError("A textured scan atlas is required")
    neutral_path = fit / "head_neutral.obj"
    if not neutral_path.exists():
        neutral_path = fit / "head_neutral.ply"
    neutral, faces = read_mesh(neutral_path)
    masks = load_masks(assets, len(neutral))
    region = region_mask(masks, len(neutral), include_ears)
    cameras = load_cameras(fit / "cameras.json")
    fitted, photos = {}, {}
    for name in ("front", "right"):
        v, f = read_mesh(fit / "fitted_views" / f"{name}.ply")
        if v.shape != neutral.shape or not np.array_equal(f, faces):
            raise ValueError(f"{name} posed mesh topology differs from neutral")
        fitted[name] = v
        with Image.open(photos_dir / f"{name}.jpg") as im:
            if im.size != cameras[name].original_size:
                raise ValueError(f"{name} original photo dimensions differ from exported camera")
            photos[name] = np.array(im.convert("RGB"))
    symmetry, sym_report = symmetry_map(neutral)
    print("[flame] generating UV atlas", flush=True)
    mapping, atlas_faces, uv, atlas_method = uv_atlas(neutral, faces, texture_size)
    if not np.array_equal(mapping[atlas_faces], faces):
        raise ValueError("Atlas changed face order; refusing incorrect barycentric transfer")
    texture, bake_report = bake(
        neutral, faces, mapping, atlas_faces, uv, fitted, cameras, photos, symmetry, masks, texture_size
    )
    print("[flame] texture baked; detecting scan render landmarks", flush=True)
    scan_lm = detect_twin_landmarks(
        scan.verts, scan.faces, welded_vertex_normals(scan.verts, scan.faces), scan.uv, scan.atlas, detector
    )
    if scan_lm is None:
        raise ValueError("MediaPipe found no scan face; alignment needs reliable landmarks")
    lm_ids, flame_lm = landmarks(assets, neutral, faces)
    stable = np.isin(lm_ids, STABLE_LM)
    aligned, alignment = align(
        neutral, masks["face"], scan.verts, scan.faces, flame_lm[stable], scan_lm[lm_ids[stable]]
    )
    print(f"[flame] scale={alignment['scale']:.5f}, landmark RMS={alignment['landmark_rms_mm']:.3f} mm", flush=True)
    # Exact welding preserves small existing features. A 1e-5m rounded weld
    # can merge genuinely distinct scan vertices and create a nonmanifold edge.
    _, first, inv = np.unique(scan.verts, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    scan_v, scan_f = scan.verts[first], inv[scan.faces]
    vertices, result_faces, removed, caps, scan_ring, flame_ring, stitch = replace_face(
        scan_v, scan_f, aligned, faces, region
    )
    # A representative UV per FLAME vertex suffices for seam colour sampling;
    # each actual inserted face retains its xatlas corner coordinates below.
    representative = np.zeros((len(neutral) + len(caps), 2), np.float32)
    representative[mapping] = uv
    for center, ring in caps:
        representative[center] = representative[ring].mean(0)
    scan_atlas, color_report = harmonize_seam(
        scan, scan_v, scan_f, removed, scan_ring, texture, representative, aligned, flame_ring
    )
    atlas, scan_uv, flame_uv = pack_atlases(scan_atlas, texture, scan.uv, representative)
    scan_corners = scan_uv[scan.faces[~removed]]
    selected = region[faces].all(1)
    offset = np.array([scan_atlas.shape[1] + 16, 0]) / atlas.shape[0]
    flame_corners = uv[atlas_faces[selected]] * texture_size / atlas.shape[0] + offset
    n_regular = len(scan_corners) + len(flame_corners)
    vertex_uv = np.vstack((scan_uv[first], flame_uv))
    tile_size = min(1024, texture_size, atlas.shape[0] - texture_size - 32)
    extras = extra_charts(
        atlas, result_faces[n_regular:], vertex_uv, (scan_atlas.shape[1] + 16, texture_size + 16), tile_size
    )
    corner_uv = np.concatenate((scan_corners, flame_corners, extras))
    out_v, out_f, out_uv = split_uv(vertices, result_faces, corner_uv)
    # Check nonzero area and winding consistency as well as edge multiplicity.
    tri = out_v[out_f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    stitch["degenerate_triangles"] = int((area < 1e-12).sum())
    stitch["degenerate_added_triangles"] = int((area[len(scan_corners) :] < 1e-12).sum())
    if stitch["degenerate_added_triangles"]:
        raise ValueError(f"Stitch produced {stitch['degenerate_added_triangles']} degenerate triangles")
    e, count = edge_table(result_faces)
    directed = np.concatenate((result_faces[:, [0, 1]], result_faces[:, [1, 2]], result_faces[:, [2, 0]]))
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    _, indices = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    winding = np.bincount(indices, weights=signs)
    stitch["inconsistent_winding_edges"] = int(((count == 2) & (winding != 0)).sum())
    if stitch["inconsistent_winding_edges"]:
        raise ValueError("Stitch introduced inconsistent face winding")
    provenance = (
        json.loads((fit / "provenance.json").read_text(encoding="utf-8")) if (fit / "provenance.json").exists() else {}
    )
    marker = {
        "version": 1,
        "method": "Pixel3DMM/FLAME local face transplant",
        "fitProvenance": provenance,
        "neutralSha256": hashlib.sha256(neutral_path.read_bytes()).hexdigest(),
        "alignment": alignment,
        "stitch": stitch,
        "includeEars": include_ears,
        "license": "FLAME/Pixel3DMM non-commercial local use",
    }
    result = Scan(out_v, out_f, out_uv, atlas, "image/png", scan.scene, scan.name)
    report = {
        "alignment": alignment,
        "stitch": stitch,
        "texture": {"atlas_method": atlas_method, **bake_report},
        "symmetry": sym_report,
        "seam_colour": color_report,
        "output": str(out),
        "output_vertices": len(out_v),
        "output_triangles": len(out_f),
        "atlas_size": list(atlas.shape[:2]),
    }
    save(out, result, inp, marker)
    if previews:
        from .previews import write_previews

        report["previews"] = write_previews(preview_dir, result, photos, cameras)
    report["seconds"] = round(time.perf_counter() - started, 2)
    (out.parent / "flame_head_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
