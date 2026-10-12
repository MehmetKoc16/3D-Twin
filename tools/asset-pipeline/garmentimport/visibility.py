"""Surface diagnostics after static body deletion, including triangle interiors."""
from __future__ import annotations

import numpy as np

from .geometry import inside, nearest, surface


def triangle_pairs(points, faces, other_points=None, other_faces=None):
    """R-tree broad phase and triangle SAT; exclude welded adjacency for self queries.

    Coplanar overlaps are included. Shared welded vertices are topological adjacency,
    not self collisions. Diagnostics include contacts within 0.1 micrometers.
    """
    self_query = other_points is None
    other_points = points if self_query else other_points
    other_faces = faces if self_query else other_faces
    target = surface(other_points, other_faces)
    triangles = points[faces]
    if self_query:
        _, weld = np.unique(np.round(points, 7), axis=0, return_inverse=True)
        welded_faces = weld[faces]
    found = []
    for start in range(0, len(faces), 256):
        pairs = []
        for i in range(start, min(start+256, len(faces))):
            tri = triangles[i]
            ids = np.array(list(target.triangles_tree.intersection(
                np.r_[tri.min(axis=0)-1e-7, tri.max(axis=0)+1e-7])), int)
            if self_query:
                ids = ids[ids > i]
                adjacent = (welded_faces[ids, :, None] == welded_faces[i, None, :]).any(axis=(1, 2))
                ids = ids[~adjacent]
            pairs.extend((i, j) for j in ids)
        if not pairs:
            continue
        pairs = np.asarray(pairs, int)
        for chunk in np.array_split(pairs, max(1, (len(pairs)+4095)//4096)):
            a, b = triangles[chunk[:, 0]], target.triangles[chunk[:, 1]]
            ea, eb = np.roll(a, -1, axis=1)-a, np.roll(b, -1, axis=1)-b
            na, nb = np.cross(ea[:, 0], ea[:, 1]), np.cross(eb[:, 0], eb[:, 1])
            axes = [na, nb]
            axes += [np.cross(ea[:, i], eb[:, j]) for i in range(3) for j in range(3)]
            # These additional axes distinguish disjoint triangles in the same plane.
            axes += [np.cross(na, ea[:, i]) for i in range(3)]
            axes += [np.cross(nb, eb[:, i]) for i in range(3)]
            overlap = np.ones(len(chunk), bool)
            for axis in axes:
                norm = np.linalg.norm(axis, axis=1)
                axis = axis/np.maximum(norm[:, None], 1e-30)
                pa, pb = np.einsum('nij,nj->ni', a, axis), np.einsum('nij,nj->ni', b, axis)
                overlap &= (norm < 1e-15) | ((pa.max(axis=1) >= pb.min(axis=1)-1e-7) &
                                           (pb.max(axis=1) >= pa.min(axis=1)-1e-7))
            found.extend(chunk[overlap].tolist())
    return np.asarray(found, int).reshape(-1, 2)


def hidden_triangles(body_faces, count, delete):
    mask = np.zeros(count, bool)
    mask[np.asarray(delete, int)] = True
    return mask[body_faces].all(axis=1)


def contact_anatomy(body, result):
    """Identify remaining contacts by immutable body skin influences and neutral-rest bounds."""
    from .pose_safety import protected_vertices
    vertices = np.unique(body.faces[result['visibleTriangleIds']])
    if not len(vertices):
        return {'dominantBoneVertices': {}, 'protectedVertices': 0, 'neutralRestBoundsM': None}
    bones = body.joints[vertices, body.weights[vertices].argmax(axis=1)]
    names, counts = np.unique(np.asarray(body.names)[bones], return_counts=True)
    return {'dominantBoneVertices': dict(zip(names.tolist(), counts.tolist())),
            'protectedVertices': int(protected_vertices(body)[vertices].sum()),
            'neutralRestBoundsM': [body.positions[vertices].min(axis=0).tolist(),
                                  body.positions[vertices].max(axis=0).tolist()]}


def diagnostics(garment, garment_faces, body, body_faces, delete, regions=None):
    """Count visible intersecting/protruding body faces and local hidden containment.

    Outside samples are triangle centroids within 40 mm of an interior garment
    projection. Open garments have no globally meaningful inside/outside volume.
    The exact intersection pass supplements these bounded surface samples.
    """
    import trimesh
    hidden = hidden_triangles(body_faces, len(body), delete)
    body_mesh = surface(body, body_faces)
    garment_mesh = surface(garment, garment_faces)
    pairs = triangle_pairs(body, body_faces, garment, garment_faces)
    crossed = np.zeros(len(body_faces), bool)
    if len(pairs):
        crossed[pairs[:, 0]] = True
    centers = body[body_faces].mean(axis=1)
    complete_body = surface(body, body_faces, weld=True)
    contained = inside(complete_body, garment)
    _, _, near_body = nearest(body_mesh, garment)
    garment_centers = garment[garment_faces].mean(axis=1)
    center_inside = inside(complete_body, garment_centers)
    _, _, center_body = nearest(body_mesh, garment_centers)
    local_inside = np.zeros(len(body_faces), bool)
    local_inside[near_body[contained]] = True
    local_inside[center_body[center_inside]] = True
    nearby = ((centers >= garment.min(axis=0)-.04) & (centers <= garment.max(axis=0)+.04)).all(axis=1)
    sample_ids = np.flatnonzero(nearby)
    closest, gap, tid = nearest(garment_mesh, centers[sample_ids])
    normals = garment_mesh.face_normals[tid].copy()
    normals[(normals*body_mesh.face_normals[sample_ids]).sum(axis=1) < 0] *= -1
    bary = trimesh.triangles.points_to_barycentric(garment_mesh.triangles[tid], closest)
    outside = np.zeros(len(body_faces), bool)
    outside[sample_ids] = (gap < .04) & (bary.min(axis=1) > .015) & (((centers[sample_ids]-closest)*normals).sum(axis=1) > .0001)
    # Nearby bare hands/skin beyond openings are not poke-through. An exterior
    # centroid must belong to a body face next to an actually contained cloth sample.
    outside &= local_inside
    visible = (crossed | outside) & ~hidden
    self_pairs = triangle_pairs(garment, garment_faces)
    self_faces = np.unique(self_pairs) if len(self_pairs) else np.zeros(0, int)
    harmless = contained & hidden[near_body]
    result = {
        'visiblePokeThroughPercent': float(visible.mean()*100),
        'visiblePokeThroughTriangles': int(visible.sum()),
        'testedBodyTriangles': len(body_faces),
        'visibleIntersectionTriangles': int((crossed & ~hidden).sum()),
        'visibleOutsideCentroids': int((outside & ~hidden).sum()),
        'selfIntersectionPercent': float(len(self_faces)/len(garment_faces)*100),
        'selfIntersectionTriangles': len(self_faces), 'selfIntersectionPairs': len(self_pairs),
        'testedGarmentTriangles': len(garment_faces),
        'insideDeletedRegionPercent': float(harmless.mean()*100),
        'insideDeletedRegionVertices': int(harmless.sum()),
        'insideVisibleRegionVertices': int((contained & ~harmless).sum()),
        'hiddenBodyTriangles': int(hidden.sum()),
        'visibleTriangleIds': np.flatnonzero(visible).tolist(),
        'selfIntersectionTriangleIds': self_faces.tolist(),
    }
    if regions is not None:
        face_regions = regions[body_faces[:, 0]]
        result['regions'] = {str(r): int((visible & (face_regions == r)).sum()) for r in np.unique(regions)}
        garment_regions = regions[body_faces[near_body, 0]][garment_faces[:, 0]]
        result['selfIntersectionRegions'] = {str(r): int((garment_regions[self_faces] == r).sum())
                                              for r in np.unique(regions)}
    return result
