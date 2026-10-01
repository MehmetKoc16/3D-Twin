import numpy as np
import trimesh
from helpers import flat_sheet, peanut

from twinrefine import meshops
from twinrefine.scan import weld_ids, welded_vertex_normals


def _closed(F: np.ndarray) -> bool:
    et = meshops.edge_table(F)
    return bool((et.count == 2).all())


def _volume(P, F):
    return trimesh.Trimesh(P, F, process=False).volume


def _xy_areas(mc):
    t = mc.P[mc.F][:, :, :2]
    return (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])


def test_corner_roundtrip_keeps_seams():
    # two quads sharing an edge but with different UVs along it: a UV seam
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]], float)
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 7], [4, 7, 5]])
    UV = np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1], [0.6, 0], [0.6, 1], [1, 0], [1, 1]], np.float32)
    mc = meshops.to_corners(V, F, UV)
    assert len(mc.P) == 6  # the seam vertices are welded
    v, f, uv = meshops.from_corners(mc)
    assert len(v) == 8  # ... and expanded again, one per distinct UV
    assert len(f) == 4
    inv, _ = weld_ids(v)
    assert inv.max() + 1 == 6


def test_refine_all_edges_keeps_closed_and_volume():
    m, _ = peanut(subdivisions=2)
    mc = meshops.to_corners(m.vertices, m.faces, np.zeros((len(m.vertices), 2)))
    et = meshops.edge_table(mc.F)
    out, is_new = meshops.refine_marked_edges(mc, np.ones(et.n_edges, bool), et)
    assert len(out.F) == 4 * len(mc.F)
    assert is_new.sum() == et.n_edges
    assert _closed(out.F)
    # midpoint subdivision does not move the surface, so the enclosed volume is unchanged
    assert abs(_volume(out.P, out.F) - _volume(mc.P, mc.F)) < 1e-9


def test_refine_partial_is_crack_free_and_uv_interpolated():
    m, _ = peanut(subdivisions=2)
    uv = np.stack([m.vertices[:, 0], m.vertices[:, 1]], axis=1).astype(np.float32)
    mc = meshops.to_corners(m.vertices, m.faces, uv)
    et = meshops.edge_table(mc.F)
    rng = np.random.default_rng(3)
    mark = rng.random(et.n_edges) < 0.3
    out, _ = meshops.refine_marked_edges(mc, mark, et)
    assert _closed(out.F)
    assert abs(_volume(out.P, out.F) - _volume(mc.P, mc.F)) < 1e-9
    # UV is a linear function of position here, so interpolated corner UVs must match it everywhere
    np.testing.assert_allclose(out.C[..., 0], out.P[out.F][..., 0], atol=1e-6)


def test_refine_with_split_fraction():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    F = np.array([[0, 1, 2]])
    mc = meshops.to_corners(V, F, V[:, :2].astype(np.float32))
    et = meshops.edge_table(mc.F)
    mark = np.zeros(et.n_edges, bool)
    mark[0] = True
    t = np.full(et.n_edges, 0.5)
    t[0] = 0.25
    out, is_new = meshops.refine_marked_edges(mc, mark, et, t)
    new = out.P[is_new][0]
    lo, hi = meshops.edge_endpoints(et)
    expected = mc.P[lo[0]] + 0.25 * (mc.P[hi[0]] - mc.P[lo[0]])
    np.testing.assert_allclose(new, expected, atol=1e-12)
    np.testing.assert_allclose(out.C[..., 0], out.P[out.F][..., 0], atol=1e-6)


def _pinch_cut(mc, x_pinch, upper_only=False):
    side = mc.P[:, 0] > x_pinch
    et = meshops.edge_table(mc.F)
    h = np.flatnonzero(et.he_pair >= 0)
    fa = side[mc.F].sum(axis=1) >= 2
    diff = fa[h // 3] != fa[et.he_pair[h] // 3]
    if upper_only:
        mid = 0.5 * (mc.P[et.he_a[h], 1] + mc.P[et.he_b[h], 1])
        diff &= mid > 0.0
    cut = np.zeros(et.n_edges, bool)
    cut[et.edge_of[h[diff]]] = True
    return et, cut


def test_cut_closed_loop_separates_two_components():
    m, x_pinch = peanut(subdivisions=3)
    mc = meshops.to_corners(m.vertices, m.faces, np.zeros((len(m.vertices), 2)))
    et, cut = _pinch_cut(mc, x_pinch)
    out, src = meshops.cut_along_edges(mc, cut, et)
    assert len(out.P) > len(mc.P)
    assert (src < len(mc.P)).all()
    n, _lab = meshops.face_components(out.F, np.ones(len(out.F), bool))
    assert n == 2
    # two open surfaces, each with one boundary loop made of the cut edges
    assert len(meshops.boundary_halfedges(out.F)) == 2 * int(cut.sum())


def test_cut_open_arc_leaves_surface_connected():
    m, x_pinch = peanut(subdivisions=3)
    mc = meshops.to_corners(m.vertices, m.faces, np.zeros((len(m.vertices), 2)))
    et, cut = _pinch_cut(mc, x_pinch, upper_only=True)  # only half of the pinch ring: an open arc
    out, _ = meshops.cut_along_edges(mc, cut, et)
    n, _lab = meshops.face_components(out.F, np.ones(len(out.F), bool))
    assert n == 1
    assert len(meshops.boundary_halfedges(out.F)) == 2 * int(cut.sum())


def test_ear_clip_area_and_orientation():
    poly = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)  # concave "L", area 3
    for p in (poly, poly[::-1]):
        tris = meshops.ear_clip(p)
        assert tris is not None and len(tris) == len(p) - 2
        area = 0.0
        for a, b, c in tris:
            area += ((p[b, 0] - p[a, 0]) * (p[c, 1] - p[a, 1]) - (p[c, 0] - p[a, 0]) * (p[b, 1] - p[a, 1])) / 2
        sign = 1.0 if meshops._area2(p) > 0 else -1.0
        assert abs(area - sign * 3.0) < 1e-12  # every triangle keeps the polygon's orientation, areas add up


def test_triangulate_loop_3d_planar_ring():
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    loop = np.stack([0.3 * np.cos(ang), 0.1 * np.sin(ang), 0.05 * np.cos(ang)], axis=1)  # tilted ellipse
    tris, extra = meshops.triangulate_loop_3d(loop)
    assert extra is None
    assert len(tris) == len(loop) - 2


def test_boundary_chain_closed_rim():
    V, F, _ = flat_sheet(6, 6)
    mc = meshops.to_corners(V, F, np.zeros((len(V), 2)))
    chains = meshops.chain_halfedges(mc.F, meshops.boundary_halfedges(mc.F))
    assert len(chains) == 1 and chains[0][1] is True
    assert len(chains[0][0]) == 4 * 5  # the rim of a 6 x 6 vertex grid


def test_flip_delaunay_keeps_orientation_and_area():
    V, F, UV = flat_sheet(8, 8, size=0.07)
    rng = np.random.default_rng(1)
    V = V.copy()
    V[:, :2] += rng.normal(0, 0.0015, (len(V), 2))
    V[:, 2] = 0.1  # flat, so the dihedral test lets every flip through
    mc = meshops.to_corners(V, F, UV.astype(np.float32))
    out, n = meshops.flip_delaunay_xy(mc)
    assert n > 0
    a0, a1 = _xy_areas(mc), _xy_areas(out)
    assert (a1 > 0).all()  # no inverted triangle
    assert abs(a0.sum() - a1.sum()) < 1e-12  # the flips re-triangulate the same region

    def min_angle(areas, m):
        t = m.P[m.F][:, :, :2]
        l2 = [((t[:, (i + 1) % 3] - t[:, i]) ** 2).sum(1) for i in range(3)]
        # smallest angle opposite to the shortest edge: sin(a) = 2 * area / (l_b * l_c)
        sa = [2 * areas / np.sqrt(l2[(i + 1) % 3] * l2[(i + 2) % 3]) for i in range(3)]
        return np.degrees(np.arcsin(np.clip(np.min(sa, axis=0), 0, 1))).min()

    assert min_angle(a1, out) >= min_angle(a0, mc) - 1e-9


def test_flip_skips_uv_seams():
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    mc = meshops.to_corners(V, F, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
    mc.C[1] += 0.25  # the two faces disagree about the UV of the shared diagonal
    out, n = meshops.flip_delaunay_xy(mc)
    assert n == 0
    np.testing.assert_array_equal(out.F, mc.F)


def test_welded_normals_point_outward():
    m, _ = peanut(subdivisions=2)
    n = welded_vertex_normals(m.vertices, m.faces)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9)
    assert (np.einsum("ij,ij->i", n, m.vertices - np.array([0.05, 0, 0])) > 0).all()


def test_zipper_triangulates_a_u_shaped_lens_with_short_rungs():
    # lens between x = 0 and x = 0.04: down the left crease, round the bottom tip, up the right crease
    ys = np.linspace(1.30, 1.18, 13)
    left = np.stack([np.zeros(13), ys, np.zeros(13)], axis=1)
    right = np.stack([0.04 * np.sin(np.linspace(0, np.pi, 11)[::-1]) * 0 + 0.04 * np.sin(np.linspace(0.3, np.pi - 0.3, 11)), np.linspace(1.18, 1.30, 11), np.zeros(11)], axis=1)
    P = np.vstack([left, right[1:]])  # the bottom vertex is shared: right[0] ~ left[-1]
    P[12] = [0.0, 1.18, 0.0]
    path = list(range(len(P)))
    tris = meshops.zipper_triangulate(path, P)
    assert tris is not None and len(tris) >= len(P) - 3
    t = P[tris]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    assert (np.abs(n[:, 2]) > 0).all()
    assert (np.sign(n[:, 2]) == np.sign(n[0, 2])).all()  # one consistent orientation
    # every triangle edge is short compared with the lens length: a ladder, not fan wedges
    edges = np.linalg.norm(np.concatenate([t[:, 1] - t[:, 0], t[:, 2] - t[:, 1], t[:, 0] - t[:, 2]]), axis=1)
    assert edges.max() < 0.06
    # a path that is not U shaped (monotone) is refused
    mono = np.stack([np.zeros(10), np.linspace(1.3, 1.2, 10), np.zeros(10)], axis=1)
    assert meshops.zipper_triangulate(list(range(10)), mono) is None
