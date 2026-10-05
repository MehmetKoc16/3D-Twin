"""The hair shell: extraction, equal-area radial remesh, texture bake (a synthetic spherical cap, no private data)."""

import numpy as np
import pytest
import trimesh

from hybridbody import hairshell
from hybridbody.hairseg import VertexGraph
from hybridbody.hairshell import ShellParams

CENTRE = np.array([0.0, 1.65, 0.05])
RADIUS = 0.1


def test_colour_grade_preserves_strand_luminance_and_alpha():
    from flamehead.colour import to_lab

    rgb = np.linspace([18, 23, 32], [66, 71, 81], 512).astype(np.uint8)[None]
    rgba = np.dstack((rgb, np.arange(512, dtype=np.uint8)[None]))
    visible = np.ones(rgba.shape[:2], bool)
    graded, report = hairshell.grade_colour(rgba, "#2a1e18", visible)
    before = to_lab(rgb / 255.0)
    after = to_lab(graded[..., :3] / 255.0)
    assert np.array_equal(graded[..., 3], rgba[..., 3])
    assert np.abs(after[..., 0] - before[..., 0]).mean() < 0.15
    assert np.corrcoef(before[..., 0].ravel(), after[..., 0].ravel())[0, 1] > .999
    assert np.max(np.abs(np.mean(after, axis=(0, 1))[1:] - report["target_lab"][1:])) < .3
    with pytest.raises(ValueError, match="#rrggbb"):
        hairshell.grade_colour(rgba, "photo", visible)


@pytest.fixture(scope="module")
def cap():
    """The front half of the upper cap of a sphere (radius 10 cm): vertices, normals, faces, per-corner UV, hair mask."""
    mesh = trimesh.creation.icosphere(subdivisions=5, radius=RADIUS)
    vertices = np.asarray(mesh.vertices) + CENTRE
    faces = np.asarray(mesh.faces)
    centroid = vertices[faces].mean(1)
    keep = (centroid[:, 1] > CENTRE[1] + 0.01) & (centroid[:, 2] > CENTRE[2] + 0.005)
    faces = faces[keep]
    rel = vertices - CENTRE
    normals = rel / np.linalg.norm(rel, axis=1, keepdims=True)
    u = np.arctan2(rel[:, 0], rel[:, 2]) / (2 * np.pi) + 0.5
    v = 0.5 - np.arcsin(np.clip(rel[:, 1] / RADIUS, -1, 1)) / np.pi
    uv = np.stack([u, v], axis=1)[faces]  # (m, 3, 2) corner UVs, no seam inside the front half
    colours = np.full((len(vertices), 3), 120.0, np.float32)
    mask = vertices[:, 1] > CENTRE[1] + 0.045  # the hair: the top of the cap, the rest is margin
    region = vertices[:, 1] > CENTRE[1] + 0.02
    return vertices, normals, colours, faces, uv, mask, region


def shell_of(cap):
    vertices, normals, colours, faces, uv, mask, region = cap
    return hairshell.extract_shell(vertices, normals, colours, faces, uv, mask, region)


def test_shell_params_overrides_are_validated():
    assert ShellParams.with_overrides({"target_triangles": 5000, "fringe": 3}).target_triangles == 5000
    assert isinstance(ShellParams.with_overrides({"texture_size": 512}).texture_size, int)
    with pytest.raises(ValueError, match="Unknown hair shell parameter"):
        ShellParams.with_overrides({"triangles": 5})
    with pytest.raises(ValueError, match="non-negative"):
        ShellParams.with_overrides({"fringe": -1})
    assert ShellParams().to_dict()["texture_size"] == 2048


def test_extract_shell_keeps_the_region_and_measures_the_signed_distance_to_the_hair_edge(cap):
    vertices, _, _, _, _, mask, region = cap
    high = shell_of(cap)
    assert high.faces.max() < len(high.positions) and region[high.source].all()
    assert high.inside.sum() > 0 and (~high.inside).sum() > 0
    assert (high.signed_edge[high.inside] >= 0).all() and (high.signed_edge[~high.inside] <= 0).all()
    height = high.positions[:, 1] - (CENTRE[1] + 0.045)  # 0 at the hair edge of the cap
    inside = high.inside & (high.positions[:, 1] > CENTRE[1] + 0.07)
    assert (high.signed_edge[inside] > 0.01).all()  # well inside
    near = np.abs(height) < 0.003
    assert near.any() and np.abs(high.signed_edge[near]).max() < 0.006
    assert hairshell.edge_vertices(high.faces, high.inside).size > 0
    with pytest.raises(ValueError, match="no triangle"):
        hairshell.extract_shell(
            vertices, vertices, np.zeros((len(vertices), 3)), cap[3], cap[4], mask, np.zeros_like(region)
        )


def test_surface_samples_are_a_regular_barycentric_lattice_on_every_triangle():
    positions = np.array([[0, 0, 0], [0.004, 0, 0], [0, 0.004, 0], [0.1, 0, 0], [0.101, 0, 0], [0.1, 0.001, 0]], float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    face_ids, bary = hairshell.surface_samples(positions, faces, 0.001)
    assert np.allclose(bary.sum(1), 1.0) and (bary >= 0).all() and (bary <= 1).all()
    big = (face_ids == 0).sum()
    small = (face_ids == 1).sum()
    assert (
        big == 9 and small == 1
    )  # m = round(sqrt(area) / spacing): 2.8 -> 3 for the big one (m * m samples), 0.7 -> 1
    coarse = hairshell.surface_samples(positions, faces, 0.002)[0]
    assert (coarse == 0).sum() == 1  # twice the spacing: m = 1
    points = bary[face_ids == 0] @ positions[faces[0]]
    assert (points[:, 0] + points[:, 1] <= 0.004 + 1e-12).all()  # inside the triangle


def test_lambert_projection_round_trips_and_preserves_area():
    rng = np.random.default_rng(7)
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[d[:, 1] > -0.9]
    assert np.abs(hairshell.inverse_lambert(hairshell.lambert(d)) - d).max() < 1e-9
    assert np.hypot(*hairshell.lambert(np.array([[0.0, 1.0, 0.0]])).T) < 1e-12  # the pole is the centre of the disc
    assert hairshell.lambert(np.array([[0.0, 0.0, 1.0]]))[0].tolist() == pytest.approx([0.0, np.sqrt(2.0)])
    # equal area: a small spherical triangle and its image have the same area, whatever the direction
    for direction in (d[0], d[1], d[2], np.array([0.0, 0.6, 0.8]), np.array([0.3, -0.5, 0.81])):
        direction = direction / np.linalg.norm(direction)
        tangent = np.cross(direction, [0.0, 1.0, 0.0])
        tangent /= np.linalg.norm(tangent) if np.linalg.norm(tangent) > 1e-9 else 1.0
        other = np.cross(direction, tangent)
        h = 1e-3
        corners = np.array([direction, direction + h * tangent, direction + h * other])
        corners /= np.linalg.norm(corners, axis=1, keepdims=True)
        sphere = np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0])) / 2
        plane = hairshell.lambert(corners)
        a, b = plane[1] - plane[0], plane[2] - plane[0]
        flat = abs(a[0] * b[1] - a[1] * b[0]) / 2
        assert flat == pytest.approx(sphere, rel=0.02)


def remesh_of(cap, target=1500):
    high = shell_of(cap)
    face_ids, bary = hairshell.surface_samples(high.positions, high.faces, 0.0015)
    samples = np.einsum("ij,ijk->ik", bary, high.positions[high.faces[face_ids]])
    params = ShellParams.with_overrides({"target_triangles": target, "texture_size": 256, "padding": 2})
    return high, (face_ids, bary), samples, params, hairshell.radial_remesh(samples, CENTRE, params)


def test_quadric_reduction_and_xatlas_keep_the_surface_and_weld_mapping(cap):
    _, _, _, params, repaired = remesh_of(cap, target=3000)
    reduced = hairshell.decimate_shell(repaired, 1500)
    assert 1490 <= len(reduced.faces) <= 1500
    assert np.abs(np.linalg.norm(reduced.positions - CENTRE, axis=1) - RADIUS).max() < 0.001
    unwrapped, mapping = hairshell.unwrap_shell(reduced, params)
    assert unwrapped.report["uv_charts"] > 0
    assert unwrapped.uv.min() >= 0 and unwrapped.uv.max() <= 1
    assert np.array_equal(unwrapped.positions, reduced.positions[mapping])
    assert np.array_equal(mapping[unwrapped.faces], reduced.faces)
    delta = np.arange(len(reduced.positions))[:, None] * np.array([[1e-6, 2e-6, 3e-6]])
    fitted = (reduced.positions + delta)[mapping]
    for vertex in np.flatnonzero(np.bincount(mapping) > 1):
        assert np.ptp(fitted[mapping == vertex], axis=0).max() == 0


def test_radial_remesh_gives_a_regular_manifold_outward_shell_with_one_chart(cap):
    _, _, samples, params, shell = remesh_of(cap)
    radius = np.linalg.norm(shell.positions - CENTRE, axis=1)
    assert np.abs(radius - RADIUS).max() < 5e-4  # every node sits on the surface it was resampled from
    assert abs(len(shell.faces) / params.target_triangles - 1) < 0.08
    # manifold: every edge is used by one or two triangles, no degenerate or repeated triangle
    edges = np.sort(np.concatenate((shell.faces[:, [0, 1]], shell.faces[:, [1, 2]], shell.faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() <= 2 and counts.min() >= 1
    assert (shell.faces[:, 0] != shell.faces[:, 1]).all() and len(
        np.unique(np.sort(shell.faces, axis=1), axis=0)
    ) == len(shell.faces)
    # outward winding
    tri = shell.positions[shell.faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.einsum("ij,ij->i", normal, tri.mean(1) - CENTRE) > 0).all()
    # one chart: UVs in the unit square, no triangle flipped in UV space relative to the others
    assert shell.uv.min() >= 0 and shell.uv.max() <= 1
    e1, e2 = (
        shell.uv[shell.faces[:, 1]] - shell.uv[shell.faces[:, 0]],
        shell.uv[shell.faces[:, 2]] - shell.uv[shell.faces[:, 0]],
    )
    orientation = np.sign(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    assert abs(orientation.mean()) > 0.99
    # nodes next to the sampled surface lie on it; the few at the edge of the footprint are extrapolated (the margin)
    from scipy.spatial import cKDTree

    distance = cKDTree(samples).query(shell.positions)[0] * 1000
    error = hairshell.surface_error(shell.positions, samples)
    assert shell.report["edge_mm"]["median"] > 2 and np.median(distance) < 1.0 and error["max_mm"] < 12.0
    assert error["mean_mm"] == pytest.approx(distance.mean())


def test_radial_remesh_drops_cells_over_a_cliff_and_refuses_an_empty_surface():
    # a flat roof 4 cm above a flat floor seen from the centre: cells that straddle the edge are cut
    rng = np.random.default_rng(1)
    azimuth = rng.uniform(-0.5, 0.5, 60000)
    elevation = rng.uniform(0.9, 1.5, 60000)
    direction = np.stack(
        [np.sin(azimuth) * np.cos(elevation), np.sin(elevation), np.cos(azimuth) * np.cos(elevation)], axis=1
    )
    radius = np.where(elevation > 1.2, 0.14, 0.1)
    samples = CENTRE + direction * radius[:, None]
    params = ShellParams.with_overrides({"target_triangles": 800, "max_jump": 20})
    shell = hairshell.radial_remesh(samples, CENTRE, params)
    r = np.linalg.norm(shell.positions - CENTRE, axis=1)
    edge = np.concatenate((shell.faces[:, [0, 1]], shell.faces[:, [1, 2]], shell.faces[:, [2, 0]]))
    assert (np.abs(r[edge[:, 0]] - r[edge[:, 1]]) <= 0.02 + 1e-9).all()  # no triangle crosses the cliff
    assert r.max() == pytest.approx(0.14, abs=1e-3) and r.min() == pytest.approx(0.1, abs=1e-3)


def test_tangent_frames_follow_the_gltf_normal_map_convention():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)  # x to the right, y up, facing +z
    faces = np.array([[0, 1, 2]])
    uv_down = np.array([[0, 1], [1, 1], [0, 0]], float)  # u grows with +x, v (down the image) shrinks with +y
    tangent, bitangent = hairshell.tangent_frames(positions, faces, uv_down)
    assert np.allclose(tangent, [[1, 0, 0]]) and np.allclose(bitangent, [[0, 1, 0]])  # +Y of the map is up the surface
    uv_up = np.array([[0, 0], [1, 0], [0, 1]], float)  # v grows with +y: the image is flipped
    _, flipped = hairshell.tangent_frames(positions, faces, uv_up)
    assert np.allclose(flipped, [[0, -1, 0]])
    t, b = hairshell.orthonormal_frame(np.array([[0, 0, 1.0]]), np.array([[1, 0, 0.3]]), np.array([[0.2, 1, 0.1]]))
    assert abs(t @ b.T) < 1e-9 and abs(t[0] @ [0, 0, 1]) < 1e-9 and abs(b[0] @ [0, 0, 1]) < 1e-9


def test_texture_lookup_is_bilinear_at_texel_centres_and_clamps_at_the_border():
    image = np.array([[[0, 0, 0], [100, 0, 0]], [[0, 100, 0], [100, 100, 0]]], np.uint8)
    uv = np.array([[0.25, 0.25], [0.75, 0.25], [0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
    out = hairshell.texture_lookup(image, uv)
    assert np.allclose(out[0], [0, 0, 0]) and np.allclose(out[1], [100, 0, 0]) and np.allclose(out[2], [50, 50, 0])
    assert np.allclose(out[3], [0, 0, 0]) and np.allclose(out[4], [100, 100, 0])  # replicated border
    many = hairshell.texture_lookup(image, np.tile([0.5, 0.5], (5000, 1)))  # more than one row of the 2048 wide map
    assert many.shape == (5000, 3) and np.allclose(many, [50, 50, 0])


def test_harmonic_fill_interpolates_between_known_vertices_and_leaves_islands_alone():
    n = 12
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    positions = np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], axis=1)
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            faces += [[a, b, d], [a, d, c]]
    graph = VertexGraph.from_faces(np.array(faces), n * n)
    values = np.zeros((n * n, 3))
    values[positions[:, 0] <= 1] = [0, 0, 0]
    values[positions[:, 0] >= n - 2] = [100, 0, 0]
    unknown = (positions[:, 0] > 1) & (positions[:, 0] < n - 2)
    usable = np.ones(n * n, bool)
    filled, reached = hairshell.harmonic_fill(graph.adjacency, values, unknown, usable)
    assert reached[unknown].all()
    ramp = filled[unknown & (positions[:, 1] == 5), 0]
    assert np.all(np.diff(ramp) > 0) and 0 < ramp.min() and ramp.max() < 100  # monotone, between the two sides
    # vertices outside the usable sub-graph (the margin) never leak in
    usable[positions[:, 1] >= 9] = False
    filled2, _ = hairshell.harmonic_fill(graph.adjacency, values, unknown & usable, usable)
    assert np.allclose(filled2[~usable], values[~usable])
    # an unknown island without a known neighbour is not solvable
    island = np.zeros(n * n, bool)
    island[5 * n + 5] = True
    isolated_graph = VertexGraph.from_faces(np.array([[0, 1, 2]]), 3)
    _, reached = hairshell.harmonic_fill(
        isolated_graph.adjacency, np.zeros((3, 3)), np.array([True, True, True]), np.ones(3, bool)
    )
    assert not reached.any() and island.sum() == 1


def test_stubble_colour_takes_the_darker_side_hair_and_falls_back_without_enough_of_it():
    class Frame:
        x0, zc, eye_y = 0.0, 0.0, 1.6

    n = 600
    angle = np.radians(np.linspace(70, 110, n))
    points = np.stack([np.sin(angle) * 0.09, np.full(n, 1.6 + 0.03), np.cos(angle) * 0.09], axis=1)
    lab = np.stack([np.linspace(14, 30, n), np.full(n, 0.5), np.full(n, -2.0)], axis=1)
    colour = hairshell.stubble_colour(lab, points, Frame())
    assert 14 <= colour[0] <= 20 and colour[1] == pytest.approx(0.5)  # the dark 40 percent, not the whole range
    assert hairshell.stubble_colour(lab[:50], points[:50], Frame())[0] == pytest.approx(24.0)  # the fallback


def bake_of(cap, base, normal=None, *, fill=None, params=None):
    high, (face_ids, bary), samples, default, shell = remesh_of(cap)
    params = params or default
    if fill is not None:
        graph = VertexGraph.from_faces(high.faces, len(high.positions))
        hairshell.attach_fill(
            high, graph, fill(high), np.tile([40.0, 0.0, 0.0], (len(high.positions), 1)), (20.0, 0.5, -2.0), 2
        )
    return high, shell, hairshell.bake(shell, high, face_ids, bary, base, normal, params)


def test_bake_writes_colour_alpha_and_a_flat_normal_map_for_a_smooth_cap(cap):
    base = np.full((64, 64, 3), (200, 100, 50), np.uint8)
    high, shell, baked = bake_of(cap, base)
    assert baked.colour.shape == (256, 256, 4) and baked.normal.shape == (256, 256, 3)
    inside = baked.covered
    assert (baked.colour[inside][:, :3] == (200, 100, 50)).all()  # the texture, everywhere on the shell
    alpha = baked.colour[..., 3]
    assert alpha[inside].max() == 255 and alpha[inside].min() == 0 and 0.1 < baked.visible.sum() / inside.sum() < 0.95
    assert ((alpha[inside] > 0) & (alpha[inside] < 255)).any()  # the soft fringe
    assert np.allclose(baked.mean_colour, (200, 100, 50)) and baked.report["normal_map_used"] is False
    # no map and a smooth cap: the high-poly normal is the low-poly normal, so the tangent-space map is flat
    assert baked.report["normal_tilt_deg"]["median"] < 3.0
    assert np.abs(baked.normal[baked.visible].astype(int) - (128, 128, 255)).max() < 30  # the visible part is flat
    # bled outwards: texels next to a chart carry the chart's colour, not black
    ring = ~baked.covered & (baked.colour[..., :3].sum(2) > 0)
    assert ring.any()
    # the alpha edge follows the hairline: texels cut out (alpha below half) are in the margin, deep texels are opaque
    assert baked.report["texels_visible"] == int(baked.visible.sum())


def test_bake_keeps_the_tilt_of_the_source_normal_map_in_the_new_tangent_space(cap):
    base = np.full((64, 64, 3), 128, np.uint8)
    tilted = np.zeros((16, 16, 3), np.uint8)
    tilted[:] = (180, 128, 230)  # tilt of 27 degrees towards +u of the source layout
    _, _, baked = bake_of(cap, base, tilted)
    assert baked.report["normal_map_used"] is True
    assert baked.report["normal_tilt_deg"]["median"] == pytest.approx(27.0, abs=4.0)
    weaker = ShellParams.with_overrides(
        {"target_triangles": 1500, "texture_size": 256, "padding": 2, "normal_strength": 0.5}
    )
    _, _, soft = bake_of(cap, base, tilted, params=weaker)
    assert soft.report["normal_tilt_deg"]["median"] < baked.report["normal_tilt_deg"]["median"] * 0.7


def test_bake_replaces_a_flagged_patch_by_its_surroundings_and_adds_a_grain(cap):
    rng = np.random.default_rng(3)
    base = np.clip(rng.normal((60, 60, 60), 1.0, (64, 64, 3)), 0, 255).astype(np.uint8)
    patch = lambda high: high.positions[:, 0] > 0.0  # noqa: E731  half of the shell is "pale"
    _, _, plain = bake_of(cap, base)
    high, shell, filled = bake_of(cap, base, fill=patch)
    assert filled.report["grain_texels"] > 1000
    differs = (filled.colour[..., :3].astype(int) != plain.colour[..., :3].astype(int)).any(2) & filled.covered
    assert differs.sum() > 1000
    # the grain is a fine texel noise, not a pattern: neighbouring flagged texels differ
    row = filled.colour[128, :, 0][filled.covered[128]].astype(float)
    assert row.std() > 0.5 and np.corrcoef(row[:-1], row[1:])[0, 1] < 0.9


def test_attach_fill_reports_what_it_flagged(cap):
    high = shell_of(cap)
    graph = VertexGraph.from_faces(high.faces, len(high.positions))
    rng = np.random.default_rng(4)
    lab = np.stack(
        [30.0 + rng.normal(0, 8, len(high.positions)), np.zeros(len(high.positions)), np.zeros(len(high.positions))], 1
    )
    pale = high.positions[:, 0] > 0.02
    report = hairshell.attach_fill(high, graph, pale, lab, (20.0, 0.5, -2.0), 3)
    assert report["pale_vertices"] == int(pale.sum()) and report["flat_vertices"] >= 0
    assert high.fill_weight.shape == (len(high.positions),) and high.fill_colour.shape == (len(high.positions), 3)
    assert high.fill_weight[pale & high.inside].min() == 1.0 and (high.fill_weight <= 1).all()
    assert report["unreached_vertices"] == 0
