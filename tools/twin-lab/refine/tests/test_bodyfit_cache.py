import pickle

import numpy as np
from helpers import neutral_fit

from twinrefine import bodyfit


def _write(tmp_path, fit, with_sidecar_for=None):
    p = tmp_path / "fit.pkl"
    with open(p, "wb") as fh:
        pickle.dump(fit, fh)
    if with_sidecar_for is not None:
        (tmp_path / "fit.pkl.sha1").write_text(bodyfit.mesh_hash(with_sidecar_for), encoding="utf-8")
    return p


def test_cache_without_sidecar_needs_matching_extent(model, tmp_path):
    fit = neutral_fit(model)
    scan = fit.posed_vertices.copy()
    p = _write(tmp_path, fit)
    assert bodyfit._load_cache(p, scan) is not None  # same body extent: accepted
    assert bodyfit._load_cache(p, scan + np.array([0.0, 0.5, 0.0])) is None  # a different (shifted) scan: refit
    assert bodyfit._load_cache(p, scan * 1.3) is None


def test_cache_with_sidecar_is_exact(model, tmp_path):
    fit = neutral_fit(model)
    scan = fit.posed_vertices.copy()
    p = _write(tmp_path, fit, with_sidecar_for=scan)
    assert bodyfit._load_cache(p, scan) is not None
    moved = scan.copy()
    moved[0, 0] += 0.01  # one vertex moved by a centimetre: another mesh
    assert bodyfit._load_cache(p, moved) is None
