"""MHCLO (MakeHuman clothes proxy) + garment OBJ + MHMAT parsing, re-implemented from the plain-text formats.

MHCLO (as understood from the data files, nothing copied from MakeHuman / MPFB code):

    # comment lines (author / license live here, e.g. `# license CC0`)
    key value ...                       header (name, obj_file, material, basemesh, tag, uuid, z_depth, ...)
    x_scale <a> <b> <dist>              scale reference: |x[a] - x[b]| of the body / dist  (dist in decimeters)
    y_scale, z_scale                    same for y and z
    verts 0
     i1 i2 i3 w1 w2 w3 ox oy oz         one line per garment (OBJ) vertex: 3 body vertices, weights, offset (dm)
     i1                                 (short form) the garment vertex sits exactly on body vertex i1
    delete_verts
     0 - 10 15 20 - 30                  body vertices hidden under the garment (single ids and inclusive ranges)

Garment vertex position = w1*b[i1] + w2*b[i2] + w3*b[i3] + (ox*sx, oy*sy, oz*sz), with s = |b[a] - b[b]|_axis / dist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Mhclo:
    header: dict[str, str]
    comments: list[str]
    indices: np.ndarray  # (N, 3) int64, MakeHuman body vertex ids
    weights: np.ndarray  # (N, 3) float64
    offsets: np.ndarray  # (N, 3) float64, decimeters
    scale: dict[str, tuple[int, int, float]]  # axis -> (a, b, dist in decimeters)
    delete_verts: np.ndarray  # sorted unique int64
    tags: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return int(self.indices.shape[0])


def parse_mhclo(text: str) -> Mhclo:
    header: dict[str, str] = {}
    comments: list[str] = []
    tags: list[str] = []
    scale: dict[str, tuple[int, int, float]] = {}
    idx: list[tuple[int, int, int]] = []
    wts: list[tuple[float, float, float]] = []
    off: list[tuple[float, float, float]] = []
    deletes: list[int] = []
    mode = "header"
    block = ""  # last data block opened (`verts` / `delete`); some files put header keys (`material`) inside it
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line[1:].strip())
            continue
        tok = line.split()
        if mode in ("verts", "delete") and not (tok[0][0].isdigit() or tok[0] == "-"):
            mode = "header"
        elif mode == "header" and block and (tok[0][0].isdigit() or tok[0] == "-"):
            mode = block  # data lines resume after an interleaved header key (e.g. `verts 0` ... `material x` ... rows)
        if mode == "header":
            key = tok[0]
            if key == "verts":
                mode = block = "verts"
            elif key == "delete_verts":
                mode = block = "delete"
            elif key in ("x_scale", "y_scale", "z_scale"):
                scale[key[0]] = (int(tok[1]), int(tok[2]), float(tok[3]))
            elif key == "tag":
                tags.append(" ".join(tok[1:]))
            else:
                header.setdefault(key, " ".join(tok[1:]))
        elif mode == "verts":
            if len(tok) == 1:
                idx.append((int(tok[0]),) * 3)
                wts.append((1.0, 0.0, 0.0))
                off.append((0.0, 0.0, 0.0))
            elif len(tok) == 9:
                idx.append((int(tok[0]), int(tok[1]), int(tok[2])))
                wts.append((float(tok[3]), float(tok[4]), float(tok[5])))
                off.append((float(tok[6]), float(tok[7]), float(tok[8])))
            else:
                raise ValueError(f"bad MHCLO vertex line: {line!r}")
        else:  # delete
            i = 0
            while i < len(tok):
                if i + 2 < len(tok) and tok[i + 1] == "-":
                    deletes.extend(range(int(tok[i]), int(tok[i + 2]) + 1))
                    i += 3
                else:
                    deletes.append(int(tok[i]))
                    i += 1
    return Mhclo(
        header=header,
        comments=comments,
        indices=np.array(idx, dtype=np.int64).reshape(-1, 3),
        weights=np.array(wts, dtype=np.float64).reshape(-1, 3),
        offsets=np.array(off, dtype=np.float64).reshape(-1, 3),
        scale=scale,
        delete_verts=np.unique(np.array(deletes, dtype=np.int64)),
        tags=tags,
    )


_LICENSE_RE = re.compile(r"license\s*:?\s*(.*)", re.IGNORECASE)


def license_text(mhclo: Mhclo, mhmat_text: str = "") -> str:
    """The licence string the asset declares itself (`# license X` line, else a comment naming a CC licence)."""
    for c in mhclo.comments:
        m = _LICENSE_RE.match(c)
        if m and m.group(1).strip():
            return m.group(1).strip()
    for c in [*mhclo.comments, *[ln.lstrip("#/ ").strip() for ln in mhmat_text.splitlines()[:3]]]:
        if re.search(r"\bCC[- ]?(0|BY)", c, re.IGNORECASE):
            return c.strip()
    return ""


def classify_license(text: str) -> str | None:
    """'CC0-1.0' / 'CC-BY-4.0' (contract spelling; upstream does not pin a CC-BY version) or None if not acceptable."""
    t = re.sub(r"[\s_]", "", text.upper())
    if re.match(r"^CC-?0", t) or "CC0" in t or "PUBLICDOMAIN" in t:
        return "CC0-1.0"
    if re.match(r"^CC-?BY(?!-?(NC|ND|SA))", t) and "NC" not in t and "ND" not in t and "SA" not in t:
        return "CC-BY-4.0"
    return None


def author_of(mhclo: Mhclo) -> str:
    for c in mhclo.comments:
        m = re.match(r"author\s*:?\s*(.*)", c, re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------------------------------------------
# Garment OBJ
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class GarmentObj:
    verts: np.ndarray  # (V, 3) decimeters
    uvs: np.ndarray  # (T, 2) OBJ convention
    faces: list[list[tuple[int, int]]]  # per face: (vertex, uv) corners, 0-based


def parse_garment_obj(text: str) -> GarmentObj:
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[list[tuple[int, int]]] = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append((float(p[1]), float(p[2]), float(p[3])))
        elif p[0] == "vt":
            uvs.append((float(p[1]), float(p[2])))
        elif p[0] == "f":
            corners = []
            for tok in p[1:]:
                a = tok.split("/")
                vi = int(a[0])
                ti = int(a[1]) if len(a) > 1 and a[1] else 0
                corners.append((vi - 1 if vi > 0 else len(verts) + vi, ti - 1 if ti > 0 else len(uvs) + ti))
            faces.append(corners)
    return GarmentObj(np.array(verts, dtype=np.float64), np.array(uvs, dtype=np.float64).reshape(-1, 2), faces)


def render_split(obj: GarmentObj):
    """UV-seam split like the body: render vertices are unique (v, vt) pairs ordered by (v, vt).

    Returns (render_obj_vertex (R,), render_uv (R, 2), tris (T, 3) into render vertices, triangulated by ear-fan).
    Quads are split along the shorter diagonal, n-gons fanned; degenerate faces (< 3 corners) are dropped.
    """
    nvt = len(obj.uvs) + 1
    keys = sorted({c[0] * nvt + c[1] for f in obj.faces for c in f})
    key_index = {k: i for i, k in enumerate(keys)}
    karr = np.array(keys, dtype=np.int64)
    render_v = karr // nvt
    render_t = karr % nvt
    tris: list[tuple[int, int, int]] = []
    for f in obj.faces:
        ids = [key_index[c[0] * nvt + c[1]] for c in f]
        if len(ids) < 3:
            continue
        if len(ids) == 4:
            p = obj.verts[render_v[ids]]
            if np.linalg.norm(p[0] - p[2]) <= np.linalg.norm(p[1] - p[3]):
                tris += [(ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])]
            else:
                tris += [(ids[0], ids[1], ids[3]), (ids[1], ids[2], ids[3])]
        else:
            tris += [(ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids) - 1)]
    uv = obj.uvs[render_t] if len(obj.uvs) else np.zeros((len(karr), 2))
    return render_v, uv, np.array(tris, dtype=np.int64)


# ---------------------------------------------------------------------------------------------------------------
# MHMAT
# ---------------------------------------------------------------------------------------------------------------


def parse_mhmat(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue
        tok = line.split(None, 1)
        if len(tok) == 2 and tok[0] not in out:
            out[tok[0]] = tok[1].strip()
    return out


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------------------------------------------
# Reconstruction (float64 reference of the runtime formula, decimeters or any unit if refs/offsets share it)
# ---------------------------------------------------------------------------------------------------------------


def axis_scales(scale: dict[str, tuple[int, int, float]], body: np.ndarray, unit: float = 1.0) -> np.ndarray:
    """(sx, sy, sz) for body positions `body` (V, 3). `unit` converts the file's reference distances to body units."""
    s = np.ones(3)
    for k, ax in (("x", 0), ("y", 1), ("z", 2)):
        a, b, dist = scale[k]
        s[ax] = abs(body[a, ax] - body[b, ax]) / (dist * unit)
    return s


def reconstruct(indices: np.ndarray, weights: np.ndarray, offsets: np.ndarray, s: np.ndarray, body: np.ndarray) -> np.ndarray:
    return (weights[:, :, None] * body[indices]).sum(axis=1) + offsets * s
