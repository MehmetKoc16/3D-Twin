"""MakeHuman .target parsing, the export catalogue and packing into the combined vertex index space.

Combined index space (contract v1.1): [render vertices of base.glb] followed by [joint points].
A MakeHuman body vertex delta is duplicated to every UV-split render copy; a joint point's delta is the mean of
the 8 deltas of its helper cube (exact, morphs are linear); all other helper vertices are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import DELTA_EPS_M, DM_TO_M, MH_DATA
from mh_obj import BaseMesh

TARGET_DIR = MH_DATA / "targets"
MH_VERTEX_COUNT = 19158

ENTRY_DTYPE = np.dtype([("i", "<u4"), ("d", "<f4", (3,))])
assert ENTRY_DTYPE.itemsize == 16

# ---------------------------------------------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------------------------------------------

MUSCLES = ("minmuscle", "averagemuscle", "maxmuscle")
WEIGHTS = ("minweight", "averageweight", "maxweight")
CUPS = ("mincup", "averagecup", "maxcup")
FIRMNESS = ("minfirmness", "averagefirmness", "maxfirmness")
GENDERS = ("female", "male")
RACES = ("caucasian", "asian", "african")


@dataclass
class TargetSpec:
    id: str
    group: str  # 'macro' | 'measure' | 'body' | 'face' | 'foot'
    files: list[tuple[Path, float]]  # (file, factor); deltas are summed
    macro_conditions: list[tuple[str, str]] | None = None


@dataclass
class ModifierSpec:
    id: str
    min: float
    max: float
    default: float
    decr: str | None
    incr: str | None


@dataclass
class Catalog:
    targets: list[TargetSpec] = field(default_factory=list)
    modifiers: list[ModifierSpec] = field(default_factory=list)


def _t(rel: str) -> Path:
    p = TARGET_DIR / (rel + ".target")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _macro_targets(cat: Catalog) -> None:
    for g in GENDERS:
        # Race pre-merged: neutral 1/3 mix of the three young race targets of this gender.
        cat.targets.append(
            TargetSpec(
                id=f"macrodetails/young-{g}",
                group="macro",
                files=[(_t(f"macrodetails/{r}-{g}-young"), 1.0 / 3.0) for r in RACES],
                macro_conditions=[("gender", g), ("age", "young")],
            )
        )
    for g in GENDERS:
        for mu in MUSCLES:
            for w in WEIGHTS:
                cond = [("gender", g), ("age", "young"), ("muscle", mu), ("weight", w)]
                cat.targets.append(
                    TargetSpec(
                        id=f"macrodetails/universal-{g}-young-{mu}-{w}",
                        group="macro",
                        files=[(_t(f"macrodetails/universal-{g}-young-{mu}-{w}"), 1.0)],
                        macro_conditions=cond,
                    )
                )
                for h in ("minheight", "maxheight"):
                    cat.targets.append(
                        TargetSpec(
                            id=f"macrodetails/height/{g}-young-{mu}-{w}-{h}",
                            group="macro",
                            files=[(_t(f"macrodetails/height/{g}-young-{mu}-{w}-{h}"), 1.0)],
                            macro_conditions=cond + [("height", h)],
                        )
                    )
    for mu in MUSCLES:
        for w in WEIGHTS:
            for cup in CUPS:
                for fm in FIRMNESS:
                    name = f"female-young-{mu}-{w}-{cup}-{fm}"
                    path = TARGET_DIR / "breast" / (name + ".target")
                    if not path.exists():  # average/average is intentionally absent upstream
                        continue
                    cat.targets.append(
                        TargetSpec(
                            id=f"breast/{name}",
                            group="macro",
                            files=[(path, 1.0)],
                            macro_conditions=[
                                ("gender", "female"),
                                ("age", "young"),
                                ("muscle", mu),
                                ("weight", w),
                                ("cupsize", cup),
                                ("firmness", fm),
                            ],
                        )
                    )


def _add_modifier(
    cat: Catalog,
    mod_id: str,
    group: str,
    decr: list[str] | None,
    incr: list[str] | None,
    target_id: str | None = None,
) -> None:
    """decr/incr are lists of target files (relative, no suffix); several files are pre-merged (l/r)."""
    base = target_id or mod_id
    dec_id = inc_id = None
    if decr:
        dec_id = base + "-decr" if incr else base + "-low"
        cat.targets.append(TargetSpec(dec_id, group, [(_t(f), 1.0) for f in decr]))
    if incr:
        inc_id = base + "-incr" if decr else base
        cat.targets.append(TargetSpec(inc_id, group, [(_t(f), 1.0) for f in incr]))
    cat.modifiers.append(
        ModifierSpec(mod_id, -1.0 if decr else 0.0, 1.0, 0.0, dec_id, inc_id)
    )


def _pair(folder: str, name: str, lo: str = "decr", hi: str = "incr") -> tuple[list[str], list[str]]:
    return [f"{folder}/{name}-{lo}"], [f"{folder}/{name}-{hi}"]


def _pair_lr(folder: str, name: str, lo: str = "decr", hi: str = "incr") -> tuple[list[str], list[str]]:
    return (
        [f"{folder}/l-{name}-{lo}", f"{folder}/r-{name}-{lo}"],
        [f"{folder}/l-{name}-{hi}", f"{folder}/r-{name}-{hi}"],
    )


MEASURE_NAMES = (
    "neck-circ",
    "neck-height",
    "upperarm-circ",
    "upperarm-length",
    "lowerarm-length",
    "wrist-circ",
    "frontchest-dist",
    "bust-circ",
    "underbust-circ",
    "waist-circ",
    "napetowaist-dist",
    "waisttohip-dist",
    "shoulder-dist",
    "hips-circ",
    "upperleg-height",
    "thigh-circ",
    "lowerleg-height",
    "calf-circ",
    "knee-circ",
    "ankle-circ",
)

HEAD_SHAPES = ("oval", "round", "rectangular", "square", "triangular", "invertedtriangular", "diamond")


def build_catalog() -> Catalog:
    cat = Catalog()
    _macro_targets(cat)

    # measure/* (single, already symmetric files)
    for n in MEASURE_NAMES:
        d, i = _pair("measure", f"measure-{n}")
        # target files are measure/measure-<n>-decr|incr ; ids keep the folder prefix
        _add_modifier(cat, f"measure/measure-{n}", "measure", d, i)

    # torso / hip / neck / stomach / buttocks / pelvis / breast (symmetric single files)
    for n in ("torso-scale-depth", "torso-scale-horiz", "torso-scale-vert", "torso-vshape",
              "torso-muscle-dorsi", "torso-muscle-pectoral"):
        d, i = _pair("torso", n)
        _add_modifier(cat, f"torso/{n}", "body", d, i)
    for n in ("hip-scale-depth", "hip-scale-horiz", "hip-scale-vert"):
        d, i = _pair("hip", n)
        _add_modifier(cat, f"hip/{n}", "body", d, i)
    d, i = _pair("hip", "hip-waist", "down", "up")
    _add_modifier(cat, "hip/hip-waist", "body", d, i)
    for n in ("neck-scale-depth", "neck-scale-horiz", "neck-scale-vert", "neck-double"):
        d, i = _pair("neck", n)
        _add_modifier(cat, f"neck/{n}", "body", d, i)
    for n in ("stomach-pregnant", "stomach-tone"):
        d, i = _pair("stomach", n)
        _add_modifier(cat, f"stomach/{n}", "body", d, i)
    d, i = _pair("buttocks", "buttocks-volume")
    _add_modifier(cat, "buttocks/buttocks-volume", "body", d, i)
    d, i = _pair("pelvis", "pelvis-tone")
    _add_modifier(cat, "pelvis/pelvis-tone", "body", d, i)
    for n, lo, hi in (("breast-dist", "decr", "incr"), ("breast-point", "decr", "incr"),
                      ("breast-trans", "down", "up"), ("breast-volume-vert", "down", "up")):
        d, i = _pair("breast", n, lo, hi)
        _add_modifier(cat, f"breast/{n}", "body", d, i)

    # armslegs: symmetric l/r targets pre-merged into one target.
    for limb in ("upperarm", "lowerarm", "upperleg", "lowerleg"):
        for n in ("scale-depth", "scale-horiz", "scale-vert", "fat", "muscle"):
            d, i = _pair_lr("armslegs", f"{limb}-{n}")
            _add_modifier(cat, f"armslegs/{limb}-{n}", "body", d, i)
    d, i = _pair_lr("armslegs", "upperarm-shoulder-muscle")
    _add_modifier(cat, "armslegs/upperarm-shoulder-muscle", "body", d, i)
    d, i = _pair_lr("armslegs", "hand-scale")
    _add_modifier(cat, "armslegs/hand-scale", "body", d, i)
    for n in ("upperlegs-height", "lowerlegs-height"):
        d, i = _pair("armslegs", n)
        _add_modifier(cat, f"armslegs/{n}", "body", d, i)
    for n in ("foot-scale", "foot-scale-depth", "foot-scale-horiz", "foot-scale-vert"):
        d, i = _pair_lr("armslegs", n)
        _add_modifier(cat, f"armslegs/{n}", "foot", d, i)

    # small face set
    for n in ("head-age", "head-fat", "head-scale-depth", "head-scale-horiz", "head-scale-vert"):
        d, i = _pair("head", n)
        _add_modifier(cat, f"head/{n}", "face", d, i)
    for s in HEAD_SHAPES:
        _add_modifier(cat, f"head/head-{s}", "face", None, [f"head/head-{s}"])
    d, i = _pair("forehead", "forehead-scale-vert")
    _add_modifier(cat, "forehead/forehead-scale-vert", "face", d, i)
    for n in ("chin-width", "chin-height", "chin-prominent", "chin-jaw-drop"):
        d, i = _pair("chin", n)
        _add_modifier(cat, f"chin/{n}", "face", d, i)
    for n in ("nose-scale-vert", "nose-scale-horiz", "nose-scale-depth", "nose-nostrils-width"):
        d, i = _pair("nose", n)
        _add_modifier(cat, f"nose/{n}", "face", d, i)
    for n in ("mouth-scale-horiz", "mouth-scale-vert", "mouth-scale-depth"):
        d, i = _pair("mouth", n)
        _add_modifier(cat, f"mouth/{n}", "face", d, i)
    d, i = _pair_lr("eyes", "eye-scale")
    _add_modifier(cat, "eyes/eye-scale", "face", d, i)
    d, i = _pair_lr("cheek", "cheek-volume")
    _add_modifier(cat, "cheek/cheek-volume", "face", d, i)
    return cat


# ---------------------------------------------------------------------------------------------------------------
# Parsing and packing
# ---------------------------------------------------------------------------------------------------------------


def parse_target(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertex ids, deltas in decimeters) of a .target file."""
    idx: list[int] = []
    val: list[tuple[float, float, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            p = line.split()
            if len(p) < 4:
                continue
            idx.append(int(p[0]))
            val.append((float(p[1]), float(p[2]), float(p[3])))
    return np.array(idx, dtype=np.int64), np.array(val, dtype=np.float64).reshape(-1, 3)


class TargetPacker:
    def __init__(self, mesh: BaseMesh, cube_vertices: np.ndarray):
        """cube_vertices: (J, 8) MakeHuman vertex ids of each joint point's helper cube."""
        self.mesh = mesh
        self.cubes = cube_vertices
        self.render_count = mesh.render_count
        self.joint_count = cube_vertices.shape[0]
        self._cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def _load(self, path: Path):
        if path not in self._cache:
            self._cache[path] = parse_target(path)
        return self._cache[path]

    def mh_delta(self, spec: TargetSpec) -> np.ndarray:
        """Summed MakeHuman-space delta, (19158, 3) decimeters."""
        acc = np.zeros((MH_VERTEX_COUNT, 3), dtype=np.float64)
        for path, factor in spec.files:
            idx, val = self._load(path)
            np.add.at(acc, idx, val * factor)
        return acc

    def combined_dense(self, mh_delta: np.ndarray) -> np.ndarray:
        """(R + J, 3) float64 meters in the combined index space."""
        out = np.empty((self.render_count + self.joint_count, 3), dtype=np.float64)
        out[: self.render_count] = mh_delta[self.mesh.render_mh]
        out[self.render_count :] = mh_delta[self.cubes].mean(axis=1)
        return out * DM_TO_M

    def pack(self, spec: TargetSpec) -> np.ndarray:
        dense = self.combined_dense(self.mh_delta(spec)).astype(np.float32)
        keep = np.abs(dense).max(axis=1) >= DELTA_EPS_M
        ids = np.nonzero(keep)[0]
        entries = np.empty(len(ids), dtype=ENTRY_DTYPE)
        entries["i"] = ids
        entries["d"] = dense[ids]
        return entries
