"""Macro variable definitions (tent buckets) and a numpy reference implementation of the morph model.

Morph model (identical to the runtime contract):  v = base + sum_i w_i * delta_i
  - macro target weight = product of tent weights of its macroConditions
  - modifier value < 0 -> |value| * decrTarget ; value > 0 -> value * incrTarget

Tent semantics: buckets are sorted by `at`; a value between two buckets splits its weight linearly between them,
outside the range the nearest bucket gets weight 1, a single bucket always has weight 1. This reproduces
MakeHuman's macro formulas exactly:  min = max(0, 1-2t), max = max(0, 2t-1), average = 1-min-max.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from targets import ENTRY_DTYPE

# MakeHuman defaults: gender 0.5 (neutral), age 0.5 (25 years), muscle/weight/height 0.5, breast 0.5.
# Age is pinned to 25 years for the MVP: only the "young" bucket exists (old/child/baby targets are not shipped).
MACRO_VARIABLES: list[dict] = [
    {"id": "gender", "min": 0.0, "max": 1.0, "default": 0.5,
     "buckets": [{"name": "female", "at": 0.0}, {"name": "male", "at": 1.0}]},
    {"id": "age", "min": 0.5, "max": 0.5, "default": 0.5,
     "buckets": [{"name": "young", "at": 0.5}]},
    {"id": "muscle", "min": 0.0, "max": 1.0, "default": 0.5,
     "buckets": [{"name": "minmuscle", "at": 0.0}, {"name": "averagemuscle", "at": 0.5},
                 {"name": "maxmuscle", "at": 1.0}]},
    {"id": "weight", "min": 0.0, "max": 1.0, "default": 0.5,
     "buckets": [{"name": "minweight", "at": 0.0}, {"name": "averageweight", "at": 0.5},
                 {"name": "maxweight", "at": 1.0}]},
    {"id": "height", "min": 0.0, "max": 1.0, "default": 0.5,
     "buckets": [{"name": "minheight", "at": 0.0}, {"name": "averageheight", "at": 0.5},
                 {"name": "maxheight", "at": 1.0}]},
    {"id": "cupsize", "min": 0.0, "max": 1.0, "default": 0.5,
     "buckets": [{"name": "mincup", "at": 0.0}, {"name": "averagecup", "at": 0.5},
                 {"name": "maxcup", "at": 1.0}]},
    {"id": "firmness", "min": 0.0, "max": 1.0, "default": 0.5,
     "buckets": [{"name": "minfirmness", "at": 0.0}, {"name": "averagefirmness", "at": 0.5},
                 {"name": "maxfirmness", "at": 1.0}]},
]


def tent_weights(buckets: list[Mapping], value: float) -> dict[str, float]:
    bs = sorted(buckets, key=lambda b: b["at"])
    out = {b["name"]: 0.0 for b in bs}
    if len(bs) == 1 or value <= bs[0]["at"]:
        out[bs[0]["name"]] = 1.0
        return out
    if value >= bs[-1]["at"]:
        out[bs[-1]["name"]] = 1.0
        return out
    for lo, hi in zip(bs, bs[1:]):
        if lo["at"] <= value <= hi["at"]:
            t = (value - lo["at"]) / (hi["at"] - lo["at"])
            out[lo["name"]] = 1.0 - t
            out[hi["name"]] = t
            return out
    raise AssertionError("unreachable")


class MorphSet:
    """Reference evaluator over a manifest dict + morphs.bin bytes + combined base positions (V, 3) meters."""

    def __init__(self, manifest: dict, morphs: bytes, base: np.ndarray):
        self.manifest = manifest
        self.base = np.asarray(base, dtype=np.float64)
        self.entries = np.frombuffer(morphs, dtype=ENTRY_DTYPE)
        self.macro_defs = {m["id"]: m for m in manifest["macroVariables"]}
        self.targets = {t["id"]: t for t in manifest["targets"]}
        self.modifiers = {m["id"]: m for m in manifest["modifiers"]}
        self._runs: dict[str, np.ndarray] = {}

    def run(self, target_id: str) -> np.ndarray:
        if target_id not in self._runs:
            t = self.targets[target_id]
            s = t["byteOffset"] // ENTRY_DTYPE.itemsize
            self._runs[target_id] = self.entries[s : s + t["count"]]
        return self._runs[target_id]

    def macro_target_weights(self, macros: Mapping[str, float] | None = None) -> dict[str, float]:
        vals = {k: d["default"] for k, d in self.macro_defs.items()}
        vals.update(macros or {})
        tents = {k: tent_weights(self.macro_defs[k]["buckets"], v) for k, v in vals.items()}
        out: dict[str, float] = {}
        for tid, t in self.targets.items():
            conds = t.get("macroConditions")
            if not conds:
                continue
            w = 1.0
            for c in conds:
                w *= tents[c["variable"]][c["bucket"]]
            if w != 0.0:
                out[tid] = w
        return out

    def modifier_target_weights(self, mods: Mapping[str, float] | None = None) -> dict[str, float]:
        out: dict[str, float] = {}
        for mid, v in (mods or {}).items():
            m = self.modifiers[mid]
            if v < 0 and m.get("decrTarget"):
                out[m["decrTarget"]] = out.get(m["decrTarget"], 0.0) + (-v)
            elif v > 0 and m.get("incrTarget"):
                out[m["incrTarget"]] = out.get(m["incrTarget"], 0.0) + v
        return out

    def positions(
        self, macros: Mapping[str, float] | None = None, mods: Mapping[str, float] | None = None
    ) -> np.ndarray:
        weights = self.macro_target_weights(macros)
        for k, v in self.modifier_target_weights(mods).items():
            weights[k] = weights.get(k, 0.0) + v
        out = self.base.copy()
        for tid, w in weights.items():
            run = self.run(tid)
            out[run["i"]] += run["d"].astype(np.float64) * w
        return out
