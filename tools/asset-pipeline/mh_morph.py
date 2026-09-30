"""Full-mesh MakeHuman morphing (float64 reference): every vertex of `base.obj`, helper geometry included.

The shipped `morphs.bin` only carries the body render vertices and the joint points. Body-part proxies (eyes, eyelashes,
some hair) are authored against helper vertices (eyeballs, lash cards, the hair cap) that are NOT part of the runtime
index space, but the raw `.target` files move them together with the body. This module evaluates the same morph model
(`v = base + sum w_i * delta_i`, ADR 0005) on all 19158 MakeHuman vertices so that the pipeline can

  * reconstruct such a proxy on the neutral body or on any other body (the "truth" it is re-bound against), and
  * check bindings on extreme bodies.

Frame: meters, MakeHuman axes (+Y up, +Z front), NOT ground-offset (add `offset_y` to y yourself).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from config import DM_TO_M
from macro import MACRO_VARIABLES, tent_weights
from mh_obj import BaseMesh
from targets import Catalog, TargetPacker


class MhMorpher:
    def __init__(self, mesh: BaseMesh, packer: TargetPacker, catalog: Catalog):
        self.mesh = mesh
        self.packer = packer
        self.catalog = catalog
        self.base = mesh.verts * DM_TO_M  # (19158, 3) meters
        self.specs = {t.id: t for t in catalog.targets}
        self.modifiers = {m.id: m for m in catalog.modifiers}
        self._macro_vars = {m["id"]: m for m in MACRO_VARIABLES}
        self._delta: dict[str, np.ndarray] = {}

    def delta(self, target_id: str) -> np.ndarray:
        """(19158, 3) meters."""
        if target_id not in self._delta:
            self._delta[target_id] = self.packer.mh_delta(self.specs[target_id]) * DM_TO_M
        return self._delta[target_id]

    def target_weights(self, macros: Mapping[str, float] | None = None, mods: Mapping[str, float] | None = None) -> dict[str, float]:
        vals = {k: d["default"] for k, d in self._macro_vars.items()}
        vals.update(macros or {})
        tents = {k: tent_weights(self._macro_vars[k]["buckets"], v) for k, v in vals.items()}
        out: dict[str, float] = {}
        for spec in self.catalog.targets:
            if not spec.macro_conditions:
                continue
            w = 1.0
            for var, bucket in spec.macro_conditions:
                w *= tents[var][bucket]
            if w != 0.0:
                out[spec.id] = w
        for mid, v in (mods or {}).items():
            m = self.modifiers[mid]
            if v < 0 and m.decr:
                out[m.decr] = out.get(m.decr, 0.0) - v
            elif v > 0 and m.incr:
                out[m.incr] = out.get(m.incr, 0.0) + v
        return out

    def positions(self, macros: Mapping[str, float] | None = None, mods: Mapping[str, float] | None = None) -> np.ndarray:
        """(19158, 3) meters (not ground-offset)."""
        out = self.base.copy()
        for tid, w in self.target_weights(macros, mods).items():
            out += self.delta(tid) * w
        return out
