"""JSON writers with a stable key order (dicts are built in the intended order; no key sorting, no timestamps)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_NUM_ARRAY = re.compile(r'\[\s*(-?\d[^\[\]{}"]*?)\s*\]')


def dumps(obj: Any) -> str:
    """Pretty JSON (indent 2) with arrays of plain numbers kept on one line."""
    text = json.dumps(obj, indent=2, ensure_ascii=True)
    text = _NUM_ARRAY.sub(lambda m: "[" + re.sub(r"\s*,\s*", ", ", m.group(1).strip()) + "]", text)
    return text + "\n"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(dumps(obj), encoding="utf-8", newline="\n")


def r6(a: np.ndarray | list[float]) -> list[float]:
    return [round(float(x), 6) for x in a]


def build_manifest(
    *,
    render_count: int,
    joint_names: list[str],
    joint_positions: np.ndarray,
    macro_variables: list[dict],
    targets: list[dict],
    modifiers: list[dict],
    provenance: dict,
) -> dict:
    return {
        "version": 1,
        "unit": "m",
        "vertexCount": render_count + len(joint_names),
        "renderVertexCount": render_count,
        "jointPoints": [{"name": n, "position": r6(p)} for n, p in zip(joint_names, joint_positions)],
        "mesh": "base.glb",
        "morphs": "morphs.bin",
        "macroVariables": macro_variables,
        "targets": targets,
        "modifiers": modifiers,
        "license": {
            "assets": "CC0-1.0",
            "source": provenance["source"],
        },
        "provenance": provenance["details"],
    }


def target_entry(spec_id: str, group: str, byte_offset: int, count: int, conditions) -> dict:
    d: dict = {"id": spec_id, "group": group, "byteOffset": byte_offset, "count": count}
    if conditions:
        d["macroConditions"] = [{"variable": v, "bucket": b} for v, b in conditions]
    return d


def modifier_entry(m) -> dict:
    d: dict = {"id": m.id, "min": m.min, "max": m.max, "default": m.default}
    if m.decr:
        d["decrTarget"] = m.decr
    if m.incr:
        d["incrTarget"] = m.incr
    return d
