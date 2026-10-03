"""Read licensed local masks and barycentrics; never load generic_model.pkl."""

import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import trimesh


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False, maintain_order=True, force="mesh")
    v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    if not np.isfinite(v).all() or len(f) == 0 or f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"Invalid mesh: {path.name}")
    return v, f


def member(root: Path, archive: str, suffix: str) -> bytes:
    direct = root / suffix
    if direct.is_file():
        return direct.read_bytes()
    with zipfile.ZipFile(root / archive) as z:
        names = [n for n in z.namelist() if n.endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f"Expected one {suffix} in {archive}")
        return z.read(names[0])


class ArrayUnpickler(pickle.Unpickler):
    """Allow only the numpy array constructors needed by the official mask dictionary."""

    def find_class(self, module, name):
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name in {"_reconstruct", "scalar"}:
            return getattr(np._core.multiarray, name)
        if module == "numpy" and name in {"ndarray", "dtype"}:
            return getattr(np, name)
        if module in {"numpy.core.numeric", "numpy._core.numeric"} and name == "_frombuffer":
            return np._core.numeric._frombuffer
        if module in {"builtins", "__builtin__"} and name in {"set", "frozenset"}:
            return {"set": set, "frozenset": frozenset}[name]
        raise ValueError(f"Unexpected mask pickle class: {module}.{name}")


def load_masks(root: Path, count: int) -> dict[str, np.ndarray]:
    raw = ArrayUnpickler(io.BytesIO(member(root, "FLAME_masks.zip", "FLAME_masks.pkl")), encoding="latin1").load()
    masks = {k: np.asarray(list(v) if isinstance(v, set) else v, np.int64).ravel() for k, v in raw.items()}
    if any(len(v) and (v.min() < 0 or v.max() >= count) for v in masks.values()):
        raise ValueError("Mask topology does not match neutral mesh")
    return masks


def region_mask(masks: dict[str, np.ndarray], count: int, ears: bool) -> np.ndarray:
    selected = np.zeros(count, bool)
    keys = [
        "face",
        "nose",
        "lips",
        "eye_region",
        "left_eye_region",
        "right_eye_region",
        "left_eyeball",
        "right_eyeball",
    ]
    if ears:
        keys += ["left_ear", "right_ear"]
    for key in keys:
        selected[masks[key]] = True
    # Keep every protected region even if masks overlap at their margins.
    for key in ["scalp", "neck", "boundary"] + ([] if ears else ["left_ear", "right_ear"]):
        selected[masks[key]] = False
    return selected


def landmarks(root: Path, verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(
        io.BytesIO(member(root, "mediapipe_landmark_embedding.zip", "mediapipe_landmark_embedding.npz")),
        allow_pickle=False,
    ) as d:
        ids = d["landmark_indices"].astype(int)
        fi, b = d["lmk_face_idx"].astype(int), d["lmk_b_coords"].astype(float)
    if b.shape != (len(ids), 3) or fi.max() >= len(faces) or fi.min() < 0 or not np.allclose(b.sum(1), 1):
        raise ValueError("Invalid MediaPipe embedding")
    return ids, np.einsum("ij,ijk->ik", b, verts[faces[fi]])
