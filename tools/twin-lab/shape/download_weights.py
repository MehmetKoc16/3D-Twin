#!/usr/bin/env python
"""Download the open Hunyuan3D-2 SHAPE weights into tools/twin-lab/shape/weights/ (gitignored).

Only model WEIGHTS are downloaded (nothing is uploaded anywhere). The layout matches what `hy3dgen` expects when the
environment variable HY3DGEN_MODELS points at the `weights/` folder: weights/<hf repo id>/<subfolder>/model.fp16.safetensors

Presets (fp16 safetensors only, .ckpt duplicates are skipped):
  mini        tencent/Hunyuan3D-2mini  hunyuan3d-dit-v2-mini                      (~3.8 GB)  single image, 50 steps + CFG
  mini-turbo  tencent/Hunyuan3D-2mini  hunyuan3d-dit-v2-mini-turbo + turbo VAE    (~4.2 GB)  single image, 5-10 steps, FlashVDM
  mv          tencent/Hunyuan3D-2mv    hunyuan3d-dit-v2-mv                        (~4.9 GB)  multi view (front/left/back)
  mv-turbo    tencent/Hunyuan3D-2mv    hunyuan3d-dit-v2-mv-turbo + turbo VAE      (~5.3 GB)  multi view, FlashVDM

Usage:  python download_weights.py mini mini-turbo        (default: mini mini-turbo)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"

# preset -> list of (repo id, subfolder)
PRESETS: dict[str, list[tuple[str, str]]] = {
    "mini": [("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini")],
    "mini-turbo": [
        ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"),
        ("tencent/Hunyuan3D-2mini", "hunyuan3d-vae-v2-mini-turbo"),
    ],
    "mv": [("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv")],
    "mv-turbo": [
        ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-turbo"),
        # FlashVDM VAE for the mv models lives in the base repo (see hy3dgen enable_flashvdm)
        ("tencent/Hunyuan3D-2", "hunyuan3d-vae-v2-0-turbo"),
    ],
}


def download(preset: str) -> None:
    from huggingface_hub import snapshot_download

    for repo, sub in PRESETS[preset]:
        target = WEIGHTS / repo
        print(f"[{preset}] {repo}/{sub} -> {target / sub}", flush=True)
        snapshot_download(
            repo_id=repo,
            allow_patterns=[f"{sub}/config.yaml", f"{sub}/model.fp16.safetensors"],
            local_dir=str(target),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("presets", nargs="*", default=["mini", "mini-turbo"], help=f"any of: {', '.join(PRESETS)}")
    args = ap.parse_args()
    for p in args.presets:
        if p not in PRESETS:
            raise SystemExit(f"unknown preset {p!r}; choose from {', '.join(PRESETS)}")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for p in args.presets:
        download(p)
    print("done")


if __name__ == "__main__":
    main()
