#!/usr/bin/env python
"""Photo(s) of one person -> clean, normalised 3D shape (mesh.glb / mesh.obj / meta.json) with Hunyuan3D-2 (shape only).

Pipeline:  background removal (rembg)  ->  Hunyuan3D-2mini / 2mv shape diffusion + VAE decode (FlashVDM optional)
           ->  mesh cleanup (weld, floaters, holes, normals, decimation)  ->  orient to the app convention
           ->  scale to --height-cm, feet on y = 0  ->  fit an orthographic camera per input view (silhouette)
           ->  export mesh.glb + mesh.obj + meta.json (+ previews).

Local only: nothing is uploaded, only the open model weights are downloaded (download_weights.py).
Output convention (same as the web app): metres, right handed, +Y up, character faces +Z (left hand = +X), feet at y=0.
See README.md for the exact meaning of the camera block in meta.json.

Examples
    python generate.py --front ../../../user-data/twin/front.png --out ../../../user-data/twin/out/shape
    python generate.py --front f.png --variant mini-turbo --octree-resolution 380 --name turbo_o380   # 5 steps, faster, less accurate
    python generate.py --front f.png --left l.png --back b.png --variant mv-turbo     # multi view
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
HY3D_SRC = HERE / ".cache" / "Hunyuan3D-2"
WEIGHTS = HERE / "weights"
DEFAULT_OUT_ROOT = REPO_ROOT / "user-data" / "twin" / "out" / "shape"

# Keep every downloaded model inside the (gitignored) weights folder.
os.environ.setdefault("HY3DGEN_MODELS", str(WEIGHTS))
os.environ.setdefault("U2NET_HOME", str(WEIGHTS / "rembg"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

VARIANTS: dict[str, dict] = {
    "mini": dict(repo="tencent/Hunyuan3D-2mini", sub="hunyuan3d-dit-v2-mini", mv=False, flashvdm=False, steps=50),
    "mini-turbo": dict(repo="tencent/Hunyuan3D-2mini", sub="hunyuan3d-dit-v2-mini-turbo", mv=False, flashvdm=True, steps=5),
    "mv": dict(repo="tencent/Hunyuan3D-2mv", sub="hunyuan3d-dit-v2-mv", mv=True, flashvdm=False, steps=50),
    "mv-turbo": dict(repo="tencent/Hunyuan3D-2mv", sub="hunyuan3d-dit-v2-mv-turbo", mv=True, flashvdm=True, steps=5),
}
VIEW_NAMES = ("front", "left", "back", "right")
OCTREE_LADDER = (512, 384, 320, 256, 192, 128)


def log(msg: str) -> None:
    print(f"[shape {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------------------------------
# privacy + GPU etiquette
# --------------------------------------------------------------------------------------------------------------------


def ensure_private_output(out: Path) -> None:
    """Outputs contain the user's likeness: refuse to write anywhere git could track them."""
    out = out.resolve()
    try:
        rel = out.relative_to(REPO_ROOT)
    except ValueError:
        return  # outside the repo
    parts = rel.parts
    if parts[:1] == ("user-data",) or (parts[:2] == ("tools", "twin-lab") and "outputs" in parts):
        return
    raise SystemExit(
        f"refusing to write user outputs to {rel}: inside the repo but not under user-data/ or a gitignored outputs/ "
        "folder (privacy rule). Use --out user-data/twin/out/shape/<name>."
    )


def _display_path(p: Path) -> str:
    """Repo-relative POSIX path when possible (keeps meta.json free of machine-specific absolute paths)."""
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.name


def gpu_used_mib() -> int:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True
    )
    return int(r.stdout.strip().splitlines()[0])


def wait_for_free_gpu(threshold_mib: int = 1024, poll_s: int = 30, timeout_s: int = 600) -> int:
    """Wait until total GPU memory in use (other processes; ours is not started yet) is <= threshold."""
    t0 = time.time()
    while True:
        used = gpu_used_mib()
        if used <= threshold_mib:
            return used
        waited = time.time() - t0
        if waited > timeout_s:
            raise SystemExit(f"GPU still busy after {timeout_s}s ({used} MiB in use by other processes); giving up (--force-gpu to ignore)")
        log(f"GPU busy ({used} MiB used by other processes), waiting {poll_s}s ({int(waited)}s so far)")
        time.sleep(poll_s)


class VramMonitor:
    """Samples device-wide used memory (all processes) in a thread; torch's own peak is read separately."""

    def __init__(self, interval: float = 0.2) -> None:
        import torch

        self.torch = torch
        self.interval = interval
        self.peak_used = 0
        self.baseline_used = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _used(self) -> int:
        free, total = self.torch.cuda.mem_get_info()
        return int((total - free) / 2**20)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_used = max(self.peak_used, self._used())
            self._stop.wait(self.interval)

    def reset_peak(self) -> None:
        self.peak_used = self._used()

    def start(self) -> None:
        self.baseline_used = self._used()
        self.peak_used = self.baseline_used
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)


# --------------------------------------------------------------------------------------------------------------------
# background removal
# --------------------------------------------------------------------------------------------------------------------


def remove_background(img, model_name: str, session_cache: dict):
    """RGB(A) PIL image -> (RGBA cutout with cleaned alpha, info). If the input already has a non-trivial alpha it is
    used as is (only cleaned)."""
    import cv2
    import numpy as np
    from PIL import Image

    rgba = img.convert("RGBA")
    a = np.asarray(rgba)[..., 3]
    has_alpha = (a < 250).mean() > 0.02
    if has_alpha:
        cut = rgba
        info = {"method": "input-alpha"}
    else:
        from rembg import new_session, remove

        if model_name not in session_cache:
            session_cache[model_name] = new_session(model_name)
        cut = remove(img.convert("RGB"), session=session_cache[model_name]).convert("RGBA")
        info = {"method": f"rembg:{model_name}"}
    arr = np.asarray(cut).copy()
    alpha = arr[..., 3]
    # clean: binarise faint alpha, keep the largest blob, fill pinholes (Hunyuan's cropper uses any alpha > 0 for the bbox)
    binary = (alpha > 127).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (lab == main).astype(np.uint8)
    inv = 1 - binary
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n2):  # holes fully enclosed by the person (not touching the border)
        x, y, w, h, area = stats2[i]
        if x > 0 and y > 0 and x + w < binary.shape[1] and y + h < binary.shape[0] and area < 400:
            binary[lab2 == i] = 1
    soft = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), 0.8)
    keep = cv2.dilate(binary, np.ones((5, 5), np.uint8)) > 0
    new_alpha = np.where(keep, np.maximum(alpha, (soft * 255).astype(np.uint8)), 0).astype(np.uint8)
    new_alpha[new_alpha < 8] = 0
    arr[..., 3] = new_alpha
    info["personAreaFraction"] = round(float(binary.mean()), 4)
    return Image.fromarray(arr, "RGBA"), binary, info


def model_input_crop(binary, border_ratio: float = 0.15, size: int = 512) -> dict:
    """Document what hy3dgen's ImageProcessorV2.recenter does to the cutout (bbox crop, square pad, 512 px)."""
    import numpy as np

    ys, xs = np.nonzero(binary)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    h, w = y1 - y0, x1 - x0
    side = max(binary.shape)
    desired = int(side * (1 - border_ratio))
    scale = desired / max(h, w)
    return {
        "personBBoxPx": [x0, y0, x1, y1],
        "note": "hy3dgen crops the alpha bbox, scales its longer side to (1 - borderRatio) of the padded square and "
        f"centres it, then resizes to {size} px. The model never sees the original framing.",
        "borderRatio": border_ratio,
        "cropScale": scale,
        "modelSizePx": size,
    }


# --------------------------------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------------------------------


def import_hy3d():
    """Import hy3dgen.shapegen from the cloned repo. pymeshlab (GPL, only used by hy3dgen's own post-processors that
    this tool does not call) is stubbed so it does not have to be installed."""
    if not HY3D_SRC.exists():
        raise SystemExit(f"{HY3D_SRC} missing: run  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 {HY3D_SRC}")
    sys.path.insert(0, str(HY3D_SRC))
    if "pymeshlab" not in sys.modules:
        stub = types.ModuleType("pymeshlab")
        stub.MeshSet = type("MeshSet", (), {})  # type: ignore[attr-defined]
        sys.modules["pymeshlab"] = stub
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    return Hunyuan3DDiTFlowMatchingPipeline


def load_pipeline(variant: str, flashvdm: bool, flashvdm_topk: str, offload: str):
    import torch

    cls = import_hy3d()
    cfg = VARIANTS[variant]
    lite = offload == "lite"
    # lite: everything is created on the CPU, only the diffusion transformer is moved to the GPU afterwards, so the
    # transient peak while loading is small too.
    pipe = cls.from_pretrained(
        cfg["repo"],
        subfolder=cfg["sub"],
        use_safetensors=True,
        variant="fp16",
        device="cpu" if lite else "cuda",
        dtype=torch.float16,
    )
    if flashvdm:
        pipe.enable_flashvdm(topk_mode=flashvdm_topk)
    if lite:
        _install_lite_offload(pipe)
    return pipe


def _install_lite_offload(pipe) -> None:
    """Keep only the diffusion transformer resident on the GPU; the DINOv2 conditioner and the VAE are moved in for
    their calls and back to the CPU afterwards (cuts the resident VRAM by roughly half). Expects a pipeline whose
    modules are still on the CPU."""
    import torch

    def sandwich(mod, methods, release_after):
        """Move `mod` to the GPU before any of `methods`, back to the CPU after the ones in `release_after`."""
        for name in methods:
            orig = getattr(mod, name)

            def make(orig=orig, name=name):
                def call(*a, **k):
                    mod.to("cuda")
                    try:
                        return orig(*a, **k)
                    finally:
                        if name in release_after:
                            mod.to("cpu")
                            torch.cuda.empty_cache()

                return call

            setattr(mod, name, make())

    sandwich(pipe.conditioner, ["forward", "unconditional_embedding"], {"forward", "unconditional_embedding"})
    sandwich(pipe.vae, ["forward", "latents2mesh"], {"latents2mesh"})
    pipe.device = torch.device("cuda")  # latents and inputs are created on the GPU
    pipe.model.to("cuda")
    torch.cuda.empty_cache()


def run_shape(pipe, images, cfg, steps, octree, num_chunks, seed, guidance):
    import torch

    kwargs = dict(
        image=images,
        num_inference_steps=steps,
        octree_resolution=octree,
        num_chunks=num_chunks,
        generator=torch.manual_seed(seed),
        output_type="trimesh",
    )
    if guidance is not None:
        kwargs["guidance_scale"] = guidance
    return pipe(**kwargs)[0]


# --------------------------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------------------------


def generate_raw(args, variant, cfg, cutouts, views, steps, timing, attempts, vram):
    """Load the model, run shape generation (with an OOM fallback ladder), record time and VRAM. Returns the raw mesh."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    if not args.force_gpu:
        used = wait_for_free_gpu()
        log(f"GPU free enough ({used} MiB used by others)")
    monitor = VramMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    pipe = load_pipeline(variant, args.use_flashvdm, args.flashvdm_topk, args.offload)
    torch.cuda.synchronize()
    timing["modelLoad"] = time.time() - t0
    vram["loadPhase"] = {
        "torchPeakAllocatedMiB": round(torch.cuda.max_memory_allocated() / 2**20),
        "deviceUsedPeakMiB": monitor.peak_used,
        "deviceUsedAfterLoadMiB": monitor._used(),
    }
    log(f"model {variant} loaded in {timing['modelLoad']:.1f}s, VRAM after load {monitor._used()} MiB (baseline {monitor.baseline_used}, load peak {monitor.peak_used})")
    torch.cuda.reset_peak_memory_stats()
    monitor.reset_peak()

    images = {n: cutouts[n] for n in views} if cfg["mv"] else cutouts["front"]
    octree, chunks = args.octree_resolution, args.num_chunks
    raw = None
    t0 = time.time()
    while True:
        try:
            log(f"generating: steps={steps} octree={octree} chunks={chunks} seed={args.seed}")
            raw = run_shape(pipe, images, cfg, steps, octree, chunks, args.seed, args.guidance)
            attempts.append({"octree": octree, "numChunks": chunks, "ok": True})
            break
        except torch.cuda.OutOfMemoryError as e:
            attempts.append({"octree": octree, "numChunks": chunks, "ok": False, "error": str(e).splitlines()[0]})
            log(f"OOM at octree={octree} chunks={chunks}")
            gc.collect()
            torch.cuda.empty_cache()
            if args.no_fallback:
                raise
            if chunks > 2000:
                chunks //= 2
            else:
                lower = [r for r in OCTREE_LADDER if r < octree]
                if not lower:
                    raise
                octree = lower[0]
                chunks = args.num_chunks
    torch.cuda.synchronize()
    timing["generation"] = time.time() - t0
    vram.update(
        {  # generation phase only (peak counters were reset after the model load)
            "torchPeakAllocatedMiB": round(torch.cuda.max_memory_allocated() / 2**20),
            "torchPeakReservedMiB": round(torch.cuda.max_memory_reserved() / 2**20),
            "deviceUsedPeakMiB": monitor.peak_used,
            "deviceUsedBaselineMiB": monitor.baseline_used,
            "deviceUsedPeakMinusBaselineMiB": monitor.peak_used - monitor.baseline_used,
            "gpuTotalMiB": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
        }
    )
    monitor.stop()
    log(f"generation {timing['generation']:.1f}s, raw mesh {len(raw.vertices)} v / {len(raw.faces)} f; VRAM {vram}")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for v in VIEW_NAMES:
        ap.add_argument(f"--{v}", type=Path, help=f"{v} view image" + (" (required)" if v == "front" else " (optional; needs a multi-view variant for the shape)"))
    ap.add_argument("--variant", default="auto", choices=["auto", *VARIANTS], help="auto: mini (single view) or mv-turbo (several views)")
    ap.add_argument("--octree-resolution", type=int, default=512, help="marching-cubes grid resolution (VRAM/time; 256..640, 512 = ~4.7 GB total VRAM on the 6 GB card)")
    ap.add_argument("--steps", type=int, default=None, help="diffusion steps (default: 5 for turbo, 50 otherwise)")
    ap.add_argument("--guidance", type=float, default=None, help="guidance scale (default: pipeline default 5.0)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--num-chunks", type=int, default=8000, help="VAE decode chunk size (lower = less VRAM)")
    ap.add_argument("--flashvdm", default="on", choices=["on", "off", "auto"], help="FlashVDM turbo VAE decoder (default on). It also works with the 50-step DiT and cuts the ~80 s volume decode to ~2 s with the same silhouette IoU; 'off' = the standard hierarchical decoder; 'auto' = only for the *-turbo variants")
    ap.add_argument("--flashvdm-topk", default="mean", choices=["mean", "merge"], help="FlashVDM adaptive kv selection mode (turbo only)")
    ap.add_argument("--offload", default="lite", choices=["none", "lite"], help="lite (default): DINOv2 conditioner + VAE live on the CPU between calls, ~1.5 GB less VRAM, ~6 %% slower; none: everything on the GPU (needs the whole 6 GB)")
    ap.add_argument("--no-fallback", action="store_true", help="do not retry with smaller chunks / octree resolution on OOM")
    ap.add_argument("--rembg-model", default="u2net_human_seg", help="rembg model (u2net_human_seg, u2net, isnet-general-use, birefnet-general, ...)")
    ap.add_argument("--height-cm", type=float, default=178.0, help="real body height; the mesh is scaled to this")
    ap.add_argument("--target-faces", type=int, default=80000, help="decimate the exported mesh to this many faces (0 = off)")
    ap.add_argument("--floater-ratio", type=float, default=0.02, help="drop components smaller than this fraction of the largest area")
    ap.add_argument("--smooth-iters", type=int, default=0, help="Taubin smoothing iterations on the high-res mesh (0 = off)")
    ap.add_argument("--source-up", choices=["auto", "+y"], default="auto", help="'+y' skips the orientation detection (assume Y up, +Z front)")
    ap.add_argument("--name", default=None, help="run name (output folder); default: <variant>_o<octree>_s<steps>")
    ap.add_argument("--out", type=Path, default=None, help=f"output folder (default: {DEFAULT_OUT_ROOT}/<name>)")
    ap.add_argument("--from-raw", type=Path, default=None, help="skip generation: post-process this raw mesh (raw.glb of an earlier run)")
    ap.add_argument("--no-previews", action="store_true")
    ap.add_argument("--force-gpu", action="store_true", help="do not wait for other GPU processes")
    args = ap.parse_args()

    if args.front is None:
        ap.error("--front is required")
    views = {n: getattr(args, n) for n in VIEW_NAMES if getattr(args, n) is not None}
    for n, p in views.items():
        if not p.exists():
            ap.error(f"--{n} {p} not found")
    variant = args.variant
    if variant == "auto":
        variant = "mv-turbo" if len(views) > 1 else "mini"
    cfg = VARIANTS[variant]
    if cfg["mv"] and len(views) < 2:
        ap.error(f"variant {variant} is multi-view and needs at least two views (front + left/back/right)")
    if not cfg["mv"] and len(views) > 1:
        log(f"variant {variant} is single-view: only --front is used for shape (other views are still used for camera fitting)")
    steps = args.steps or cfg["steps"]
    args.use_flashvdm = cfg["flashvdm"] if args.flashvdm == "auto" else args.flashvdm == "on"
    name = args.name or f"{variant}_o{args.octree_resolution}_s{steps}" + ("" if args.use_flashvdm else "_nofvdm")
    out = args.out or (DEFAULT_OUT_ROOT / name)
    ensure_private_output(out)
    out.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np
    from PIL import Image

    timing: dict[str, float] = {}
    t_total = time.time()

    # ---- 1. background removal ------------------------------------------------------------------------------------
    t0 = time.time()
    cutouts, masks, bg_info, crop_info = {}, {}, {}, {}
    sessions: dict = {}
    for n, p in views.items():
        img = Image.open(p)
        img.load()
        cut, binary, info = remove_background(img, args.rembg_model, sessions)
        cutouts[n], masks[n], bg_info[n] = cut, binary, info
        crop_info[n] = model_input_crop(binary)
        cut.save(out / f"cutout_{n}.png")
        cv2.imwrite(str(out / f"mask_{n}.png"), binary * 255)
        log(f"{n}: {img.size[0]}x{img.size[1]}, background removed with {info['method']} (person = {info['personAreaFraction']:.1%})")
    sessions.clear()
    gc.collect()
    timing["backgroundRemoval"] = time.time() - t0

    # ---- 2. shape generation --------------------------------------------------------------------------------------
    import torch

    octree, chunks = args.octree_resolution, args.num_chunks
    attempts: list[dict] = []
    vram: dict = {}
    if args.from_raw is not None:
        import trimesh

        raw = trimesh.load(str(args.from_raw), force="mesh", process=False)
        timing["generation"] = 0.0
        log(f"skipping generation, raw mesh from {args.from_raw}: {len(raw.vertices)} v / {len(raw.faces)} f")
    else:
        raw = generate_raw(args, variant, cfg, cutouts, views, steps, timing, attempts, vram)
        octree, chunks = attempts[-1]["octree"], attempts[-1]["numChunks"]
        raw.export(out / "raw.glb")

    # ---- 3. cleanup + normalisation + camera fit ------------------------------------------------------------------
    import meshops

    t0 = time.time()
    raw_v = np.asarray(raw.vertices, dtype=np.float64)
    if args.source_up == "auto":
        det = meshops.detect_frame(raw_v)
    else:
        det = {"upAxis": 1, "upSign": 1, "forwardAxis": 2, "forwardSign": 1, "confident": True, "cues": {"forced": True}}
    log(f"frame detection: {det}")
    if not det["confident"]:
        log("WARNING: orientation cues ambiguous, check the previews")

    mesh, hires, stats = meshops.cleanup(
        raw, floater_ratio=args.floater_ratio, smooth_iters=args.smooth_iters, target_faces=args.target_faces
    )
    height_m = args.height_cm / 100.0
    fv, norm = meshops.normalize(np.asarray(mesh.vertices), det, height_m)
    hv = np.asarray(hires.vertices) @ norm.rotation.T * norm.scale + norm.translation
    import trimesh

    def build(vv, ff):
        # winding is preserved because R is a proper rotation (det +1) and the scale is positive
        m = trimesh.Trimesh(vertices=vv, faces=np.asarray(ff), process=False)
        return m

    mesh_n = build(fv, mesh.faces)
    hires_n = build(hv, hires.faces)
    if not mesh_n.is_winding_consistent or mesh_n.volume < 0:
        trimesh.repair.fix_normals(mesh_n)
    _ = mesh_n.vertex_normals  # cache so the GLB carries normals
    _ = hires_n.vertex_normals
    timing["cleanup"] = time.time() - t0

    t0 = time.time()
    cameras = {}
    for n in views:
        try:
            cameras[n] = meshops.fit_orthographic_camera(
                np.asarray(mesh_n.vertices), np.asarray(mesh_n.faces), masks[n], meshops.VIEW_YAW_DEG[n]
            )
            c = cameras[n]
            cv2.imwrite(str(out / f"camera_{n}_overlay.png"), meshops.overlay_image(np.asarray(mesh_n.vertices), np.asarray(mesh_n.faces), c, masks[n]))
            log(
                f"camera {n}: {c['pxPerMeter']:.1f} px/m, origin {c['originPx'][0]:.1f},{c['originPx'][1]:.1f}px, "
                f"silhouette IoU {c['silhouetteIoU']} (initial {c['silhouetteIoUInitialGuess']}, mirrored mesh {c['silhouetteIoUMirroredMesh']}, "
                f"anisotropic {c['anisotropicFit']['silhouetteIoU']} at h/v {c['anisotropicFit']['horizontalOverVertical']})"
            )
        except Exception as e:  # noqa: BLE001
            cameras[n] = {"error": repr(e)}
            log(f"camera fit for {n} failed: {e!r}")
    timing["cameraFit"] = time.time() - t0

    # ---- 4. export ------------------------------------------------------------------------------------------------
    mesh_n.export(out / "mesh.glb")
    mesh_n.export(out / "mesh.obj", include_normals=True)
    if len(hires_n.faces) != len(mesh_n.faces):
        hires_n.export(out / "mesh_hires.glb")
    timing["total"] = time.time() - t_total

    fb = norm.final_bbox
    meta = {
        "schema": "twin-shape/1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "coordinateSystem": {
            "units": "meters",
            "handedness": "right",
            "up": "+Y",
            "forward": "+Z (the character faces +Z)",
            "characterLeft": "+X",
            "origin": "x,z = middle of the bbox of the lowest 3 % of the mesh (feet contact patch); y = 0 at the lowest vertex (soles)",
        },
        "inputs": {
            n: {
                "file": _display_path(views[n]),
                "backgroundRemoval": bg_info[n],
                "cutout": f"cutout_{n}.png",
                "mask": f"mask_{n}.png",
                "modelInput": crop_info[n],
            }
            for n in views
        },
        "model": {
            "variant": variant,
            "repo": cfg["repo"],
            "subfolder": cfg["sub"],
            "multiView": cfg["mv"],
            "viewsUsedForShape": list(views) if cfg["mv"] else ["front"],
            "steps": steps,
            "octreeResolutionRequested": args.octree_resolution,
            "octreeResolutionUsed": octree,
            "numChunksUsed": chunks,
            "seed": args.seed,
            "guidanceScale": args.guidance,
            "flashvdm": args.use_flashvdm,
            "flashvdmTopk": args.flashvdm_topk if args.use_flashvdm else None,
            "offload": args.offload,
            "dtype": "float16",
            "attempts": attempts,
        },
        "performance": {"seconds": {k: round(v, 2) for k, v in timing.items()}, "vram": vram},
        "normalization": {
            "sourceFrameDetection": det,
            "rotationSourceToStandard": norm.rotation.tolist(),
            "scaleMetersPerSourceUnit": norm.scale,
            "translationMeters": norm.translation.tolist(),
            "matrix4x4RowMajor": norm.matrix().tolist(),
            "formula": "p_final = (R @ p_source) * scale + translation   (R proper rotation, so triangle winding is unchanged)",
            "heightCm": args.height_cm,
            "heightNote": "total mesh height (soles to the highest hair point) is scaled to heightCm",
            "sourceBBox": norm.source_bbox,
            "finalBBox": {"min": fb["min"], "max": fb["max"], "sizeMeters": [b - a for a, b in zip(fb["min"], fb["max"], strict=True)]},
        },
        "cameras": {
            "assumption": "orthographic. World point (x,y,z) [final frame, metres] -> image pixel  u = originPx[0] + pxPerMeter * x',"
            "  v = originPx[1] - pxPerMeter * y,  with  x' = x*cos(yaw) - z*sin(yaw)  (yaw 0 front, 90 left side, 180 back, 270 right side;"
            " pixel coordinates in the ORIGINAL input image, v downwards). pxPerMeter / originPx were fitted by maximising the"
            " silhouette IoU against the cleaned person mask (mask_<view>.png). Real photos are perspective, so treat it as a first"
            " estimate and refine per pixel row if needed.",
            "views": cameras,
        },
        "mesh": {
            "faces": len(mesh_n.faces),
            "vertices": len(mesh_n.vertices),
            "watertight": bool(mesh_n.is_watertight),
            "hiresFaces": len(hires_n.faces),
            "cleanup": stats,
            "targetFaces": args.target_faces,
            "floaterRatio": args.floater_ratio,
            "smoothIters": args.smooth_iters,
        },
        "files": {
            "glb": "mesh.glb",
            "obj": "mesh.obj",
            "hiresGlb": "mesh_hires.glb" if len(hires_n.faces) != len(mesh_n.faces) else None,
            "previews": "previews/" if not args.no_previews else None,
        },
        "licenseNote": "Shape generated with Tencent Hunyuan3D-2 (Tencent Hunyuan 3D 2.0 Community License; not licensed in the EU, UK and South Korea).",
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"wrote {out / 'mesh.glb'} ({len(mesh_n.faces)} faces), mesh.obj, meta.json")

    if not args.no_previews:
        import preview

        files = preview.make_previews(out / "mesh.glb", out / "previews")
        log(f"previews: {out / 'previews'} ({len(files)} files)")
    log(f"done in {timing['total']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
