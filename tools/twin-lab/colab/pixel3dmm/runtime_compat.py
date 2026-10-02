"""Original T4-safe execution policy, shared by the worker and preprocessing."""

import os


def configure() -> None:
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import torch
    from timm.layers import set_fused_attn

    torch.set_default_dtype(torch.float32)
    torch.set_autocast_dtype("cuda", torch.float16)  # No T4-incompatible bf16 default.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_fused_attn(False)  # timm DINO uses its ordinary matmul/softmax attention.


if __name__ == "__main__":
    import runpy
    import sys
    from pathlib import Path

    configure()
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    import torch
    with torch.autocast(device_type="cuda", enabled=False):
        runpy.run_path(str(script), run_name="__main__")
