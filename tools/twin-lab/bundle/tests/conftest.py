"""Only synthetic inputs and the documented CC0 stand-in are used in this suite."""

import importlib.util
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bundle = load_module("write_twin_glb", LAB / "bundle/write_twin_glb.py")
runner = load_module("run_all", LAB / "run_all.py")
