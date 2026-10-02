"""Show bounded stack traces, never raw upstream image/array/debug output."""

from collections import deque
from pathlib import Path
import json
import re


def traceback_tail(path: Path, limit: int = 120) -> list:
    if not path.is_file():
        return []
    # Upstream prints detections, arrays and progress bars. Never dump raw logs.
    with path.open(encoding="utf-8", errors="replace") as stream:
        lines = deque(stream, maxlen=8000)
    result = []
    for raw in lines:
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw).strip()
        if line.startswith(("Traceback (", "During handling of the above exception", "The above exception")):
            result.append(line)
        elif re.fullmatch(r'File "[^"\n]+", line \d+(?:, in [\w<>.]+)?', line):
            result.append("  " + line)
        elif re.match(r"^(?:[\w.]+(?:Error|Exception)|KeyboardInterrupt|SystemExit|AssertionError):", line):
            kind, _, message = line.partition(":")
            # Do not print array/tensor/bytes reprs, long payloads, numeric dumps,
            # credentials, or arbitrary strings embedded in an exception message.
            unsafe = (len(message) > 600 or bool(re.search(
                r"array\(|tensor\(|\[|\{|data:image|base64|\\x[0-9a-fA-F]{2}|"
                r"(?i:password|credential|token\s*[=:])|(?:\d[., ]+){8}", message)))
            if unsafe:
                message = " [message payload omitted]"
            else:
                message = re.sub(r"(['\"]).*?\1", "<quoted value>", message)
            result.append(kind + ":" + message)
    return result[-limit:]


def step_context(root: Path, step: str, view: str = "all", log_name=None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "current_step.json").write_text(json.dumps({"step": step, "view": view, "log": log_name}), encoding="utf-8")


def install_log_tail(path: Path, limit: int = 60) -> list:
    """Pre-upload installation logs only. Never call this for photo processing."""
    if not path.is_file():
        return []
    result = deque(maxlen=limit)
    # newline='' preserves CR progress updates rather than treating them as lines.
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        pending = ""
        for chunk in iter(lambda: stream.read(8192), ""):
            pending += chunk
            parts = pending.split("\n")
            pending = parts.pop()
            for raw in parts:
                result.append(clean_install_line(raw))
        if pending:
            result.append(clean_install_line(pending))
    return list(result)


def clean_install_line(raw: str) -> str:
    line = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", raw)
    line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
    # Strip a trailing CR from CRLF; last remaining CR starts final progress state.
    line = line.rstrip("\r").rsplit("\r", 1)[-1]
    if re.search(r"(?i)token|password|credential|authorization|bearer|secret|api[_-]?key|hf_[A-Za-z0-9]+", line):
        return "[credential-related line redacted]"
    # URLs can carry HTTP basic credentials without naming the field.
    line = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", line)
    return line[:300]


def print_diagnostics(root: Path, limit: int = 120, allow_install_logs: bool = False) -> None:
    info = {}
    context = root / "current_step.json"
    if context.is_file():
        info = json.loads(context.read_text(encoding="utf-8"))
        # Values come from our labels, but still exclude arbitrary user input.
        step = re.sub(r"[^a-zA-Z0-9_.: -]", "?", str(info.get("step", "unknown")))[:80]
        view = info.get("view", "all")
        if view not in {"all", "front", "left", "right", "back"}:
            view = "all"
        print(f"Failed step: {step}; view: {view}")
    traces = []
    logs = sorted((root / "logs").glob("*.log"), key=lambda path: path.stat().st_mtime)
    # Per-step logs contain the underlying exception, worker.log its calling stack.
    for path in logs:
        tail = traceback_tail(path, limit)
        if tail:
            traces.extend(["Traceback log: " + path.name, *tail])
    print(f"Last {limit} filtered traceback lines (raw image/debug output omitted):")
    print("\n".join(traces[-limit:]) if traces else "No Python stack trace captured; the process may have been killed or failed in native code.")
    # Both the caller's pre-upload flag AND our install step prefix are required.
    # Default is fail-closed, including photo_upload and archive_download failures.
    log_name = info.get("log")
    if (allow_install_logs and str(info.get("step", "")).startswith("install:")
            and isinstance(log_name, str) and re.fullmatch(r"install-[a-z0-9-]+-all\.log", log_name)):
        print("Last 60 installation log lines (pre-upload only; credentials redacted):")
        print("\n".join(install_log_tail(root / "logs" / log_name)))


def print_warnings(root: Path) -> None:
    path = root / "warnings.json"
    if not path.is_file():
        return
    for item in json.loads(path.read_text(encoding="utf-8")):
        if item.get("view") in {"left", "right", "back"}:
            reason = re.sub(r"[^a-zA-Z0-9_ -]", "?", str(item.get("reason", "preprocessing_failed")))[:80]
            step = re.sub(r"[^a-zA-Z0-9_ -]", "?", str(item.get("step", "detection")))[:80]
            print(f"Warning: skipped {item['view']} at {step}: {reason}.")
