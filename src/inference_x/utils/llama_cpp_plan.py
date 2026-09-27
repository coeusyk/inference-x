"""llama.cpp planning helpers (add-llama-cpp-backend D2).

The llama.cpp counterpart of `utils/vllm_pool_config.py`: locating the
`llama-server` binary and the GGUF file, and asking llama.cpp's own planner
how many layers fit. Nothing here starts a server or downloads a model.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

BINARY_ENV = "INFERENCE_X_LLAMA_SERVER"


def find_llama_server() -> str | None:
    """The `llama-server` binary: `INFERENCE_X_LLAMA_SERVER`, else `PATH`."""
    configured = os.environ.get(BINARY_ENV, "").strip()
    if configured:
        return configured if os.access(configured, os.X_OK) else None
    return shutil.which("llama-server")


def cached_gguf_path(model_path: str, gguf_file: str | None) -> Path | None:
    """Local path of an entry's GGUF file if it is already on disk, else None.
    Never downloads (plan/doctor/models are read-only)."""
    if gguf_file is None:
        path = Path(model_path)
        return path if path.is_file() else None
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(model_path, gguf_file)
    return Path(found) if isinstance(found, str) else None


def fit_gpu_layers(
    gguf: Path, max_model_len: int, n_gpu_layers: int | None
) -> tuple[int | None, str | None]:
    """(GPU layers, error) from llama.cpp's own planner, `llama-fit-params`,
    which ships beside `llama-server`. -1 means every layer fits on the GPU.
    Read-only: loads GGUF metadata, starts no server (add-llama-cpp-backend D2).
    """
    server = find_llama_server()
    if server is None:
        return None, f"llama-server not found (set {BINARY_ENV} or put it on PATH)"
    fit = Path(server).with_name("llama-fit-params")
    if not fit.is_file():
        return None, f"llama-fit-params not found beside {Path(server).name}"
    args = [str(fit), "-m", str(gguf), "-c", str(max_model_len), "-np", "1"]
    if n_gpu_layers is not None:
        args += ["-ngl", str(n_gpu_layers)]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"llama-fit-params failed: {exc}"
    fitted = out.stdout.split()
    if out.returncode != 0 or "-ngl" not in fitted:
        return None, f"llama-fit-params failed (exit {out.returncode})"
    return int(fitted[fitted.index("-ngl") + 1]), None


def gguf_size_gib(model_path: str, gguf_file: str | None) -> float | None:
    """Size of the entry's GGUF file when it is on disk, else None."""
    path = cached_gguf_path(model_path, gguf_file)
    return round(path.stat().st_size / 2**30, 3) if path is not None else None
