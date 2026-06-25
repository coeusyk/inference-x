#!/usr/bin/env bash
# Install a site-packages .pth hook so vLLM worker subprocesses apply WSL patches.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

site="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
pth="${site}/inferencex_vllm.pth"

cat >"$pth" <<EOF
import os
if os.environ.get("INFERENCE_X_VLLM_PATCH_APPLIED") == "1":
    try:
        import vllm.platforms.interface as _ix_vllm_platform
        _ix_vllm_platform.in_wsl = lambda: False
    except Exception:
        pass
EOF

echo "Installed vLLM worker patch: ${pth}"
