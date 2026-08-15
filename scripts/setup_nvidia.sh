#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install/repair the NVIDIA driver before continuing." >&2
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
uv sync --project "$ROOT/nvidia" "$@"
uv run --project "$ROOT/nvidia" mlx-indextts-nvidia doctor

cat <<'EOF'

NVIDIA runtime is ready.
Download IndexTTS 2.5:
  uv run --project nvidia mlx-indextts-nvidia download --version 2.5

Generate:
  uv run --project nvidia mlx-indextts-nvidia generate \
    --model-dir checkpoints --device cuda:0 --precision bf16 \
    --ref-audio reference.wav --language en \
    --text "Hello from NVIDIA CUDA." --output outputs/nvidia.wav
EOF
