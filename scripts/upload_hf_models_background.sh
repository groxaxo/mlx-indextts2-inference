#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/outputs/hf_upload_logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

MODELS=(
  "vanch007/mlx-indextts2-standard-fp32|models/mlx-indexTTS2-standard-fp32"
  "vanch007/mlx-indextts2-standard-fp16|models/mlx-indexTTS2-standard-fp16"
  "vanch007/mlx-indextts2-standard-8bit|models/mlx-indexTTS2-standard-8bit"
  "vanch007/mlx-indextts2-vietnamese-fp32|models/mlx-indexTTS2-vietnamese-fp32"
  "vanch007/mlx-indextts2-vietnamese-fp16|models/mlx-indexTTS2-vietnamese-fp16"
  "vanch007/mlx-indextts2-vietnamese-8bit|models/mlx-indexTTS2-vietnamese-8bit"
)

KEEP_DIRS=(
  "models/mlx-indexTTS2-standard-8bit"
  "models/mlx-indexTTS2-vietnamese-8bit"
)

DELETE_AFTER_SUCCESS=(
  "models/mlx-indexTTS2-standard-fp32"
  "models/mlx-indexTTS2-standard-fp16"
  "models/mlx-indexTTS2-vietnamese-fp32"
  "models/mlx-indexTTS2-vietnamese-fp16"
)

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

verify_repo() {
  local repo="$1"
  "$ROOT_DIR/.venv/bin/python" - "$repo" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
info = HfApi().repo_info(repo_id=repo_id, repo_type="model")
names = {s.rfilename for s in info.siblings}
required = {
    "README.md",
    "gpt.safetensors",
    "s2mel.safetensors",
    "bigvgan.safetensors",
    "vq2emb.safetensors",
    "tokenizer.model",
    "config.yaml",
    "config.json",
    "feat1.pt",
    "feat2.pt",
    "wav2vec2bert_stats.pt",
}
missing = sorted(required - names)
if missing:
    raise SystemExit(f"{repo_id} missing files: {missing}")
print(f"{repo_id} verified: {len(names)} files")
PY
}

log "Hugging Face upload started. Logs: $LOG_DIR"
log "Endpoint check:"
huggingface-cli env | grep -E 'Who am I|ENDPOINT|HF_HUB_DISABLE_XET|HF_HUB_ENABLE_HF_TRANSFER' || true

for item in "${MODELS[@]}"; do
  repo="${item%%|*}"
  dir="${item#*|}"
  model_log="$LOG_DIR/${repo#vanch007/}.log"
  if [[ ! -d "$dir" ]]; then
    log "SKIP upload for $repo because local dir is missing: $dir"
    verify_repo "$repo" | tee "$model_log"
    log "DONE $repo"
    continue
  fi
  log "UPLOAD $repo <= $dir"
  huggingface-cli upload-large-folder "$repo" "$dir" \
    --repo-type model \
    --num-workers 4 \
    --no-bars \
    --no-report \
    2>&1 | tee "$model_log"
  verify_repo "$repo" | tee -a "$model_log"
  log "DONE $repo"
done

log "All uploads verified. Keeping 8bit model directories:"
printf '  %s\n' "${KEEP_DIRS[@]}"

log "Deleting local non-8bit model directories after successful upload."
for dir in "${DELETE_AFTER_SUCCESS[@]}"; do
  if [[ -d "$dir" ]]; then
    log "DELETE $dir"
    rm -rf "$dir"
  fi
done

log "Final local model directories:"
find models -maxdepth 1 -type d -name 'mlx-indexTTS2-*' -print | sort
log "Upload lifecycle complete."
