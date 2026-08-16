# NVIDIA CUDA backend

This repository supports two first-class execution paths:

- **Apple Silicon:** the native MLX implementation in `mlx_indextts`.
- **NVIDIA:** a thin, pinned adapter around the official IndexTTS PyTorch runtime.

The CUDA path intentionally reuses the official model implementation instead of maintaining a second, drifting Torch port. The wrapper adds stable CLI/API contracts, validation, reproducible upstream pinning, and one-model-per-GPU batch distribution.

## Supported NVIDIA modes

| Capability | IndexTTS 2.5 | IndexTTS 2.0 |
| --- | --- | --- |
| Precision | BF16 or FP32 | FP16 or FP32 |
| Languages | Chinese, English, Japanese, Spanish, Arabic | Chinese and English |
| Emotion audio/vector/text | Yes | Yes |
| BigVGAN CUDA kernel | Optional | Optional |
| DeepSpeed | Optional | Optional |
| GPT acceleration | Optional | Optional |
| `torch.compile` S2Mel | Optional | Optional |
| Multi-GPU batch | One persistent process per selected GPU | One persistent process per selected GPU |

## Requirements

- Linux or Windows with a CUDA-capable NVIDIA GPU.
- NVIDIA driver compatible with CUDA 12.8.
- Python 3.10 or 3.11.
- `uv` and `git`.
- At least roughly 10 GiB of VRAM; 24 GiB cards such as the RTX 3090 are a strong fit.

The NVIDIA environment is isolated under `nvidia/` because the official CUDA runtime pins PyTorch 2.8, while MLX conversion/development extras in the main project use a newer Torch toolchain. Keeping them separate prevents dependency compromise on either platform.

## Install

```bash
./scripts/setup_nvidia.sh
```

Install optional acceleration dependencies only when you intend to benchmark them:

```bash
./scripts/setup_nvidia.sh --extra accel
./scripts/setup_nvidia.sh --extra deepspeed
# Or both:
./scripts/setup_nvidia.sh --extra full
```

Equivalent manual setup:

```bash
uv sync --project nvidia
uv run --project nvidia mlx-indextts-nvidia doctor
```

The environment pins the official source revision:

```text
index-tts/index-tts@4f8792ff120cd3ea470dd511e997a17c86cddd10
```

## Download checkpoints

```bash
# IndexTTS 2.5 -> ./checkpoints
uv run --project nvidia mlx-indextts-nvidia download --version 2.5

# The public IndexTTS 2.5 config contains stale internal checkpoint paths.
# The NVIDIA runtime normalizes them to the local downloaded artifacts.

# IndexTTS 2.0 -> ./checkpoints_2
uv run --project nvidia mlx-indextts-nvidia download --version 2.0
```

## Single-GPU generation

RTX 3090 / Ampere default for IndexTTS 2.5:

```bash
uv run --project nvidia mlx-indextts-nvidia generate \
  --version 2.5 \
  --model-dir checkpoints \
  --device cuda:0 \
  --precision bf16 \
  --ref-audio reference.wav \
  --language en \
  --text "Hello from IndexTTS 2.5 on NVIDIA CUDA." \
  --output outputs/nvidia_25.wav
```

Spanish should be explicit because automatic detection conservatively treats Latin-script text as English:

```bash
uv run --project nvidia mlx-indextts-nvidia generate \
  --model-dir checkpoints --device cuda:0 --precision bf16 \
  --ref-audio reference.wav --language es \
  --text "Hola, esta voz se está ejecutando localmente." \
  --output outputs/es.wav
```

Text-guided emotion loads the Qwen emotion model:

```bash
uv run --project nvidia mlx-indextts-nvidia generate \
  --model-dir checkpoints --device cuda:0 --precision bf16 \
  --qwen-emotion \
  --ref-audio reference.wav --language en \
  --text "I cannot believe we finally made it." \
  --emotion-text "relieved, joyful, and slightly overwhelmed" \
  --emo-alpha 0.6 \
  --output outputs/emotional.wav
```

## Three-RTX-3090 throughput mode

IndexTTS fits on one 24 GiB GPU, so tensor-parallel sharding would add synchronization without solving a capacity problem. For offline production, the highest-value topology is one warm model process per GPU and job-level distribution across all cards.

Create `jobs.jsonl`:

```jsonl
{"text":"First line","ref_audio":"voices/a.wav","language":"en","output":"001.wav"}
{"text":"Segunda línea","ref_audio":"voices/b.wav","language":"es","output":"002.wav"}
{"text":"第三句话","ref_audio":"voices/c.wav","language":"zh","output":"003.wav"}
```

Run one persistent worker on each 3090:

```bash
uv run --project nvidia mlx-indextts-nvidia batch \
  --version 2.5 \
  --model-dir checkpoints \
  --devices cuda:0,cuda:1,cuda:2 \
  --precision bf16 \
  --input jobs.jsonl \
  --output-dir outputs/three_gpu
```

`--devices auto` uses every visible CUDA device. Jobs are assigned round-robin, each worker loads the model once, and `manifest.jsonl` records success, failure, device, precision, language, output, and elapsed time.

## FastAPI server

```bash
uv run --project nvidia mlx-indextts-nvidia serve \
  --model-dir checkpoints \
  --device cuda:0 \
  --precision bf16 \
  --host 0.0.0.0 \
  --port 7863
```

Endpoints:

- `GET /health`
- `POST /generate`
- `GET /audio/{filename}`

Example:

```bash
curl -X POST http://127.0.0.1:7863/generate \
  -H 'content-type: application/json' \
  -d '{
    "text": "Hello from the CUDA API.",
    "ref_audio": "/absolute/path/reference.wav",
    "language": "en"
  }'
```

The model is protected by an in-process lock because the official runtime reuses mutable speaker/emotion caches. For concurrent production traffic, run one API instance per GPU and place a small reverse proxy or queue in front of the three ports.

## Performance switches

Start with the stable baseline:

```text
IndexTTS 2.5: --precision bf16 --no-cuda-kernel
IndexTTS 2.0: --precision fp16 --no-cuda-kernel
```

Then benchmark these independently:

- `--cuda-kernel`: fused BigVGAN activation; falls back to Torch if compilation/loading fails.
- `--torch-compile`: can improve S2Mel after warm-up, but increases first-request latency.
- `--accel`: official GPT acceleration engine; install with `--extra accel` first.
- `--deepspeed`: install with `--extra deepspeed`; hardware/workload dependent, so do not assume it is faster.

Avoid enabling every switch at once. Benchmark warm generation RTF, peak VRAM, first-request latency, and output equivalence separately.

## Troubleshooting

### `torch.cuda.is_available()` is false

Run:

```bash
nvidia-smi
uv run --project nvidia python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

If the CUDA build is missing, remove `nvidia/.venv` and rerun `./scripts/setup_nvidia.sh` so uv installs PyTorch from the CUDA 12.8 index.

### BigVGAN custom kernel fails

Use `--no-cuda-kernel`. The official runtime falls back to the standard Torch implementation; generation remains functional.

### Out of memory

- Disable `--qwen-emotion` unless text-driven emotion is required.
- Use the correct half precision: BF16 for 2.5, FP16 for 2.0.
- Reduce `max_text_tokens` / `max_mel_tokens` and split long text.
- Ensure only one model process is resident on each 24 GiB card.

### Multiple GPUs but only one is used

Single generation intentionally uses one GPU. Use the `batch --devices ...` command for throughput, or run separate API instances per GPU.
