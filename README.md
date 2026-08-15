<p align="center">
  <img src="docs/assets/mlx-indextts-banner.svg" alt="IndexTTS on Apple Silicon MLX and NVIDIA CUDA" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/ml-explore/mlx"><img alt="Apple MLX" src="https://img.shields.io/badge/Apple%20Silicon-MLX-000000?style=flat-square&logo=apple&logoColor=white"></a>
  <a href="docs/nvidia.md"><img alt="NVIDIA CUDA" src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white"></a>
  <a href="https://github.com/index-tts/index-tts"><img alt="IndexTTS 2 and 2.5" src="https://img.shields.io/badge/IndexTTS-2.0%20%7C%202.5-0891B2?style=flat-square"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/github/license/groxaxo/mlx-indextts2-inference?style=flat-square"></a>
</p>

<p align="center">
  <strong>Expressive multilingual voice cloning on the hardware you already own.</strong><br />
  Native MLX inference for Apple Silicon, plus a pinned official PyTorch backend for NVIDIA CUDA.
</p>

<p align="center">
  <a href="#choose-your-backend">Backends</a> ·
  <a href="#apple-silicon-quick-start">Apple quick start</a> ·
  <a href="#nvidia-quick-start">NVIDIA quick start</a> ·
  <a href="#capabilities">Capabilities</a> ·
  <a href="#documentation">Documentation</a>
</p>

## Choose your backend

| | Apple Silicon / MLX | NVIDIA / CUDA |
| --- | --- | --- |
| Runtime | Native MLX implementation in this repository | Pinned official IndexTTS PyTorch runtime |
| Primary platform | M-series Macs | Linux/Windows NVIDIA GPUs |
| IndexTTS 2.5 precision | Persistent 3/4/5/6/8-bit GPT, fp16/fp32 components | BF16 or FP32 |
| IndexTTS 2.0 precision | Quantized/fp16/fp32 MLX | FP16 or FP32 |
| Serving | CLI, batch, FastAPI, Gradio, completed-segment streaming | CLI, FastAPI, multi-GPU batch |
| Multi-device strategy | Unified-memory MLX execution | One warm model worker per GPU |

The CUDA implementation intentionally wraps the official upstream classes at a pinned revision rather than maintaining a separate Torch fork. That keeps NVIDIA behavior aligned with official CUDA kernels, DeepSpeed, GPT acceleration, `torch.compile`, model downloads, and checkpoint formats.

## Capabilities

- Zero-shot voice cloning from reference audio.
- IndexTTS 2.5 multilingual synthesis in Chinese, English, Japanese, Spanish, and Arabic.
- Independent speaker and emotion control from audio, named/mixed vectors, or text descriptions.
- Pinyin, CMU phoneme, and Japanese Kana pronunciation guidance.
- Batch generation with model reuse instead of one model load per utterance.
- FastAPI endpoints for both platform families.
- Subtitle-oriented duration controls and pitch-preserving output fitting on MLX.
- Three-GPU CUDA throughput mode that keeps one model resident on each selected GPU.

> [!NOTE]
> IndexTTS 2.5 is the primary target. IndexTTS 2.0 remains supported, including the repository's local Vietnamese MLX profile.

## Apple Silicon quick start

### Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/groxaxo/mlx-indextts2-inference.git
cd mlx-indextts2-inference
uv sync --extra v25
```

### Download and convert IndexTTS 2.5

```bash
uv run hf download IndexTeam/IndexTTS-2.5 \
  --revision d0aa86e75bb6f3437f3831e95056fa72842d89ef \
  --local-dir models/IndexTTS-2.5-source

uv run mlx-indextts convert \
  --model-dir models/IndexTTS-2.5-source \
  --output models/mlx-IndexTTS-2.5-8bit \
  --dtype float16 \
  --quantize 8 \
  --source-revision d0aa86e75bb6f3437f3831e95056fa72842d89ef
```

### Generate

```bash
uv run mlx-indextts generate \
  --profile v25 \
  --language en \
  --ref-audio reference.wav \
  --text "Hello from a local MLX voice." \
  --output outputs/apple.wav
```

See [the complete IndexTTS 2.5 MLX guide](docs/indextts-2.5.md).

## NVIDIA quick start

The CUDA stack lives in an isolated uv project so official PyTorch 2.8/CUDA 12.8 requirements cannot conflict with the newer Torch tooling used by MLX conversion and validation.

### Install and validate CUDA

```bash
./scripts/setup_nvidia.sh

# Optional acceleration dependencies:
./scripts/setup_nvidia.sh --extra accel
./scripts/setup_nvidia.sh --extra deepspeed
```

### Download IndexTTS 2.5

```bash
uv run --project nvidia mlx-indextts-nvidia download --version 2.5
```

### Generate on one RTX GPU

```bash
uv run --project nvidia mlx-indextts-nvidia generate \
  --version 2.5 \
  --model-dir checkpoints \
  --device cuda:0 \
  --precision bf16 \
  --ref-audio reference.wav \
  --language en \
  --text "Hello from IndexTTS on NVIDIA CUDA." \
  --output outputs/nvidia.wav
```

### Saturate three RTX 3090s for batch throughput

```bash
uv run --project nvidia mlx-indextts-nvidia batch \
  --model-dir checkpoints \
  --devices cuda:0,cuda:1,cuda:2 \
  --precision bf16 \
  --input jobs.jsonl \
  --output-dir outputs/three_gpu
```

Each GPU gets one long-lived process and one model load; jobs are distributed round-robin. See the [full NVIDIA guide](docs/nvidia.md) for JSONL/CSV schemas, API serving, performance switches, and troubleshooting.

## Interfaces

### MLX CLI

```bash
uv run mlx-indextts --help
```

Commands include conversion, generation, speaker-cache creation, batch planning, batch synthesis, video/emotion reference libraries, and denoising.

### NVIDIA CLI

```bash
uv run --project nvidia mlx-indextts-nvidia --help
uv run --project nvidia mlx-indextts-nvidia doctor
```

Commands:

- `doctor`: verify the CUDA PyTorch build, visible GPUs, VRAM, BF16, and official runtime.
- `download`: download IndexTTS 2.5 or 2.0 checkpoints.
- `generate`: model-resident single-GPU synthesis.
- `batch`: one persistent model worker per selected GPU.
- `serve`: local FastAPI service.

### Python API — Apple MLX

```python
from mlx_indextts import IndexTTSv25

tts = IndexTTSv25("models/mlx-IndexTTS-2.5-8bit")
tts.generate(
    text="Hola, esta es una prueba.",
    reference_audio="reference.wav",
    output_path="outputs/es.wav",
    language="es",
)
```

### Python API — NVIDIA CUDA

```python
from mlx_indextts import NvidiaGenerateRequest, NvidiaIndexTTS, NvidiaRuntimeConfig

runtime = NvidiaIndexTTS(
    NvidiaRuntimeConfig(
        model_dir="checkpoints",
        version="2.5",
        device="cuda:0",
        precision="bf16",
    )
)

result = runtime.generate(
    NvidiaGenerateRequest(
        text="Hola, esta es una prueba.",
        ref_audio="reference.wav",
        output_path="outputs/es_cuda.wav",
        language="es",
    )
)
print(result.as_dict())
```

## Model profiles

| Profile | Backend | Languages | Notes |
| --- | --- | --- | --- |
| IndexTTS 2.5 | MLX + CUDA | `zh`, `en`, `ja`, `es`, `ar` | Primary multilingual target |
| IndexTTS 2.0 standard | MLX + CUDA | Chinese, English | Preserved compatibility path |
| IndexTTS 2.0 Vietnamese | MLX | Vietnamese | Local repository extension |

## Performance posture

For Apple Silicon, persistent 8-bit GPT is the default local balance; codec, S2Mel/DiT, and BigVGAN retain the selected conversion dtype.

For RTX 3090-class CUDA systems, start with BF16 on IndexTTS 2.5 and FP16 on IndexTTS 2.0. Benchmark `--cuda-kernel`, `--torch-compile`, `--accel`, and `--deepspeed` independently. For multiple 24 GiB cards, job-level parallelism generally provides more useful throughput than tensor parallelism because one model already fits on one GPU.

## Documentation

- [NVIDIA CUDA setup, API, and multi-GPU guide](docs/nvidia.md)
- [IndexTTS 2.5 MLX conversion and generation](docs/indextts-2.5.md)
- [Executed MLX validation record](docs/indextts-2.5-validation.md)
- [Persistent quantized model guide](docs/quantized-models.md)
- [Performance bottleneck analysis](docs/performance-bottlenecks.md)

## License and responsible use

Repository code uses the license in this repository. Official IndexTTS weights, source, and converted derivatives remain governed by their upstream model/source licenses, including commercial thresholds, downstream notices, and use restrictions.

Only clone voices you own or have explicit permission to use. Do not use the software for impersonation, deception, privacy infringement, fraud, unlawful content, or prohibited high-risk deployment.

## Acknowledgments

- [IndexTTS](https://github.com/index-tts/index-tts) — official model and NVIDIA PyTorch runtime
- [MLX](https://github.com/ml-explore/mlx) — Apple machine-learning framework
- [PyTorch](https://pytorch.org/) — CUDA execution platform
