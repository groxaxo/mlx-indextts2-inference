# MLX-IndexTTS2

IndexTTS for Apple Silicon using MLX. Zero-shot text-to-speech with voice cloning capabilities.

## Features

- Run IndexTTS 1.5/2.0 natively on Apple Silicon
- RTF ~0.5 (2x faster than real-time on M2 Max)
- Voice cloning from reference audio
- **v2.0**: Emotion control (8 emotions)
- Auto-detect model version (1.5/2.0)

## Official Feature Parity

Checked against the official IndexTTS2 capability set: zero-shot cloning,
speaker/emotion disentanglement, audio/text/manual emotion control, duration
control, Chinese/English generation, API/WebUI usage, and batch workflows.

| Official capability | Local MLX status | Notes |
| --- | --- | --- |
| IndexTTS 1.5 / IndexTTS2 model conversion | Preserved | `convert` auto-detects 1.5 vs 2.0. |
| Zero-shot speaker cloning | Preserved | Raw WAV refs and precomputed `.npz` speaker caches are supported. |
| Speaker / emotion disentanglement | Preserved | `--ref-audio` and `--emotion-ref-audio` are separate; emotion sources are mutually exclusive. |
| Manual 8-emotion vectors / mixes | Preserved | `--emotion happy`, mixed weights, and `--emo-alpha` are exposed. |
| Text emotion via official Qwen emotion model | Preserved | Converted MLX model at `models/qwen0.6bemo4-merge-mlx-8bit`. |
| Audio emotion reference | Preserved | CLI/API/WebUI/batch CSV support `emotion_ref_audio`. |
| Duration control | Partially preserved | `--target-duration` now maps seconds to a mel-token budget; `--fit-duration` explicitly applies pitch-preserving stretch when exact file length matters. |
| Batch / novel / dialogue generation | Extended | Adds planner, merged WAV, Qwen smoothing, scene emotion library, and crosstalk catalogs. |
| API and WebUI | Preserved as local replacements | Lightweight FastAPI and Gradio entrypoints share the same runtime cache. |
| Streaming | Not production | `generate_stream` is still a placeholder, not a true low-latency stream. |
| Vietnamese profile | Local extension | Uses a separate Vietnamese checkpoint/profile beyond the official zh/en baseline. |

## Current Crosstalk Benchmark Role

This project is the content-fidelity and emotion-control baseline in the four-backend MLX TTS comparison. It owns the shared 64-row crosstalk manifest and reference-library outputs used by the runner:

- shared manifest: `/Users/vanch/mlx-indextts2/outputs/groupchat_crosstalk_20260509_scene_ref/audio/manifest.csv`
- clean speaker refs catalog: `/Users/vanch/mlx-indextts2/outputs/fjymb_library_final/catalog.csv`
- latest full64 runner output: `/Users/vanch/tts_benchmarks/mlx_four_backend_full64_cleanrefs_20260516/indextts`
- latest full64 metrics: 64 rows, 236.203s audio, 183.469s generation time, RTF `0.7767`

Use IndexTTS2 when the task needs separated speaker and emotion references, Vietnamese routing, Qwen text-emotion conversion, or the highest priority is exact content with explicit emotion control.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/user/mlx-indextts.git
cd mlx-indextts

# Basic install (generation only)
uv sync

# With model conversion support (requires torch)
uv sync --extra convert
```

## Quick Start

### Local 8bit Defaults

This checkout has been tuned for Apple Silicon IndexTTS2 migration work on an
M3 Max Mac. If `--model` is omitted, the CLI now uses local 8bit IndexTTS2
models by default:

- standard / Chinese / English: `models/mlx-indexTTS2-standard-8bit`
- Vietnamese: `models/mlx-indexTTS2-vietnamese-8bit`

Vietnamese is auto-selected when the input text contains Vietnamese tone marks.
You can override the default paths with environment variables:

```bash
export MLX_INDEXTTS_STANDARD_MODEL=/path/to/standard-8bit
export MLX_INDEXTTS_VIETNAMESE_MODEL=/path/to/vietnamese-8bit
```

### 1. Convert Model (auto-detects version)

```bash
# Convert IndexTTS 1.5
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-1.5 \
    -o models/mlx-indexTTS-1.5

# Convert IndexTTS 2.0
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-2 \
    -o models/mlx-indexTTS-2.0
```

### 2. Generate Speech (auto-detects version)

```bash
# v1.5
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r reference.wav \
    -t "你好，这是一个语音合成测试。" \
    -o output.wav

# v2.0
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r reference.wav \
    -t "你好，这是一个语音合成测试。" \
    -o output.wav

# v2.0 with emotion control
uv run mlx-indextts generate \
    -r reference.wav \
    -t "今天真是太开心了！" \
    -o output.wav \
    --emotion happy --emo-alpha 0.6

# v2.0 with MLX-native Qwen text emotion
uv sync --extra qwen
uv run python scripts/convert_qwen_emotion_mlx.py
uv run mlx-indextts generate \
    -r speaker_v20.npz \
    -t "我今天很开心，终于见到你了。" \
    -o output_qwen.wav \
    --auto-emotion

# Vietnamese uses the local Vietnamese 8bit model automatically
uv run mlx-indextts generate \
    -r speaker_vietnamese.npz \
    -t "Đêm nay gió rất nhẹ, ánh đèn ngoài cửa sổ chậm rãi sáng lên." \
    -o output_vi.wav

# v2.0 duration-oriented generation
uv run mlx-indextts generate \
    -r speaker_v20.npz \
    -t "这一句需要贴近视频时长。" \
    -o output_duration.wav \
    --target-duration 3.2

# exact file-length fitting is explicit and uses pitch-preserving time-stretch
uv run mlx-indextts generate \
    -r speaker_v20.npz \
    -t "这一句需要严格对齐三点二秒。" \
    -o output_fit_duration.wav \
    --target-duration 3.2 \
    --fit-duration
```

### 3. Pre-compute Speaker (Faster Inference)

Pre-compute speaker conditioning to skip audio preprocessing on subsequent generations.

```bash
# v1.5
uv run mlx-indextts speaker \
    -m models/mlx-indexTTS-1.5 \
    -r reference.wav \
    -o speaker_v15.npz

# v2.0
uv run mlx-indextts speaker \
    -m models/mlx-indexTTS-2.0 \
    -r reference.wav \
    -o speaker_v20.npz

# Use pre-computed speaker (much faster loading)
uv run mlx-indextts generate \
    -r speaker_v20.npz \
    -t "你好，世界！" \
    -o output.wav
```

**Note**: v1.5 and v2.0 speaker files are incompatible - each version requires its own .npz file.

### 4. Batch Generation

Batch generation migrates the core batch workflow from the PyTorch IndexTTS
project while keeping the MLX model loaded once.

Input can be a plain text file with one utterance per line, or a CSV with a
`text` column:

```bash
uv run mlx-indextts batch \
    -i novel_lines.txt \
    -r speaker_vietnamese.npz \
    -o outputs/novel_vi \
    --profile vietnamese \
    --auto-emotion \
    --combine
```

The batch command writes per-line WAV files plus `manifest.csv`; with
`--combine`, it also writes `combined.wav`. When `--auto-emotion` is enabled,
Qwen emotion analysis runs for the whole batch first, the Qwen model is released,
and `manifest.csv` includes `emotion_json`, `dominant_emotion`, and
`emotion_source`.

## Python API

```python
# v1.5
from mlx_indextts.generate import IndexTTS

tts = IndexTTS.load_model("models/mlx-indexTTS-1.5")
audio = tts.generate(text="你好", ref_audio="reference.wav")
tts.save_audio(audio, "output.wav")

# v2.0
from mlx_indextts.generate_v2 import IndexTTSv2

tts = IndexTTSv2("models/mlx-indexTTS-2.0")
audio = tts.generate(
    text="你好",
    reference_audio="reference.wav",
    output_path="output.wav",
    emotion="happy",
    emo_alpha=0.6,
)
```

## CLI Options

```
mlx-indextts generate [OPTIONS]

Required:
  -r, --ref-audio    Reference audio (.wav or .npz)
  -t, --text         Text to synthesize
  -o, --output       Output file

Common options:
  -m, --model        Model directory (optional; local 8bit defaults if omitted)
  --profile          auto / standard / vietnamese / vi
  --max-tokens       Max mel tokens (default: 800 for v1.5, 1500 for v2.0)
  --temperature      Sampling temperature (default: 1.0 for v1.5, 0.8 for v2.0)
  --seed, -s         Random seed for reproducibility
  -v, --verbose      Verbose output
  -p, --play         Play audio after generation
  --quantize, -q     Runtime quantization: 4, 8, or fp32

v2.0 only:
  --emotion          Emotion: happy/sad/angry/afraid/disgusted/melancholic/surprised/calm/auto-qwen
  --auto-emotion     Use the MLX-native Qwen text emotion model before TTS
  --qwen-emotion-model
                     Converted Qwen emotion model path (default: models/qwen0.6bemo4-merge-mlx-8bit)
  --qwen-unload-after / --no-qwen-unload-after
                     Release Qwen after emotion analysis (default: true)
  --emo-alpha        Emotion intensity 0.0-1.0 (default: 0.6, recommend ≤ 0.8)
  --diffusion-steps  Diffusion steps (default: 16)
  --cfg-rate         CFG rate (default: 0.7)
```

### Qwen Text Emotion

The official IndexTTS2 Qwen emotion checkpoint can be converted to MLX and used
as a text preprocessor:

```bash
uv sync --extra qwen
uv run python scripts/convert_qwen_emotion_mlx.py
```

Default paths:

- Source for reconversion: `checkpoints/qwen0.6bemo4-merge` or pass `--source`
- Source override: `MLX_INDEXTTS_QWEN_EMOTION_SOURCE=/path/to/qwen0.6bemo4-merge`
- Converted MLX 8bit: `models/qwen0.6bemo4-merge-mlx-8bit`

The Qwen classifier returns the same 8 emotion weights used by IndexTTS2:
`happy`, `angry`, `sad`, `afraid`, `disgusted`, `melancholic`, `surprised`,
and `calm`. The runtime unloads Qwen after analysis by default so it does not
remain resident with the TTS model.

### emotion2vec Audio Emotion Library

`emotion2vec_plus_large` is the recommended offline audio-emotion tagger for
building an emotion reference library. It runs as preprocessing, not in the
MLX TTS hot path.

```bash
uv sync --extra emotion2vec
uv run mlx-indextts emotion2vec \
  --input /path/to/audio_clips \
  --output-dir outputs/emotion_library
```

This writes `catalog.csv`, `emotion_refs_by_emotion.json`, `summary.md`, and
an optional `clips/` mirror for portable later use.

You can then let the planner auto-fill `emotion_ref_audio` from the catalog:

```bash
uv run mlx-indextts plan \
  -i script.txt \
  -o outputs/batch_plan.csv \
  --emotion-library outputs/emotion_library/catalog.csv
```

### Video / YouTube Scene Library

For crosstalk or other long-form dialogue videos, the `video-library` command
downloads the source audio, extracts vocals when Demucs is available, runs ASR
sentence splitting, tags each sentence with emotion plus age/gender, and writes
a reusable scene-emotion-gender-age catalog. Emotion tagging uses
`emotion2vec_plus_large`; age/gender uses an audEERING model when available and
falls back to a lightweight heuristic when the model is not requested.

```bash
uv sync --extra library
uv run mlx-indextts video-library \
  --source "https://www.youtube.com/watch?v=FjY-mbHMGvI" \
  --scene crosstalk \
  --output-dir outputs/crosstalk_library
```

If the first `emotion2vec_plus_large` download is too slow, build the catalog
with `--emotion-model emotion2vec/emotion2vec_plus_base` first, then rerun with
the large model when the cache is ready.

Outputs:

- `catalog.csv`
- `emotion_refs_by_emotion.json`
- `emotion_refs_by_scene_emotion_gender_age.json`
- `summary.md`
- `clips/<scene>_...wav`

This catalog is compatible with `plan --emotion-library ...` for later
emotion-reference auto-fill.

For group-chat crosstalk generation, write the adapted script as speaker-prefixed
dialogue such as `逗哏：...` and `捧哏：...`, then let `plan` fill both the
voice reference and the crosstalk emotion reference:

```bash
uv run mlx-indextts plan \
  -i outputs/groupchat_crosstalk/script.txt \
  -o outputs/groupchat_crosstalk/batch.csv \
  --emotion-library outputs/fjymb_library_final/catalog.csv \
  --emotion-scene crosstalk \
  --speaker-ref "逗哏=/path/to/dougen.wav" \
  --speaker-ref "捧哏=/path/to/penggen.wav" \
  --speaker-profile "逗哏=male:adult" \
  --speaker-profile "捧哏=male:adult"
```

If you want the scene library to auto-pick a more distinct duo, use
`--auto-duo-refs`. It chooses two acoustically different clips from the
crosstalk catalog and assigns the lower voice to `逗哏` and the higher voice to
`捧哏`.

Speaker references are denoised by default before saving or generation, so the
runtime uses cleaned vocal stems unless you explicitly pass `--no-denoise-ref`.

```bash
uv run mlx-indextts batch \
  -i outputs/groupchat_crosstalk/batch.csv \
  -o outputs/groupchat_crosstalk/audio \
  --combine
```

Scene libraries match `scene + emotion + gender + age_band` first, then fall
back to looser scene/emotion matches when an exact demographic bucket is absent.

### API Server

Install API extras and start the local server:

```bash
uv sync --extra api
uv run mlx-indextts-api
```

Endpoints:

- `GET /health`
- `GET /profiles`
- `POST /generate`
- `POST /speaker`
- `GET /audio?path=outputs/api_output.wav`

Example:

```bash
curl -X POST http://127.0.0.1:7862/generate \
  -H 'content-type: application/json' \
  -d '{
    "text": "Đêm nay gió rất nhẹ.",
    "ref_audio": "speakers/ban_khoe_vietnamese_v2.npz",
    "profile": "vietnamese",
    "output_path": "outputs/api_vi.wav",
    "auto_emotion": true,
    "diffusion_steps": 16
  }'
```

### WebUI

Install WebUI extras and start the lightweight Gradio interface:

```bash
uv sync --extra webui
uv run mlx-indextts-webui
```

The WebUI uses the same single-model cache and 8bit defaults as the CLI/API, so
standard and Vietnamese models are not kept loaded at the same time.

## Version Comparison

| Feature | v1.5 | v2.0 |
|---------|------|------|
| Sample rate | 24000 Hz | 22050 Hz |
| Max tokens | 800 | 1815 |
| Default temperature | 1.0 | 0.8 |
| Emotion control | ❌ | ✅ 8 emotions |
| S2Mel (CFM) | ❌ | ✅ |
| BigVGAN | Custom | nvidia pretrained |
| Runtime quantization | ✅ | ✅ |
| Speaker pre-compute | ✅ | ✅ |

## Supported Emotions (v2.0)

| English | 中文 |
|---------|------|
| happy | 高兴 |
| angry | 愤怒 |
| sad | 悲伤 |
| afraid | 恐惧 |
| disgusted | 反感 |
| melancholic | 低落 |
| surprised | 惊讶 |
| calm | 自然 |

Mixed emotions: `--emotion "happy:0.6,sad:0.4"`

## Performance

| Metric | v1.5 | v2.0 |
|--------|------|------|
| RTF (M2 Max) | ~0.5 | ~1.3 |
| Load time (.wav) | ~0.3s | ~9s |
| Load time (.npz) | ~0.3s | ~1.5s |

### Local M3 Max Benchmark

These results were measured in `/Users/vanch/mlx-indextts2` with precomputed
speaker `.npz`, `memory_limit=24GB`, `diffusion_steps=16`, and emotion `calm`.
RTF lower is faster.

| Case | fp32 MLX | fp16 MLX | 8bit MLX | optimized PyTorch MPS |
|------|---------:|---------:|---------:|----------------------:|
| zh short | 1.127 | 1.538 | 0.966 | 1.446 |
| zh long | 1.232 | 1.584 | 1.035 | 1.699 |
| en short | 1.157 | 1.462 | 0.914 | 2.192 |
| en long | 1.193 | 1.511 | 0.956 | 1.783 |
| vi short | 1.562 | 1.471 | 0.976 | 2.329 |
| vi long | 1.557 | 1.500 | 0.965 | 1.822 |

8bit is the default local choice because it was fastest in every benchmark case.
The upstream 8bit conversion quantizes GPT only; S2Mel and BigVGAN remain fp32.

## License

MIT License

## Acknowledgments

- [IndexTTS](https://github.com/index-tts/index-tts) - Original PyTorch implementation
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
