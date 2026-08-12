# MLX-IndexTTS

Native Apple Silicon MLX inference for IndexTTS 2.0 and IndexTTS 2.5, with
zero-shot voice cloning, disentangled emotion control, multilingual generation,
batch/API/WebUI entrypoints, and persistent quantized conversion.

## Features

- Native MLX GPT, codec, S2Mel/DiT, and BigVGAN inference for IndexTTS 2.5
- IndexTTS 2.0 standard and local Vietnamese profiles remain supported
- Chinese, English, Japanese, Spanish, and Arabic synthesis in 2.5
- Cross-lingual voice transfer and version-safe speaker caches
- Separate audio, manual eight-value, and Qwen text emotion controls
- Pinyin, CMU phoneme, and Kana pronunciation annotations
- Completed-segment streaming, batch generation, FastAPI, and Gradio
- fp32, fp16, and persistent 4-, 5-, 6-, or 8-bit GPT model conversion

IndexTTS 1.5 files may remain for compatibility, but 1.5 is no longer maintained
or part of the regression target. See [the complete 2.5 guide](docs/indextts-2.5.md)
and [the executed validation record](docs/indextts-2.5-validation.md).

## Official Feature Parity

Checked against the pinned public IndexTTS 2.5 source/model revisions and the
preserved 2.0 capability set.

| Official capability | Local MLX status | Notes |
| --- | --- | --- |
| 2.0 / 2.5 conversion | Implemented | Strict 2.5 manifests plus preserved 2.0 conversion. |
| Zero-shot speaker cloning | Implemented | Raw WAV refs and version-safe `.npz` caches. |
| Five-language / cross-lingual 2.5 | Implemented | `zh`, `en`, `ja`, `es`, `ar`; explicit `en`/`es` is recommended. |
| Speaker / emotion disentanglement | Implemented | Speaker, emotion audio, manual vector, and Qwen modes are mutually exclusive. |
| Qwen emotion text | Implemented | Supports synthesis text or independent `--emotion-text`. |
| Pronunciation annotations | Implemented | Chinese Pinyin, English CMU, Japanese Kana. |
| Duration controls | Implemented with caveat | 2.5 `duration_factor`, token budgets, and explicit pitch-preserving fitting; no unsupported upstream precision claim. |
| Batch / API / WebUI | Implemented | Per-row language/references, combined WAV, REST, and Gradio. |
| Streaming | Implemented | Completed safe text segments; not token-level waveform streaming. |
| Vietnamese profile | Preserved local extension | Continues on the 2.0 Vietnamese checkpoint. |

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
git clone https://github.com/vanch007/mlx-indextts2.git
cd mlx-indextts2

# IndexTTS 2.5 generation
uv sync --extra v25

# Complete local interfaces
uv sync --extra v25 --extra qwen --extra api --extra webui
```

## Quick Start

### Local 8-bit Defaults

This checkout has been tuned for Apple Silicon IndexTTS2 migration work on an
M3 Max Mac. If `--model` is omitted, the CLI uses these local models:

- IndexTTS 2.5 / non-Vietnamese: `models/mlx-IndexTTS-2.5-8bit`
- preserved IndexTTS 2.0 standard: `models/mlx-indexTTS2-standard-8bit`
- local Vietnamese 2.0 profile: `models/mlx-indexTTS2-vietnamese-8bit`

Vietnamese is auto-selected when the input text contains Vietnamese tone marks.
You can override the default paths with environment variables:

```bash
export MLX_INDEXTTS_STANDARD_MODEL=/path/to/standard-8bit
export MLX_INDEXTTS_VIETNAMESE_MODEL=/path/to/vietnamese-8bit
export MLX_INDEXTTS_V25_MODEL=/path/to/mlx-IndexTTS-2.5-8bit
```

### 1. Convert Model

```bash
# Download the pinned official 2.5 snapshot
hf download IndexTeam/IndexTTS-2.5 \
    --revision d0aa86e75bb6f3437f3831e95056fa72842d89ef \
    --local-dir models/IndexTTS-2.5-source

# Convert IndexTTS 2.5 to the recommended persistent 8-bit model
uv run mlx-indextts convert \
    --model-dir models/IndexTTS-2.5-source \
    --output models/mlx-IndexTTS-2.5-8bit \
    --dtype float16 --quantize 8 \
    --source-revision d0aa86e75bb6f3437f3831e95056fa72842d89ef

# Existing IndexTTS 2.0 conversion remains available
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-2 \
    -o models/mlx-indexTTS-2.0
```

The 2.5 converter also supports full fp16/fp32 and persistent 4-, 5-, 6-, or
8-bit GPT output, strict
tensor coverage, resumable staging, and atomic publication. See the
[2.5 conversion guide](docs/indextts-2.5.md#download-and-convert).

For persistent 4-, 5-, 6-, and 8-bit GPT variants, benchmark reproduction,
and the Parakeet WER sanity-check method, see
[the quantized-model guide](docs/quantized-models.md).

### 2. Generate Speech

```bash
# IndexTTS 2.5 multilingual voice cloning
uv run mlx-indextts generate \
    --profile v25 \
    -r reference.wav \
    --language en \
    -t "Hello, this is a cross-lingual voice cloning test." \
    -o output_v25.wav

# Japanese, Spanish, and Arabic use the same 2.5 model
uv run mlx-indextts generate --profile v25 \
    -r reference.wav --language ja \
    -t "こんにちは、これは日本語の音声合成テストです。" \
    -o output_ja.wav

# Preserved IndexTTS 2.0 standard profile
uv run mlx-indextts generate \
    --profile standard \
    -r reference.wav \
    -t "你好，这是一个语音合成测试。" \
    -o output_v20.wav

# 2.0 and 2.5 share named/mixed emotion controls
uv run mlx-indextts generate \
    --profile v25 --language zh \
    -r reference.wav \
    -t "今天真是太开心了！" \
    -o output.wav \
    --emotion happy --emo-alpha 0.6

# Qwen can analyze the synthesis text or an independent emotion description
uv sync --extra qwen
uv run python scripts/convert_qwen_emotion_mlx.py
uv run mlx-indextts generate \
    --profile v25 --language zh \
    -r speaker_v25.npz \
    -t "我今天很开心，终于见到你了。" \
    -o output_qwen.wav \
    --emotion-text "一种终于重逢后的快乐和激动"

# Vietnamese uses the local Vietnamese 8bit model automatically
uv run mlx-indextts generate \
    -r speaker_vietnamese.npz \
    -t "Đêm nay gió rất nhẹ, ánh đèn ngoài cửa sổ chậm rãi sáng lên." \
    -o output_vi.wav

# Duration-oriented generation remains available for both versions
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
# v2.0
uv run mlx-indextts speaker \
    --profile standard \
    -r reference.wav \
    -o speaker_v20.npz

# v2.5
uv run mlx-indextts speaker \
    --profile v25 \
    -r reference.wav \
    -o speaker_v25.npz

# Use pre-computed 2.5 speaker conditioning
uv run mlx-indextts generate \
    --profile v25 -r speaker_v25.npz --language zh \
    -t "你好，世界！" \
    -o output.wav
```

**Note**: 2.0 and 2.5 speaker files are incompatible. The 2.5 cache embeds its
schema, source revision, preprocessing metadata, tensor shapes, and audio hash.

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

For subtitle-aligned generation, CSV rows may set `target_duration` /
`target_duration_s`, `fit_duration`, and `max_tokens` / `max_mel_tokens`.
These controls stay on the model-resident batch path, avoiding one IndexTTS2
restart per subtitle.

`target_duration` does not reduce the GPT generation cap below the default
320-token content-safety floor. This lets short cross-language subtitle lines
reach their natural EOS instead of losing opening or trailing words; exact
alignment remains the responsibility of opt-in `fit_duration` and its stretch
guard. An explicit smaller `max_tokens` value remains a hard caller limit.

## Python API

```python
# v2.0
from mlx_indextts import IndexTTSv2, IndexTTSv25

tts = IndexTTSv2("models/mlx-indexTTS-2.0")
audio = tts.generate(
    text="你好",
    reference_audio="reference.wav",
    output_path="output.wav",
    emotion="happy",
    emo_alpha=0.6,
)

# v2.5
tts25 = IndexTTSv25("models/mlx-IndexTTS-2.5-8bit")
audio = tts25.generate(
    text="Hola, esta es una prueba.",
    reference_audio="speaker_v25.npz",
    output_path="output_es.wav",
    language="es",
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
  --profile          auto / v25 / 2.5 / standard / vietnamese / vi
  --language         auto / zh / en / ja / es / ar (2.5)
  --max-tokens       Max semantic tokens (default: 1500 for v2.0/v2.5)
  --temperature      Sampling temperature (default: 0.8 for v2.0/v2.5)
  --seed, -s         Random seed for reproducibility
  -v, --verbose      Verbose output
  -p, --play         Play audio after generation
  --quantize, -q     Runtime quantization: 4, 5, 6, 8, or fp32

v2.0 and v2.5:
  --emotion          Emotion: happy/sad/angry/afraid/disgusted/melancholic/surprised/calm/auto-qwen
  --auto-emotion     Use the MLX-native Qwen text emotion model before TTS
  --emotion-text     Derive emotion from an independent text description
  --use-random       Randomly select emotion prototypes
  --emotion-ref-audio
                     Separate emotion reference audio or compatible speaker cache
  --qwen-emotion-model
                     Converted Qwen emotion model path (default: models/qwen0.6bemo4-merge-mlx-8bit)
  --qwen-unload-after / --no-qwen-unload-after
                     Release Qwen after emotion analysis (default: true)
  --emo-alpha        Emotion intensity 0.0-1.0 (default: 0.6, recommend ≤ 0.8)
  --diffusion-steps  Diffusion steps (default: 16)
  --cfg-rate         CFG rate (default: 0.7)

v2.5:
  --text-normalization / --no-text-normalization
  --duration-factor  S2Mel target-length multiplier
  --use-gpt-latent   Enable the optional upstream GPT latent branch
  --stream           Yield completed safe text segments and assemble the WAV
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
- `POST /generate/stream` (newline-delimited completed-segment WAV events)
- `POST /speaker`
- `POST /batch`
- `POST /plan`
- `GET /audio?path=outputs/api_output.wav`

Example:

```bash
curl -X POST http://127.0.0.1:7862/generate \
  -H 'content-type: application/json' \
  -d '{
    "text": "Hola, esta es una prueba.",
    "ref_audio": "speakers/reference_v25.npz",
    "profile": "v25",
    "language": "es",
    "output_path": "outputs/api_es.wav",
    "diffusion_steps": 25
  }'
```

### WebUI

Install WebUI extras and start the lightweight Gradio interface:

```bash
uv sync --extra webui
uv run mlx-indextts-webui
```

The WebUI uses the same single-model cache and 8-bit defaults as the CLI/API.
It exposes 2.5 language selection, pronunciation guidance, all emotion modes,
normalization/duration controls, and completed-segment progress. Only one 2.0
or 2.5 model is kept loaded at a time.

## Version Comparison

| Feature | v2.0 | v2.5 |
|---------|------|------|
| Sample rate | 22050 Hz | 22050 Hz |
| Languages | Chinese, English; local Vietnamese profile | Chinese, English, Japanese, Spanish, Arabic |
| Cross-lingual transfer | Chinese / English | Chinese reference → en / ja / es / ar validated |
| Emotion control | 8 emotions + audio + Qwen | 8 emotions + audio + Qwen + independent emotion text |
| Pronunciation annotations | No | Pinyin / CMU / Kana |
| Completed-segment streaming | No | Yes |
| Persistent GPT quantization | Yes | Yes |
| Speaker pre-compute | Legacy cache | Revision/schema-safe cache |

## Supported Emotions (v2.0 / v2.5)

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

Matched local M3 Max validation used the same raw Chinese reference, the same
Chinese/English texts, persistent 8-bit GPT models, 16 diffusion steps, and
version-specific speaker caches:

| Metric | v2.0 | v2.5 |
| --- | ---: | ---: |
| Cold model load | 1.484s | 0.226s |
| Raw reference preprocessing | 10.837s | 4.293s |
| Warm Chinese RTF | 1.388 | 1.043 |
| Warm English RTF | 1.840 | 1.074 |
| Mean warm RTF | 1.614 | 1.059 |

Both versions passed matched ASR sanity checks. The 2.5 five-language default
25-step validation produced RTF 1.70–2.19; English, Japanese, Spanish, and
Arabic had zero ASR edit errors on the selected short cases, while Chinese CER
was 0.227 due mainly to product-name/number spelling. These are environment- and
sample-specific engineering results, not a claim of the paper's 2.28× speedup.

### Historical 2.0 Quantization Benchmark

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

8-bit remains the default local choice. Persistent 2.5 quantization applies to
GPT; EnhancedCodec, S2Mel/DiT, and BigVGAN retain the selected conversion dtype.

Reproducible reports are written by `scripts/validate_v25_matrix.py`,
`scripts/asr_validate_v25.py`, `scripts/similarity_validate_v25.py`, and
`scripts/benchmark_v20_v25.py` under `outputs/validation/`.

## License and Responsible Use

Repository code uses the license in this repository. Official IndexTTS weights
and converted/quantized derivatives are governed by the upstream
**bilibili Model Use License Agreement**, including its commercial thresholds,
downstream notice obligations, and use restrictions. Keep the upstream license
with every model copy.

Only clone voices you own or have explicit permission to use. The model does
not verify identity or consent; do not use it for impersonation, deception,
privacy infringement, unlawful content, or prohibited high-risk deployment.

## Acknowledgments

- [IndexTTS](https://github.com/index-tts/index-tts) - Original PyTorch implementation
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
