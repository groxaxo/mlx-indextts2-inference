# IndexTTS 2.5 on MLX

This document covers the native Apple Silicon implementation of the public
IndexTTS 2.5 release. IndexTTS 2.0 remains supported. IndexTTS 1.5 source code
may still exist for compatibility, but it is no longer maintained or included
in the regression target.

## Evidence baseline

The implementation is pinned to:

- official source repository: `index-tts/index-tts`;
- audited source commit: `9c87c46b84bd0e75ecaefb461e7e8f69bc9ecf44`;
- Hugging Face model: `IndexTeam/IndexTTS-2.5`;
- audited model revision: `d0aa86e75bb6f3437f3831e95056fa72842d89ef`;
- BigVGAN: `nvidia/bigvgan_v2_22khz_80band_256x` at
  `633ff708ed5b74903e86ff1298cf4a98e921c513`.

The technical report mentions a Zipformer S2Mel upgrade, while the published
checkpoint, configuration, model card, and executable inference path use
GPT + DiT/S2Mel + BigVGAN and publish no portable Zipformer weights. This port
follows the released artifacts and does not claim a nonexistent Zipformer
conversion.

## Install

```bash
uv sync --extra v25

# Add only the interfaces you use.
uv sync --extra v25 --extra qwen --extra api --extra webui
```

The 2.5 frontend uses `tiktoken`, `fugashi`, and `unidic-lite`. Core synthesis
runs GPT, EnhancedCodec, S2Mel/DiT, and BigVGAN in MLX. W2V-BERT and CampPlus
remain PyTorch preprocessing dependencies and are loaded lazily only when a raw
reference must be analyzed. A versioned `.npz` speaker cache avoids that work on
later calls.

## Download and convert

Download the exact audited source revision:

```bash
hf download IndexTeam/IndexTTS-2.5 \
  --revision d0aa86e75bb6f3437f3831e95056fa72842d89ef \
  --local-dir models/IndexTTS-2.5-source
```

Convert to the recommended persistent 8-bit model:

```bash
uv run mlx-indextts convert \
  --model-dir models/IndexTTS-2.5-source \
  --output models/mlx-IndexTTS-2.5-8bit \
  --dtype float16 \
  --quantize 8 \
  --source-revision d0aa86e75bb6f3437f3831e95056fa72842d89ef
```

Full fp16 and fp32 outputs use the same converter:

```bash
uv run mlx-indextts convert \
  --model-dir models/IndexTTS-2.5-source \
  --output models/mlx-IndexTTS-2.5 \
  --dtype float16 --quantize fp32

uv run mlx-indextts convert \
  --model-dir models/IndexTTS-2.5-source \
  --output models/mlx-IndexTTS-2.5-fp32 \
  --dtype float32 --quantize fp32
```

Persistent quantization applies to the GPT transformer. EnhancedCodec,
S2Mel/DiT, and BigVGAN retain the selected conversion dtype. Conversion writes
to a sibling staging directory, verifies strict tensor coverage and component
loads, writes a manifest/report, and only then publishes the final directory.
An existing destination is never silently overwritten. `--resume` reuses a
matching partial conversion; `--force` archives the old destination instead of
deleting it.

## Converted model contract

A complete 2.5 directory contains:

```text
config.json
config.yaml
model_manifest.json
conversion_report.json
gpt.safetensors
codec.safetensors
s2mel.safetensors
bigvgan.safetensors
multilingual_zh_ja_yue_char_del.tiktoken
feat1.pt
feat2.pt
wav2vec2bert_stats.pt
```

`model_manifest.json` is authoritative. It records the model/source revisions,
dtype, quantization, tensor coverage, tokenizer vocabulary, languages,
auxiliary resources, semantic frame rate, and speaker-cache schema. A stale
`version` value in an upstream YAML file is not enough to identify a model.

## CLI generation

Non-Vietnamese `--profile auto` requests prefer the local 2.5 8-bit model.
Vietnamese text continues to route to the existing 2.0 Vietnamese profile.
Use an explicit language for English and Spanish because Latin-script auto
detection is inherently ambiguous.

```bash
uv run mlx-indextts generate \
  --profile v25 \
  --ref-audio reference.wav \
  --language en \
  --text "Hello, this is a cross-lingual voice cloning test." \
  --output outputs/hello.wav
```

Supported 2.5 languages are `zh`, `en`, `ja`, `es`, and `ar`.

### Speaker cache

```bash
uv run mlx-indextts speaker \
  --profile v25 \
  --ref-audio reference.wav \
  --output speakers/reference_v25.npz

uv run mlx-indextts generate \
  --profile v25 \
  --ref-audio speakers/reference_v25.npz \
  --language ja \
  --text "こんにちは。" \
  --output outputs/ja.wav
```

2.0 and 2.5 speaker caches are intentionally incompatible. A cache records the
source revision, preprocessing metadata, audio fingerprint, tensor names, and
shapes; passing it to the wrong model fails with a corrective error.

### Pronunciation annotations

Annotations are protected during normalization and long-text segmentation:

```text
他在银<行|XING2>里行走。
He had a <minute|M IH1 . N AH0 T> to think.
<今日|きょう>は良い天気です。
```

Chinese annotations use uppercase Pinyin entries, English annotations use CMU
phonemes, and Japanese replacements must be Hiragana or Katakana.

### Emotion controls

Exactly one explicit emotion source may be active:

```bash
# Separate emotion reference audio
uv run mlx-indextts generate --profile v25 \
  -r speaker.wav --emotion-ref-audio sad.wav --emo-alpha 0.8 \
  --language zh -t "这是一段情感语音。" -o outputs/emotion_ref.wav

# Named or weighted manual vector
uv run mlx-indextts generate --profile v25 \
  -r speaker.wav --emotion "happy:0.8,surprised:0.1" --emo-alpha 0.8 \
  --language zh -t "太好了！" -o outputs/emotion_manual.wav

# Qwen derives emotion from synthesis text
uv run mlx-indextts generate --profile v25 \
  -r speaker.wav --auto-emotion \
  --language zh -t "快躲起来！" -o outputs/emotion_auto.wav

# Qwen derives emotion from a separate description
uv run mlx-indextts generate --profile v25 \
  -r speaker.wav --emotion-text "令人非常害怕和紧张的场景" \
  --language zh -t "快躲起来！" -o outputs/emotion_text.wav
```

With no explicit mode, the speaker reference supplies the emotion, matching
upstream behavior. `--use-random` samples emotion prototypes but may reduce
voice-cloning fidelity. The official emotion order is `happy`, `angry`, `sad`,
`afraid`, `disgusted`, `melancholic`, `surprised`, `calm`.

### Duration and streaming

`--duration-factor` changes the 2.5 S2Mel target length. The existing local
`--target-duration` token budget and opt-in `--fit-duration` post-processing
remain available. Upstream has not enabled its advertised precise duration
control, so this project does not claim model-native exact timing.

```bash
uv run mlx-indextts generate --profile v25 \
  -r speakers/reference_v25.npz --language zh \
  -t "第一段。第二段。第三段。" \
  -o outputs/stream.wav --stream --max-text-tokens 20
```

Streaming returns completed safe text segments while keeping the model loaded.
It is not token-level waveform streaming. Each event reports the segment index,
count, language, sample rate, and whether the request is complete.

## Python API

Use the shared router when an application serves both 2.0 and 2.5:

```python
from mlx_indextts.runtime import GenerateOptions, TTSRuntime

runtime = TTSRuntime()
result = runtime.generate(
    text="Hola, esta es una prueba.",
    ref_audio="speakers/reference_v25.npz",
    output_path="outputs/es.wav",
    profile="v25",
    options=GenerateOptions(language="es", diffusion_steps=25),
)
```

The version-specific class is also public:

```python
from mlx_indextts import IndexTTSv25

tts = IndexTTSv25("models/mlx-IndexTTS-2.5-8bit")
audio = tts.generate(
    text="مرحبا بالعالم.",
    reference_audio="speakers/reference_v25.npz",
    output_path="outputs/ar.wav",
    language="ar",
    emotion=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
)
```

## Batch, API, and WebUI

Batch CSV accepts per-row `language`/`lang`, `ref_audio`,
`emotion_ref_audio`, `emotion`, `emo_alpha`, `emotion_text`/`emo_text`,
duration controls, and mel-token caps:

```bash
uv run mlx-indextts batch \
  --profile v25 --input multilingual.csv \
  --output-dir outputs/multilingual --combine
```

FastAPI endpoints include `/generate`, `/generate/stream`, `/speaker`, `/batch`,
`/plan`, `/health`, `/profiles`, and the restricted `/audio` reader.
`/generate/stream` returns newline-delimited JSON with a base64 WAV for each
completed segment. Output paths must remain under `outputs/`.

```bash
uv sync --extra v25 --extra api
uv run mlx-indextts-api
```

The Gradio UI exposes model/profile, language, pronunciation guidance, speaker
and emotion references, manual/Qwen emotion, separate emotion text, sampling,
duration, normalization, and completed-segment progress:

```bash
uv sync --extra v25 --extra webui
uv run mlx-indextts-webui
```

## Compatibility

| Capability | IndexTTS 2.0 | IndexTTS 2.5 |
| --- | --- | --- |
| Chinese / English | Yes | Yes |
| Japanese / Spanish / Arabic | No | Yes |
| Local Vietnamese profile | Yes | Routed to 2.0 |
| Separate emotion audio | Yes | Yes |
| Manual and Qwen emotion | Yes | Yes |
| Independent Qwen emotion text | Yes | Yes |
| Pronunciation annotations | No | Pinyin / CMU / Kana |
| Versioned speaker cache | Legacy 2.0 schema | Strict 2.5 schema |
| Completed-segment streaming | No | Yes |
| Batch / API / WebUI | Yes | Yes |

## Verification

The repository includes reproducible validation commands:

```bash
uv run python scripts/validate_v25_matrix.py \
  --ref-audio speakers/reference_v25.npz --scope full

uv run --with mlx-whisper python scripts/asr_validate_v25.py \
  --functional-report outputs/validation/indextts25/functional_report.json

uv run python scripts/benchmark_v20_v25.py \
  --ref-audio reference.wav
```

The current local evidence covers strict fp16/8-bit/fp32 loads, numerical
component parity, five-language synthesis and ASR, four cross-lingual targets,
pronunciation annotations, all emotion sources, six-segment streaming, speaker
cache, multilingual batch/combined output, API/WebUI smoke, and matched 2.0/2.5
performance. ASR and cosine similarity are engineering sanity checks, not a
substitute for human listening.

## License, consent, and known limitations

Repository code uses the license in this repository. Official IndexTTS weights
and converted or quantized derivatives remain governed by the upstream
**bilibili Model Use License Agreement**, not merely this repository's code
license. Keep the upstream copyright notice and license with every model copy,
and review its commercial-use thresholds and downstream obligations.

Only clone voices you own or have explicit permission to use. The model does
not verify identity or consent. Do not use it for impersonation, deception,
privacy infringement, or unlawful/high-risk deployment.

Known constraints:

- English and Spanish auto detection is ambiguous; select `--language`.
- NeMo text normalization is optional and deliberately falls back to raw text
  when its grammar/runtime is unavailable, matching upstream failure behavior.
- Streaming is completed-segment streaming, not codec-token streaming.
- The paper's 2.28× claim has not been reproduced here and is not claimed.
- Generated speech still requires listening review for production use.
