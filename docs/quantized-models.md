# IndexTTS 2.5 MLX 4/5/6-bit variants

These variants persistently quantize the GPT transformer linear layers using
MLX group quantization (`group_size=64`). EnhancedCodec, S2Mel/DiT, and
BigVGAN are retained in float16. This is native **MLX** quantization, not an
oMLX `oQ` artifact.

## Reproduce

```bash
uv sync --extra v25 --extra dev
uv run mlx-indextts convert \
  --model-dir /path/to/IndexTTS-2.5-source \
  --output /path/to/IndexTTS-2.5-MLX-5bit \
  --dtype float16 --quantize 5 \
  --source-revision <source-revision>
```

The converter atomically publishes only after strict component mapping and
load checks. Its `model_manifest.json` records source revision, converter
revision, GPT quantization, component coverage, and every artifact size.

## Intelligibility check

Run a controlled prompt through a local Parakeet OpenAI-compatible endpoint:

```bash
uv run python scripts/transcribe_and_score.py \
  --audio output.wav \
  --expected 'Esta prueba mide la claridad de cada modelo cuantizado.'
```

WER is a content-intelligibility sanity check only. It does not establish
speaker similarity, emotional fidelity, naturalness, or safety for production
voice-cloning use.

## Licensing and consent

IndexTTS weights and all quantized derivatives are governed by the upstream
bilibili Model Use License Agreement. Include it with every redistributed
model, preserve notices, comply with downstream obligations and restrictions,
and obtain authorization for every voice reference used in synthesis.
