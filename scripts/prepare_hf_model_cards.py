#!/usr/bin/env python3
from pathlib import Path


MODELS = [
    {
        "dir": "models/mlx-indexTTS2-standard-fp32",
        "repo": "vanch007/mlx-indextts2-standard-fp32",
        "profile": "standard",
        "precision": "fp32",
        "size": "4.1GB",
        "source": "Local standard IndexTTS2 checkpoint converted before PyTorch project cleanup.",
        "note": "Baseline standard IndexTTS2 MLX conversion.",
    },
    {
        "dir": "models/mlx-indexTTS2-standard-fp16",
        "repo": "vanch007/mlx-indextts2-standard-fp16",
        "profile": "standard",
        "precision": "fp16",
        "size": "2.0GB",
        "source": "Derived from local standard MLX conversion before PyTorch project cleanup.",
        "note": "All floating MLX weights cast to fp16 from the standard fp32 conversion.",
    },
    {
        "dir": "models/mlx-indexTTS2-standard-8bit",
        "repo": "vanch007/mlx-indextts2-standard-8bit",
        "profile": "standard",
        "precision": "8bit",
        "size": "2.8GB",
        "source": "Derived from local standard MLX conversion before PyTorch project cleanup.",
        "note": "Upstream MLX GPT-only 8-bit quantization; S2Mel and BigVGAN remain fp32.",
    },
    {
        "dir": "models/mlx-indexTTS2-vietnamese-fp32",
        "repo": "vanch007/mlx-indextts2-vietnamese-fp32",
        "profile": "vietnamese",
        "precision": "fp32",
        "size": "2.4GB",
        "source": "Local Vietnamese IndexTTS2 checkpoint converted before PyTorch project cleanup.",
        "note": "Vietnamese IndexTTS2 MLX conversion. The source GPT checkpoint was already fp16.",
    },
    {
        "dir": "models/mlx-indexTTS2-vietnamese-fp16",
        "repo": "vanch007/mlx-indextts2-vietnamese-fp16",
        "profile": "vietnamese",
        "precision": "fp16",
        "size": "2.0GB",
        "source": "Derived from local Vietnamese MLX conversion before PyTorch project cleanup.",
        "note": "Vietnamese model with S2Mel, BigVGAN, and vq2emb also cast to fp16.",
    },
    {
        "dir": "models/mlx-indexTTS2-vietnamese-8bit",
        "repo": "vanch007/mlx-indextts2-vietnamese-8bit",
        "profile": "vietnamese",
        "precision": "8bit",
        "size": "2.0GB",
        "source": "Derived from local Vietnamese MLX conversion before PyTorch project cleanup.",
        "note": "Vietnamese model with upstream MLX GPT-only 8-bit quantization.",
    },
]


BENCHMARKS = {
    "standard": [
        ("zh short", "1.127", "1.538", "0.966", "1.446"),
        ("zh long", "1.232", "1.584", "1.035", "1.699"),
        ("en short", "1.157", "1.462", "0.914", "2.192"),
        ("en long", "1.193", "1.511", "0.956", "1.783"),
    ],
    "vietnamese": [
        ("vi short", "1.562", "1.471", "0.976", "2.329"),
        ("vi long", "1.557", "1.500", "0.965", "1.822"),
    ],
}


ASR_NOTES = {
    "standard": (
        "ASR validation with local `mlx_whisper` + `whisper-large-v3-turbo` found no empty audio, "
        "wrong-language output, or obvious missing sentences. Chinese long-form ASR showed a minor "
        "`她/他` homophone difference; English long-form 8-bit ASR showed a minor tense difference."
    ),
    "vietnamese": (
        "ASR validation with local `mlx_whisper` + `whisper-large-v3-turbo` found no empty audio, "
        "wrong-language output, or obvious missing sentences. Vietnamese long-form ASR still showed "
        "minor tone/word-ending differences, so subjective listening is recommended for production use."
    ),
}


def benchmark_table(profile: str) -> str:
    rows = [
        "| Case | fp32 MLX RTF | fp16 MLX RTF | 8bit MLX RTF | PyTorch MPS RTF |",
        "|---|---:|---:|---:|---:|",
    ]
    for case, fp32, fp16, q8, torch_mps in BENCHMARKS[profile]:
        rows.append(f"| {case} | {fp32} | {fp16} | {q8} | {torch_mps} |")
    return "\n".join(rows)


def precision_details(precision: str) -> str:
    if precision == "fp32":
        return "Converted with `mlx-indextts convert --quantize fp32`. Some source tensors may already be lower precision depending on the original checkpoint."
    if precision == "fp16":
        return "Derived locally by casting floating MLX safetensors to `float16`; this is not an upstream CLI quantization mode."
    return "Converted with `mlx-indextts convert --quantize 8`. In the current upstream implementation this quantizes GPT only; S2Mel and BigVGAN stay fp32."


def card(model: dict) -> str:
    repo_name = model["repo"].split("/", 1)[1]
    title = repo_name
    profile_label = "Vietnamese" if model["profile"] == "vietnamese" else "Standard multilingual"
    lang_tags = "vi\n- text-to-speech\n- apple-silicon\n- mlx" if model["profile"] == "vietnamese" else "zh\n- en\n- text-to-speech\n- apple-silicon\n- mlx"
    return f"""---
library_name: mlx
pipeline_tag: text-to-speech
tags:
- indextts2
- mlx-indextts
- voice-cloning
- {model['precision']}
- {lang_tags}
license: mit
---

# {title}

This is a converted MLX IndexTTS2 model for Apple Silicon inference with [`solar2ain/mlx-indextts`](https://github.com/solar2ain/mlx-indextts).

It was prepared during the local IndexTTS2 Apple Silicon optimization work, where the goal was stable Vietnamese and multilingual TTS on an M3 Max Mac without PyTorch MPS memory crashes.

## Variant

- Profile: **{profile_label}**
- Precision / quantization: **{model['precision']}**
- Approx local size: **{model['size']}**
- Source checkpoint directory during conversion: `{model['source']}`
- Note: {model['note']}
- Conversion detail: {precision_details(model['precision'])}

## Expected Files

The repository root is a ready-to-use MLX IndexTTS2 model directory:

- `gpt.safetensors`
- `s2mel.safetensors`
- `bigvgan.safetensors`
- `vq2emb.safetensors`
- `tokenizer.model`
- `config.yaml`
- `config.json`
- `feat1.pt`
- `feat2.pt`
- `wav2vec2bert_stats.pt`

## Usage

Install and use `mlx-indextts`:

```bash
git clone https://github.com/solar2ain/mlx-indextts.git
cd mlx-indextts
uv sync --extra convert --extra v2

huggingface-cli download {model['repo']} \\
  --local-dir models/{repo_name} \\
  --local-dir-use-symlinks False

uv run mlx-indextts generate \\
  -m models/{repo_name} \\
  -r /path/to/reference_or_speaker.npz \\
  -t \"Your text here\" \\
  -o output.wav \\
  --memory-limit 24 \\
  --diffusion-steps 16
```

For repeated generation, precompute speaker conditioning first:

```bash
uv run mlx-indextts speaker \\
  -m models/{repo_name} \\
  -r /path/to/reference.wav \\
  -o speaker.npz \\
  --memory-limit 24
```

## Benchmark

Benchmarked on a 128GB unified-memory M3 Max Mac using:

- `mlx-indextts` from `solar2ain/mlx-indextts`
- precomputed `.npz` speaker conditioning
- `memory_limit=24GB`
- `diffusion_steps=16`
- emotion=`calm`, `emo_alpha=0.6`
- same text set across fp32 / fp16 / 8bit / optimized PyTorch MPS

RTF lower is faster:

{benchmark_table(model['profile'])}

Summary from the local comparison:

- 8bit was the fastest MLX route in this test set.
- fp16 saved space but was slower than fp32 for the standard profile.
- Vietnamese fp16 was slightly faster than Vietnamese fp32, but Vietnamese 8bit was fastest.

## ASR Validation

{ASR_NOTES[model['profile']]}

ASR was used only as an automated sanity check. Final production selection should still include human listening, especially for long-form Vietnamese narration.

## Provenance and Scope

This is an MLX conversion for local Apple Silicon inference, not the original PyTorch release. The original implementation and model family are associated with IndexTTS / IndexTTS2; the MLX runtime used here is `solar2ain/mlx-indextts`.

The benchmark numbers are environment-specific and should be treated as local M3 Max results, not universal performance guarantees.
"""


def main() -> None:
    for model in MODELS:
        path = Path(model["dir"]) / "README.md"
        path.write_text(card(model), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
