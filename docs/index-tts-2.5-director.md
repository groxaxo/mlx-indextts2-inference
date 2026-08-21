# IndexTTS 2.5 sentence director and RTX 3090 quality path

The sentence director adapts the exact-source and every-sentence invariant from
`xai-sentence-tagger` to native IndexTTS 2.5 controls. It never inserts
unsupported stage directions into the spoken transcript.

## Contract

For every successful plan:

- every source sentence has exactly one control row;
- source offsets and characters reconstruct the original input exactly;
- missing, duplicate, unknown, malformed, or out-of-range LLM rows are audited
  and repaired deterministically, unless `--no-fallback` is selected;
- model-supplied rewrites are ignored;
- the eight emotion values are normalized to a total mass of `0.8`;
- `alpha`, speed, and pauses are bounded;
- a final N/N audit runs after continuity smoothing and before synthesis.

The external LLM returns metadata only:

```json
{
  "sentences": [
    {
      "index": 0,
      "emotion": "warm calm",
      "emotion_vector": {
        "happy": 0.10,
        "angry": 0.00,
        "sad": 0.00,
        "afraid": 0.00,
        "disgusted": 0.00,
        "melancholic": 0.00,
        "surprised": 0.00,
        "calm": 0.70
      },
      "alpha": 0.44,
      "speed": 1.00,
      "pause_after_ms": 150
    }
  ]
}
```

The runtime sends the original source slice to IndexTTS and maps controls to
native `emo_vector`, `emo_alpha`, `duration_factor = 1 / speed`, and external
silence. It sets `use_random=false`. Do not send `[laughs]`, `[whispers]`, SSML,
or the human-readable `[SEG]` intermediate format to IndexTTS.

## Print the master system prompt

```bash
uv run --project nvidia mlx-indextts-director prompt
```

Save it for another LLM or orchestration system:

```bash
uv run --project nvidia mlx-indextts-director prompt \
  --output prompts/indextts25-director.txt
```

## Create a direction plan

Any OpenAI-compatible `/v1/chat/completions` endpoint can be used:

```bash
export INDEXTTS_DIRECTOR_BASE_URL=http://127.0.0.1:12434/v1
export INDEXTTS_DIRECTOR_API_KEY=not-needed
export INDEXTTS_DIRECTOR_MODEL=your-local-model

uv run --project nvidia mlx-indextts-director tag \
  --language en \
  --input script.txt \
  --output outputs/script.plan.json
```

Use `--format markup` for a human-readable operator view. Use
`--heuristic-only` for deterministic offline direction. Add `--no-fallback` when
a malformed or incomplete LLM response must fail the job instead of being
repaired.

## Recommended RTX 3090 quality baseline

```bash
uv run --project nvidia mlx-indextts-director generate \
  --model-dir checkpoints \
  --device cuda:0 \
  --precision bf16 \
  --quality-preset natural-hq \
  --cuda-profile quality \
  --no-cuda-kernel \
  --ref-audio voices/identity.wav \
  --language en \
  --input script.txt \
  --plan-output outputs/script.plan.json \
  --output outputs/script.wav
```

`natural-hq` uses:

| Control | Value |
| --- | ---: |
| precision | BF16 |
| temperature | `0.72` |
| top-p | `0.80` |
| top-k | `30` |
| repetition penalty | `10.0` |
| maximum text tokens | `90` |
| maximum mel tokens | `1500` |
| random emotion exemplar | disabled |
| QwenEmotion helper | unloaded |
| TF32 substitutions | disabled by the `quality` profile |
| cuDNN shape benchmarking | disabled by the `quality` profile |

This is a conservative engineering baseline, not a universal perceptual
optimum. Blind-test the actual reference voice, language, and content.
`studio-stable` lowers sampling variance; `expressive-hq` permits more variation.
FP32 is available for maximum numerical fidelity, but it consumes materially
more VRAM and should not be claimed audibly superior without a blind A/B result.

## Reference audio

Reference quality usually matters more than small sampling adjustments. Use one
canonical identity sample per voice:

- approximately 8–12 seconds; the official runtime clips at 15 seconds;
- one speaker, dry close microphone, little room reflection;
- no music, clipping, fan noise, denoiser warble, or aggressive compression;
- natural conversational pitch and enough phonetic variety;
- reuse the same path across adjacent chunks so the official speaker cache is
  reused.

## Semantic chunking

Generating every sentence separately can create audible resets; generating a
whole chapter prevents local control changes and increases drift risk. The
runtime conservatively coalesces adjacent rows only when emotion-vector
distance, alpha, speed, paragraph structure, and size limits are compatible.
Defaults are at most three sentences and 320 characters per synthesis call.

Use `--no-coalesce` for diagnostics and `--keep-segments` to retain component
WAV files for an audit.

## Optimization ladder

Establish the quality baseline first. Measure warm real-time factor, peak VRAM,
failure rate, and output equivalence while changing one option at a time:

1. `--cuda-kernel` — optional BigVGAN fused CUDA path.
2. `--torch-compile` — potentially useful after warm-up; shape-sensitive.
3. `--accel` — official GPT acceleration extra.
4. `--deepspeed` — benchmark independently; one model already fits one 3090.
5. `--cuda-profile balanced` — allows TF32 for eligible FP32 matrix work but
   leaves cuDNN shape benchmarking off.
6. `--cuda-profile throughput` — also enables cuDNN shape benchmarking.

Do not enable every acceleration switch at once and then assume quality or speed
improved. Keep only changes that win repeated warm benchmarks and pass audio
comparison.

## Three RTX 3090s

Do not tensor-parallelize a normal utterance merely because three cards exist.
IndexTTS 2.5 fits on one 24 GiB card, and cross-GPU synchronization does not
improve voice quality. For production throughput, use one persistent worker and
one warm model per GPU, then distribute independent jobs:

```text
GPU 0 -> jobs 0, 3, 6, ...
GPU 1 -> jobs 1, 4, 7, ...
GPU 2 -> jobs 2, 5, 8, ...
```

Keep the lightweight sentence-director LLM outside those TTS workers when
possible. Because the director supplies native vectors, each worker keeps the
official Qwen emotion helper unloaded.

## Python API

```python
from mlx_indextts.director import IndexTTSDirector, OpenAICompatibleDirector
from mlx_indextts.directed_runtime import synthesize_direction_plan
from mlx_indextts.nvidia_runtime import NvidiaIndexTTS, NvidiaRuntimeConfig

annotator = OpenAICompatibleDirector(
    base_url="http://127.0.0.1:12434/v1",
    api_key="not-needed",
    model="your-local-model",
)
plan = IndexTTSDirector(annotator).direct(
    "I wasn't sure you'd come. I'm really glad you did.",
    language="en",
    style_prompt="Warm, restrained, natural conversation.",
)

runtime = NvidiaIndexTTS(
    NvidiaRuntimeConfig(
        model_dir="checkpoints",
        version="2.5",
        device="cuda:0",
        precision="bf16",
        use_qwen_emotion=False,
    )
)
try:
    result = synthesize_direction_plan(
        runtime,
        plan,
        ref_audio="voices/identity.wav",
        output_path="outputs/directed.wav",
        language="en",
        preset="natural-hq",
        cuda_profile="quality",
    )
finally:
    runtime.close()

print(result.to_json())
```

## Validation

The deterministic tests cover exact source reconstruction, abbreviations,
pronunciation annotations, URLs, decimals, N/N repair, duplicate and unknown
rows, strict no-fallback behavior, ignored model rewrites, control bounds,
continuity smoothing, paragraph boundaries, semantic chunking, native request
mapping, deterministic seed progression, WAV concatenation, and CUDA policy.

```bash
PYTHONPATH=. pytest -q tests/test_director.py tests/test_directed_runtime.py
PYTHONPATH=. python -m compileall -q mlx_indextts tests
```
