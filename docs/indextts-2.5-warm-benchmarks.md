# IndexTTS 2.5 warm MLX benchmarks

These measurements compare resident IndexTTS 2.5 MLX variants on an Apple M5
MacBook Air with 24 GB unified memory, running macOS 26.6.1. They isolate warm
generation: model loading and the first generation are excluded from the timed
result.

## Results

| GPT variant | Diffusion steps | Warm generation | Audio duration | RTF |
| --- | ---: | ---: | ---: | ---: |
| 3-bit | 8 | 7.916 s | 6.304 s | 1.256 |
| 4-bit | 8 | 8.076 s | 6.420 s | 1.258 |
| 4-bit | 16 | 9.603 s | 6.420 s | 1.496 |
| 6-bit | 16 | 21.348 s | 6.780 s | 3.149 |

The 3-bit and 4-bit 8-step runs were effectively tied: 3-bit was 0.160 seconds
faster in wall time, while their RTFs differed by 0.002. Quantizing only the GPT
does not reduce the cost of reference processing, S2Mel diffusion, or BigVGAN.

## Method

- One untimed warmup followed by one timed generation in the same process.
- MLX synchronization immediately before and after the timed generation.
- `seed=42`, `max_mel_tokens=256`, English, and an 8 GB MLX memory limit.
- The same private prompt and voice reference were used for every row. Their
  contents and local paths are intentionally not published.
- MLX reported `Device(gpu, 0)`.

The prompt was held constant, but quantized semantic generation produced
slightly different audio lengths. Compare both wall time and RTF; the rows are
not identical-duration workloads. No MOS, speaker-similarity, or emotion-quality
evaluation was performed, so these results measure latency only.

## Reproduce the measurement contract

The runner keeps the model resident, performs explicit warmups, synchronizes MLX,
and emits JSON without the prompt or reference path:

```bash
uv run python scripts/benchmark_indextts25_warm.py \
  --model /path/to/IndexTTS-2.5-MLX-4bit \
  --reference /path/to/authorized-reference.wav \
  --text 'A representative English benchmark sentence.' \
  --warmups 1 --runs 1 \
  --diffusion-steps 8 \
  --seed 42 --max-mel-tokens 256 \
  --memory-limit-gb 8 \
  --json outputs/warm-q4.json
```

Use an authorized voice reference. The emitted report records only the model
directory name, text length, generation settings, device, and timing metrics.
