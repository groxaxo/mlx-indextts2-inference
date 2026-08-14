# Inference Performance Bottleneck Audit

This note records the performance audit started against `main` at
`0b28a63e745d4290ed57d260a4f0b50cb4e01846` on 2026-08-14 and the fixes now
landed on `main`.

## Completed work

### P0 — GPT KV-cache growth — fixed

`GPT2Attention` previously appended every new key/value tensor with
`mx.concatenate()` at every layer and every autoregressive token. For a generated
sequence of length `T`, that repeatedly copied the historical cache and turned
cache maintenance into quadratic memory traffic.

The runtime now keeps a chunked cache, grows capacity only when a chunk is
exhausted, and writes new K/V slices into allocated MLX arrays. The public
`cache[layer][0/1]` indexing contract is preserved for compatibility.

The GPT attention path also uses `mx.fast.scaled_dot_product_attention` when the
installed MLX exposes it, with the explicit matmul/softmax implementation retained
as a compatibility fallback.

### P1 — Full-vocabulary nucleus work after top-k — fixed

The previous sampler applied top-k and then performed a full-vocabulary `argsort`
for nucleus sampling on every token. The new shared sampler selects candidates with
`mx.argpartition`, gathers only the top-k candidate logits, and performs top-p
sorting over that bounded set. With the default `top_k=30`, nucleus sorting is now
bounded to 30 values rather than the complete semantic vocabulary.

The old final `softmax()` followed by `log()` was also removed. MLX categorical
sampling consumes unnormalized logits directly. Token IDs are mapped back from the
candidate set after sampling, so the public result remains a vocabulary ID.

### P1 — Repetition-history reconstruction — fixed for v2.5, reduced for v2.0

The previous repetition penalty rebuilt and sorted a Python `set` from the entire
generated-token history on every autoregressive step.

IndexTTS 2.5 now uses a list-compatible `RepetitionPenaltyState` with an incremental
MLX boolean seen-token mask. It preserves the ordered Python list needed by the
rest of the pipeline while eliminating per-token set construction and sorting.
The v2.0 compatibility path no longer creates or sorts a set, although it still
receives the existing Python history list from the legacy generation loop.

### P1 — IndexTTS 2.5 attention-mask growth — fixed

`IndexTTSv25._generate_semantic_codes()` previously appended one element to its
attention mask with `mx.concatenate()` after every generated token. It now allocates
the maximum generation mask once and passes an active prefix view for each key
length. This removes repeated allocation and graph construction from the decode
loop.

### P1 — IndexTTS 2.5 safe compile hook — fixed

IndexTTS 2.5 now invokes the same best-effort `_compile_hotpaths()` policy already
used by v2.0 after loading S2Mel and BigVGAN. Unsupported MLX versions still fall
back to eager execution. Benchmark cold compile latency separately from warm
inference because the first resident request may include compilation.

## Remaining bottlenecks

### P2 — Per-token host synchronization

Both v2.0 and v2.5 materialize the sampled token through `.item()` for EOS handling.
That synchronizes the host once per generated semantic token. Eliminating this
requires a larger device-side generation-loop design so EOS, position updates, and
history bookkeeping can remain on-device.

### P2 — CFM per-step allocations

The CFM path already hoists invariant CFG tensors. It still creates `stacked_x`,
timestep arrays, and prompt-zeroed tensors inside the Euler loop and forces
`mx.eval(x)` every step. Revisit this after measuring the new GPT decode profile.

### P2 — Raw reference-audio preprocessing

Raw WAV references cross librosa/PyTorch/torchaudio and MLX boundaries. The speaker
`.npz` cache is the production fast path for repeated voices and avoids repeating
W2V-BERT, CAMPPlus, resampling, and tensor-bridge work.

## Validation

Run focused regression and lint checks on Apple Silicon:

```bash
uv run pytest \
  tests/test_gpt2_cache_perf.py \
  tests/test_sampling_perf.py \
  tests/test_generate_v25.py \
  tests/test_models.py \
  -q

uv run ruff check \
  mlx_indextts/models/gpt2.py \
  mlx_indextts/models/gpt_v2.py \
  mlx_indextts/models/sampling.py \
  mlx_indextts/generate_v25.py \
  tests/test_gpt2_cache_perf.py \
  tests/test_sampling_perf.py
```

Run the resident-model benchmark before and after the optimization commits:

```bash
uv run python scripts/benchmark_v20_v25.py \
  --ref-audio /path/to/reference.wav \
  --output-dir outputs/validation/v20-v25-benchmark
```

For detailed phase timing on v2.0, use:

```bash
uv run python scripts/test_benchmark.py --v20-only --mlx-only
```

Record cold load, first-resident inference, warm RTF, GPT generation time, S2Mel
time, and BigVGAN time separately. The KV-cache improvement should scale with
semantic-token count; the sampling improvement should be visible as lower per-token
host and vocabulary-processing overhead.
