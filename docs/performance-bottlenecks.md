# Inference Performance Bottleneck Audit

This audit started against `main` at
`0b28a63e745d4290ed57d260a4f0b50cb4e01846` on 2026-08-14. The sections below
track the fixes subsequently landed on `main` and the boundary of the remaining
architectural work.

## Completed work

### P0 — GPT KV-cache growth

`GPT2Attention` previously appended every new key/value tensor with
`mx.concatenate()` at every layer and autoregressive token. The runtime now keeps
a chunked cache, grows capacity only when a chunk is exhausted, and writes new
K/V slices into allocated arrays. The historical `cache[layer][0/1]` contract is
preserved.

GPT attention also uses `mx.fast.scaled_dot_product_attention` when available,
with the explicit matmul/softmax implementation retained as a compatibility
fallback.

### P1 — Full-vocabulary nucleus work after top-k

Sampling now uses `mx.argpartition` to gather only the top-k candidates before
top-p sorting. With the default `top_k=30`, nucleus sorting operates on 30 values
rather than the complete semantic vocabulary.

The redundant final `softmax()` followed by `log()` was also removed because MLX
categorical sampling accepts unnormalized logits.

### P1 — Repetition-history reconstruction

IndexTTS 2.5 already used a list-compatible `RepetitionPenaltyState` with an
incremental MLX seen-token mask. The legacy v1.5/v2.0 list contract is now mirrored
through a bounded append-only state cache, so all GPT runtimes avoid rebuilding and
sorting the complete Python history on each token.

The cache is model-scoped through weak references and bounded LRU entries. It does
not add arrays to the model parameter tree or retain unbounded request history.

### P1 — Decode token/cache materialization

Each GPT decode step now submits the sampled token and KV-cache tree together with
`mx.async_eval` when supported. The caller still performs one scalar EOS inspection,
but the cache is materialized in the same graph submission instead of requiring a
separate evaluation path. `MLX_INDEXTTS_ASYNC_EVAL=0` restores blocking evaluation.

IndexTTS 2.5 also special-cases the normal one-query cached attention mask. Causality
adds no exclusions when the only query is the final key position, so the runtime
returns the broadcastable padding mask directly rather than constructing per-token
position arrays and comparisons.

### P1 — IndexTTS 2.5 mask growth and compile parity

The 2.5 semantic decoder allocates its maximum generation mask once and passes an
active prefix for each key length instead of concatenating a new mask element after
every token.

IndexTTS 2.5 also invokes the same best-effort S2Mel projection and BigVGAN compile
policy used by v2.0.

### P1 — S2Mel attention and diffusion loop

S2Mel DiT now uses fused scaled-dot-product attention when supported and keeps its
valid-key mask in broadcastable `(B, 1, 1, T)` form rather than materializing an
`(B, 1, T, T)` matrix on every diffusion step.

CFM now:

- lazily compiles the DiT estimator, with shapeless and eager fallbacks;
- uses broadcast/reshape views for duplicated CFG state;
- hoists prompt, style, content, and length invariants;
- uses a broadcast state mask instead of rebuilding the zero prompt prefix;
- submits Euler states asynchronously to bound the lazy graph without a host
  barrier after every step.

Set `MLX_INDEXTTS_COMPILE_CFM=0` to disable estimator compilation.

### P1 — BigVGAN layout churn

BigVGAN previously moved between NCL and NLC around virtually every convolution
inside every residual block and anti-aliased activation. Its public input/output
contract remains NCL, but the full internal vocoder hot path now stays NLC, matching
MLX convolution kernels. Only the initial mel input and final waveform output are
transposed.

Snake/SnakeBeta also replace generic `power(x, 2)` calls with direct multiplication,
and the anti-alias up/down samplers support either NCL or NLC while preserving the
existing parameter names and generated Kaiser filters.

### P2 — Reference-audio preprocessing

Raw WAV references still cross librosa, PyTorch/torchaudio, and MLX while extracting
W2V-BERT, CAMPPlus, mel, and prompt features. Repeated voices should use the existing
speaker `.npz` cache. Within one resident runtime, canonical reference features and
their evaluated MLX forms are already reused.

Automatically persisting every arbitrary raw reference is deliberately not enabled:
API uploads may be temporary or sensitive, and an implicit disk cache would require
an explicit retention, location, invalidation, and privacy contract.

## Remaining architectural boundary

### One scalar EOS synchronization per semantic token

Autoregressive generation still needs to inspect the sampled token to stop exactly
at EOS. The runtime now co-submits token and cache, so any subsequent cache check
reuses that materialization. Fully removing the scalar synchronization requires moving
the entire variable-length generation loop—including EOS control flow, position
updates, and repetition state—into a device-side loop primitive.

Generating a fixed `max_mel_tokens` sequence and trimming after EOS was rejected:
it wastes the largest part of inference for short utterances and changes runtime
behavior under memory limits.

## Validation

Run the focused suite on an MLX-capable host:

```bash
uv run pytest \
  tests/test_gpt2_cache_perf.py \
  tests/test_sampling_perf.py \
  tests/test_s2mel_perf.py \
  tests/test_remaining_perf.py \
  tests/test_gpt_v25.py \
  tests/test_generate_v25.py \
  tests/test_models.py \
  -q
```

Run local lint without relying on remote automation:

```bash
uv run ruff check \
  mlx_indextts/performance.py \
  mlx_indextts/models/sampling.py \
  mlx_indextts/models/gpt.py \
  mlx_indextts/models/gpt_v2.py \
  mlx_indextts/models/gpt_v25.py \
  mlx_indextts/models/s2mel/cfm.py \
  mlx_indextts/models/activations.py \
  mlx_indextts/models/bigvgan_v2.py \
  tests/test_remaining_perf.py
```

Benchmark cold load, first-resident inference, and warm inference separately:

```bash
uv run python scripts/benchmark_v20_v25.py \
  --ref-audio /path/to/reference.wav \
  --output-dir outputs/validation/v20-v25-benchmark
```

For v2.0 phase timing:

```bash
uv run python scripts/test_benchmark.py --v20-only --mlx-only
```

Record GPT generation, S2Mel, BigVGAN, total wall time, audio duration, and RTF.
Warm results are the relevant comparison for compiled CFM and BigVGAN paths.
