# Inference Performance Bottleneck Audit

This note records the performance audit performed against `main` at
`0b28a63e745d4290ed57d260a4f0b50cb4e01846` on 2026-08-14.

## Priority order

### P0 — GPT KV-cache growth — fixed in this change

`GPT2Attention` previously appended every new key/value tensor with
`mx.concatenate()` at every layer and every autoregressive token. For a generated
sequence of length `T`, that repeatedly copies the historical cache and turns
cache maintenance into quadratic memory traffic.

The runtime now keeps a chunked cache (256 tokens by default), grows capacity only
when a chunk is exhausted, and writes new K/V slices into the allocated MLX arrays.
The public `cache[layer][0/1]` indexing contract is preserved for compatibility.

The GPT attention path also uses `mx.fast.scaled_dot_product_attention` when the
installed MLX exposes it, with the existing explicit implementation retained as a
compatibility fallback.

### P1 — Sampling performs more vocabulary work than necessary

`UnifiedVoiceV2._sample()` applies top-k and then performs a full-vocabulary
`argsort` for nucleus sampling on every token. With the default `top_k=30`, top-p
only needs to operate over the surviving candidates. It also computes
`softmax(logits)` and then `log()` before `mx.random.categorical`, although MLX
categorical accepts logits directly.

Recommended follow-up: implement top-p on the top-k candidate set and sample from
filtered logits directly. Keep a deterministic parity test around fixed logits and
seed before changing this because sampling changes can alter generated speech.

### P1 — Repetition penalty rebuilds Python history state every token

`_apply_repetition_penalty()` reconstructs and sorts a Python `set` from the full
`generated_tokens` list on every autoregressive step. That adds another quadratic
host-side component as utterances get longer.

Recommended follow-up: maintain a per-generation seen-token mask/set incrementally,
or keep the state entirely on MLX tensors.

### P1 — IndexTTS 2.5 grows its attention mask with per-token concatenation

`IndexTTSv25._generate_semantic_codes()` appends one element to `current_mask`
with `mx.concatenate()` on every generated token. The final size is small compared
with the KV cache, but it is still avoidable allocation/graph churn in the hottest
loop.

Recommended follow-up: allocate the maximum padding/generation mask once and pass
a prefix view for the active key length.

### P1 — IndexTTS 2.5 misses the existing safe compile hook

IndexTTS 2.0 calls `_compile_hotpaths()` after loading S2Mel and BigVGAN. The 2.5
loader initializes the equivalent modules but does not invoke that inherited hook.
This leaves BigVGAN eager in 2.5 and also leaves `s2mel.gpt_layer` eager when GPT
latent fusion is enabled.

Recommended follow-up: enable the same best-effort compile hook for 2.5 and measure
first-run compile latency separately from warm inference.

### P2 — Per-token host synchronization

Both v2.0 and v2.5 materialize the sampled token through `.item()` for EOS handling.
That synchronizes the host once per generated semantic token. Eliminating this
requires a larger device-side generation-loop design and should be benchmarked
carefully rather than patched locally.

### P2 — CFM still allocates per diffusion step

The CFM path already hoists invariant CFG tensors, which is good. It still creates
`stacked_x`, timestep arrays, and prompt-zeroed tensors inside the Euler loop and
forces `mx.eval(x)` every step. This may be worth revisiting after GPT decode is no
longer dominant.

### P2 — Raw reference audio preprocessing

Raw WAV references cross PyTorch/librosa/torchaudio and MLX boundaries. The repo's
speaker `.npz` cache already provides the right fast path. Production/API usage
should prefer precomputed speaker caches for repeated voices.

## Validation

Run the focused regression tests:

```bash
uv run pytest tests/test_gpt2_cache_perf.py tests/test_models.py -q
```

Run the resident-model benchmark on Apple Silicon before/after the commit:

```bash
uv run python scripts/benchmark_v20_v25.py \
  --ref-audio /path/to/reference.wav \
  --output-dir outputs/validation/v20-v25-benchmark
```

For detailed phase timing on v2.0, use:

```bash
uv run python scripts/test_benchmark.py --v20-only --mlx-only
```

Record cold-load separately from first-resident and warm RTF. The expected gain
from this commit should be most visible in GPT generation time and should increase
with semantic-token count because the removed cache-copy cost grew with history.
