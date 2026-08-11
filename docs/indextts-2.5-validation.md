# IndexTTS 2.5 MLX Validation Record

Date: 2026-08-11
Host: Apple M3 Max, 128 GB unified memory
Status: automated implementation gates pass; human listening review pending

This record distinguishes executed evidence from unverified claims. Generated
WAVs and machine-readable reports are under `outputs/validation/` and are
intentionally excluded from source distributions.

## Source evidence

| Item | Revision | Status |
| --- | --- | --- |
| `index-tts/index-tts` | `9c87c46b84bd0e75ecaefb461e7e8f69bc9ecf44` | pass |
| `IndexTeam/IndexTTS-2.5` | `d0aa86e75bb6f3437f3831e95056fa72842d89ef` | pass |
| NVIDIA BigVGAN | `633ff708ed5b74903e86ff1298cf4a98e921c513` | pass |
| Public artifact inventory | GPT, EnhancedCodec, S2Mel/DiT, BigVGAN resources, tokenizer, feature matrices | pass |
| Zipformer discrepancy | Paper describes Zipformer; released executable artifacts use DiT and publish no Zipformer checkpoint | documented |

## Conversion

All three full model variants were generated from the pinned source and passed
strict model loading:

| Variant | Core safetensors bytes | Quantization | Strict load |
| --- | ---: | --- | --- |
| fp16 | 2,157,580,265 | none | pass |
| fp16 + GPT 8-bit | 1,715,230,963 | 8-bit, group size 64 | pass |
| fp32 | 4,315,011,467 | none | pass |

Required model parameter coverage had no missing or unexpected tensors:

| Component | Source tensors | Saved tensors | Explicit non-parameter handling |
| --- | ---: | ---: | --- |
| GPT | 456 | 457 | fixed positional buffer excluded; one source tensor maps to multiple MLX tensors |
| EnhancedCodec | 243 | 241 | weight-normalization pairs are fused |
| S2Mel/DiT | 284 | 264 | fixed positional/cache tensors rebuilt |
| BigVGAN | 783 | 449 | 334 fixed anti-alias filters rebuilt deterministically |

Repeat conversion to `models/mlx-IndexTTS-2.5-8bit-repeat` produced identical
SHA-256 values for all ten deterministic artifacts: four safetensors files, two
normalized configs, tokenizer, both feature matrices, and W2V statistics.
Timestamp-bearing manifests/reports were intentionally excluded from bytewise
comparison.

## Numeric component parity

Comparisons used the pinned upstream implementation, deterministic inputs, and
Transformers 4.52.1 where the official GPT code depends on Transformers
semantics.

| Component | Evidence | Result |
| --- | --- | ---: |
| Tokenizer/language IDs | exact token and language-ID fixtures | exact |
| GPT, full 24 layers | maximum absolute difference | `1.37e-6` |
| GPT speaker projection | maximum absolute difference | `5.72e-6` |
| EnhancedCodec codes | integer codes | exact |
| EnhancedCodec quantized latent | maximum absolute difference | `2.38e-7` |
| EnhancedCodec decode | maximum absolute difference | `5.07e-6` |
| S2Mel GPT layer | maximum absolute difference | `1.19e-7` |
| S2Mel length regulator | maximum absolute difference | `3.35e-8` |
| S2Mel CFM estimator | maximum absolute difference | `1.26e-5` |
| BigVGAN | maximum absolute waveform difference | `4.04e-5` |

These tolerances are consistent with fp16 conversion and MLX/PyTorch operation
ordering. Shape-only checks were not used as parity evidence.

## 2.5 functional matrix

The final default 8-bit matrix is recorded in
`outputs/validation/indextts25-8bit-full-final/functional_report.json`.
Thirteen scenarios passed:

- basic Chinese synthesis;
- Chinese-reference cross-lingual synthesis into English, Japanese, Spanish,
  and Arabic;
- Chinese Pinyin, English CMU, and Japanese Kana annotations;
- speaker-reference emotion fallback;
- separate emotion-reference audio;
- manual eight-value/mixed emotion with random prototype selection;
- Qwen emotion from a separate emotion description (`afraid` dominant);
- six-segment long-text streaming with only the final segment marked complete;
- bilingual per-row batch generation plus `combined.wav`;
- versioned precomputed 2.5 speaker cache;
- Python runtime, CLI, FastAPI generation/batch, and Gradio construction.

Every generated file was non-empty, finite, mono 22,050 Hz audio. The full
8-step automation run completed in 83.20 seconds. A separate five-language run
with the upstream 25-step diffusion default also passed, with RTF from 1.70 to
2.19 on the selected short cases.

FastAPI real-model evidence:

- `/generate`: HTTP 200, 1.788 seconds of audio, RTF 0.917;
- `/batch`: HTTP 200, two language-specific rows and a valid 3.409-second
  combined WAV;
- `/health`: correct model revision, five languages, and persistent 8-bit
  quantization metadata.

## Content and similarity checks

MLX Whisper `whisper-large-v3-turbo` transcribed the final five-language matrix:

| Language | Metric | Error rate | Result |
| --- | --- | ---: | --- |
| Chinese | CER | 0.2273 | pass; differences centered on `IndexTTS` and `2.5` spelling |
| English | WER | 0.0000 | pass |
| Japanese | CER | 0.0000 | pass |
| Spanish | WER | 0.0000 | pass |
| Arabic | WER | 0.0000 | pass |

CampPlus speaker cosine similarity against the Chinese reference:

| Output language | Cosine |
| --- | ---: |
| Chinese | 0.6897 |
| English | 0.5757 |
| Japanese | 0.5939 |
| Spanish | 0.6384 |
| Arabic | 0.6894 |

The separate-emotion output's learned GPT emotion-vector cosine was 0.7923 to
the emotion reference and 0.7623 to the original speaker reference. ASR and
embedding cosines are engineering sanity checks, not substitutes for blinded
human listening or a publication-grade evaluation set.

## 2.0 regression and performance

Executed 2.0 regression evidence includes:

- standard Chinese and English generation;
- Vietnamese model routing and real generation (3.367 seconds, RTF 1.290);
- separate emotion reference, manual mixed emotion, and Qwen emotion generation;
- existing duration/dynamic-token, batch, API, and WebUI unit/integration tests.

Matched 8-bit performance used the same raw Chinese reference, Chinese/English
texts, version-specific precomputed caches, and 16 diffusion steps:

| Metric | 2.0 | 2.5 |
| --- | ---: | ---: |
| Cold model load | 1.4842s | 0.2259s |
| Reference preprocessing | 10.8371s | 4.2927s |
| Warm Chinese RTF | 1.3884 | 1.0429 |
| Warm English RTF | 1.8403 | 1.0741 |
| Mean warm RTF | 1.6143 | 1.0585 |

Matched ASR was exact for both English outputs, exact for 2.0 Chinese, and had
one Chinese homophone substitution for 2.5 (CER 0.0625). This run does not
reproduce or claim the paper's 2.28× result.

## Repository and package checks

| Check | Result |
| --- | --- |
| `pytest` | 167 passed, 7 skipped |
| Ruff | pass |
| `git diff --check` | pass |
| lock resolution | pass, 171 packages |
| sdist and wheel | pass, version 0.2.0 |
| wheel content | all 2.5 modules present; local models/outputs/caches absent |
| isolated Python 3.13 install | pass with `torch==2.10.0`, `torchaudio==2.10.0` |
| isolated `[v25,api]` imports | `IndexTTSv25`, FastAPI, and eight-value request vector pass |
| WebUI construction | Gradio `Blocks` pass |
| OpenAPI construction | pass |

The seven skipped tests are weight-dependent legacy integration fixtures. They
are covered for 2.5 by the separately executed real-model matrix above. The
remaining evidence gap is subjective human listening; no listening claim is
made.
