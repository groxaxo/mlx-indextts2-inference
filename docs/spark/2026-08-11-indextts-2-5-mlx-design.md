# IndexTTS-2.5 MLX Support Design

Date: 2026-08-11
Status: Approved architecture; pending written-spec review
Owner: vanch
Target project: /Users/vanch/mlx-indextts2

## 1. Decision Summary

The project will keep IndexTTS-2.0 fully supported and add IndexTTS-2.5 as an
isolated, versioned MLX implementation. IndexTTS-1.5 is no longer a target for
new work or regression coverage. Existing 1.5 source files do not need to be
deleted as part of this upgrade unless they directly block the 2.0/2.5
architecture.

The selected architecture is additive:

- preserve the existing 2.0 conversion, inference, CLI, batch, API, WebUI, and
  local extensions;
- add version-specific 2.5 conversion and runtime components;
- route both versions through a shared public control surface;
- reject incompatible cross-version caches rather than silently coercing them;
- implement every portable upstream user capability;
- map CUDA-only acceleration concepts to MLX-native optimization instead of
  reproducing CUDA, DeepSpeed, TensorRT, Triton, or vLLM backends.

No completion claim is valid until the acceptance matrix in section 13 is
supported by current artifacts and test evidence.

## 2. Authoritative Evidence Baseline

Implementation decisions are grounded in these pinned public sources:

- Official source repository:
  https://github.com/index-tts/index-tts
- Audited source commit:
  9c87c46b84bd0e75ecaefb461e7e8f69bc9ecf44
- IndexTTS-2.5 release commit:
  583d6d4
- Official Hugging Face model:
  https://huggingface.co/IndexTeam/IndexTTS-2.5
- Audited Hugging Face revision:
  d0aa86e75bb6f3437f3831e95056fa72842d89ef
- Technical report:
  https://arxiv.org/abs/2601.03888

The audited model repository contains these primary 2.5 resources:

- gpt.pth
- codec.pth
- s2mel.pth
- feat1.pt and feat2.pt
- wav2vec2bert_stats.pt
- multilingual_zh_ja_yue_char_del.tiktoken
- qwen0.6bemo4-merge
- config.yaml

The source and model revisions must be captured in every converted model
manifest. A later upstream revision may be supported, but it must be audited as
a new source revision rather than silently treated as equivalent.

## 3. Upstream Evidence Conflict

The IndexTTS-2.5 technical report describes a Zipformer-based Semantic-to-Mel
upgrade. The public model card identifies the released model as GPT + DiT +
BigVGAN, the released configuration selects DiT, and the audited public source
contains no Zipformer implementation in the inference path.

Therefore:

- the MLX port follows the actual public checkpoint, public model card, and
  public loader behavior;
- it must not invent or claim a Zipformer conversion path without public
  weights and executable source evidence;
- the discrepancy remains documented as an upstream evidence conflict;
- if a later official revision publishes Zipformer artifacts, it requires a
  separate evidence audit and model-format revision.

The Hugging Face config also carries stale or invalid fields, including a 2.0
version value in one published revision and internal absolute checkpoint paths.
The converter must normalize these fields deterministically from the published
artifact set and checkpoint shapes.

## 4. Scope

### 4.1 Required 2.5 capabilities

- conversion from IndexTeam/IndexTTS-2.5 PyTorch artifacts to MLX;
- fp32, fp16, and supported MLX quantized output formats through one converter;
- deterministic model-version detection and source manifest generation;
- zero-shot voice cloning from raw reference audio;
- Chinese, English, Japanese, Spanish, and Arabic synthesis;
- explicit language selection and documented auto-language limitations;
- cross-lingual voice transfer;
- separate speaker and emotion reference audio;
- manual eight-value emotion vectors and named/mixed local emotion controls;
- text-derived emotion using the official Qwen emotion model;
- speaker-reference emotion fallback matching upstream behavior;
- Chinese Pinyin, English CMU phoneme, and Japanese Kana annotations;
- long-text token-budget segmentation that protects pronunciation annotations;
- generation sampling controls exposed by the existing MLX runtime;
- segment-level streaming behavior equivalent to the public upstream generator;
- precomputed speaker/reference conditioning with version safety;
- single generation, batch generation, concatenated output, Python API,
  FastAPI server, and Gradio WebUI;
- MLX-native performance paths for quantization, compiled safe hot paths,
  reference caching, KV caching, and unified-memory limits;
- regression preservation for IndexTTS-2.0 and existing local extensions.

### 4.2 Existing local extensions to preserve

- standard and Vietnamese 2.0 profile routing;
- model-resident batch generation and manifest output;
- per-row speaker and emotion references;
- duration target and opt-in duration fitting;
- dynamic per-row mel-token limits;
- denoised reference workflow;
- Qwen emotion smoothing and normalization;
- emotion2vec libraries and dialogue/novel planning;
- video-library integration;
- restricted output paths and serialized API inference;
- M3 Max memory tuning and current MLX caching optimizations.

### 4.3 Non-goals

- new IndexTTS-1.5 features or regression work;
- training, fine-tuning, reinforcement-learning post-training, or dataset
  reproduction;
- CUDA custom kernels;
- DeepSpeed;
- TensorRT or Triton serving;
- vLLM serving;
- claiming the paper's 2.28x speedup unless the local benchmark measures it;
- claiming true token-level low-latency streaming if upstream only yields
  completed segment audio;
- deleting user files, existing model directories, or current uncommitted work.

## 5. Version and Artifact Contract

A converted model is identified by a versioned manifest, not by config.yaml
alone.

Required manifest fields:

- format_version;
- model_family;
- model_version;
- source_repository;
- source_revision;
- source_file names, sizes, and available hashes or LFS object identifiers;
- converter version or Git commit;
- conversion timestamp;
- dtype and quantization configuration;
- tensor counts by component;
- mapped, ignored, and missing tensor lists;
- tokenizer type and vocabulary size;
- supported languages;
- semantic codec frame rate;
- required auxiliary resources;
- speaker-cache schema version.

IndexTTS-2.5 detection requires corroborating evidence:

- a 2.5 multilingual tiktoken vocabulary;
- codec.pth using the EnhancedCodec layout;
- GPT text-vocabulary and language-conditioning shapes compatible with 2.5;
- the expected 2.5 artifact set.

The converter must reject an ambiguous or internally inconsistent source rather
than infer 2.5 from one stale config field.

Converted output is written to a task-owned staging directory. The final model
directory becomes visible only after required files, tensor coverage, and a load
smoke pass. Existing directories are never overwritten without an explicit
force option and a clear destination check.

## 6. Component Boundaries

The expected source boundaries are:

- model-version registry and manifest handling;
- 2.5 config normalizer and converter;
- 2.5 multilingual tokenizer and text frontend;
- 2.5 GPT with language embedding and CampPlus speaker conditioning;
- 2.5 EnhancedCodec;
- versioned 2.5 inference orchestrator;
- shared S2Mel/DiT and BigVGAN components only where checkpoint coverage proves
  exact architectural compatibility;
- shared runtime router;
- CLI, API, WebUI, and batch adapters;
- parity and end-to-end validation scripts.

Likely new modules:

- mlx_indextts/model_version.py
- mlx_indextts/convert_v25.py
- mlx_indextts/tokenizer_v25.py
- mlx_indextts/text_frontend_v25.py
- mlx_indextts/models/gpt_v25.py
- mlx_indextts/models/codec_v25.py
- mlx_indextts/generate_v25.py

Likely integration points:

- mlx_indextts/config.py
- mlx_indextts/cli.py
- mlx_indextts/runtime.py
- mlx_indextts/api_server.py
- mlx_indextts/webui.py
- pyproject.toml
- README.md
- tests/
- scripts/

These names are design boundaries, not permission for unrelated refactoring.
Existing dirty changes in overlapping files must be preserved and integrated
deliberately.

## 7. Conversion Data Flow

Source snapshot
→ artifact and revision verification
→ version detection
→ config normalization
→ component-specific PyTorch checkpoint loading
→ deterministic key and layout mapping
→ MLX tensor serialization
→ optional quantization
→ tokenizer and auxiliary-resource copy/link
→ manifest and coverage report
→ isolated load smoke
→ atomic publication of the converted directory

Conversion rules:

- tensor-name transformations are explicit and unit-tested;
- all required tensors must be mapped with shape evidence;
- unexpected required tensors are failures, not warnings;
- explicitly unused training-only tensors may be ignored only through a named
  allowlist with a reason;
- transpositions and convolution layouts receive component parity tests;
- quantization applies only to supported layers and records exclusions;
- the Qwen emotion model is converted separately through the existing MLX-native
  path;
- if the 2.5 Qwen checkpoint is byte-identical to the validated local checkpoint,
  the converter may reuse it after hash evidence;
- the converter never deletes source weights or pre-existing converted models.

## 8. Inference Data Flow

Text and language
→ pronunciation-annotation protection
→ language-specific normalization and Japanese G2P where applicable
→ multilingual tiktoken encoding and language IDs
→ token-budget-safe segmentation
→ speaker and optional emotion reference preprocessing
→ CampPlus speaker conditioning and emotion conditioning
→ language-aware GPT semantic-token generation with KV cache
→ EnhancedCodec semantic decoding at the released 2.5 frame rate
→ length regulation and released S2Mel/DiT flow-matching path
→ BigVGAN waveform generation
→ segment concatenation or segment streaming
→ output validation and runtime metrics

Language values are normalized to:

- auto
- zh
- en
- ja
- es
- ar

Explicit language selection is authoritative. Auto may safely detect distinctive
scripts, but English and Spanish Latin-script text can be ambiguous. The API
must report the resolved language, and documentation must recommend explicit
selection for ambiguous inputs.

## 9. Speaker and Emotion Contract

Raw reference audio remains reusable across 2.0 and 2.5. Precomputed caches do
not.

Every cache stores:

- model family and version;
- source model revision or model fingerprint;
- cache schema version;
- sample-rate and preprocessing metadata;
- conditioning tensor names and shapes;
- source-audio fingerprint where available.

A 2.0 cache passed to 2.5, or a 2.5 cache passed to 2.0, fails with an actionable
error.

Emotion modes are mutually exclusive at the public control surface:

- speaker-reference fallback;
- separate emotion reference audio;
- explicit emotion vector or named/mixed local vector;
- Qwen text-derived emotion.

The runtime must not silently combine incompatible modes. Emotion ordering stays
compatible with the existing project:

happy, angry, sad, afraid, disgusted, melancholic, surprised, calm.

## 10. Public Interfaces

### 10.1 CLI

The existing mlx-indextts entrypoint remains. Conversion and inference gain
version-aware behavior without replacing current batch/planner commands.

Required additions include:

- 2.5 source detection and conversion;
- explicit model/version reporting;
- language selection;
- 2.5 pronunciation annotations;
- 2.5-compatible speaker cache generation;
- streaming/segment output where exposed;
- actionable model-resource checks.

### 10.2 Python API

The public runtime keeps a common GenerateOptions-style contract. Version
specific model classes remain importable for direct component work, while the
shared runtime selects 2.0 or 2.5 from the model manifest.

### 10.3 FastAPI

Health and profile responses include model version, model revision, supported
languages, and resolved quantization. Generate and batch requests accept
language and 2.5 controls. Existing output-path restrictions remain mandatory.

### 10.4 WebUI

The WebUI exposes:

- model/profile selection;
- model version;
- language;
- speaker reference;
- emotion source;
- pronunciation annotation guidance;
- long-text segmentation controls;
- sampling controls;
- streaming/segment progress consistent with actual runtime semantics.

It must not expose CUDA-only toggles in the MLX application.

## 11. Streaming Semantics

The public upstream 2.5 inference generator is the semantic reference.

The first accepted MLX implementation may yield completed audio per safe text
segment while keeping the model loaded and preserving order. It must:

- start returning before the entire multi-segment request completes;
- include sample rate, segment index, and completion state;
- preserve pronunciation-annotation boundaries;
- propagate errors without presenting a partial stream as complete;
- avoid claiming token-level waveform streaming.

True codec-token streaming is a separate future capability unless the public
upstream contract and checkpoint behavior provide evidence for it.

## 12. Error Handling and Safety

Required failures:

- ambiguous or incomplete 2.5 source snapshot;
- stale config paths that cannot be normalized to published artifacts;
- missing tokenizer, codec, GPT, S2Mel, or required auxiliary resource;
- incomplete required tensor mapping;
- incompatible tensor shape;
- unsupported quantization target;
- incompatible speaker cache;
- unsupported or unresolved language when explicit selection is required;
- conflicting emotion sources;
- output path outside the allowed root for API operations;
- overwrite attempt without explicit force;
- incomplete stream marked as final.

Errors name the missing artifact, expected version, and corrective action.
Warnings are reserved for non-fatal behavior such as auto-language ambiguity or
generation reaching a configured token cap.

Model and voice-cloning documentation retains the upstream license and consent
warnings. No source or generated voice is presented as consent-verified.

## 13. Acceptance Matrix

### Gate A: source evidence

- official GitHub and Hugging Face revisions recorded;
- source file inventory recorded;
- config normalization decisions tested;
- Zipformer/DiT discrepancy documented.

### Gate B: conversion

- 100% required tensor coverage for GPT, EnhancedCodec, S2Mel/DiT, and BigVGAN
  components used by the runtime;
- every ignored tensor is allowlisted with a reason;
- converted manifest is complete;
- fp32 conversion loads;
- fp16 conversion loads;
- at least one supported quantized conversion loads;
- conversion is repeatable for the same source revision.

### Gate C: component parity

Using fixed fixtures and deterministic inputs:

- tokenizer IDs and language IDs match upstream;
- pronunciation annotation transformation matches upstream;
- speaker and emotion conditioning shapes and selected numeric outputs match
  within declared tolerances;
- GPT prefill logits and one-step decode logits match within declared
  tolerances;
- EnhancedCodec quantize/decode outputs match within declared tolerances;
- S2Mel/DiT component outputs match within declared tolerances;
- BigVGAN output matches within a declared numeric or audio-domain tolerance.

Each tolerance must be justified by dtype and quantization. Shape-only checks do
not prove numeric parity.

### Gate D: 2.5 functional matrix

- basic generation in zh, en, ja, es, and ar;
- cross-lingual generation using at least one Chinese reference into en, ja,
  es, and ar;
- speaker-reference fallback emotion;
- separate emotion audio;
- manual vector or named/mixed emotion;
- Qwen text emotion;
- Chinese Pinyin annotation;
- English CMU annotation;
- Japanese Kana annotation;
- long text spanning multiple safe segments;
- segment streaming;
- precomputed 2.5 speaker cache;
- batch with per-row language and references;
- concatenated batch output;
- Python API;
- FastAPI generation and batch;
- WebUI smoke.

Generated files alone are insufficient. Evidence records non-empty valid WAV
structure, duration, runtime status, and applicable ASR/listening results.

### Gate E: 2.0 regression

- existing 2.0 standard and Vietnamese model routing;
- existing 2.0 generation;
- separate emotion reference;
- manual and Qwen emotion;
- duration and dynamic token controls;
- batch, API, and WebUI tests relevant to touched paths;
- current dirty performance/caching changes remain present and tested.

IndexTTS-1.5 is not part of this regression gate.

### Gate F: quality and performance

- cold-load and warm steady-state RTF reported separately;
- 2.0 and 2.5 compared on matched Chinese and English inputs and reference
  strategy;
- 2.5 five-language content fidelity checked with language-appropriate ASR when
  available;
- speaker similarity and emotion similarity reported where valid;
- no claim of the paper's 2.28x speedup unless reproduced locally;
- any slower-than-2.0 2.5 steady-state result is investigated and documented
  before completion.

### Gate G: documentation and packaging

- installation and conversion instructions;
- official source revision and license notice;
- model directory format;
- 2.0/2.5 compatibility table;
- language and annotation examples;
- CLI, Python, API, WebUI, batch, and streaming examples;
- known limitations and evidence conflicts;
- clean package build and applicable test/lint commands.

## 14. Storage and Local Validation Policy

The audited machine had approximately 35 GiB free before implementation.
Validation must avoid unnecessary duplicate full-precision copies.

- keep the official source snapshot and one final validated local model;
- use task-owned staging for component conversion;
- quantize component-by-component when supported;
- reuse identical auxiliary/Qwen artifacts only after hash evidence;
- remove only task-owned transient staging after the final artifact is safely
  published;
- do not delete source weights, current models, user outputs, or existing
  checkpoints without separate authorization.

The converter itself remains capable of fp32, fp16, and supported quantized
outputs even if local end-to-end audio validation prioritizes the 8-bit model
used by this project.

## 15. Verification Status at Design Time

- Official 2.5 release and source revision: pass.
- Official Hugging Face artifact inventory: pass.
- Local 2.0 project and dirty-worktree inventory: pass.
- Ruff baseline: pass.
- Pytest baseline: missing evidence because pytest is not installed in the
  current environment.
- 2.5 weights downloaded locally: pending.
- 2.5 MLX conversion: pending.
- Component parity: pending.
- End-to-end five-language generation: pending.
- 2.0 regression after implementation: pending.
- API/WebUI/streaming parity: pending.
- Performance and quality report: pending.

These pending items are implementation acceptance gates, not evidence of current
completion.

## 16. Approved Design Decisions

The user approved:

- the additive versioned architecture;
- preserving IndexTTS-2.0;
- adding IndexTTS-2.5;
- excluding IndexTTS-1.5 from new work;
- implementing every portable upstream capability;
- replacing CUDA-specific acceleration with MLX-native optimization;
- grounding implementation in the actual public release artifacts.

The written specification itself remains subject to user review before
implementation begins.
