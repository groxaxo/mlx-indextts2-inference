"""Sentence-directed IndexTTS 2.5 synthesis and RTX 3090 quality profiles.

The director produces one immutable control row per source sentence.  This
module optionally coalesces adjacent, compatible rows into short semantic
chunks, calls the existing model-resident NVIDIA runtime, and concatenates WAV
segments without loudness normalization or resampling.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from .director_core import (
    EMOTION_ORDER,
    DirectionPlan,
    SentenceDirection,
    normalize_emotion_vector,
    validate_plan,
)


@dataclass(slots=True, frozen=True)
class QualityPreset:
    """Generation controls chosen for a particular quality/variance posture."""

    name: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_text_tokens: int
    max_mel_tokens: int
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_text_tokens": self.max_text_tokens,
            "max_mel_tokens": self.max_mel_tokens,
            "description": self.description,
        }


QUALITY_PRESETS: dict[str, QualityPreset] = {
    "natural-hq": QualityPreset(
        name="natural-hq",
        temperature=0.72,
        top_p=0.80,
        top_k=30,
        repetition_penalty=10.0,
        max_text_tokens=90,
        max_mel_tokens=1500,
        description=(
            "Best default for natural production speech: restrained variance, "
            "short semantic chunks, and upstream-safe acoustic limits."
        ),
    ),
    "studio-stable": QualityPreset(
        name="studio-stable",
        temperature=0.68,
        top_p=0.76,
        top_k=24,
        repetition_penalty=10.0,
        max_text_tokens=80,
        max_mel_tokens=1500,
        description="Lower-variance delivery for repeatable narration and product voices.",
    ),
    "expressive-hq": QualityPreset(
        name="expressive-hq",
        temperature=0.78,
        top_p=0.85,
        top_k=36,
        repetition_penalty=10.0,
        max_text_tokens=80,
        max_mel_tokens=1500,
        description="More acoustic variation for dialogue while retaining safe token limits.",
    ),
}

CUDA_PROFILES = ("quality", "balanced", "throughput", "unchanged")


class NvidiaRuntimeProtocol(Protocol):
    version: str
    device: str
    precision: str

    def generate(self, request: Any) -> Any: ...


@dataclass(slots=True, frozen=True)
class DirectedChunk:
    chunk_index: int
    sentence_indexes: tuple[int, ...]
    start: int
    end: int
    text: str
    emotion: str
    emotion_vector: tuple[float, ...]
    alpha: float
    speed: float
    pause_after_ms: int

    @property
    def duration_factor(self) -> float:
        return 1.0 / self.speed

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "sentence_indexes": list(self.sentence_indexes),
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "emotion": self.emotion,
            "emotion_vector": {
                name: round(value, 4)
                for name, value in zip(
                    EMOTION_ORDER, self.emotion_vector, strict=True
                )
            },
            "alpha": round(self.alpha, 4),
            "speed": round(self.speed, 4),
            "duration_factor": round(self.duration_factor, 4),
            "pause_after_ms": self.pause_after_ms,
        }


@dataclass(slots=True)
class DirectedSynthesisResult:
    output_path: str
    elapsed_seconds: float
    device: str
    precision: str
    preset: str
    cuda_profile: str
    sentence_count: int
    chunk_count: int
    sample_rate: int
    peak: float
    plan: DirectionPlan
    chunks: list[DirectedChunk]
    cuda_tuning: dict[str, Any] = field(default_factory=dict)
    segment_results: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self, *, include_plan: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "output_path": self.output_path,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "device": self.device,
            "precision": self.precision,
            "preset": self.preset,
            "cuda_profile": self.cuda_profile,
            "sentence_count": self.sentence_count,
            "chunk_count": self.chunk_count,
            "sample_rate": self.sample_rate,
            "peak": round(self.peak, 6),
            "cuda_tuning": dict(self.cuda_tuning),
            "segments": list(self.segment_results),
            "chunks": [chunk.as_dict() for chunk in self.chunks],
        }
        if include_plan:
            payload["direction_plan"] = self.plan.as_dict()
        return payload

    def to_json(self, *, include_plan: bool = True, indent: int = 2) -> str:
        return json.dumps(
            self.as_dict(include_plan=include_plan),
            ensure_ascii=False,
            indent=indent,
        )


def get_quality_preset(name: str | None) -> QualityPreset:
    normalized = str(name or "natural-hq").strip().lower()
    try:
        return QUALITY_PRESETS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown quality preset {name!r}; choose "
            + ", ".join(QUALITY_PRESETS)
        ) from exc


def apply_cuda_profile(profile: str, torch_module: Any | None = None) -> dict[str, Any]:
    """Apply explicit process-level CUDA math policy.

    ``quality`` is the conservative RTX 3090 default: BF16 model execution still
    uses Tensor Cores, while TF32 substitutions for FP32 operations are disabled.
    The other profiles are opt-in throughput experiments.
    """

    normalized = str(profile or "quality").strip().lower()
    if normalized not in CUDA_PROFILES:
        raise ValueError(
            f"Unknown CUDA profile {profile!r}; choose " + ", ".join(CUDA_PROFILES)
        )
    if normalized == "unchanged":
        return {"profile": normalized, "changed": False}

    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except (ImportError, OSError):
            return {
                "profile": normalized,
                "changed": False,
                "warning": "PyTorch is unavailable; CUDA math policy was not changed.",
            }

    quality = normalized == "quality"
    benchmark = normalized == "throughput"
    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision("highest" if quality else "high")

    backends = getattr(torch_module, "backends", None)
    cuda = getattr(backends, "cuda", None)
    matmul = getattr(cuda, "matmul", None)
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        matmul.allow_tf32 = not quality
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        if hasattr(cudnn, "allow_tf32"):
            cudnn.allow_tf32 = not quality
        if hasattr(cudnn, "benchmark"):
            cudnn.benchmark = benchmark
        if hasattr(cudnn, "deterministic"):
            cudnn.deterministic = False

    return {
        "profile": normalized,
        "changed": True,
        "float32_matmul_precision": "highest" if quality else "high",
        "allow_tf32": not quality,
        "cudnn_benchmark": benchmark,
    }


def _l1_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right, strict=True))


def _crosses_paragraph(text: str, left: SentenceDirection, right: SentenceDirection) -> bool:
    return bool(re.search(r"\n\s*\n", text[left.end : right.start]))


def _weighted_vector(directions: Sequence[SentenceDirection]) -> tuple[float, ...]:
    weights = [max(1, len(item.text.strip())) for item in directions]
    total = sum(weights)
    values = [
        sum(
            item.emotion_vector[position] * weight
            for item, weight in zip(directions, weights, strict=True)
        )
        / total
        for position in range(len(EMOTION_ORDER))
    ]
    return normalize_emotion_vector(values)


def _weighted_scalar(
    directions: Sequence[SentenceDirection], attribute: str
) -> float:
    weights = [max(1, len(item.text.strip())) for item in directions]
    total = sum(weights)
    return (
        sum(
            float(getattr(item, attribute)) * weight
            for item, weight in zip(directions, weights, strict=True)
        )
        / total
    )


def _chunk_from_group(
    text: str,
    group: Sequence[SentenceDirection],
    chunk_index: int,
) -> DirectedChunk:
    first, last = group[0], group[-1]
    dominant = max(
        range(len(EMOTION_ORDER)),
        key=lambda index: sum(item.emotion_vector[index] for item in group),
    )
    labels = list(dict.fromkeys(item.emotion for item in group))
    emotion = labels[0] if len(labels) == 1 else f"blended {EMOTION_ORDER[dominant]} calm"
    return DirectedChunk(
        chunk_index=chunk_index,
        sentence_indexes=tuple(item.index for item in group),
        start=first.start,
        end=last.end,
        text=text[first.start : last.end],
        emotion=emotion,
        emotion_vector=_weighted_vector(group),
        alpha=round(_weighted_scalar(group, "alpha"), 4),
        speed=round(_weighted_scalar(group, "speed"), 4),
        pause_after_ms=last.pause_after_ms,
    )


def build_synthesis_chunks(
    plan: DirectionPlan,
    *,
    coalesce: bool = True,
    max_sentences_per_chunk: int = 3,
    max_characters_per_chunk: int = 320,
    vector_distance_limit: float = 0.22,
    alpha_delta_limit: float = 0.08,
    speed_delta_limit: float = 0.04,
) -> list[DirectedChunk]:
    """Coalesce only adjacent controls whose delivery is genuinely compatible."""

    errors = validate_plan(plan)
    if errors:
        raise ValueError("Invalid direction plan: " + "; ".join(errors))
    if max_sentences_per_chunk <= 0 or max_characters_per_chunk <= 0:
        raise ValueError("Chunk limits must be positive")
    if not plan.directions:
        return []

    groups: list[list[SentenceDirection]] = []
    current: list[SentenceDirection] = []
    for direction in plan.directions:
        if not current:
            current = [direction]
            continue
        previous = current[-1]
        candidate_characters = direction.end - current[0].start
        compatible = (
            coalesce
            and len(current) < max_sentences_per_chunk
            and candidate_characters <= max_characters_per_chunk
            and previous.pause_after_ms <= 200
            and not _crosses_paragraph(plan.original_text, previous, direction)
            and _l1_distance(previous.emotion_vector, direction.emotion_vector)
            <= vector_distance_limit
            and abs(previous.alpha - direction.alpha) <= alpha_delta_limit
            and abs(previous.speed - direction.speed) <= speed_delta_limit
        )
        if compatible:
            current.append(direction)
        else:
            groups.append(current)
            current = [direction]
    if current:
        groups.append(current)

    return [
        _chunk_from_group(plan.original_text, group, index)
        for index, group in enumerate(groups)
    ]


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise RuntimeError(f"Generated audio is empty: {path}")
    if not np.isfinite(audio).all():
        raise RuntimeError(f"Generated audio contains NaN or Inf: {path}")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _edge_fade(audio: np.ndarray, sample_rate: int, milliseconds: float) -> np.ndarray:
    samples = min(int(sample_rate * max(0.0, milliseconds) / 1000.0), audio.shape[0] // 2)
    if samples <= 1:
        return audio
    result = audio.copy()
    fade_in = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
    fade_out = np.linspace(1.0, 0.0, samples, dtype=np.float32)[:, None]
    result[:samples] *= fade_in
    result[-samples:] *= fade_out
    return result


def _silence(sample_rate: int, channels: int, milliseconds: int) -> np.ndarray:
    count = max(0, int(round(sample_rate * milliseconds / 1000.0)))
    return np.zeros((count, channels), dtype=np.float32)


def synthesize_direction_plan(
    runtime: NvidiaRuntimeProtocol,
    plan: DirectionPlan,
    *,
    ref_audio: str | Path,
    output_path: str | Path,
    language: str = "auto",
    preset: str | QualityPreset = "natural-hq",
    cuda_profile: str = "quality",
    seed: int | None = None,
    coalesce: bool = True,
    max_sentences_per_chunk: int = 3,
    max_characters_per_chunk: int = 320,
    edge_fade_ms: float = 4.0,
    keep_segments: bool = False,
) -> DirectedSynthesisResult:
    """Synthesize a validated plan through the existing NVIDIA runtime."""

    if str(getattr(runtime, "version", "")) != "2.5":
        raise ValueError("Sentence-directed synthesis requires IndexTTS 2.5")
    selected = preset if isinstance(preset, QualityPreset) else get_quality_preset(preset)
    tuning = apply_cuda_profile(cuda_profile)
    chunks = build_synthesis_chunks(
        plan,
        coalesce=coalesce,
        max_sentences_per_chunk=max_sentences_per_chunk,
        max_characters_per_chunk=max_characters_per_chunk,
    )
    if not chunks:
        raise ValueError("Direction plan contains no synthesis chunks")

    speaker_reference = Path(ref_audio).expanduser().resolve()
    if not speaker_reference.is_file():
        raise FileNotFoundError(f"Speaker reference audio does not exist: {speaker_reference}")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    from .nvidia_runtime import NvidiaGenerateRequest

    started = time.perf_counter()
    segment_results: list[dict[str, Any]] = []
    waveforms: list[np.ndarray] = []
    sample_rate: int | None = None
    channels: int | None = None

    segment_dir: Path | None = None
    if keep_segments:
        segment_dir = destination.parent / f"{destination.stem}.segments"
        segment_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="indextts25-directed-") as temporary:
        temporary_path = Path(temporary)
        for chunk in chunks:
            segment_path = temporary_path / f"{chunk.chunk_index:05d}.wav"
            request = NvidiaGenerateRequest(
                text=chunk.text,
                ref_audio=speaker_reference,
                output_path=segment_path,
                language=language,
                emotion_vector=list(chunk.emotion_vector),
                emo_alpha=chunk.alpha,
                use_random=False,
                interval_silence_ms=0,
                max_text_tokens=selected.max_text_tokens,
                duration_factor=chunk.duration_factor,
                text_normalization=True,
                max_mel_tokens=selected.max_mel_tokens,
                temperature=selected.temperature,
                top_p=selected.top_p,
                top_k=selected.top_k,
                repetition_penalty=selected.repetition_penalty,
                seed=None if seed is None else seed + chunk.chunk_index,
                verbose=False,
            )
            result = runtime.generate(request)
            audio, current_rate = _read_wav(segment_path)
            if sample_rate is None:
                sample_rate = current_rate
                channels = audio.shape[1]
            elif current_rate != sample_rate or audio.shape[1] != channels:
                raise RuntimeError(
                    "IndexTTS returned inconsistent sample rate or channel count across chunks"
                )
            audio = _edge_fade(audio, current_rate, edge_fade_ms)
            waveforms.append(audio)
            if chunk.chunk_index < len(chunks) - 1 and chunk.pause_after_ms:
                waveforms.append(_silence(current_rate, audio.shape[1], chunk.pause_after_ms))

            result_payload = (
                result.as_dict() if hasattr(result, "as_dict") else {"result": str(result)}
            )
            retained_path: str | None = None
            if segment_dir is not None:
                retained = segment_dir / segment_path.name
                shutil.copy2(segment_path, retained)
                retained_path = str(retained)
            result_payload.pop("output_path", None)
            result_payload.update(
                {
                    "chunk_index": chunk.chunk_index,
                    "sentence_indexes": list(chunk.sentence_indexes),
                    "pause_after_ms": chunk.pause_after_ms,
                    "retained_output_path": retained_path,
                }
            )
            segment_results.append(result_payload)

    assert sample_rate is not None and channels is not None
    combined = np.concatenate(waveforms, axis=0)
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if not math.isfinite(peak):
        raise RuntimeError("Combined audio peak is not finite")
    sf.write(destination, combined, sample_rate, subtype="PCM_16")
    if not destination.is_file():
        raise RuntimeError(f"Failed to create directed output: {destination}")

    elapsed = time.perf_counter() - started
    return DirectedSynthesisResult(
        output_path=str(destination),
        elapsed_seconds=elapsed,
        device=str(getattr(runtime, "device", "unknown")),
        precision=str(getattr(runtime, "precision", "unknown")),
        preset=selected.name,
        cuda_profile=cuda_profile,
        sentence_count=plan.sentence_count,
        chunk_count=len(chunks),
        sample_rate=sample_rate,
        peak=peak,
        plan=plan,
        chunks=chunks,
        cuda_tuning=tuning,
        segment_results=segment_results,
    )
