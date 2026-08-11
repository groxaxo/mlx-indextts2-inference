"""Reusable runtime helpers for MLX-IndexTTS apps.

This module keeps UI/API code thin and ensures all entrypoints use the same
8bit default model routing and generation defaults as the CLI.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from mlx_indextts.cli import detect_mlx_version, resolve_default_model
from mlx_indextts.performance import configure_mlx_runtime, resolve_mlx_memory_limits


@dataclass
class GenerateOptions:
    """Generation options shared by CLI, API, and WebUI."""

    max_tokens: int | None = None
    max_text_tokens: int = 120
    interval_silence: int = 0
    temperature: float | None = None
    top_k: int = 30
    top_p: float = 0.8
    repetition_penalty: float = 10.0
    diffusion_steps: int = 16
    cfg_rate: float = 0.7
    emotion: str | dict[str, float] | list[float] | None = None
    emo_alpha: float = 0.6
    emotion_ref_audio: str | None = None
    auto_emotion: bool = False
    use_emo_text: bool = False
    emo_text: str | None = None
    use_random: bool = False
    qwen_emotion_model: str | None = None
    qwen_unload_after: bool = True
    smooth_emotion: bool = True
    denoise_ref_audio: bool = True
    denoise_emotion_ref_audio: bool = True
    seed: int | None = None
    segment_overlap: int = 50
    speed: float = 1.0
    target_duration: float | None = None
    fit_duration: bool = False
    max_fit_stretch_ratio: float = 2.0
    verbose: bool = False
    dynamic_max_tokens: bool = False
    tokens_per_char: float = 14.0
    min_max_tokens: int = 320
    language: str = "auto"
    text_normalization: bool = True
    duration_factor: float = 1.0
    use_gpt_latent: bool = False


def _safe_id(value: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or fallback


def _normalize_emotion_total(weights: dict[str, float], target_total: float = 1.0) -> dict[str, float]:
    """Scale auto-emotion weights so they fully override reference emotion."""
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0.0:
        normalized = {key: 0.0 for key in weights}
        normalized["calm"] = target_total
        return normalized
    scale = target_total / total
    return {key: round(max(0.0, float(value)) * scale, 4) for key, value in weights.items()}


def validate_emotion_source(options: GenerateOptions, *, has_row_emotion_refs: bool = False) -> None:
    """Ensure only one explicit emotion source is active.

    Valid modes are speaker-reference fallback, separate emotion reference,
    manual emotion vector/name, or Qwen text emotion.
    """
    auto = (
        options.auto_emotion
        or options.use_emo_text
        or options.emotion == "auto-qwen"
    )
    manual = bool(options.emotion and options.emotion != "auto-qwen")
    emotion_ref = bool(options.emotion_ref_audio) or has_row_emotion_refs
    active = sum(bool(value) for value in (auto, manual, emotion_ref))
    if active > 1:
        raise ValueError(
            "Emotion sources are mutually exclusive: use only one of "
            "auto_emotion/auto-qwen, manual emotion, or emotion_ref_audio."
        )


def _batch_value(
    row: dict[str, Any] | tuple[str, ...],
    key: str | tuple[str, ...],
    index: int,
    default: Any = "",
) -> Any:
    if isinstance(row, dict):
        if isinstance(key, tuple):
            for candidate in key:
                value = row.get(candidate, default)
                if value is not None and str(value).strip():
                    return value
            value = default
        else:
            value = row.get(key, default)
    else:
        value = row[index] if len(row) > index else default
    return value if value is not None else default


def _batch_bool(value: Any, default: bool) -> bool:
    """Parse optional CSV booleans without treating ``"false"`` as true."""
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "y"}:
        return True
    if normalized in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _default_memory_limit(version: str) -> float:
    limits = resolve_mlx_memory_limits()
    if limits.memory_limit_gb is not None:
        return limits.memory_limit_gb
    return 24.0 if version in {"2.0", "2.5"} else 8.0


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass


def estimate_mel_tokens_for_duration(duration_s: float, *, version: str) -> int:
    """Estimate IndexTTS mel-token budget from target seconds.

    IndexTTS2 commonly lands near 50 GPT mel tokens per output second in local
    benchmarks. This is a generation cap, not guaranteed exact duration.
    """
    if version == "2.5":
        tokens_per_second = 25.0
    elif version == "2.0":
        tokens_per_second = 50.0
    else:
        tokens_per_second = 45.0
    return max(1, int(math.ceil(max(0.1, duration_s) * tokens_per_second)))


def duration_fit_allowed(
    source_duration: float, target_duration: float, max_ratio: float = 2.0
) -> bool:
    """Reject post-hoc stretching large enough to destroy speech quality."""
    if source_duration <= 0 or target_duration <= 0 or max_ratio < 1.0:
        return False
    ratio = source_duration / target_duration
    return (1.0 / max_ratio) <= ratio <= max_ratio


def estimate_mel_tokens_for_text(
    text: str,
    *,
    hard_max: int,
    tokens_per_char: float = 14.0,
    min_tokens: int = 320,
) -> int:
    """Estimate a safe per-utterance mel-token cap from visible text length."""
    visible = re.sub(r"\s+", "", text or "")
    visible = re.sub(r"[，。！？、,.!?;；:：\"'“”‘’（）()【】\[\]…—~-]", "", visible)
    if not visible:
        return max(1, min(int(hard_max), int(min_tokens)))
    budget = int(math.ceil(len(visible) * max(1.0, tokens_per_char)))
    budget = max(int(min_tokens), budget)
    return max(1, min(int(hard_max), budget))


def resolve_mel_token_budget(
    text: str,
    *,
    requested_max: int,
    target_duration: float | None,
    version: str,
    min_content_tokens: int = 320,
) -> int:
    """Resolve a duration-aware cap without truncating short utterances.

    ``target_duration`` is an alignment target, not proof that the GPT can
    finish the requested text inside the corresponding number of mel tokens.
    Keep a content-safe floor so short subtitle windows cannot stop generation
    before the model emits EOS. An explicitly smaller ``requested_max`` remains
    a hard caller limit.
    """
    hard_max = max(1, int(requested_max))
    if not target_duration:
        return hard_max
    duration_budget = estimate_mel_tokens_for_duration(target_duration, version=version)
    content_budget = estimate_mel_tokens_for_text(
        text,
        hard_max=hard_max,
        min_tokens=min_content_tokens,
    )
    return min(hard_max, max(duration_budget, content_budget))


class TTSRuntime:
    """Single-model cache for MLX IndexTTS.

    The cache intentionally keeps only one model alive. This matches the memory
    safety goals for long local runs and avoids standard/Vietnamese double-loads.
    """

    def __init__(self, memory_limit_gb: float | None = None, quantize: str = "8"):
        self.memory_limit_gb = memory_limit_gb
        self.quantize = quantize
        self._model: Any | None = None
        self._model_path: str | None = None
        self._version: str | None = None

    @property
    def model_path(self) -> str | None:
        return self._model_path

    @property
    def version(self) -> str | None:
        return self._version

    def resolve_model(self, profile: str = "auto", text: str = "", model: str | None = None) -> str:
        return model or resolve_default_model(profile, text)

    def unload(self) -> None:
        self._model = None
        self._model_path = None
        self._version = None
        gc.collect()
        _clear_mlx_cache()

    def load(self, model_path: str, memory_limit_gb: float | None = None) -> Any:
        model_path = str(Path(model_path))
        if self._model is not None and self._model_path == model_path:
            return self._model

        self.unload()
        version = detect_mlx_version(Path(model_path))
        memory_limit = memory_limit_gb
        if memory_limit is None:
            memory_limit = self.memory_limit_gb or _default_memory_limit(version)
        limits = configure_mlx_runtime(memory_limit_gb=memory_limit)
        memory_limit = limits.memory_limit_gb or memory_limit

        quantize_bits = None if self.quantize.lower() == "fp32" else int(self.quantize)
        if version == "2.5":
            from mlx_indextts.generate_v25 import IndexTTSv25

            model_obj = IndexTTSv25(
                model_dir=model_path,
                memory_limit_gb=memory_limit,
                quantize_bits=quantize_bits,
            )
        elif version == "2.0":
            from mlx_indextts.generate_v2 import IndexTTSv2

            model_obj = IndexTTSv2(
                model_dir=model_path,
                memory_limit_gb=memory_limit,
                quantize_bits=quantize_bits,
            )
        else:
            from mlx_indextts.generate import IndexTTS

            model_obj = IndexTTS.load_model(
                model_path,
                memory_limit_gb=memory_limit,
                quantize_bits=quantize_bits,
            )

        self._model = model_obj
        self._model_path = model_path
        self._version = version
        return model_obj

    def _resolve_auto_emotion(
        self,
        text: str,
        options: GenerateOptions,
    ) -> tuple[dict[str, float] | str | None, dict[str, Any]]:
        should_auto = (
            options.auto_emotion
            or options.use_emo_text
            or options.emotion == "auto-qwen"
        )
        if not should_auto:
            if options.emotion_ref_audio:
                emotion_source = "emotion_reference"
            elif options.emotion:
                emotion_source = "manual"
            else:
                emotion_source = "speaker_reference"
            return options.emotion, {
                "emotion_source": emotion_source,
                "emotion_json": "",
                "dominant_emotion": "",
                "qwen_elapsed_s": "",
                "qwen_raw_text": "",
            }

        # Keep Qwen and TTS from staying resident together during auto analysis.
        self.unload()
        from mlx_indextts.qwen_emotion import (
            DEFAULT_QWEN_EMOTION_MODEL,
            adaptive_emo_alpha,
            get_qwen_emotion,
            smooth_emotion_sequence_by_speaker,
            unload_qwen_emotion,
        )

        qwen_model_path = options.qwen_emotion_model or DEFAULT_QWEN_EMOTION_MODEL
        result = get_qwen_emotion(qwen_model_path).inference(
            options.emo_text or text
        )
        if options.qwen_unload_after:
            unload_qwen_emotion(qwen_model_path)
        weights = _normalize_emotion_total(result.weights)
        return weights, {
            "emotion_source": result.source,
            "emotion_json": json.dumps(weights, ensure_ascii=False),
            "dominant_emotion": max(weights.items(), key=lambda item: item[1])[0],
            "qwen_elapsed_s": result.elapsed_s,
            "qwen_raw_text": result.raw_text,
        }

    def generate(
        self,
        text: str,
        ref_audio: str,
        output_path: str,
        profile: str = "auto",
        model: str | None = None,
        options: GenerateOptions | None = None,
    ) -> dict[str, Any]:
        options = options or GenerateOptions()
        validate_emotion_source(options)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output_path = str(output)
        model_path = self.resolve_model(profile=profile, text=text, model=model)
        version = detect_mlx_version(Path(model_path))

        if options.denoise_ref_audio or options.denoise_emotion_ref_audio:
            from mlx_indextts.audio_denoise import maybe_denoise_reference

            ref_audio = maybe_denoise_reference(
                ref_audio,
                enabled=options.denoise_ref_audio,
                suffix="speaker",
            ) or ref_audio
            options = GenerateOptions(**options.__dict__)
            options.emotion_ref_audio = maybe_denoise_reference(
                options.emotion_ref_audio,
                enabled=options.denoise_emotion_ref_audio,
                suffix="emotion",
            )

        max_tokens = options.max_tokens
        if max_tokens is None:
            max_tokens = 1500 if version in {"2.0", "2.5"} else 800
        max_tokens = resolve_mel_token_budget(
            text,
            requested_max=max_tokens,
            target_duration=options.target_duration,
            version=version,
        )
        temperature = options.temperature
        if temperature is None:
            temperature = 0.8 if version in {"2.0", "2.5"} else 1.0

        resolved_emotion, emotion_meta = self._resolve_auto_emotion(text, options)
        from mlx_indextts.qwen_emotion import adaptive_emo_alpha
        effective_emo_alpha = options.emo_alpha
        if emotion_meta.get("emotion_source") == "qwen-mlx" and isinstance(resolved_emotion, dict):
            effective_emo_alpha = adaptive_emo_alpha(resolved_emotion, base_alpha=options.emo_alpha)
        tts = self.load(model_path)
        start = time.perf_counter()
        if version == "2.5":
            tts.use_gpt_latent = options.use_gpt_latent
            audio = tts.generate(
                text=text,
                reference_audio=ref_audio,
                emotion_reference_audio=options.emotion_ref_audio,
                output_path=output_path,
                language=options.language,
                max_mel_tokens=max_tokens,
                max_text_tokens_per_segment=options.max_text_tokens,
                interval_silence=options.interval_silence,
                temperature=temperature,
                top_k=options.top_k,
                top_p=options.top_p,
                repetition_penalty=options.repetition_penalty,
                diffusion_steps=options.diffusion_steps,
                cfg_rate=options.cfg_rate,
                emotion=resolved_emotion,
                emo_alpha=effective_emo_alpha,
                use_random=options.use_random,
                text_normalization=options.text_normalization,
                duration_factor=options.duration_factor,
                seed=options.seed,
                verbose=options.verbose,
                segment_overlap_ms=options.segment_overlap,
                speed=(options.speed if not options.fit_duration else 1.0),
            )
        elif version == "2.0":
            audio = tts.generate(
                text=text,
                reference_audio=ref_audio,
                emotion_reference_audio=options.emotion_ref_audio,
                output_path=output_path,
                max_mel_tokens=max_tokens,
                max_text_tokens_per_segment=options.max_text_tokens,
                interval_silence=options.interval_silence,
                temperature=temperature,
                top_k=options.top_k,
                top_p=options.top_p,
                repetition_penalty=options.repetition_penalty,
                diffusion_steps=options.diffusion_steps,
                cfg_rate=options.cfg_rate,
                emotion=resolved_emotion,
                emo_alpha=effective_emo_alpha,
                use_random=options.use_random,
                seed=options.seed,
                verbose=options.verbose,
                segment_overlap_ms=options.segment_overlap,
                speed=(options.speed if not options.fit_duration else 1.0),
            )
        else:
            audio = tts.generate(
                text=text,
                ref_audio=ref_audio,
                max_mel_tokens=max_tokens,
                max_text_tokens_per_segment=options.max_text_tokens,
                interval_silence=options.interval_silence,
                temperature=temperature,
                top_k=options.top_k,
                top_p=options.top_p,
                repetition_penalty=options.repetition_penalty,
                seed=options.seed,
                verbose=options.verbose,
                segment_overlap_ms=options.segment_overlap,
                speed=(options.speed if not options.fit_duration else 1.0),
            )
            tts.save_audio(audio, output_path)

        fit_duration_applied = False
        fit_duration_skip_reason = ""
        if options.fit_duration and options.target_duration and audio is not None:
            duration_before = len(audio) / 22050.0
            if duration_fit_allowed(
                duration_before,
                options.target_duration,
                options.max_fit_stretch_ratio,
            ):
                from mlx_indextts.generate import time_stretch_wsola

                fit_rate = duration_before / options.target_duration
                audio = time_stretch_wsola(audio, rate=fit_rate, sample_rate=22050)
                sf.write(output_path, audio, 22050)
                fit_duration_applied = True
            else:
                fit_duration_skip_reason = (
                    "requested duration requires destructive stretch beyond "
                    f"{options.max_fit_stretch_ratio:.2f}x"
                )

        elapsed = time.perf_counter() - start
        duration = len(audio) / 22050.0 if isinstance(audio, np.ndarray) else sf.info(output_path).duration
        return {
            "output_path": str(output_path),
            "model": model_path,
            "version": version,
            "duration_s": round(duration, 3),
            "elapsed_s": round(elapsed, 3),
            "rtf": round(elapsed / duration, 4) if duration > 0 else None,
            "max_tokens": max_tokens,
            "target_duration_s": options.target_duration or "",
            "fit_duration": options.fit_duration,
            "fit_duration_applied": fit_duration_applied,
            "fit_duration_skip_reason": fit_duration_skip_reason,
            "language": getattr(tts, "last_generation_info", {}).get(
                "resolved_language", options.language
            ),
            "language_ambiguous": getattr(tts, "last_generation_info", {}).get(
                "language_ambiguous", False
            ),
            "model_revision": getattr(tts, "model_revision", ""),
            **emotion_meta,
        }

    def save_speaker(
        self,
        ref_audio: str,
        output_path: str,
        profile: str = "auto",
        model: str | None = None,
        text: str = "",
    ) -> dict[str, Any]:
        model_path = self.resolve_model(profile=profile, text=text, model=model)
        tts = self.load(model_path)
        start = time.perf_counter()
        tts.save_speaker(ref_audio, output_path)
        return {
            "output_path": str(output_path),
            "model": model_path,
            "version": self._version,
            "elapsed_s": round(time.perf_counter() - start, 3),
        }

    def stream(
        self,
        text: str,
        ref_audio: str,
        profile: str = "auto",
        model: str | None = None,
        options: GenerateOptions | None = None,
    ):
        """Yield completed 2.5 audio segments without claiming token streaming."""
        options = options or GenerateOptions()
        validate_emotion_source(options)
        if options.fit_duration:
            raise ValueError("fit_duration is not available for segment streaming")
        model_path = self.resolve_model(profile=profile, text=text, model=model)
        version = detect_mlx_version(Path(model_path))
        if version != "2.5":
            raise ValueError("segment streaming currently requires IndexTTS 2.5")

        if options.denoise_ref_audio or options.denoise_emotion_ref_audio:
            from mlx_indextts.audio_denoise import maybe_denoise_reference

            ref_audio = maybe_denoise_reference(
                ref_audio,
                enabled=options.denoise_ref_audio,
                suffix="speaker",
            ) or ref_audio
            options = GenerateOptions(**options.__dict__)
            options.emotion_ref_audio = maybe_denoise_reference(
                options.emotion_ref_audio,
                enabled=options.denoise_emotion_ref_audio,
                suffix="emotion",
            )

        resolved_emotion, emotion_meta = self._resolve_auto_emotion(text, options)
        from mlx_indextts.qwen_emotion import adaptive_emo_alpha

        effective_emo_alpha = options.emo_alpha
        if emotion_meta.get("emotion_source") == "qwen-mlx" and isinstance(
            resolved_emotion, dict
        ):
            effective_emo_alpha = adaptive_emo_alpha(
                resolved_emotion,
                base_alpha=options.emo_alpha,
            )
        tts = self.load(model_path)
        tts.use_gpt_latent = options.use_gpt_latent
        max_tokens = resolve_mel_token_budget(
            text,
            requested_max=options.max_tokens or 1500,
            target_duration=options.target_duration,
            version="2.5",
        )
        for chunk in tts.stream(
            text=text,
            reference_audio=ref_audio,
            language=options.language,
            emotion_reference_audio=options.emotion_ref_audio,
            emotion=resolved_emotion,
            emo_alpha=effective_emo_alpha,
            use_random=options.use_random,
            max_mel_tokens=max_tokens,
            max_text_tokens_per_segment=options.max_text_tokens,
            interval_silence=options.interval_silence,
            temperature=options.temperature if options.temperature is not None else 0.8,
            top_k=options.top_k,
            top_p=options.top_p,
            repetition_penalty=options.repetition_penalty,
            diffusion_steps=options.diffusion_steps,
            cfg_rate=options.cfg_rate,
            duration_factor=options.duration_factor,
            text_normalization=options.text_normalization,
            seed=options.seed,
            speed=options.speed,
        ):
            tts.last_generation_info.update(emotion_meta)
            yield chunk

    def batch(
        self,
        rows: list[dict[str, Any] | tuple[str, ...]],
        ref_audio: str | None,
        output_dir: str,
        profile: str = "auto",
        model: str | None = None,
        options: GenerateOptions | None = None,
        combine: bool = False,
        combine_silence_ms: int = 120,
    ) -> dict[str, Any]:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        all_text = "\n".join(str(_batch_value(row, "text", 1)) for row in rows)
        model_path = self.resolve_model(profile=profile, text=all_text, model=model)
        model_version = detect_mlx_version(Path(model_path))
        options = options or GenerateOptions()
        speakers = [str(_batch_value(row, ("speaker", "role", "说话人", "角色"), 0, "")).strip() for row in rows]
        has_row_emotion_refs = any(bool(_batch_value(row, "emotion_ref_audio", 3)) for row in rows)
        has_row_manual_emotions = any(bool(_batch_value(row, "emotion", 4)) for row in rows)
        if has_row_manual_emotions and (
            options.auto_emotion
            or options.use_emo_text
            or options.emotion == "auto-qwen"
        ):
            raise ValueError(
                "CSV row emotion cannot be combined with auto_emotion/use_emo_text/auto-qwen."
            )
        validate_emotion_source(options, has_row_emotion_refs=has_row_emotion_refs)
        emotion_payloads: list[dict[str, Any]] | None = None
        if options.auto_emotion or options.use_emo_text or options.emotion == "auto-qwen":
            self.unload()
            from mlx_indextts.qwen_emotion import (
                DEFAULT_QWEN_EMOTION_MODEL,
                adaptive_emo_alpha,
                get_qwen_emotion,
                smooth_emotion_sequence,
                smooth_emotion_sequence_by_speaker,
                unload_qwen_emotion,
            )

            qwen_model_path = options.qwen_emotion_model or DEFAULT_QWEN_EMOTION_MODEL
            qwen = get_qwen_emotion(qwen_model_path)
            qwen_inputs = []
            for row, speaker in zip(rows, speakers):
                text_value = str(_batch_value(row, "text", 1))
                emotion_text = str(
                    _batch_value(
                        row,
                        ("emotion_text", "emo_text"),
                        10,
                        options.emo_text or "",
                    )
                ).strip()
                qwen_input = emotion_text or text_value
                qwen_inputs.append(
                    f"{speaker}：{qwen_input}" if speaker and not emotion_text else qwen_input
                )
            raw_results = [qwen.inference(text_value) for text_value in qwen_inputs]
            weights_list = [result.weights for result in raw_results]
            if options.smooth_emotion:
                if any(speakers) and len(set(speakers)) > 1:
                    weights_list = smooth_emotion_sequence_by_speaker(weights_list, speakers)
                else:
                    weights_list = smooth_emotion_sequence(weights_list)
            weights_list = [_normalize_emotion_total(weights) for weights in weights_list]
            if options.qwen_unload_after:
                unload_qwen_emotion(qwen_model_path)
            emotion_payloads = []
            for result, weights in zip(raw_results, weights_list):
                dominant = max(weights.items(), key=lambda item: item[1])[0]
                emotion_payloads.append({
                    "emotion": weights,
                    "emotion_source": result.source,
                    "emotion_json": json.dumps(weights, ensure_ascii=False),
                    "dominant_emotion": dominant,
                    "qwen_elapsed_s": result.elapsed_s,
                    "qwen_raw_text": result.raw_text,
                })
        manifest_rows = []
        wav_paths = []
        start_all = time.perf_counter()
        for idx, row in enumerate(rows, start=1):
            row_id = str(_batch_value(row, "id", 0, f"{idx:04d}"))
            text = str(_batch_value(row, "text", 1))
            row_ref_audio = _batch_value(row, "ref_audio", 2) or ref_audio
            row_emotion_ref_audio = _batch_value(row, "emotion_ref_audio", 3) or options.emotion_ref_audio
            row_emotion = _batch_value(row, "emotion", 4) or options.emotion
            row_emo_alpha = _batch_value(row, "emo_alpha", 5, options.emo_alpha)
            row_target_duration = (
                _batch_value(row, ("target_duration_s", "target_duration", "duration"), 6, "")
                or options.target_duration
            )
            if row_emotion_ref_audio and row_emotion:
                # A scene emotion catalog uses the text emotion only to select
                # the reference clip. TTS must receive a single emotion source.
                row_emotion = None
            if not row_ref_audio:
                raise ValueError("Batch row is missing ref_audio and no default ref_audio was provided")
            output_path = output_root / f"{idx:04d}_{_safe_id(row_id, f'{idx:04d}')}.wav"
            row_options = GenerateOptions(**options.__dict__)
            row_options.emotion_ref_audio = row_emotion_ref_audio
            row_options.emotion = row_emotion
            try:
                row_options.emo_alpha = float(row_emo_alpha)
            except (TypeError, ValueError):
                row_options.emo_alpha = options.emo_alpha
            try:
                row_options.target_duration = float(row_target_duration) if row_target_duration else None
            except (TypeError, ValueError):
                row_options.target_duration = options.target_duration
            row_options.fit_duration = _batch_bool(
                _batch_value(row, "fit_duration", 8, ""),
                options.fit_duration,
            )
            row_options.language = str(
                _batch_value(row, ("language", "lang", "语言"), 9, options.language)
                or options.language
            )
            row_max_tokens = _batch_value(row, ("max_tokens", "max_mel_tokens"), 7, "")
            if row_max_tokens:
                try:
                    row_options.max_tokens = int(row_max_tokens)
                except (TypeError, ValueError):
                    row_options.max_tokens = options.max_tokens
            elif row_options.dynamic_max_tokens and row_options.target_duration is None:
                hard_max = options.max_tokens or (
                    1500 if model_version in {"2.0", "2.5"} else 800
                )
                row_options.max_tokens = estimate_mel_tokens_for_text(
                    text,
                    hard_max=hard_max,
                    tokens_per_char=row_options.tokens_per_char,
                    min_tokens=row_options.min_max_tokens,
                )
            row_meta: dict[str, Any] | None = None
            if emotion_payloads is not None:
                row_meta = emotion_payloads[idx - 1].copy()
                row_options.auto_emotion = False
                row_options.use_emo_text = False
                row_options.emo_text = None
                row_options.emotion = row_meta.pop("emotion")
                row_options.emo_alpha = adaptive_emo_alpha(row_options.emotion, base_alpha=options.emo_alpha)
            result = self.generate(
                text=text,
                ref_audio=row_ref_audio,
                output_path=str(output_path),
                profile=profile,
                model=model_path,
                options=row_options,
            )
            if row_meta is not None:
                result.update(row_meta)
            wav_paths.append(output_path)
            manifest_rows.append({
                "index": idx,
                "id": row_id,
                "speaker": _batch_value(row, ("speaker", "role", "说话人", "角色"), 0, ""),
                "text": text,
                "ref_audio": row_ref_audio,
                "emotion_ref_audio": row_emotion_ref_audio or "",
                "emotion": row_options.emotion or "",
                "emo_alpha": row_options.emo_alpha,
                "target_duration_s": row_options.target_duration or "",
                "fit_duration": row_options.fit_duration,
                "max_tokens": row_options.max_tokens or "",
                "language": row_options.language,
                **result,
            })

        manifest_path = output_root / "manifest.csv"
        if manifest_rows:
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(manifest_rows)

        combined_path = None
        if combine and wav_paths:
            from mlx_indextts.cli import _combine_wavs

            combined_path = output_root / "combined.wav"
            _combine_wavs(wav_paths, combined_path, silence_ms=combine_silence_ms)

        return {
            "output_dir": str(output_root),
            "manifest_path": str(manifest_path),
            "combined_path": str(combined_path) if combined_path else None,
            "items": len(rows),
            "elapsed_s": round(time.perf_counter() - start_all, 3),
        }
