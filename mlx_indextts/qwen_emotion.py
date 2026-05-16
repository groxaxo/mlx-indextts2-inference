"""MLX-native Qwen text emotion classifier for IndexTTS2.

The official PyTorch IndexTTS2 runtime uses a small Qwen3 model to map text to
the 8 IndexTTS2 emotion weights. This module provides the same front-end
contract through mlx-lm so the main TTS path can remain MLX-native.
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_QWEN_EMOTION_MODEL = "models/qwen0.6bemo4-merge-mlx-8bit"
DEFAULT_QWEN_EMOTION_SOURCE = "checkpoints/qwen0.6bemo4-merge"
OFFICIAL_QWEN_EMOTION_SOURCE = os.environ.get(
    "MLX_INDEXTTS_QWEN_EMOTION_SOURCE",
    DEFAULT_QWEN_EMOTION_SOURCE,
)

EMOTION_ORDER = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)

CN_TO_EN = {
    "高兴": "happy",
    "愤怒": "angry",
    "悲伤": "sad",
    "恐惧": "afraid",
    "反感": "disgusted",
    "低落": "melancholic",
    "惊讶": "surprised",
    "自然": "calm",
}

EN_ALIASES = {
    "happy": "happy",
    "joy": "happy",
    "joyful": "happy",
    "angry": "angry",
    "anger": "angry",
    "sad": "sad",
    "sadness": "sad",
    "afraid": "afraid",
    "fear": "afraid",
    "fearful": "afraid",
    "disgusted": "disgusted",
    "disgust": "disgusted",
    "melancholic": "melancholic",
    "melancholy": "melancholic",
    "depressed": "melancholic",
    "surprised": "surprised",
    "surprise": "surprised",
    "calm": "calm",
    "neutral": "calm",
    "natural": "calm",
}

MELANCHOLIC_WORDS = {
    "低落",
    "melancholy",
    "melancholic",
    "depression",
    "depressed",
    "gloomy",
}


@dataclass
class EmotionResult:
    """Result returned by Qwen text emotion inference."""

    weights: dict[str, float]
    raw_text: str
    elapsed_s: float
    model: str
    source: str = "qwen-mlx"

    @property
    def dominant_emotion(self) -> str:
        return max(self.weights.items(), key=lambda item: item[1])[0]


def clamp_score(value: Any, min_score: float = 0.0, max_score: float = 1.2) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(min_score, min(max_score, number))


def normalize_emotion_dict(content: dict[str, Any], text_input: str = "") -> dict[str, float]:
    """Normalize Qwen output to the IndexTTS2 English emotion order."""
    normalized = {key: 0.0 for key in EMOTION_ORDER}
    for key, value in content.items():
        clean_key = str(key).strip().strip('"').strip("'")
        target = CN_TO_EN.get(clean_key) or EN_ALIASES.get(clean_key.lower())
        if target:
            normalized[target] = clamp_score(value)

    text_lower = str(text_input or "").lower()
    if any(word in text_lower for word in MELANCHOLIC_WORDS):
        normalized["sad"], normalized["melancholic"] = (
            normalized["melancholic"],
            normalized["sad"],
        )

    if all(value <= 0.0 for value in normalized.values()):
        normalized["calm"] = 1.0
    return normalized


def parse_qwen_emotion_output(raw_text: str, text_input: str = "") -> dict[str, float]:
    """Parse JSON or loose key/value output from QwenEmotion."""
    content = str(raw_text or "").strip()
    if not content:
        return normalize_emotion_dict({}, text_input)

    json_candidate = content
    if "{" in content and "}" in content:
        json_candidate = content[content.find("{") : content.rfind("}") + 1]

    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError:
        parsed = {
            match.group(1).strip(): float(match.group(2))
            for match in re.finditer(r'([^":,，。{}\n]+?)"?\s*[:：]\s*([+-]?\d+(?:\.\d+)?)', content)
        }
    if not isinstance(parsed, dict):
        parsed = {}
    return normalize_emotion_dict(parsed, text_input)


def smooth_emotion_sequence(
    weights_list: list[dict[str, float]],
    max_step: float = 0.18,
    total_cap: float = 1.2,
) -> list[dict[str, float]]:
    """Limit adjacent emotion jumps for long-form narration."""
    if not weights_list:
        return []

    smoothed: list[dict[str, float]] = []
    previous: dict[str, float] | None = None
    for weights in weights_list:
        current = {key: clamp_score(weights.get(key, 0.0)) for key in EMOTION_ORDER}
        if previous is not None:
            limited = {}
            for key in EMOTION_ORDER:
                prev_value = previous[key]
                target = current[key]
                delta = target - prev_value
                if abs(delta) > max_step:
                    target = prev_value + max_step if delta > 0 else prev_value - max_step
                limited[key] = max(0.0, round(target, 4))
            current = limited

        total = sum(current.values())
        if total > total_cap:
            scale = total_cap / total
            current = {key: round(value * scale, 4) for key, value in current.items()}
        smoothed.append(current)
        previous = current
    return smoothed


def adaptive_emo_alpha(
    weights: dict[str, float],
    *,
    base_alpha: float = 0.65,
    min_alpha: float = 0.58,
    max_alpha: float = 0.82,
) -> float:
    """Scale emotion intensity to keep Qwen expressive but not unstable."""
    ordered = sorted((clamp_score(value) for value in weights.values()), reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    spread = max(0.0, top1 - top2)
    strength = max(0.0, top1)
    alpha = max(base_alpha, 0.62)
    alpha += 0.16 * max(0.0, strength - 0.45)
    alpha += 0.10 * spread
    return round(max(min_alpha, min(max_alpha, alpha)), 3)


def smooth_emotion_sequence_by_speaker(
    weights_list: list[dict[str, float]],
    speakers: list[str],
    *,
    max_step: float = 0.16,
    cross_speaker_step: float = 0.24,
    speaker_inertia: float = 0.72,
    total_cap: float = 1.2,
) -> list[dict[str, float]]:
    """Smooth emotions while preserving per-speaker continuity.

    This keeps the same role from jumping wildly between adjacent lines, while
    still allowing different speakers to diverge naturally.
    """
    if not weights_list:
        return []

    smoothed: list[dict[str, float]] = []
    speaker_state: dict[str, dict[str, float]] = {}
    previous_global: dict[str, float] | None = None

    for idx, weights in enumerate(weights_list):
        speaker = str(speakers[idx]).strip() if idx < len(speakers) else ""
        current = {key: clamp_score(weights.get(key, 0.0)) for key in EMOTION_ORDER}
        reference = speaker_state.get(speaker)
        limit = max_step
        if reference is None and previous_global is not None:
            reference = previous_global
            limit = cross_speaker_step

        if reference is not None:
            blended = {}
            for key in EMOTION_ORDER:
                target = speaker_inertia * reference[key] + (1.0 - speaker_inertia) * current[key]
                delta = target - reference[key]
                if abs(delta) > limit:
                    target = reference[key] + limit if delta > 0 else reference[key] - limit
                blended[key] = max(0.0, round(target, 4))
            current = blended

        total = sum(current.values())
        if total > total_cap:
            scale = total_cap / total
            current = {key: round(value * scale, 4) for key, value in current.items()}

        smoothed.append(current)
        if speaker:
            speaker_state[speaker] = current
        previous_global = current

    return smoothed


class QwenEmotionMLX:
    """Lazy MLX-native Qwen emotion model wrapper."""

    def __init__(self, model_path: str = DEFAULT_QWEN_EMOTION_MODEL, max_tokens: int = 128):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.prompt = "文本情感分类"
        self._model = None
        self._tokenizer = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        if self.loaded:
            return
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError("Install Qwen emotion dependencies first: uv sync --extra qwen") from exc

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Qwen emotion MLX model not found: {path}. "
                "Run: uv run python scripts/convert_qwen_emotion_mlx.py"
            )
        self._model, self._tokenizer = load(str(path))

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        except Exception:
            pass

    def _render_prompt(self, text_input: str) -> str:
        self.load()
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": str(text_input)},
        ]
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return f"{self.prompt}\n{text_input}\n"

    def _generate_text(self, prompt: str) -> str:
        from mlx_lm import generate

        kwargs = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "verbose": False,
        }
        signature = inspect.signature(generate)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if "sampler" in signature.parameters or accepts_kwargs:
            try:
                from mlx_lm.sample_utils import make_sampler

                kwargs["sampler"] = make_sampler(temp=0.0)
            except Exception:
                pass
        elif "temp" in signature.parameters:
            kwargs["temp"] = 0.0
        elif "temperature" in signature.parameters:
            kwargs["temperature"] = 0.0
        return str(generate(self._model, self._tokenizer, **kwargs))

    def inference(self, text_input: str) -> EmotionResult:
        start = time.perf_counter()
        prompt = self._render_prompt(text_input)
        raw_text = self._generate_text(prompt)
        weights = parse_qwen_emotion_output(raw_text, text_input)
        elapsed = time.perf_counter() - start
        return EmotionResult(
            weights=weights,
            raw_text=raw_text,
            elapsed_s=round(elapsed, 3),
            model=self.model_path,
        )


_QWEN_EMOTION_CACHE: dict[str, QwenEmotionMLX] = {}


def get_qwen_emotion(model_path: str = DEFAULT_QWEN_EMOTION_MODEL) -> QwenEmotionMLX:
    if model_path not in _QWEN_EMOTION_CACHE:
        _QWEN_EMOTION_CACHE[model_path] = QwenEmotionMLX(model_path)
    return _QWEN_EMOTION_CACHE[model_path]


def unload_qwen_emotion(model_path: str | None = None) -> None:
    if model_path is not None:
        model = _QWEN_EMOTION_CACHE.get(model_path)
        if model:
            model.unload()
        return
    for model in _QWEN_EMOTION_CACHE.values():
        model.unload()
