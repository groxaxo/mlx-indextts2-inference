"""Source-preserving sentence controls shared by the IndexTTS 2.5 director."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
EMOTION_VECTOR_MASS = 0.8

_BLOCK_RE = re.compile(r"\S(?:.*?\S)?(?=(?:\n\s*\n)|(?:\s*\Z))", re.DOTALL)
_SENTENCE_RE = re.compile(
    r".+?(?:[.!?！？。؟…]+(?:[\"'»”’）)\]】』」]*)|$)", re.DOTALL
)
_ALWAYS_CONTINUE_RE = re.compile(
    r"(?:(?i:\b(?:mr|mrs|ms|dr|prof|sr|sra|srta|dra|jr|st|ud|uds|no|núm|fig|dept)\.)|(?:\b[A-ZÁÉÍÓÚÑ]\.){1,3})\s*$"
)
_CONTEXTUAL_RE = re.compile(
    r"\b(?:vs|etc|inc|ltd|e\.g|i\.e|a\.m|p\.m)\.\s*$", re.IGNORECASE
)
_PRONUNCIATION_RE = re.compile(r"<[^|>\n]+\|[^>\n]+>")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_DECIMAL_RE = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)+(?!\w)")
_DOTTED_TOKEN_RE = re.compile(r"\b(?:[A-Za-z0-9_-]+\.)+[A-Za-z0-9_-]+\b")
_PROTECTED_PUNCTUATION = frozenset(".!?！？。؟…")
_PROTECTED_PLACEHOLDER = "\ue000"
_URL_TRAILING_PUNCTUATION = frozenset(".!?！？。؟…,;:)]}”’\"'")


@dataclass(slots=True, frozen=True)
class SentenceUnit:
    index: int
    start: int
    end: int
    text: str


@dataclass(slots=True, frozen=True)
class SentenceDirection:
    index: int
    start: int
    end: int
    text: str
    emotion: str
    emotion_vector: tuple[float, ...]
    alpha: float
    speed: float
    pause_after_ms: int
    source: str = "llm"
    warning: str | None = None

    def vector_dict(self) -> dict[str, float]:
        return {
            name: round(value, 4)
            for name, value in zip(EMOTION_ORDER, self.emotion_vector, strict=True)
        }

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "emotion": self.emotion,
            "emotion_vector": self.vector_dict(),
            "alpha": round(self.alpha, 4),
            "speed": round(self.speed, 4),
            "duration_factor": round(1.0 / self.speed, 4),
            "pause_after_ms": self.pause_after_ms,
            "source": self.source,
        }
        if self.warning:
            result["warning"] = self.warning
        return result


@dataclass(slots=True)
class DirectionPlan:
    original_text: str
    language: str
    style_prompt: str
    model: str | None
    directions: list[SentenceDirection]
    warnings: list[str] = field(default_factory=list)

    @property
    def sentence_count(self) -> int:
        return len(self.directions)

    @property
    def directed_sentence_count(self) -> int:
        return len(self.directions)

    @property
    def undirected_sentence_indexes(self) -> list[int]:
        return []

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "language": self.language,
            "style_prompt": self.style_prompt,
            "model": self.model,
            "sentence_count": self.sentence_count,
            "directed_sentence_count": self.directed_sentence_count,
            "undirected_sentence_indexes": self.undirected_sentence_indexes,
            "warnings": list(self.warnings),
            "directions": [item.as_dict() for item in self.directions],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=indent)

    def to_markup(self) -> str:
        blocks: list[str] = []
        for item in self.directions:
            vector = ",".join(
                f"{name}:{value:.3f}"
                for name, value in zip(
                    EMOTION_ORDER, item.emotion_vector, strict=True
                )
                if value > 0.0005
            )
            blocks.append(
                f'[SEG index="{item.index}" emotion="{item.emotion}" '
                f'vector="{vector}" alpha="{item.alpha:.3f}" '
                f'speed="{item.speed:.3f}" pause_after="{item.pause_after_ms}"]\n'
                f"{item.text}\n[/SEG]"
            )
        return "\n".join(blocks)


def _should_merge(fragment: str, following: str) -> bool:
    stripped, next_text = fragment.rstrip(), following.lstrip()
    left = re.search(r"\b([apei])\.$", stripped, re.IGNORECASE)
    right = re.match(r"([mge])\.", next_text, re.IGNORECASE)
    if left and right and (left.group(1) + right.group(1)).lower() in {
        "am",
        "pm",
        "eg",
        "ie",
    }:
        return True
    if _ALWAYS_CONTINUE_RE.search(stripped):
        return True
    if _CONTEXTUAL_RE.search(stripped) or re.search(r"\b\d+\.$", stripped):
        return bool(next_text and next_text[0].islower())
    return False


def _protected_text(block: str) -> str:
    """Mask sentence punctuation inside tokens that must remain atomic."""

    characters = list(block)
    patterns = (
        _PRONUNCIATION_RE,
        _URL_RE,
        _EMAIL_RE,
        _DECIMAL_RE,
        _DOTTED_TOKEN_RE,
    )
    for pattern in patterns:
        for match in pattern.finditer(block):
            start, end = match.span()
            # URLs commonly occur immediately before sentence punctuation.  Keep
            # that final punctuation visible to the sentence splitter.
            if pattern is _URL_RE:
                while end > start and block[end - 1] in _URL_TRAILING_PUNCTUATION:
                    end -= 1
            for index in range(start, end):
                if characters[index] in _PROTECTED_PUNCTUATION:
                    characters[index] = _PROTECTED_PLACEHOLDER
    return "".join(characters)


def _segment_block(block: str) -> list[str]:
    protected = _protected_text(block)
    spans = [(match.start(), match.end()) for match in _SENTENCE_RE.finditer(protected)]
    raw = [block[start:end] for start, end in spans] or [block]
    merged: list[str] = []
    for segment in raw:
        if merged and _should_merge(merged[-1], segment):
            merged[-1] += segment
        else:
            merged.append(segment)
    return merged


def segment_text(text: str) -> list[SentenceUnit]:
    """Segment source text while retaining exact character offsets."""

    units: list[SentenceUnit] = []
    index = 0
    for match in _BLOCK_RE.finditer(text):
        block, cursor = match.group(0), 0
        for segment in _segment_block(block):
            relative = block.find(segment, cursor)
            if relative < 0:
                relative = cursor
            start = match.start() + relative
            end = start + len(segment)
            cursor = relative + len(segment)
            if segment.strip():
                units.append(SentenceUnit(index, start, end, segment))
                index += 1
    if not units and text.strip():
        start, end = len(text) - len(text.lstrip()), len(text.rstrip())
        units.append(SentenceUnit(0, start, end, text[start:end]))
    return units


def reconstruct_source(text: str, units: Sequence[SentenceUnit]) -> str:
    if not units:
        return text
    output: list[str] = []
    cursor = 0
    for unit in units:
        output.extend((text[cursor : unit.start], unit.text))
        cursor = unit.end
    output.append(text[cursor:])
    return "".join(output)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def finite_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def vector_from_label(label: str) -> tuple[float, ...]:
    text = str(label or "").lower()
    weights = {name: 0.0 for name in EMOTION_ORDER}
    rules: tuple[tuple[str, float, tuple[str, ...]], ...] = (
        ("happy", 0.55, ("happy", "joy", "excited", "alegre", "feliz")),
        ("happy", 0.25, ("warm", "amused", "playful", "affectionate")),
        (
            "angry",
            0.55,
            ("angry", "frustrated", "firm", "furious", "enoj", "ira", "molest"),
        ),
        ("sad", 0.45, ("sad", "grief", "sorrow", "triste", "dolor")),
        (
            "afraid",
            0.45,
            ("afraid", "fear", "tense", "worried", "nervous", "miedo", "preocup"),
        ),
        ("disgusted", 0.45, ("disgust", "repuls", "asco")),
        (
            "melancholic",
            0.40,
            ("melanch", "reflective", "subdued", "low", "nostalg", "deprim"),
        ),
        ("surprised", 0.55, ("surpris", "astonish", "shock", "sorpr")),
        (
            "calm",
            0.70,
            (
                "calm",
                "neutral",
                "natural",
                "professional",
                "serious",
                "confident",
                "thoughtful",
                "relaxed",
                "tranquil",
                "curious",
            ),
        ),
    )
    matched = False
    for emotion, weight, words in rules:
        if any(word in text for word in words):
            weights[emotion] += weight
            matched = True
    if not matched:
        weights["calm"] = 1.0
    elif not weights["calm"]:
        weights["calm"] = 0.35
    total = sum(weights.values())
    return tuple(
        round(weights[name] * EMOTION_VECTOR_MASS / total, 6)
        for name in EMOTION_ORDER
    )


def normalize_emotion_vector(
    value: Mapping[str, Any] | Sequence[Any] | None,
    *,
    fallback_label: str = "calm",
) -> tuple[float, ...]:
    values: list[float]
    if isinstance(value, Mapping):
        lowered = {str(key).strip().lower(): raw for key, raw in value.items()}
        values = [
            max(0.0, finite_float(lowered.get(name), 0.0))
            for name in EMOTION_ORDER
        ]
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        values = (
            [max(0.0, finite_float(item, 0.0)) for item in value]
            if len(value) == len(EMOTION_ORDER)
            else []
        )
    else:
        values = []
    if not values or sum(values) <= 0.0:
        return vector_from_label(fallback_label)
    scale = EMOTION_VECTOR_MASS / sum(values)
    return tuple(round(item * scale, 6) for item in values)



def direction_row_issues(row: Mapping[str, Any]) -> list[str]:
    """Describe contract violations that the deterministic normalizer will repair."""

    issues: list[str] = []
    emotion = row.get("emotion") or row.get("delivery")
    if not isinstance(emotion, str) or not emotion.strip():
        issues.append("missing or invalid emotion label")

    vector = row.get("emotion_vector")
    values: list[float] = []
    if isinstance(vector, Mapping):
        lowered = {str(key).strip().lower(): raw for key, raw in vector.items()}
        missing = [name for name in EMOTION_ORDER if name not in lowered]
        if missing:
            issues.append("emotion vector omitted: " + ", ".join(missing))
        for name in EMOTION_ORDER:
            raw = lowered.get(name, 0.0)
            number = finite_float(raw, math.nan)
            if not math.isfinite(number) or number < 0.0:
                issues.append(f"emotion vector has invalid {name} value")
                number = 0.0
            values.append(number)
    elif isinstance(vector, Sequence) and not isinstance(
        vector, (str, bytes, bytearray)
    ):
        if len(vector) != len(EMOTION_ORDER):
            issues.append("emotion vector does not contain exactly eight values")
        else:
            for position, raw in enumerate(vector):
                number = finite_float(raw, math.nan)
                if not math.isfinite(number) or number < 0.0:
                    issues.append(
                        f"emotion vector has invalid {EMOTION_ORDER[position]} value"
                    )
                    number = 0.0
                values.append(number)
    else:
        issues.append("missing or invalid emotion vector")

    if values:
        total = sum(values)
        if total <= 0.0:
            issues.append("emotion vector has zero total mass")
        elif not math.isclose(total, EMOTION_VECTOR_MASS, abs_tol=0.02):
            issues.append(
                f"emotion vector total {total:.4f} was normalized to {EMOTION_VECTOR_MASS:.1f}"
            )

    numeric_contracts = (
        ("alpha", 0.30, 0.70),
        ("speed", 0.90, 1.10),
        ("pause_after_ms", 60.0, 450.0),
    )
    for name, lower, upper in numeric_contracts:
        if name not in row:
            issues.append(f"missing {name}")
            continue
        value = finite_float(row.get(name), math.nan)
        if not math.isfinite(value):
            issues.append(f"{name} is not finite")
        elif not lower <= value <= upper:
            issues.append(f"{name} is outside {lower:g}-{upper:g}")

    supplied = row.get("text") or row.get("tagged_text")
    if supplied is not None:
        issues.append("model returned spoken text even though metadata-only output was required")
    return list(dict.fromkeys(issues))


_POSITIVE_RE = re.compile(
    r"\b(happy|glad|great|excellent|amazing|love|finally|success|feliz|alegre|"
    r"genial|excelente|increíble|encanta|por fin|logramos)\b",
    re.IGNORECASE,
)
_ANGER_RE = re.compile(
    r"\b(angry|furious|rage|hate this|unacceptable|enojad|furioso|rabia|"
    r"odio esto|inaceptable)\b",
    re.IGNORECASE,
)
_SAD_RE = re.compile(
    r"\b(sad|grief|heartbroken|lost|miss you|triste|dolor|desolad|perdimos|"
    r"te extraño)\b",
    re.IGNORECASE,
)
_FEAR_RE = re.compile(
    r"\b(afraid|terrified|panic|danger|worried|nervous|miedo|terror|pánico|"
    r"peligro|preocupad|nervios)\b",
    re.IGNORECASE,
)
_DISGUST_RE = re.compile(
    r"\b(disgust|revolting|repulsive|asco|repugnante)\b", re.IGNORECASE
)
_SURPRISE_RE = re.compile(
    r"\b(surprised|unexpected|cannot believe|no way|sorprendid|inesperad|"
    r"no puedo creer|imposible)\b",
    re.IGNORECASE,
)
_REFLECTIVE_RE = re.compile(
    r"\b(maybe|perhaps|wonder|remember|sometimes|quizás|tal vez|me pregunto|"
    r"recuerdo|a veces)\b",
    re.IGNORECASE,
)


def heuristic_direction(
    unit: SentenceUnit, *, warning: str | None = None
) -> SentenceDirection:
    text = unit.text.strip()
    label, alpha, speed = "warm calm", 0.40, 1.0
    if _ANGER_RE.search(text):
        label, alpha, speed = "restrained angry serious", 0.55, 1.01
    elif _FEAR_RE.search(text):
        label, alpha, speed = "concerned tense calm", 0.49, 0.98
    elif _SAD_RE.search(text):
        label, alpha, speed = "subdued sad melancholic calm", 0.48, 0.95
    elif _DISGUST_RE.search(text):
        label, alpha, speed = "restrained disgusted calm", 0.49, 0.98
    elif _SURPRISE_RE.search(text):
        label, alpha, speed = "pleasantly surprised calm", 0.49, 1.02
    elif _POSITIVE_RE.search(text):
        label, alpha, speed = "warm gently happy", 0.48, 1.02
    elif _REFLECTIVE_RE.search(text) or "..." in text or "…" in text:
        label, alpha, speed = "thoughtful melancholic calm", 0.43, 0.96
    elif text.endswith(("?", "？", "؟")):
        label, alpha, speed = "curious calm", 0.41, 1.01
    elif text.endswith(("!", "！")):
        label, alpha, speed = "confident warm calm", 0.45, 1.02

    if text.endswith(("...", "…")):
        pause = 250
    elif text.endswith(("?", "？", "؟")):
        pause = 180
    elif text.endswith(("!", "！")):
        pause = 170
    elif text.endswith((";", "；", ":", "：")):
        pause = 125
    else:
        pause = 155
    return SentenceDirection(
        unit.index,
        unit.start,
        unit.end,
        unit.text,
        label,
        vector_from_label(label),
        alpha,
        speed,
        pause,
        "heuristic",
        warning,
    )


def direction_from_row(
    unit: SentenceUnit, row: Mapping[str, Any]
) -> SentenceDirection:
    emotion = re.sub(
        r"[\r\n\t]+",
        " ",
        str(row.get("emotion") or row.get("delivery") or "warm calm").strip(),
    )[:96] or "warm calm"
    emotion = emotion.replace('"', "'").replace("<", "").replace(">", "")
    issues = direction_row_issues(row)
    supplied = row.get("text") or row.get("tagged_text")
    if supplied is not None and str(supplied) != unit.text:
        issues.append("model-supplied text was ignored; original source text was retained")
    warning = "; ".join(dict.fromkeys(issues)) or None
    return SentenceDirection(
        unit.index,
        unit.start,
        unit.end,
        unit.text,
        emotion,
        normalize_emotion_vector(row.get("emotion_vector"), fallback_label=emotion),
        round(clamp(finite_float(row.get("alpha"), 0.43), 0.30, 0.70), 4),
        round(clamp(finite_float(row.get("speed"), 1.0), 0.90, 1.10), 4),
        int(
            round(
                clamp(
                    finite_float(row.get("pause_after_ms"), 155.0),
                    60.0,
                    450.0,
                )
            )
        ),
        "llm-repair" if warning else "llm",
        warning,
    )


def make_batches(
    units: Sequence[SentenceUnit], max_sentences: int, max_characters: int
) -> list[list[SentenceUnit]]:
    batches: list[list[SentenceUnit]] = []
    current: list[SentenceUnit] = []
    characters = 0
    for unit in units:
        overflow = current and (
            len(current) >= max_sentences
            or characters + len(unit.text) > max_characters
        )
        if overflow:
            batches.append(current)
            current, characters = [], 0
        current.append(unit)
        characters += len(unit.text)
    if current:
        batches.append(current)
    return batches


def validate_plan(plan: DirectionPlan) -> list[str]:
    errors: list[str] = []
    units = segment_text(plan.original_text)
    if reconstruct_source(plan.original_text, units) != plan.original_text:
        errors.append("Sentence spans do not reconstruct the original text exactly")
    if len(units) != len(plan.directions):
        errors.append(f"Direction count mismatch: {len(plan.directions)}/{len(units)}")
        return errors
    if [item.index for item in plan.directions] != [unit.index for unit in units]:
        errors.append("Directions are missing, duplicated, or out of source order")
    for unit, item in zip(units, plan.directions, strict=True):
        if (item.start, item.end, item.text) != (unit.start, unit.end, unit.text):
            errors.append(f"Direction {unit.index} does not reference exact source text")
        if len(item.emotion_vector) != len(EMOTION_ORDER) or not math.isclose(
            sum(item.emotion_vector), EMOTION_VECTOR_MASS, abs_tol=1e-4
        ):
            errors.append(f"Direction {unit.index} has an invalid emotion vector")
        if not 0.30 <= item.alpha <= 0.70:
            errors.append(f"Direction {unit.index} alpha is outside 0.30-0.70")
        if not 0.90 <= item.speed <= 1.10:
            errors.append(f"Direction {unit.index} speed is outside 0.90-1.10")
        if not 60 <= item.pause_after_ms <= 450:
            errors.append(f"Direction {unit.index} pause is outside 60-450 ms")
    return errors
