"""LLM-backed, fail-closed sentence direction for native IndexTTS 2.5 controls."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .director_core import (
    DirectionPlan,
    SentenceDirection,
    SentenceUnit,
    direction_from_row,
    direction_row_issues,
    heuristic_direction,
    make_batches,
    reconstruct_source,
    segment_text,
    validate_plan,
)

INDEXTTS25_DIRECTOR_SYSTEM_PROMPT = r"""You are the speech director for IndexTTS 2.5.

For every supplied sentence, return restrained natural delivery metadata. Never
answer, summarize, translate, correct, censor, or rewrite the source. Return one
JSON object and nothing else:

{"sentences":[{"index":0,"emotion":"warm calm","emotion_vector":{"happy":0.10,"angry":0.00,"sad":0.00,"afraid":0.00,"disgusted":0.00,"melancholic":0.00,"surprised":0.00,"calm":0.70},"alpha":0.44,"speed":1.00,"pause_after_ms":150}]}

HARD CONTRACT
- Return every supplied index exactly once and no unknown index.
- Never return spoken/rewritten text, SSML, XML, Markdown, stage directions,
  pseudo-tags such as [laughs], or commentary.
- Use only: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm.
- Emotion values are finite and non-negative and total 0.80.
- alpha: prefer 0.34-0.56; allowed 0.30-0.70.
- speed is a speech-rate multiplier: prefer 0.96-1.04; allowed 0.90-1.10.
- pause_after_ms: prefer 110-220; allowed 60-450.
- Under-act. Ordinary speech is mostly calm; strong emotion is rare.
- Read the complete supplied window first. Keep adjacent delivery continuous.
- A question is not automatically surprise. Profanity is not automatically anger.
- Sarcasm is normally dry/restrained. Professional speech is calm/confident.
- Use longer pauses and slightly slower speed for reflection or sadness; slightly
  faster speed for genuine excitement or urgency. Never translate.
- Do not reproduce pronunciation annotations; they remain in the source text.

When uncertain, choose lower alpha, speed closer to 1.00, and a calm-heavy vector.
Meaning controls emotion; punctuation is only supporting evidence."""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class DirectionError(RuntimeError):
    pass


class LLMDirectionError(DirectionError):
    pass


@dataclass(slots=True, frozen=True)
class DirectorSettings:
    batch_sentences: int = 24
    batch_characters: int = 8_000
    fallback_on_llm_error: bool = True
    max_alpha_step: float = 0.12
    max_speed_step: float = 0.05

    def validate(self) -> None:
        if self.batch_sentences <= 0:
            raise ValueError("batch_sentences must be positive")
        if self.batch_characters < 256:
            raise ValueError("batch_characters must be at least 256")
        if self.max_alpha_step <= 0 or self.max_speed_step <= 0:
            raise ValueError("continuity limits must be positive")


class DirectionAnnotator(Protocol):
    model: str | None

    def annotate(
        self,
        units: Sequence[SentenceUnit],
        *,
        language: str,
        style_prompt: str,
        context_before: str,
        context_after: str,
    ) -> list[Mapping[str, Any]]: ...


class HeuristicAnnotator:
    model = None

    def annotate(self, units: Sequence[SentenceUnit], **_: Any) -> list[Mapping[str, Any]]:
        return [heuristic_direction(unit).as_dict() for unit in units]


class OpenAICompatibleDirector:
    """Strict JSON client for any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "not-needed",
        model: str = "default",
        temperature: float = 0.1,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        max_tokens: int = 4096,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url cannot be blank")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "not-needed"
        self.model = model or "default"
        self.temperature = max(0.0, min(0.5, float(temperature)))
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self.max_tokens = max(256, int(max_tokens))

    @property
    def endpoint(self) -> str:
        return (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )

    @staticmethod
    def _extract_payload(content: str) -> dict[str, Any]:
        raw = str(content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_RE.search(raw)
            if not match:
                raise LLMDirectionError("The model did not return a JSON object") from None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise LLMDirectionError("The model returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise LLMDirectionError("The response JSON must be an object")
        return payload

    def _post(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode())
        if not isinstance(payload, dict):
            raise LLMDirectionError("Endpoint returned a non-object response")
        return payload

    @staticmethod
    def _message_content(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise LLMDirectionError("Response is missing choices")
        choice = choices[0]
        message = choice.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                joined = "".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, Mapping)
                ).strip()
                if joined:
                    return joined
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls and isinstance(calls[0], Mapping):
                function = calls[0].get("function")
                if isinstance(function, Mapping) and isinstance(function.get("arguments"), str):
                    return str(function["arguments"])
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and "{" in reasoning:
                return reasoning
        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return text
        raise LLMDirectionError("Response contains no usable content")

    def annotate(
        self,
        units: Sequence[SentenceUnit],
        *,
        language: str,
        style_prompt: str,
        context_before: str,
        context_after: str,
    ) -> list[Mapping[str, Any]]:
        user_payload = {
            "language": language,
            "style_prompt": style_prompt,
            "context_before": context_before,
            "context_after": context_after,
            "sentences": [{"index": unit.index, "text": unit.text} for unit in units],
        }
        base: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": INDEXTTS25_DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            body = dict(base)
            if attempt == 0:
                body["response_format"] = {"type": "json_object"}
            try:
                parsed = self._extract_payload(self._message_content(self._post(body)))
                rows = parsed.get("sentences")
                if not isinstance(rows, list):
                    raise LLMDirectionError("Response is missing a sentences array")
                return [row for row in rows if isinstance(row, Mapping)]
            except (LLMDirectionError, OSError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(0.5 * 2**attempt, 2.0))
        raise LLMDirectionError(str(last_error or "Unknown direction failure"))


class IndexTTSDirector:
    """Generate a validated N/N source-preserving IndexTTS 2.5 plan."""

    def __init__(
        self,
        annotator: DirectionAnnotator | None = None,
        *,
        settings: DirectorSettings | None = None,
    ) -> None:
        self.annotator = annotator or HeuristicAnnotator()
        self.settings = settings or DirectorSettings()
        self.settings.validate()

    def _annotate_batch(
        self,
        batch: Sequence[SentenceUnit],
        **context: str,
    ) -> tuple[list[SentenceDirection], list[str]]:
        warnings: list[str] = []
        if isinstance(self.annotator, HeuristicAnnotator):
            return [heuristic_direction(unit) for unit in batch], warnings
        try:
            rows = self.annotator.annotate(batch, **context)
        except Exception as exc:
            if not self.settings.fallback_on_llm_error:
                raise
            message = f"Direction model failed: {exc}"
            return [heuristic_direction(unit, warning=message) for unit in batch], [message]

        valid_indexes = {unit.index for unit in batch}
        indexed: dict[int, Mapping[str, Any]] = {}
        duplicates: set[int] = set()
        structural_warnings: list[str] = []
        for position, row in enumerate(rows):
            raw_index = row.get("index")
            index: int | None = None
            if isinstance(raw_index, int) and not isinstance(raw_index, bool):
                index = raw_index
            elif isinstance(raw_index, str) and re.fullmatch(r"[+-]?\d+", raw_index.strip()):
                index = int(raw_index)
            if index is None:
                structural_warnings.append(
                    f"Direction row {position} has an invalid index and was ignored."
                )
                continue
            if index not in valid_indexes:
                structural_warnings.append(
                    f"Direction row references unknown sentence {index} and was ignored."
                )
                continue
            if index in indexed:
                duplicates.add(index)
                continue
            indexed[index] = row

        if duplicates:
            structural_warnings.extend(
                f"Duplicate directions for sentence {index}; deterministic repair used."
                for index in sorted(duplicates)
            )
        if structural_warnings and not self.settings.fallback_on_llm_error:
            raise LLMDirectionError(" ".join(structural_warnings))
        warnings.extend(structural_warnings)

        output: list[SentenceDirection] = []
        for unit in batch:
            if unit.index in duplicates:
                message = (
                    f"Duplicate directions for sentence {unit.index}; "
                    "deterministic repair used."
                )
                output.append(heuristic_direction(unit, warning=message))
                continue
            row = indexed.get(unit.index)
            if row is None:
                message = f"Missing direction for sentence {unit.index}; deterministic repair used."
                if not self.settings.fallback_on_llm_error:
                    raise LLMDirectionError(message)
                warnings.append(message)
                output.append(heuristic_direction(unit, warning=message))
                continue

            issues = direction_row_issues(row)
            if issues and not self.settings.fallback_on_llm_error:
                raise LLMDirectionError(
                    f"Invalid direction for sentence {unit.index}: " + "; ".join(issues)
                )
            try:
                direction = direction_from_row(unit, row)
            except Exception as exc:
                message = (
                    f"Invalid direction for sentence {unit.index}: {exc}; "
                    "deterministic repair used."
                )
                if not self.settings.fallback_on_llm_error:
                    raise LLMDirectionError(message) from exc
                warnings.append(message)
                output.append(heuristic_direction(unit, warning=message))
                continue
            output.append(direction)
            if direction.warning:
                warnings.append(f"Sentence {unit.index}: {direction.warning}.")
        return output, warnings

    def _continuity(
        self, text: str, directions: Sequence[SentenceDirection]
    ) -> list[SentenceDirection]:
        output: list[SentenceDirection] = []
        previous: SentenceDirection | None = None
        for index, current in enumerate(directions):
            if previous:
                alpha = max(
                    previous.alpha - self.settings.max_alpha_step,
                    min(previous.alpha + self.settings.max_alpha_step, current.alpha),
                )
                speed = max(
                    previous.speed - self.settings.max_speed_step,
                    min(previous.speed + self.settings.max_speed_step, current.speed),
                )
                current = replace(current, alpha=round(alpha, 4), speed=round(speed, 4))
            if index + 1 < len(directions):
                between = text[current.end : directions[index + 1].start]
                if re.search(r"\n\s*\n", between):
                    current = replace(current, pause_after_ms=max(current.pause_after_ms, 260))
            elif current.pause_after_ms < 180:
                current = replace(current, pause_after_ms=180)
            output.append(current)
            previous = current
        return output

    def direct(
        self,
        text: str,
        *,
        language: str = "auto",
        style_prompt: str = "",
    ) -> DirectionPlan:
        if not str(text or "").strip():
            raise ValueError("text cannot be blank")
        units = segment_text(text)
        if reconstruct_source(text, units) != text:
            raise DirectionError("Source segmentation failed exact reconstruction")
        batches = make_batches(
            units, self.settings.batch_sentences, self.settings.batch_characters
        )
        position = {unit.index: index for index, unit in enumerate(units)}
        directions: list[SentenceDirection] = []
        warnings: list[str] = []
        for batch in batches:
            first, last = position[batch[0].index], position[batch[-1].index]
            rows, batch_warnings = self._annotate_batch(
                batch,
                language=language,
                style_prompt=style_prompt,
                context_before=units[first - 1].text if first else "",
                context_after=units[last + 1].text if last + 1 < len(units) else "",
            )
            directions.extend(rows)
            warnings.extend(batch_warnings)
        directions.sort(key=lambda item: item.index)
        plan = DirectionPlan(
            text,
            language,
            style_prompt,
            getattr(self.annotator, "model", None),
            self._continuity(text, directions),
            list(dict.fromkeys(warnings)),
        )
        errors = validate_plan(plan)
        if errors:
            raise DirectionError("Final direction plan failed validation: " + "; ".join(errors))
        return plan


__all__ = [
    "INDEXTTS25_DIRECTOR_SYSTEM_PROMPT",
    "DirectionError",
    "LLMDirectionError",
    "DirectorSettings",
    "HeuristicAnnotator",
    "OpenAICompatibleDirector",
    "IndexTTSDirector",
    "DirectionPlan",
    "SentenceDirection",
    "SentenceUnit",
    "segment_text",
    "validate_plan",
]
