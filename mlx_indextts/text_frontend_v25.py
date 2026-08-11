"""Pure text preparation for the public IndexTTS-2.5 inference contract.

The annotation and segmentation order follows:
https://github.com/index-tts/index-tts/blob/9c87c46b84bd0e75ecaefb461e7e8f69bc9ecf44/indextts/infer_v2_5.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from mlx_indextts.tokenizer_v25 import language_id, normalize_v25_language

_PRONUNCIATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
_PROTECTED_PATTERN = re.compile(
    r"<\|SPECIAL_TOKEN_(\d+)\|>.*?<\|SPECIAL_TOKEN_\1\|>"
)
_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|([^|]+)\|>")
_KANA_PATTERN = re.compile(r"^[\u3040-\u309f\u30a0-\u30ff]+$")
_ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_KANA_SEARCH = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_HAN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_SPANISH_PATTERN = re.compile(r"[áéíóúüñ¿¡ÁÉÍÓÚÜÑ]")
_LATIN_PATTERN = re.compile(r"[A-Za-z]")
_PUNCTUATION_SPLIT = re.compile(r"(?<=[，。！？、；：,\.!?;:\n])")


class TokenCounter(Protocol):
    def token_count(self, text: str) -> int: ...

    def encode(self, text: str, *, allowed_special: str = "all") -> list[int]: ...


@dataclass(frozen=True)
class LanguageResolution:
    language: str
    ambiguous: bool = False


@dataclass(frozen=True)
class PreparedV25Text:
    language: str
    language_id: int
    language_ambiguous: bool
    normalized_text: str
    segments: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]


def is_kana(text: str) -> bool:
    return bool(_KANA_PATTERN.fullmatch(text))


def apply_pronunciation_annotations(text: str) -> str:
    """Convert public <text|pronunciation> annotations to model special tokens."""

    def replace(match: re.Match[str]) -> str:
        word = match.group(1)
        pronunciation = match.group(2).upper()
        if is_kana(pronunciation):
            return f" {pronunciation} "
        marker = "SPECIAL_TOKEN_2" if _HAN_PATTERN.search(word) else "SPECIAL_TOKEN_1"
        return f"<|{marker}|>{pronunciation}<|{marker}|>"

    return _PRONUNCIATION_PATTERN.sub(replace, text)


def resolve_v25_language(text: str, requested: str = "auto") -> LanguageResolution:
    """Resolve one released 2.5 language and report Latin-script ambiguity."""

    if requested.strip().lower() != "auto":
        return LanguageResolution(normalize_v25_language(requested))
    if not text or not text.strip():
        raise ValueError("Cannot auto-detect IndexTTS-2.5 language from empty text")

    has_arabic = bool(_ARABIC_PATTERN.search(text))
    has_kana = bool(_KANA_SEARCH.search(text))
    has_han = bool(_HAN_PATTERN.search(text))
    distinctive = sum((has_arabic, has_kana or has_han))
    if distinctive > 1:
        raise ValueError("Mixed Arabic and CJK text requires an explicit language")
    if has_arabic:
        return LanguageResolution("ar")
    if has_kana:
        return LanguageResolution("ja")
    if has_han:
        return LanguageResolution("zh")
    if _SPANISH_PATTERN.search(text):
        return LanguageResolution("es")
    if _LATIN_PATTERN.search(text):
        return LanguageResolution("en", ambiguous=True)
    raise ValueError("Cannot auto-detect IndexTTS-2.5 language; choose zh/en/ja/es/ar")


def _atomic_pieces(text: str) -> list[tuple[str, bool]]:
    pieces: list[tuple[str, bool]] = []
    position = 0
    for match in _PROTECTED_PATTERN.finditer(text):
        if match.start() > position:
            pieces.append((text[position : match.start()], False))
        pieces.append((match.group(0), True))
        position = match.end()
    if position < len(text):
        pieces.append((text[position:], False))
    return pieces


def split_text_by_tokens(
    text: str,
    *,
    tokenizer: TokenCounter,
    max_tokens: int,
    language_prefix: str,
    text_position_capacity: int = 602,
) -> list[str]:
    """Split text within the GPT position budget without breaking annotations."""

    prefix_tokens = tokenizer.token_count(language_prefix)
    budget = max(1, min(max_tokens, text_position_capacity - 2) - prefix_tokens)
    if tokenizer.token_count(text) <= budget:
        return [text]

    chunks: list[str] = []
    for piece, atomic in _atomic_pieces(text):
        if atomic:
            chunks.append(piece)
            continue
        for part in _PUNCTUATION_SPLIT.split(piece):
            if not part:
                continue
            if tokenizer.token_count(part) <= budget:
                chunks.append(part)
                continue
            current = ""
            for character in part:
                if current and tokenizer.token_count(current + character) > budget:
                    chunks.append(current)
                    current = character
                else:
                    current += character
            if current:
                chunks.append(current)

    segments: list[str] = []
    current = ""
    for chunk in chunks:
        if current and tokenizer.token_count(current + chunk) > budget:
            segments.append(current)
            current = chunk
        else:
            current += chunk
    if current:
        segments.append(current)
    return segments or [text]


def prepare_v25_tokens(
    text: str,
    *,
    tokenizer: TokenCounter,
    language: str = "auto",
    max_text_tokens_per_segment: int = 120,
    text_position_capacity: int = 602,
) -> PreparedV25Text:
    """Prepare already-normalized text for language-aware GPT inference."""

    resolution = resolve_v25_language(text, language)
    prepared = text
    if resolution.language in {"zh", "en", "ja"}:
        prepared = prepared.lower()
    elif resolution.language == "es":
        prepared = prepared.upper()
    prepared = apply_pronunciation_annotations(prepared)
    prepared = _SPECIAL_TOKEN_PATTERN.sub(
        lambda match: f"<|{match.group(1).upper()}|>",
        prepared,
    )

    prefix = f"<|{resolution.language}|> "
    segments = split_text_by_tokens(
        prepared,
        tokenizer=tokenizer,
        max_tokens=max_text_tokens_per_segment,
        language_prefix=prefix,
        text_position_capacity=text_position_capacity,
    )
    token_ids = tuple(
        tuple(tokenizer.encode(prefix + segment, allowed_special="all") + [1])
        for segment in segments
    )
    return PreparedV25Text(
        language=resolution.language,
        language_id=language_id(resolution.language),
        language_ambiguous=resolution.ambiguous,
        normalized_text=prepared,
        segments=tuple(segments),
        token_ids=token_ids,
    )
