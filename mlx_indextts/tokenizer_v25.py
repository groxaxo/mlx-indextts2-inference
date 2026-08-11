"""IndexTTS-2.5 multilingual tiktoken frontend.

Token ordering mirrors the official implementation at:
https://github.com/index-tts/index-tts/blob/9c87c46b84bd0e75ecaefb461e7e8f69bc9ecf44/indextts/utils/tokenizer.py
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import AbstractSet

import tiktoken

from mlx_indextts.model_version import (
    V25_LANGUAGES,
    V25_TEXT_VOCAB_SIZE,
    V25_TOKENIZER,
    ModelFormatError,
)

LANGUAGE_CODES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh",
    "common",
)
LANGUAGE_IDS = {code: index for index, code in enumerate(LANGUAGE_CODES)}
LANGUAGE_ALIASES = {
    "chinese": "zh",
    "mandarin": "zh",
    "cn": "zh",
    "zh-cn": "zh",
    "zhen": "zh",
    "english": "en",
    "japanese": "ja",
    "spanish": "es",
    "castilian": "es",
    "arabic": "ar",
}

_AUDIO_EVENTS = (
    "ASR",
    "AED",
    "SER",
    "Speech",
    "/Speech",
    "BGM",
    "/BGM",
    "Laughter",
    "/Laughter",
    "Applause",
    "/Applause",
)
_EMOTIONS = ("HAPPY", "SAD", "ANGRY", "NEUTRAL")
_TTS_VOCAL_TOKENS = (
    "TTS/B",
    "TTS/O",
    "TTS/Q",
    "TTS/A",
    "TTS/CO",
    "TTS/CL",
    "TTS/H",
    *(f"TTS/SP{i:02d}" for i in range(1, 14)),
)
_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|"
    r"\s+(?!\S)|\s+"
)


def normalize_v25_language(language: str) -> str:
    """Normalize one explicit language to the five released 2.5 languages."""

    normalized = str(language).strip().lower()
    normalized = LANGUAGE_ALIASES.get(normalized, normalized)
    if normalized not in V25_LANGUAGES:
        raise ValueError(
            f"Unsupported IndexTTS-2.5 language: {language}; "
            f"expected one of: {', '.join(V25_LANGUAGES)}"
        )
    return normalized


def language_id(language: str) -> int:
    """Return the official categorical language ID."""

    return LANGUAGE_IDS[normalize_v25_language(language)]


def _special_token_strings(num_languages: int = 99) -> list[str]:
    return [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *(f"<|{language}|>" for language in LANGUAGE_CODES[:num_languages]),
        *(f"<|{event}|>" for event in _AUDIO_EVENTS),
        *(f"<|{emotion}|>" for emotion in _EMOTIONS),
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *(f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)),
        *(f"<|{token}|>" for token in _TTS_VOCAL_TOKENS),
        *(f"<|{i * 0.02:.2f}|>" for i in range(1501)),
    ]


@lru_cache(maxsize=None)
def _load_encoding(vocab_path: str, num_languages: int = 99) -> tiktoken.Encoding:
    path = Path(vocab_path)
    try:
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (
                line.split()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            )
        }
    except (OSError, ValueError) as exc:
        raise ModelFormatError(f"Cannot read IndexTTS-2.5 tokenizer: {path}") from exc

    special_tokens: dict[str, int] = {}
    next_id = len(ranks)
    for token in _special_token_strings(num_languages):
        special_tokens[token] = next_id
        next_id += 1

    return tiktoken.Encoding(
        name=path.name,
        explicit_n_vocab=next_id,
        pat_str=_PATTERN,
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


class IndexTTS25Tokenizer:
    """Small runtime wrapper around the official 2.5 tiktoken encoding."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        expected_vocab_size: int | None = V25_TEXT_VOCAB_SIZE,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.vocab_path = self.model_dir / V25_TOKENIZER
        self.encoding = _load_encoding(str(self.vocab_path.resolve()))
        self.vocab_size = self.encoding.n_vocab
        if expected_vocab_size is not None and self.vocab_size != expected_vocab_size:
            raise ModelFormatError(
                f"IndexTTS-2.5 tokenizer has {self.vocab_size} tokens; "
                f"expected {expected_vocab_size}: {self.vocab_path}"
            )

    def encode(
        self,
        text: str,
        *,
        allowed_special: AbstractSet[str] | str = "all",
    ) -> list[int]:
        return self.encoding.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)

    def token_count(self, text: str) -> int:
        return len(self.encode(text, allowed_special="all"))


@lru_cache(maxsize=None)
def get_tokenizer_v25(model_dir: str | Path) -> IndexTTS25Tokenizer:
    return IndexTTS25Tokenizer(Path(model_dir).resolve())
