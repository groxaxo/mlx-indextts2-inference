"""Japanese segmentation and optional Kana replacement for IndexTTS-2.5.

Behavior follows the official processor at:
https://github.com/index-tts/index-tts/blob/9c87c46b84bd0e75ecaefb461e7e8f69bc9ecf44/indextts/utils/ja_g2p.py
"""

from __future__ import annotations

import random
import re
from typing import Any


class JapaneseG2PProcessor:
    """Tokenize Japanese and optionally replace Kanji tokens with Hiragana."""

    def __init__(
        self,
        g2p_ratio: float = 0.0,
        *,
        tagger: Any | None = None,
        random_generator: random.Random | None = None,
    ) -> None:
        if not 0.0 <= g2p_ratio <= 1.0:
            raise ValueError("g2p_ratio must be between 0 and 1")
        self.g2p_ratio = g2p_ratio
        self.random = random_generator or random
        self.tagger = tagger or self._create_tagger()

    @staticmethod
    def _create_tagger():
        try:
            import fugashi
        except ImportError as exc:
            raise RuntimeError(
                "IndexTTS-2.5 Japanese support requires the v25 extra: "
                "uv sync --extra v25"
            ) from exc
        return fugashi.Tagger()

    def tokenize(self, text: str) -> list[tuple[str, str]]:
        """Return surface and Katakana reading pairs."""

        tokens: list[tuple[str, str]] = []
        for token in self.tagger(text):
            surface = token.surface
            try:
                reading = token.feature.kana
                if not reading or reading == "*":
                    reading = surface
            except AttributeError:
                features = token.feature.split(",")
                reading = (
                    features[7]
                    if len(features) > 7 and features[7] != "*"
                    else surface
                )
            tokens.append((surface, reading))
        return tokens

    @staticmethod
    def katakana_to_hiragana(text: str) -> str:
        return "".join(
            chr(ord(character) - 0x60)
            if 0x30A1 <= ord(character) <= 0x30F6
            else character
            for character in text
        )

    @staticmethod
    def _has_kanji(text: str) -> bool:
        return any("\u4e00" <= character <= "\u9fff" for character in text)

    def _process_segment(self, text: str) -> str:
        tokens = self.tokenize(text)
        kanji_indices = [
            index
            for index, (surface, _) in enumerate(tokens)
            if self._has_kanji(surface)
        ]
        replace_count = int(len(kanji_indices) * self.g2p_ratio)
        if (
            replace_count == 0
            and kanji_indices
            and self.random.random() < self.g2p_ratio
        ):
            replace_count = 1
        replace_indices = set(
            self.random.sample(
                kanji_indices,
                min(replace_count, len(kanji_indices)),
            )
        )
        return " ".join(
            self.katakana_to_hiragana(reading)
            if index in replace_indices
            else surface
            for index, (surface, reading) in enumerate(tokens)
        )

    def process(self, text: str) -> str:
        """Process non-space spans while preserving original space positions."""

        parts = re.split(r"( +)", text)
        return "".join(
            self._process_segment(part) if part.strip() else part
            for part in parts
        )
