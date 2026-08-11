"""Language normalization pipeline for IndexTTS-2.5."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

from mlx_indextts.japanese_v25 import JapaneseG2PProcessor
from mlx_indextts.normalize import TextNormalizer
from mlx_indextts.text_frontend_v25 import PreparedV25Text, prepare_v25_tokens
from mlx_indextts.tokenizer_v25 import IndexTTS25Tokenizer, get_tokenizer_v25

logger = logging.getLogger(__name__)


class OptionalNemoNormalizer:
    """Lazy optional NeMo wrapper matching the upstream pass-through fallback.

    Source:
    https://github.com/index-tts/index-tts/blob/9c87c46b84bd0e75ecaefb461e7e8f69bc9ecf44/indextts/utils/nemo_tn.py
    """

    def __init__(self, factory: Callable[[str], Any] | None = None) -> None:
        self.factory = factory
        self._cache: dict[str, Any | None] = {}

    @staticmethod
    def _default_factory(language: str):
        from nemo_text_processing.text_normalization.normalize import Normalizer

        return Normalizer(input_case="cased", lang=language)

    def normalize(self, text: str, language: str) -> str:
        if language != "es":
            return text
        if language not in self._cache:
            try:
                factory = self.factory or self._default_factory
                self._cache[language] = factory(language)
            except Exception as exc:
                logger.warning(
                    "NeMo text normalization unavailable for %s; using raw text: %s",
                    language,
                    exc,
                )
                self._cache[language] = None
        normalizer = self._cache[language]
        if normalizer is None:
            return text
        try:
            return str(normalizer.normalize(text, verbose=False))
        except Exception as exc:
            logger.warning(
                "NeMo text normalization failed for %s; using raw text: %s",
                language,
                exc,
            )
            return text


class IndexTTS25TextNormalizer:
    """Apply the released language-specific normalization policy."""

    def __init__(
        self,
        *,
        zh_en_normalizer: TextNormalizer | None = None,
        nemo_normalizer: OptionalNemoNormalizer | None = None,
    ) -> None:
        self.zh_en = zh_en_normalizer or TextNormalizer(enable_glossary=True)
        self.nemo = nemo_normalizer or OptionalNemoNormalizer()
        self._zh_en_loaded = False
        replacements = sorted(
            TextNormalizer.CHAR_REP_MAP,
            key=len,
            reverse=True,
        )
        self._clean_pattern = re.compile("|".join(map(re.escape, replacements)))

    def load_glossary(self, path: str | Path) -> bool:
        return self.zh_en.load_glossary_from_yaml(str(path))

    def __call__(self, text: str, language: str, enabled: bool = True) -> str:
        cleaned = self._clean_pattern.sub(
            lambda match: TextNormalizer.CHAR_REP_MAP[match.group()],
            text,
        )
        if not enabled:
            return cleaned
        if language in {"zh", "en"}:
            if not self._zh_en_loaded:
                self.zh_en.load()
                self._zh_en_loaded = True
            return self.zh_en.normalize(cleaned)
        if language == "es":
            return self.nemo.normalize(cleaned, language)
        return cleaned


class IndexTTS25TextFrontend:
    """Complete tokenizer-facing text frontend for a converted 2.5 model."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        tokenizer: IndexTTS25Tokenizer | None = None,
        normalizer: IndexTTS25TextNormalizer | None = None,
        japanese_processor: JapaneseG2PProcessor | None = None,
    ) -> None:
        self.tokenizer = tokenizer or get_tokenizer_v25(model_dir)
        self.normalizer = normalizer or IndexTTS25TextNormalizer()
        self.japanese = japanese_processor or JapaneseG2PProcessor(g2p_ratio=0)

    def prepare(
        self,
        text: str,
        *,
        language: str = "auto",
        text_normalization: bool = True,
        max_text_tokens_per_segment: int = 120,
        text_position_capacity: int = 602,
    ) -> PreparedV25Text:
        return prepare_v25_tokens(
            text,
            tokenizer=self.tokenizer,
            language=language,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            text_position_capacity=text_position_capacity,
            text_normalization=text_normalization,
            normalizer=self.normalizer,
            japanese_processor=self.japanese.process,
        )
