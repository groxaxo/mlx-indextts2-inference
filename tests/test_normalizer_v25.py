"""Tests for the full IndexTTS-2.5 normalization pipeline."""

from pathlib import Path

from mlx_indextts.model_version import V25_TOKENIZER
from mlx_indextts.normalizer_v25 import (
    IndexTTS25TextFrontend,
    IndexTTS25TextNormalizer,
    OptionalNemoNormalizer,
)
from mlx_indextts.tokenizer_v25 import IndexTTS25Tokenizer


class _FakeZhEnNormalizer:
    CHAR_REP_MAP = {",": ","}

    def __init__(self):
        self.load_count = 0

    def load(self):
        self.load_count += 1

    def normalize(self, text: str) -> str:
        return f"normalized:{text}"

    def load_glossary_from_yaml(self, path: str) -> bool:
        del path
        return True


class _FakeNemo:
    def normalize(self, text: str, *, verbose: bool = False) -> str:
        del verbose
        return f"nemo:{text}"


def test_zh_en_normalizer_is_loaded_once():
    fake = _FakeZhEnNormalizer()
    normalizer = IndexTTS25TextNormalizer(zh_en_normalizer=fake)

    assert normalizer("Hello", "en") == "normalized:Hello"
    assert normalizer("你好", "zh") == "normalized:你好"
    assert fake.load_count == 1


def test_spanish_nemo_is_lazy_and_cached():
    calls = []

    def factory(language: str):
        calls.append(language)
        return _FakeNemo()

    normalizer = OptionalNemoNormalizer(factory=factory)

    assert normalizer.normalize("25%", "es") == "nemo:25%"
    assert normalizer.normalize("12", "es") == "nemo:12"
    assert calls == ["es"]


def test_missing_nemo_falls_back_to_raw_text():
    def unavailable(language: str):
        raise ImportError(language)

    normalizer = OptionalNemoNormalizer(factory=unavailable)

    assert normalizer.normalize("Hola 25%", "es") == "Hola 25%"
    assert normalizer.normalize("Hola 25%", "es") == "Hola 25%"


def test_disabled_normalization_still_applies_character_cleanup():
    normalizer = IndexTTS25TextNormalizer(zh_en_normalizer=_FakeZhEnNormalizer())

    assert normalizer("你好，世界", "zh", False) == "你好,世界"


def test_full_frontend_matches_official_japanese_segmentation():
    model_dir = Path("models/IndexTTS-2.5-source")
    if not (model_dir / V25_TOKENIZER).is_file():
        return
    tokenizer = IndexTTS25Tokenizer(model_dir)
    frontend = IndexTTS25TextFrontend(model_dir, tokenizer=tokenizer)

    prepared = frontend.prepare(
        "今日はいい天気です。",
        language="ja",
        text_normalization=False,
    )

    assert prepared.normalized_text == "今日 は いい 天気 です ."
    assert prepared.language == "ja"
    assert prepared.token_ids[0][-1] == 1
