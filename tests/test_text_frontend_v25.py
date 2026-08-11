"""Tests for the IndexTTS-2.5 pure text frontend."""

from pathlib import Path

import pytest

from mlx_indextts.model_version import V25_TOKENIZER
from mlx_indextts.text_frontend_v25 import (
    apply_pronunciation_annotations,
    prepare_v25_tokens,
    resolve_v25_language,
    split_text_by_tokens,
)
from mlx_indextts.tokenizer_v25 import IndexTTS25Tokenizer


class _CharacterTokenizer:
    def token_count(self, text: str) -> int:
        return len(text)

    def encode(self, text: str, *, allowed_special: str = "all") -> list[int]:
        del allowed_special
        return [ord(character) for character in text]


def test_pronunciation_annotations_match_upstream_markers():
    text = "go <going|G OW1 . IH0 NG> 银<行|XING2> <今日|きょう>"

    result = apply_pronunciation_annotations(text)

    assert "<|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|>" in result
    assert "<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>" in result
    assert " きょう " in result


@pytest.mark.parametrize(
    ("text", "expected", "ambiguous"),
    [
        ("你好，世界", "zh", False),
        ("今日はいい天気です", "ja", False),
        ("مرحبا بالعالم", "ar", False),
        ("¡Hola, señor!", "es", False),
        ("Hello world", "en", True),
    ],
)
def test_auto_language_resolution(text: str, expected: str, ambiguous: bool):
    resolution = resolve_v25_language(text)

    assert resolution.language == expected
    assert resolution.ambiguous is ambiguous


def test_explicit_language_wins_for_ambiguous_latin_text():
    resolution = resolve_v25_language("Hola mundo", "es")

    assert resolution.language == "es"
    assert resolution.ambiguous is False


def test_mixed_arabic_and_cjk_requires_explicit_language():
    with pytest.raises(ValueError, match="explicit"):
        resolve_v25_language("你好 مرحبا")


def test_split_keeps_pronunciation_markers_paired():
    tokenizer = _CharacterTokenizer()
    unit = "语音合成很重要。<|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|>"
    text = unit * 6

    segments = split_text_by_tokens(
        text,
        tokenizer=tokenizer,
        max_tokens=80,
        language_prefix="<|zh|> ",
    )

    assert len(segments) > 1
    assert all(segment.count("<|SPECIAL_TOKEN_1|>") % 2 == 0 for segment in segments)
    assert "".join(segments) == text


def test_prepare_official_zh_tokens_match_pinned_upstream_vector():
    model_dir = Path("models/IndexTTS-2.5-source")
    if not (model_dir / V25_TOKENIZER).is_file():
        pytest.skip("Pinned official IndexTTS-2.5 tokenizer is not downloaded")
    tokenizer = IndexTTS25Tokenizer(model_dir)

    prepared = prepare_v25_tokens("你好", tokenizer=tokenizer, language="zh")

    assert prepared.language == "zh"
    assert prepared.language_id == 1
    assert prepared.token_ids == ((58839, 220, 48934, 50371, 1),)


def test_prepare_spanish_uses_upstream_uppercase_rule(tmp_path: Path):
    del tmp_path
    tokenizer = _CharacterTokenizer()

    prepared = prepare_v25_tokens(
        "¡Hola, señor!",
        tokenizer=tokenizer,
        language="es",
    )

    assert prepared.normalized_text == "¡HOLA, SEÑOR!"
    assert prepared.token_ids[0][-1] == 1


def test_prepare_calls_normalizer_before_case_and_annotations():
    tokenizer = _CharacterTokenizer()
    calls = []

    def normalizer(text: str, language: str, enabled: bool) -> str:
        calls.append((text, language, enabled))
        return "<行|xing2>"

    prepared = prepare_v25_tokens(
        "ignored",
        tokenizer=tokenizer,
        language="zh",
        normalizer=normalizer,
    )

    assert calls == [("ignored", "zh", True)]
    assert prepared.normalized_text == (
        "<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>"
    )
