"""Parity tests for the IndexTTS-2.5 tokenizer."""

import base64
from pathlib import Path

import pytest

from mlx_indextts.model_version import V25_TOKENIZER
from mlx_indextts.tokenizer_v25 import (
    IndexTTS25Tokenizer,
    language_id,
    normalize_v25_language,
)


def _write_byte_vocab(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{base64.b64encode(bytes([value])).decode()} {value}"
        for value in range(256)
    ]
    (root / V25_TOKENIZER).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_language_ids_match_official_model():
    assert language_id("EN") == 0
    assert language_id("zh") == 1
    assert language_id("es") == 3
    assert language_id("ja") == 7
    assert language_id("ar") == 13
    assert normalize_v25_language("Mandarin") == "zh"


def test_unsupported_language_is_explicit_error():
    with pytest.raises(ValueError, match="Unsupported"):
        language_id("vi")


def test_minimal_vocab_roundtrips_unicode_and_special_tokens(tmp_path: Path):
    _write_byte_vocab(tmp_path)
    tokenizer = IndexTTS25Tokenizer(tmp_path, expected_vocab_size=None)
    text = "<|zh|> 你好<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>"

    token_ids = tokenizer.encode(text, allowed_special="all")

    assert tokenizer.decode(token_ids) == text
    assert tokenizer.token_count(text) == len(token_ids)


def test_official_vocab_matches_pinned_upstream_vectors():
    model_dir = Path("models/IndexTTS-2.5-source")
    if not (model_dir / V25_TOKENIZER).is_file():
        pytest.skip("Pinned official IndexTTS-2.5 tokenizer is not downloaded")
    tokenizer = IndexTTS25Tokenizer(model_dir)

    assert tokenizer.vocab_size == 60509
    assert tokenizer.encode("<|zh|> 你好", allowed_special="all") == [
        58839,
        220,
        48934,
        50371,
    ]
    assert tokenizer.encode("<|en|> Hello", allowed_special="all") == [58838, 2415]
    assert tokenizer.encode(
        "<|zh|> <|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>",
        allowed_special="all",
    ) == [58839, 220, 58959, 55, 2997, 17, 58959]
