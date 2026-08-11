"""Parity tests for the IndexTTS-2.5 Japanese frontend."""

from mlx_indextts.japanese_v25 import JapaneseG2PProcessor


def test_official_ratio_zero_segmentation():
    processor = JapaneseG2PProcessor(g2p_ratio=0)

    assert processor.process("今日はいい天気です。") == "今日 は いい 天気 です 。"
    assert processor.process("銀行へ行きます。") == "銀行 へ 行き ます 。"


def test_official_ratio_one_converts_kanji_to_hiragana():
    processor = JapaneseG2PProcessor(g2p_ratio=1)

    assert processor.process("今日はいい天気です。") == "きょう は いい てんき です 。"
    assert processor.process("銀行へ行きます。") == "ぎんこう へ いき ます 。"


def test_existing_spaces_are_preserved():
    processor = JapaneseG2PProcessor(g2p_ratio=0)

    assert processor.process("東京 で 会いましょう。") == "東京 で 会い ましょう 。"


def test_katakana_to_hiragana():
    assert JapaneseG2PProcessor.katakana_to_hiragana("トウキョウ ABC") == "とうきょう ABC"
