"""Tests for MLX-native Qwen emotion helpers."""

from mlx_indextts.qwen_emotion import (
    EMOTION_ORDER,
    adaptive_emo_alpha,
    normalize_emotion_dict,
    parse_qwen_emotion_output,
    smooth_emotion_sequence,
    smooth_emotion_sequence_by_speaker,
)


def test_parse_json_output_in_official_order():
    weights = parse_qwen_emotion_output('{"高兴": 0.4, "自然": 0.3, "愤怒": 2.0}')

    assert list(weights.keys()) == list(EMOTION_ORDER)
    assert weights["happy"] == 0.4
    assert weights["calm"] == 0.3
    assert weights["angry"] == 1.2


def test_parse_loose_output_fallback():
    weights = parse_qwen_emotion_output("高兴:0.1, 悲伤: 0.6, 自然:0.2")

    assert weights["happy"] == 0.1
    assert weights["sad"] == 0.6
    assert weights["calm"] == 0.2


def test_empty_output_defaults_to_calm():
    weights = normalize_emotion_dict({})

    assert weights["calm"] == 1.0
    assert sum(value for key, value in weights.items() if key != "calm") == 0.0


def test_melancholic_text_swaps_sad_to_melancholic():
    weights = parse_qwen_emotion_output('{"悲伤": 0.8, "低落": 0.1}', "她的心情很低落")

    assert weights["sad"] == 0.1
    assert weights["melancholic"] == 0.8


def test_smooth_emotion_sequence_limits_adjacent_jumps():
    rows = [
        {"happy": 1.0},
        {"angry": 1.0},
    ]
    smoothed = smooth_emotion_sequence(rows, max_step=0.18)

    assert smoothed[1]["happy"] >= 0.82
    assert smoothed[1]["angry"] <= 0.18


def test_speaker_aware_smoothing_keeps_same_role_stable():
    rows = [
        {"happy": 1.0},
        {"angry": 1.0},
        {"angry": 1.0},
    ]
    speakers = ["A", "B", "A"]

    smoothed = smooth_emotion_sequence_by_speaker(rows, speakers)

    assert smoothed[2]["happy"] >= smoothed[1]["happy"]
    assert smoothed[2]["angry"] <= 0.5


def test_adaptive_emo_alpha_scales_with_emotion_strength():
    weak = adaptive_emo_alpha({"calm": 0.4, "happy": 0.2})
    strong = adaptive_emo_alpha({"angry": 0.9, "calm": 0.1})

    assert strong >= weak
    assert 0.58 <= weak <= 0.82
