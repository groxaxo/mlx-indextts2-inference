"""Tests for separate emotion reference plumbing."""

import csv
from pathlib import Path

import mlx.core as mx


def test_batch_csv_reads_per_row_emotion_reference(tmp_path: Path):
    from mlx_indextts.cli import _read_batch_items

    csv_path = tmp_path / "dialogue.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speaker", "text", "ref_audio", "emotion_ref_audio"])
        writer.writeheader()
        writer.writerow({
            "speaker": "a",
            "text": "hello",
            "ref_audio": "speaker_a.npz",
            "emotion_ref_audio": "calm_ref.npz",
        })

    assert _read_batch_items(str(csv_path)) == [{
        "id": "a",
        "speaker": "a",
        "text": "hello",
        "ref_audio": "speaker_a.npz",
        "emotion_ref_audio": "calm_ref.npz",
        "emotion": "",
        "emo_alpha": "0.6",
    }]


def test_batch_csv_reads_original_chinese_columns(tmp_path: Path):
    from mlx_indextts.cli import _read_batch_items

    csv_path = tmp_path / "batch.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["文本", "音色参考音频", "情感权重", "喜", "怒", "平静"],
        )
        writer.writeheader()
        writer.writerow({
            "文本": "你好",
            "音色参考音频": "speaker.npz",
            "情感权重": "0.7",
            "喜": "0.2",
            "怒": "0.1",
            "平静": "0.4",
        })

    assert _read_batch_items(str(csv_path))[0] == {
        "id": "0001",
        "speaker": "",
        "text": "你好",
        "ref_audio": "speaker.npz",
        "emotion_ref_audio": "",
        "emotion": "happy:0.2,angry:0.1,calm:0.4",
        "emo_alpha": "0.7",
    }


def test_batch_csv_reads_role_alias(tmp_path: Path):
    from mlx_indextts.cli import _read_batch_items

    csv_path = tmp_path / "dialogue_role.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "text", "ref_audio"])
        writer.writeheader()
        writer.writerow({
            "role": "逗哏",
            "text": "先来一句",
            "ref_audio": "speaker.npz",
        })

    assert _read_batch_items(str(csv_path))[0]["speaker"] == "逗哏"


def test_batch_csv_preserves_duration_fit_fields(tmp_path: Path):
    from mlx_indextts.cli import _read_batch_items

    csv_path = tmp_path / "timed_dialogue.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "ref_audio", "target_duration_s", "fit_duration", "max_mel_tokens"],
        )
        writer.writeheader()
        writer.writerow({
            "text": "Timed line",
            "ref_audio": "speaker.npz",
            "target_duration_s": "2.75",
            "fit_duration": "true",
            "max_mel_tokens": "180",
        })

    item = _read_batch_items(str(csv_path))[0]
    assert item["target_duration"] == "2.75"
    assert item["fit_duration"] == "true"
    assert item["max_tokens"] == "180"


def test_auto_emotion_weights_are_scaled_to_full_total():
    from mlx_indextts.runtime import _normalize_emotion_total

    weights = _normalize_emotion_total({"happy": 0.2, "calm": 0.3, "angry": 0.0})

    assert weights["happy"] == 0.4
    assert weights["calm"] == 0.6
    assert round(sum(weights.values()), 4) == 1.0


def test_runtime_reports_separate_emotion_reference_source():
    from mlx_indextts.runtime import GenerateOptions, TTSRuntime

    emotion, metadata = TTSRuntime()._resolve_auto_emotion(
        "text",
        GenerateOptions(emotion_ref_audio="emotion.wav"),
    )

    assert emotion is None
    assert metadata["emotion_source"] == "emotion_reference"


def test_emotion_sources_are_mutually_exclusive():
    import pytest

    from mlx_indextts.runtime import GenerateOptions, validate_emotion_source

    validate_emotion_source(GenerateOptions(emotion_ref_audio="calm.npz"))
    validate_emotion_source(GenerateOptions(emotion="angry"))
    validate_emotion_source(GenerateOptions(auto_emotion=True))
    validate_emotion_source(GenerateOptions(use_emo_text=True, emo_text="fear"))

    with pytest.raises(ValueError):
        validate_emotion_source(GenerateOptions(auto_emotion=True, emotion_ref_audio="calm.npz"))
    with pytest.raises(ValueError):
        validate_emotion_source(GenerateOptions(emotion="angry", emotion_ref_audio="calm.npz"))
    with pytest.raises(ValueError):
        validate_emotion_source(GenerateOptions(emotion="auto-qwen"), has_row_emotion_refs=True)
    with pytest.raises(ValueError):
        validate_emotion_source(
            GenerateOptions(use_emo_text=True, emo_text="fear", emotion="angry")
        )


def test_gpt_v2_merge_emovec_formula():
    from mlx_indextts.models.gpt_v2 import UnifiedVoiceV2

    class Dummy:
        def get_emovec(self, value, lengths=None):
            return value

    base = mx.array([[1.0, 3.0]])
    emotion = mx.array([[5.0, 7.0]])
    merged = UnifiedVoiceV2.merge_emovec(Dummy(), base, emotion, alpha=0.25)

    assert merged.tolist() == [[2.0, 4.0]]
