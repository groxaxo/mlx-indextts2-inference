"""Tests for emotion2vec library helpers."""

import csv
import json
from pathlib import Path


def test_emotion2vec_raw_to_indextts_mapping():
    from mlx_indextts.emotion2vec import map_emotion2vec_to_indextts

    mapped = map_emotion2vec_to_indextts(
        {
            "angry": 0.4,
            "disgusted": 0.2,
            "fearful": 0.3,
            "happy": 0.6,
            "neutral": 0.1,
            "other": 0.2,
            "sad": 0.5,
            "surprised": 0.15,
            "unknown": 0.05,
        }
    )

    assert mapped["angry"] == 0.4
    assert mapped["disgusted"] == 0.2
    assert mapped["afraid"] == 0.3
    assert mapped["happy"] == 0.6
    assert mapped["sad"] == 0.5
    assert mapped["surprised"] == 0.15
    assert mapped["calm"] == 0.35000000000000003


def test_emotion_catalog_prefers_best_reference(tmp_path: Path):
    from mlx_indextts.emotion2vec import Emotion2VecCatalog

    catalog_path = tmp_path / "catalog.csv"
    rows = [
        {
            "index": "1",
            "source_path": str(tmp_path / "a.wav"),
            "copied_path": "clips/a.wav",
            "duration_s": "4.000",
            "dominant_emotion": "happy",
            "confidence": "0.82",
            "melancholic_hint": "0.10",
            "raw_scores_json": json.dumps({}),
            "indextts_scores_json": json.dumps({"happy": 0.88, "calm": 0.05}),
            "elapsed_s": "0.2",
            "model": "iic/emotion2vec_plus_large",
            "hub": "hf",
        },
        {
            "index": "2",
            "source_path": str(tmp_path / "b.wav"),
            "copied_path": "clips/b.wav",
            "duration_s": "4.000",
            "dominant_emotion": "happy",
            "confidence": "0.95",
            "melancholic_hint": "0.10",
            "raw_scores_json": json.dumps({}),
            "indextts_scores_json": json.dumps({"happy": 0.92, "calm": 0.02}),
            "elapsed_s": "0.2",
            "model": "iic/emotion2vec_plus_large",
            "hub": "hf",
        },
    ]
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    catalog = Emotion2VecCatalog.load(catalog_path)
    refs = catalog.emotion_refs()

    assert refs["happy"].endswith("clips/b.wav")
    assert "calm" in refs


def test_novel_planner_uses_emotion_refs_by_emotion():
    from mlx_indextts.novel_planner import NovelLine, NovelScriptPlanner

    planner = NovelScriptPlanner(max_chars_per_line=80)
    rows = planner.to_batch_rows(
        [
            NovelLine(speaker="A", text="你好", emotion_label="happy", emotion="happy:0.8", emo_alpha=0.8),
            NovelLine(speaker="A", text="别闹", emotion_label="angry", emotion="angry:0.7", emo_alpha=0.7),
        ],
        emotion_refs_by_emotion={"happy": "happy.wav", "angry": "angry.wav"},
    )

    assert rows[0]["emotion_ref_audio"] == "happy.wav"
    assert rows[1]["emotion_ref_audio"] == "angry.wav"


def test_novel_planner_resolver_overrides_emotion_refs():
    from mlx_indextts.novel_planner import NovelLine, NovelScriptPlanner

    planner = NovelScriptPlanner(max_chars_per_line=80)
    rows = planner.to_batch_rows(
        [NovelLine(speaker="逗哏", text="别闹", emotion_label="angry", emotion="angry:0.7", emo_alpha=0.7)],
        speaker_refs={"逗哏": "speaker.wav"},
        emotion_refs_by_emotion={"angry": "generic_angry.wav"},
        emotion_ref_resolver=lambda row: f"crosstalk_{row.emotion_label}.wav",
    )

    assert rows[0]["ref_audio"] == "speaker.wav"
    assert rows[0]["emotion_ref_audio"] == "crosstalk_angry.wav"
