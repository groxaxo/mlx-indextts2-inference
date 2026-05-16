"""Tests for the YouTube/video emotion library workflow."""

import csv
from pathlib import Path


def test_split_asr_segments_into_sentences():
    from mlx_indextts.video_library import split_asr_segments_into_sentences

    segments = [
        {"text": "你好啊！我来了。", "start": 1.0, "end": 4.0},
        {"text": "别闹", "start": 5.0, "end": 6.0},
    ]

    sentences = split_asr_segments_into_sentences(segments)

    assert [item.text for item in sentences] == ["你好啊！", "我来了。", "别闹"]
    assert sentences[0].start_s == 1.0
    assert sentences[1].end_s == 4.0
    assert sentences[2].source_segment_index == 2


def test_scene_emotion_catalog_best_for_and_emotion_refs(tmp_path: Path):
    from mlx_indextts.video_library import SceneEmotionCatalog

    rows = [
        {
            "scene": "crosstalk",
            "video_id": "v1",
            "video_title": "demo",
            "source_url": "https://example.com",
            "source_path": "/tmp/a.wav",
            "vocal_path": "/tmp/a.wav",
            "vocal_source": "demucs",
            "segment_index": "1",
            "source_segment_index": "1",
            "start_s": "0.000",
            "end_s": "1.000",
            "duration_s": "1.000",
            "sentence": "你好",
            "emotion_label": "happy",
            "emotion_confidence": "0.9000",
            "emotion_json": "{}",
            "emotion_raw_json": "{}",
            "melancholic_hint": "0.0000",
            "gender_label": "male",
            "gender_confidence": "0.8100",
            "gender_probs_json": "{}",
            "age_score": "0.3300",
            "age_years": "33.0",
            "age_band": "adult",
            "clip_path": "clips/a.wav",
            "composite_key": "crosstalk|happy|male|adult",
            "library_score": "0.72",
            "duration_ratio": "0.1",
        },
        {
            "scene": "crosstalk",
            "video_id": "v1",
            "video_title": "demo",
            "source_url": "https://example.com",
            "source_path": "/tmp/b.wav",
            "vocal_path": "/tmp/b.wav",
            "vocal_source": "demucs",
            "segment_index": "2",
            "source_segment_index": "2",
            "start_s": "2.000",
            "end_s": "3.000",
            "duration_s": "1.000",
            "sentence": "太好笑了",
            "emotion_label": "happy",
            "emotion_confidence": "0.9700",
            "emotion_json": "{}",
            "emotion_raw_json": "{}",
            "melancholic_hint": "0.0000",
            "gender_label": "male",
            "gender_confidence": "0.8400",
            "gender_probs_json": "{}",
            "age_score": "0.3500",
            "age_years": "35.0",
            "age_band": "adult",
            "clip_path": "clips/b.wav",
            "composite_key": "crosstalk|happy|male|adult",
            "library_score": "0.81",
            "duration_ratio": "0.1",
        },
        {
            "scene": "crosstalk",
            "video_id": "v1",
            "video_title": "demo",
            "source_url": "https://example.com",
            "source_path": "/tmp/c.wav",
            "vocal_path": "/tmp/c.wav",
            "vocal_source": "demucs",
            "segment_index": "3",
            "source_segment_index": "3",
            "start_s": "4.000",
            "end_s": "5.000",
            "duration_s": "1.000",
            "sentence": "太惨了",
            "emotion_label": "sad",
            "emotion_confidence": "0.9300",
            "emotion_json": "{}",
            "emotion_raw_json": "{}",
            "melancholic_hint": "0.6000",
            "gender_label": "male",
            "gender_confidence": "0.8400",
            "gender_probs_json": "{}",
            "age_score": "0.3500",
            "age_years": "35.0",
            "age_band": "adult",
            "clip_path": "clips/c.wav",
            "composite_key": "crosstalk|sad|male|adult",
            "library_score": "0.79",
            "duration_ratio": "0.1",
        },
    ]
    catalog_path = tmp_path / "catalog.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    catalog = SceneEmotionCatalog.load(catalog_path)

    best = catalog.best_for(scene="crosstalk", emotion="happy", gender="male", age_band="adult")
    assert best is not None
    assert best["clip_path"] == "clips/b.wav"
    assert catalog.best_ref_for(scene="crosstalk", emotion="happy", gender="female", age_band="adult").endswith("clips/b.wav")
    assert catalog.best_ref_for(scene="crosstalk", emotion="melancholic", gender="male", age_band="adult").endswith("clips/c.wav")
    assert catalog.emotion_refs()["happy"].endswith("clips/b.wav")
    assert "crosstalk|happy|male|adult" in catalog.composite_refs()


def test_scene_emotion_catalog_recommended_duo_refs_prefers_distinct_pitch(tmp_path: Path, monkeypatch):
    import mlx_indextts.video_library as video_library

    rows = []
    for idx, (clip, pitch, gender) in enumerate(
        [
            ("a.wav", 110.0, "male"),
            ("b.wav", 220.0, "female"),
            ("c.wav", 150.0, "male"),
        ],
        start=1,
    ):
        rows.append(
            {
                "scene": "crosstalk",
                "video_id": "v1",
                "video_title": "demo",
                "source_url": "https://example.com",
                "source_path": f"/tmp/{clip}",
                "vocal_path": f"/tmp/{clip}",
                "vocal_source": "demucs",
                "segment_index": str(idx),
                "source_segment_index": str(idx),
                "start_s": "0.000",
                "end_s": "1.000",
                "duration_s": "1.000",
                "sentence": clip,
                "emotion_label": "happy",
                "emotion_confidence": "0.9000",
                "emotion_json": "{}",
                "emotion_raw_json": "{}",
                "melancholic_hint": "0.0000",
                "gender_label": gender,
                "gender_confidence": "0.8100",
                "gender_probs_json": "{}",
                "age_score": "0.3300",
                "age_years": "33.0",
                "age_band": "adult",
                "clip_path": f"clips/{clip}",
                "composite_key": f"crosstalk|happy|{gender}|adult",
                "library_score": "0.90",
                "duration_ratio": "0.1",
            }
        )

    catalog_path = tmp_path / "catalog.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    catalog = video_library.SceneEmotionCatalog.load(catalog_path)
    pitch_map = {"a.wav": 110.0, "b.wav": 220.0, "c.wav": 150.0}
    monkeypatch.setattr(video_library, "_estimate_median_f0", lambda path: pitch_map[Path(path).name])
    duo = catalog.recommended_duo_refs(scene="crosstalk")
    assert duo["逗哏"].endswith("clips/a.wav")
    assert duo["捧哏"].endswith("clips/b.wav")
