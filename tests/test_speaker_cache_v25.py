"""Tests for versioned IndexTTS 2.5 speaker cache files."""

from pathlib import Path

import numpy as np
import pytest


def _features():
    return {
        "spk_cond_emb": np.zeros((1, 5, 1024), dtype=np.float32),
        "ref_mel": np.zeros((1, 80, 9), dtype=np.float32),
        "style": np.zeros((1, 192), dtype=np.float32),
        "prompt_condition": np.zeros((1, 9, 512), dtype=np.float32),
    }


def test_v25_speaker_cache_roundtrip(tmp_path: Path):
    from mlx_indextts.speaker_cache_v25 import load_speaker_cache, save_speaker_cache

    source_audio = tmp_path / "voice.wav"
    source_audio.write_bytes(b"fixture-audio")
    cache_path = tmp_path / "voice-v25.npz"

    save_speaker_cache(
        cache_path,
        _features(),
        model_revision="official-revision",
        source_audio=source_audio,
        preprocessing={"max_seconds": 15, "semantic_sample_rate": 16000},
    )
    loaded = load_speaker_cache(cache_path, model_revision="official-revision")

    assert loaded.metadata["model_version"] == "2.5"
    assert loaded.metadata["cache_schema_version"] == 1
    assert loaded.metadata["source_audio_sha256"]
    assert loaded.metadata["preprocessing"]["max_seconds"] == 15
    assert set(loaded.features) == set(_features())
    np.testing.assert_array_equal(loaded.features["style"], _features()["style"])


def test_v25_cache_rejects_model_revision_mismatch(tmp_path: Path):
    from mlx_indextts.speaker_cache_v25 import (
        SpeakerCacheError,
        load_speaker_cache,
        save_speaker_cache,
    )

    cache_path = tmp_path / "voice.npz"
    save_speaker_cache(cache_path, _features(), model_revision="revision-a")

    with pytest.raises(SpeakerCacheError, match="model revision"):
        load_speaker_cache(cache_path, model_revision="revision-b")


def test_v25_cache_rejects_legacy_v20_npz(tmp_path: Path):
    from mlx_indextts.speaker_cache_v25 import SpeakerCacheError, load_speaker_cache

    cache_path = tmp_path / "legacy.npz"
    np.savez(cache_path, version=np.array([2.0]), **_features())

    with pytest.raises(SpeakerCacheError, match="IndexTTS 2.5 cache"):
        load_speaker_cache(cache_path, model_revision="revision")


def test_v25_cache_rejects_invalid_tensor_shape(tmp_path: Path):
    from mlx_indextts.speaker_cache_v25 import SpeakerCacheError, save_speaker_cache

    features = _features()
    features["style"] = np.zeros((1, 128), dtype=np.float32)

    with pytest.raises(SpeakerCacheError, match="style"):
        save_speaker_cache(
            tmp_path / "invalid.npz",
            features,
            model_revision="revision",
        )
