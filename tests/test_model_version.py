"""Tests for version and artifact contracts."""

import json
from pathlib import Path

import pytest
import yaml

from mlx_indextts.model_version import (
    V20_SOURCE_FILES,
    V25_CONVERTED_FILES,
    V25_SOURCE_FILES,
    ModelFormatError,
    detect_converted_version,
    detect_source_version,
    normalize_v25_config,
)


def _touch_all(root: Path, names: set[str] | frozenset[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"fixture")


def _official_stale_config() -> dict:
    return {
        "version": 2.0,
        "dataset": {"bpe_model": "bpe.model", "sample_rate": 24000},
        "gpt": {"number_text_tokens": 60509},
        "semantic_codec": {"codebook_size": 8192},
        "gpt_checkpoint": "/cubefs/internal/checkpoint/gpt.pth",
        "s2mel_checkpoint": "/cubefs/internal/checkpoint/s2mel.pth",
        "w2v_stat": "/cubefs/internal/checkpoint/wav2vec2bert_stats.pt",
        "emo_matrix": "/cubefs/internal/checkpoint/feat2.pt",
        "spk_matrix": "/cubefs/internal/checkpoint/feat1.pt",
        "vocoder": {"name": "/cubefs/internal/checkpoint/bigvgan_generator.pt"},
    }


def test_normalize_v25_config_repairs_public_release_fields():
    raw = _official_stale_config()
    normalized = normalize_v25_config(raw)

    assert normalized["version"] == 2.5
    assert normalized["dataset"]["bpe_model"].endswith(".tiktoken")
    assert normalized["dataset"]["tokenizer_type"] == "tiktoken"
    assert normalized["supported_languages"] == ["zh", "en", "ja", "es", "ar"]
    assert normalized["gpt_checkpoint"] == "gpt.pth"
    assert normalized["s2mel_checkpoint"] == "s2mel.pth"
    assert normalized["w2v_stat"] == "wav2vec2bert_stats.pt"
    assert normalized["semantic_codec"]["frame_rate"] == 25
    assert normalized["vocoder"]["name"] == "bigvgan_generator.pt"
    assert raw["version"] == 2.0
    assert raw["dataset"]["bpe_model"] == "bpe.model"


def test_normalize_v25_config_rejects_wrong_text_vocabulary():
    raw = _official_stale_config()
    raw["gpt"]["number_text_tokens"] = 12000

    with pytest.raises(ModelFormatError, match="60509"):
        normalize_v25_config(raw)


def test_detect_complete_v25_source(tmp_path: Path):
    _touch_all(tmp_path, V25_SOURCE_FILES - {"config.yaml"})
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(_official_stale_config()), encoding="utf-8"
    )

    assert detect_source_version(tmp_path) == "2.5"


def test_detect_v25_source_rejects_missing_artifact(tmp_path: Path):
    _touch_all(tmp_path, V25_SOURCE_FILES - {"config.yaml", "feat2.pt"})
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(_official_stale_config()), encoding="utf-8"
    )

    with pytest.raises(ModelFormatError, match="feat2.pt"):
        detect_source_version(tmp_path)


def test_detect_complete_v20_source(tmp_path: Path):
    _touch_all(tmp_path, V20_SOURCE_FILES)

    assert detect_source_version(tmp_path) == "2.0"


def test_detect_converted_version_prefers_manifest(tmp_path: Path):
    (tmp_path / "model_manifest.json").write_text(
        json.dumps({"model_version": "2.5"}), encoding="utf-8"
    )

    assert detect_converted_version(tmp_path) == "2.5"


def test_detect_complete_converted_v25_model(tmp_path: Path):
    _touch_all(tmp_path, V25_CONVERTED_FILES)

    assert detect_converted_version(tmp_path) == "2.5"


def test_detect_converted_v25_rejects_partial_artifacts(tmp_path: Path):
    _touch_all(tmp_path, {"codec.safetensors"})

    with pytest.raises(ModelFormatError, match="gpt.safetensors"):
        detect_converted_version(tmp_path)
