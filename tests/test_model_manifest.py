"""Tests for converted-model manifest contracts."""

import json
from pathlib import Path

import pytest

from mlx_indextts.model_manifest import (
    ManifestError,
    build_v25_manifest,
    collect_source_files,
    load_manifest,
    validate_manifest,
    write_manifest,
)


def _components(*, missing: list[str] | None = None) -> dict:
    return {
        "gpt": {
            "source_tensors": 10,
            "mapped_tensors": 10,
            "ignored": [],
            "missing": missing or [],
        },
        "codec": {
            "source_tensors": 4,
            "mapped_tensors": 4,
            "ignored": [],
            "missing": [],
        },
    }


def _manifest(**overrides):
    values = {
        "source_revision": "d0aa86e75bb6f3437f3831e95056fa72842d89ef",
        "source_files": [{"name": "gpt.pth", "size": 123}],
        "converter_revision": "test-revision",
        "dtype": "float16",
        "quantization": {"bits": 8, "group_size": 64},
        "components": _components(),
        "converted_at": "2026-08-11T12:00:00+00:00",
    }
    values.update(overrides)
    return build_v25_manifest(**values)


def test_build_v25_manifest_has_versioned_runtime_contract():
    manifest = _manifest()

    assert manifest["format_version"] == 1
    assert manifest["model_version"] == "2.5"
    assert manifest["source_repository"] == "IndexTeam/IndexTTS-2.5"
    assert manifest["tokenizer"]["vocab_size"] == 60509
    assert manifest["supported_languages"] == ["zh", "en", "ja", "es", "ar"]
    assert manifest["semantic_codec_frame_rate"] == 25
    assert manifest["speaker_cache_schema_version"] == 1


def test_collect_source_files_records_size_and_available_hashes(tmp_path: Path):
    (tmp_path / "gpt.pth").write_bytes(b"checkpoint")

    records = collect_source_files(
        tmp_path,
        ["gpt.pth"],
        metadata={"gpt.pth": {"lfs_oid": "sha256:official-object"}},
    )

    assert records == [
        {
            "name": "gpt.pth",
            "size": len(b"checkpoint"),
            "lfs_oid": "sha256:official-object",
        }
    ]


def test_build_manifest_rejects_unmapped_required_tensor():
    with pytest.raises(ManifestError, match="unmapped required tensors"):
        _manifest(components=_components(missing=["gpt.lang_embedding.weight"]))


def test_validate_manifest_rejects_missing_contract_field():
    manifest = _manifest()
    del manifest["source_revision"]

    with pytest.raises(ManifestError, match="source_revision"):
        validate_manifest(manifest)


def test_write_and_load_manifest_roundtrip(tmp_path: Path):
    manifest = _manifest()

    path = write_manifest(tmp_path, manifest)
    loaded = load_manifest(tmp_path)

    assert path.name == "model_manifest.json"
    assert loaded == manifest
    assert not (tmp_path / ".model_manifest.json.tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["model_version"] == "2.5"


def test_load_manifest_rejects_invalid_json(tmp_path: Path):
    (tmp_path / "model_manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(ManifestError, match="Cannot read"):
        load_manifest(tmp_path)
