"""Versioned manifest for converted IndexTTS MLX models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from mlx_indextts.model_version import V25_LANGUAGES, V25_TEXT_VOCAB_SIZE, V25_TOKENIZER

MANIFEST_FILENAME = "model_manifest.json"
MANIFEST_FORMAT_VERSION = 1
SOURCE_REPOSITORY = "IndexTeam/IndexTTS-2.5"
SPEAKER_CACHE_SCHEMA_VERSION = 1

REQUIRED_FIELDS = frozenset(
    {
        "format_version",
        "model_family",
        "model_version",
        "source_repository",
        "source_revision",
        "source_files",
        "converter_revision",
        "converted_at",
        "dtype",
        "quantization",
        "components",
        "tokenizer",
        "supported_languages",
        "semantic_codec_frame_rate",
        "required_auxiliary_resources",
        "speaker_cache_schema_version",
    }
)


class ManifestError(ValueError):
    """Raised when a converted-model manifest violates its contract."""


def collect_source_files(
    model_dir: str | Path,
    names: Iterable[str],
    *,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Collect deterministic file records without hashing multi-gigabyte weights."""

    root = Path(model_dir)
    metadata = metadata or {}
    records: list[dict[str, Any]] = []
    for name in sorted(set(names)):
        path = root / name
        if not path.is_file():
            raise ManifestError(f"Cannot record missing source artifact: {path}")
        record: dict[str, Any] = {"name": name, "size": path.stat().st_size}
        for key in ("sha256", "lfs_oid"):
            value = metadata.get(name, {}).get(key)
            if value:
                record[key] = str(value)
        records.append(record)
    return records


def build_v25_manifest(
    *,
    source_revision: str,
    source_files: list[dict[str, Any]],
    converter_revision: str,
    dtype: str,
    quantization: dict[str, Any] | None,
    components: dict[str, dict[str, Any]],
    converted_at: str | None = None,
    required_auxiliary_resources: list[str] | None = None,
) -> dict[str, Any]:
    """Build and validate an IndexTTS-2.5 converted-model manifest."""

    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "model_family": "IndexTTS",
        "model_version": "2.5",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": source_revision,
        "source_files": source_files,
        "converter_revision": converter_revision,
        "converted_at": converted_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dtype": dtype,
        "quantization": quantization,
        "components": components,
        "tokenizer": {
            "type": "tiktoken",
            "filename": V25_TOKENIZER,
            "vocab_size": V25_TEXT_VOCAB_SIZE,
        },
        "supported_languages": list(V25_LANGUAGES),
        "semantic_codec_frame_rate": 25,
        "required_auxiliary_resources": required_auxiliary_resources
        or [
            "facebook/w2v-bert-2.0",
            "funasr/campplus",
        ],
        "speaker_cache_schema_version": SPEAKER_CACHE_SCHEMA_VERSION,
    }
    validate_manifest(manifest, require_complete=True)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> None:
    """Validate manifest structure and optional load-readiness."""

    missing_fields = sorted(REQUIRED_FIELDS - set(manifest))
    if missing_fields:
        raise ManifestError(f"Manifest missing fields: {', '.join(missing_fields)}")
    if manifest["format_version"] != MANIFEST_FORMAT_VERSION:
        raise ManifestError(
            f"Unsupported manifest format_version: {manifest['format_version']}"
        )
    if str(manifest["model_version"]) not in {"2.0", "2.5"}:
        raise ManifestError(f"Unsupported model_version: {manifest['model_version']}")
    if not str(manifest["source_revision"]).strip():
        raise ManifestError("Manifest source_revision must not be empty")
    if not isinstance(manifest["source_files"], list) or not manifest["source_files"]:
        raise ManifestError("Manifest source_files must be a non-empty list")
    if not isinstance(manifest["components"], dict) or not manifest["components"]:
        raise ManifestError("Manifest components must be a non-empty mapping")

    for name, component in manifest["components"].items():
        if not isinstance(component, Mapping):
            raise ManifestError(f"Component record must be a mapping: {name}")
        for field in ("source_tensors", "mapped_tensors", "ignored", "missing"):
            if field not in component:
                raise ManifestError(f"Component {name} missing field: {field}")
        if int(component["source_tensors"]) < 0 or int(component["mapped_tensors"]) < 0:
            raise ManifestError(f"Component {name} has a negative tensor count")
        if require_complete and component["missing"]:
            raise ManifestError(
                f"Component {name} has unmapped required tensors: "
                f"{', '.join(map(str, component['missing']))}"
            )

    tokenizer = manifest["tokenizer"]
    if not isinstance(tokenizer, Mapping) or not tokenizer.get("filename"):
        raise ManifestError("Manifest tokenizer record is invalid")
    languages = manifest["supported_languages"]
    if not isinstance(languages, list) or not languages:
        raise ManifestError("Manifest supported_languages must be a non-empty list")


def write_manifest(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Atomically publish a validated manifest in a converted model directory."""

    validate_manifest(manifest, require_complete=True)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_FILENAME
    staging = root / f".{MANIFEST_FILENAME}.tmp"
    staging.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(path)
    return path


def load_manifest(
    model_dir: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Load and validate a converted-model manifest."""

    path = Path(model_dir) / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read model manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"Model manifest must be a JSON object: {path}")
    validate_manifest(manifest, require_complete=require_complete)
    return manifest
