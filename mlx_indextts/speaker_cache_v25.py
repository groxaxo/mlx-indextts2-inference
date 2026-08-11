"""Versioned speaker-conditioning cache contract for IndexTTS 2.5."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mlx_indextts.model_manifest import SPEAKER_CACHE_SCHEMA_VERSION

CACHE_MODEL_VERSION = "2.5"
METADATA_KEY = "metadata_json"
REQUIRED_TENSORS = (
    "spk_cond_emb",
    "ref_mel",
    "style",
    "prompt_condition",
)


class SpeakerCacheError(ValueError):
    """Raised when a speaker cache is incompatible or malformed."""


@dataclass(frozen=True)
class SpeakerCacheV25:
    metadata: dict[str, Any]
    features: dict[str, np.ndarray]


def audio_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_features(features: Mapping[str, Any]) -> dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_TENSORS if name not in features]
    if missing:
        raise SpeakerCacheError(f"speaker cache missing tensors: {', '.join(missing)}")
    arrays = {name: np.asarray(features[name]) for name in REQUIRED_TENSORS}
    shape_rules = {
        "spk_cond_emb": lambda value: value.ndim == 3 and value.shape[0] == 1 and value.shape[2] == 1024,
        "ref_mel": lambda value: value.ndim == 3 and value.shape[0] == 1 and value.shape[1] == 80,
        "style": lambda value: value.ndim == 2 and value.shape == (1, 192),
        "prompt_condition": lambda value: value.ndim == 3 and value.shape[0] == 1 and value.shape[2] == 512,
    }
    for name, array in arrays.items():
        if not shape_rules[name](array):
            raise SpeakerCacheError(
                f"invalid {name} tensor shape for IndexTTS 2.5: {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise SpeakerCacheError(f"{name} must be a floating-point tensor")
    if arrays["ref_mel"].shape[2] != arrays["prompt_condition"].shape[1]:
        raise SpeakerCacheError("ref_mel and prompt_condition lengths must match")
    return arrays


def save_speaker_cache(
    output_path: str | Path,
    features: Mapping[str, Any],
    *,
    model_revision: str,
    source_audio: str | Path | None = None,
    preprocessing: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a model-bound 2.5 speaker cache."""
    if not str(model_revision).strip():
        raise SpeakerCacheError("model_revision must not be empty")
    arrays = _validate_features(features)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_family": "IndexTTS",
        "model_version": CACHE_MODEL_VERSION,
        "model_revision": str(model_revision),
        "cache_schema_version": SPEAKER_CACHE_SCHEMA_VERSION,
        "source_audio_sha256": audio_fingerprint(source_audio) if source_audio else None,
        "preprocessing": dict(preprocessing or {}),
        "tensors": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        },
    }
    payload = dict(arrays)
    payload[METADATA_KEY] = np.array(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_speaker_cache(
    cache_path: str | Path,
    *,
    model_revision: str,
) -> SpeakerCacheV25:
    """Load and validate a cache against the active converted model."""
    path = Path(cache_path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if METADATA_KEY not in archive:
                raise SpeakerCacheError(
                    f"not an IndexTTS 2.5 cache (legacy or missing metadata): {path}"
                )
            metadata = json.loads(str(archive[METADATA_KEY].item()))
            features = {name: np.array(archive[name]) for name in REQUIRED_TENSORS}
    except SpeakerCacheError:
        raise
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SpeakerCacheError(f"cannot read IndexTTS 2.5 cache: {path}") from exc
    if not isinstance(metadata, dict):
        raise SpeakerCacheError("speaker cache metadata must be an object")
    if str(metadata.get("model_version")) != CACHE_MODEL_VERSION:
        raise SpeakerCacheError(
            f"speaker cache model version is {metadata.get('model_version')}, expected 2.5"
        )
    if metadata.get("cache_schema_version") != SPEAKER_CACHE_SCHEMA_VERSION:
        raise SpeakerCacheError(
            "speaker cache schema is incompatible with this runtime"
        )
    if str(metadata.get("model_revision")) != str(model_revision):
        raise SpeakerCacheError(
            "speaker cache model revision does not match the active 2.5 model"
        )
    validated = _validate_features(features)
    for name, array in validated.items():
        recorded = metadata.get("tensors", {}).get(name, {})
        if recorded.get("shape") != list(array.shape) or recorded.get("dtype") != str(array.dtype):
            raise SpeakerCacheError(f"speaker cache metadata does not match tensor: {name}")
    return SpeakerCacheV25(metadata=metadata, features=validated)
