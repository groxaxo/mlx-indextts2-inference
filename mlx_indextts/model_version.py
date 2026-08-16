"""Version and artifact contracts for IndexTTS 2.0 and 2.5.

The 2.5 rules follow the official release artifacts and loader at the audited
upstream revision:
https://github.com/index-tts/index-tts/tree/9c87c46b84bd0e75ecaefb461e7e8f69bc9ecf44
https://huggingface.co/IndexTeam/IndexTTS-2.5/tree/d0aa86e75bb6f3437f3831e95056fa72842d89ef
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

V25_TOKENIZER = "multilingual_zh_ja_yue_char_del.tiktoken"
V25_LANGUAGES = ("zh", "en", "ja", "es", "ar")
V25_TEXT_VOCAB_SIZE = 60509

V25_SOURCE_FILES = frozenset(
    {
        "config.yaml",
        "gpt.pth",
        "codec.pth",
        "s2mel.pth",
        "feat1.pt",
        "feat2.pt",
        "wav2vec2bert_stats.pt",
        V25_TOKENIZER,
    }
)
V20_SOURCE_FILES = frozenset(
    {
        "config.yaml",
        "gpt.pth",
        "s2mel.pth",
        "feat1.pt",
        "feat2.pt",
        "wav2vec2bert_stats.pt",
        "bpe.model",
    }
)
V25_CONVERTED_FILES = frozenset(
    {
        "config.json",
        "config.yaml",
        "gpt.safetensors",
        "codec.safetensors",
        "s2mel.safetensors",
        "bigvgan.safetensors",
        "feat1.pt",
        "feat2.pt",
        "wav2vec2bert_stats.pt",
        V25_TOKENIZER,
    }
)


class ModelFormatError(ValueError):
    """Raised when a model directory is incomplete or internally inconsistent."""


def _missing_files(model_dir: Path, required: frozenset[str]) -> list[str]:
    return sorted(name for name in required if not (model_dir / name).is_file())


def _load_yaml_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ModelFormatError(f"Cannot read IndexTTS config: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelFormatError(f"IndexTTS config must be a mapping: {path}")
    return data


def normalize_v25_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Return a converted-model config normalized for the public 2.5 artifacts.

    The first public Hugging Face revision contains a stale version value, a
    missing bpe.model reference, and internal absolute checkpoint paths.
    Artifact names and the 60,509-token vocabulary are authoritative here.
    """

    config = copy.deepcopy(raw_config)
    config["version"] = 2.5
    config["supported_languages"] = list(V25_LANGUAGES)

    dataset = config.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        raise ModelFormatError("IndexTTS 2.5 dataset config must be a mapping")
    dataset["bpe_model"] = V25_TOKENIZER
    dataset["tokenizer_type"] = "tiktoken"

    gpt = config.setdefault("gpt", {})
    if not isinstance(gpt, dict):
        raise ModelFormatError("IndexTTS 2.5 GPT config must be a mapping")
    vocab_size = int(gpt.get("number_text_tokens", 0))
    if vocab_size != V25_TEXT_VOCAB_SIZE:
        raise ModelFormatError(
            "IndexTTS 2.5 requires gpt.number_text_tokens="
            f"{V25_TEXT_VOCAB_SIZE}, got {vocab_size}"
        )

    # Public v2.5 releases contain stale internal paths. These are the
    # authoritative filenames in the downloaded Hugging Face artifact.
    config["gpt_checkpoint"] = "gpt.pth"
    config["s2mel_checkpoint"] = "s2mel.pth"
    config["w2v_stat"] = "wav2vec2bert_stats.pt"
    config["emo_matrix"] = "feat2.pt"
    config["spk_matrix"] = "feat1.pt"

    semantic_codec = config.setdefault("semantic_codec", {})
    if not isinstance(semantic_codec, dict):
        raise ModelFormatError("IndexTTS 2.5 semantic_codec config must be a mapping")
    semantic_codec["frame_rate"] = 25

    vocoder = config.setdefault("vocoder", {})
    if not isinstance(vocoder, dict):
        raise ModelFormatError("IndexTTS 2.5 vocoder config must be a mapping")
    vocoder["name"] = "bigvgan_generator.pt"
    return config


def detect_source_version(model_dir: str | Path) -> str:
    """Detect a complete PyTorch source snapshot as 1.5, 2.0, or 2.5."""

    root = Path(model_dir)
    has_v25_marker = (root / "codec.pth").exists() or (root / V25_TOKENIZER).exists()
    if has_v25_marker:
        missing = _missing_files(root, V25_SOURCE_FILES)
        if missing:
            raise ModelFormatError(
                f"Incomplete IndexTTS 2.5 source at {root}; missing: {', '.join(missing)}"
            )
        normalize_v25_config(_load_yaml_config(root))
        return "2.5"

    if (root / "s2mel.pth").exists():
        missing = _missing_files(root, V20_SOURCE_FILES)
        if missing:
            raise ModelFormatError(
                f"Incomplete IndexTTS 2.0 source at {root}; missing: {', '.join(missing)}"
            )
        return "2.0"

    if (root / "gpt.pth").is_file() and (root / "bpe.model").is_file():
        return "1.5"
    raise ModelFormatError(f"Unrecognized IndexTTS source directory: {root}")


def detect_converted_version(model_dir: str | Path) -> str:
    """Detect an MLX model, preferring its versioned manifest when available."""

    root = Path(model_dir)
    manifest_path = root / "model_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            version = str(manifest["model_version"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelFormatError(f"Invalid converted-model manifest: {manifest_path}") from exc
        if version not in {"2.0", "2.5"}:
            raise ModelFormatError(f"Unsupported converted model version: {version}")
        return version

    has_v25_marker = (root / "codec.safetensors").exists() or (
        root / V25_TOKENIZER
    ).exists()
    if has_v25_marker:
        missing = _missing_files(root, V25_CONVERTED_FILES)
        if missing:
            raise ModelFormatError(
                f"Incomplete converted IndexTTS 2.5 model at {root}; "
                f"missing: {', '.join(missing)}"
            )
        return "2.5"
    if (root / "s2mel.safetensors").is_file():
        return "2.0"
    if (root / "gpt.safetensors").is_file():
        return "1.5"
    raise ModelFormatError(f"Unrecognized converted IndexTTS directory: {root}")
