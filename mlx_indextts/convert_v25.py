"""Strict conversion utilities for the public IndexTTS 2.5 checkpoints."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from collections.abc import Iterable, Mapping
from pathlib import Path
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml

from mlx_indextts.config import IndexTTSConfig
from mlx_indextts.convert import _quantize_weights, convert_gpt_weights
from mlx_indextts.convert_v2 import (
    convert_bigvgan_v2_weights,
    convert_s2mel_weights,
)
from mlx_indextts.model_manifest import (
    build_v25_manifest,
    collect_source_files,
    write_manifest,
)
from mlx_indextts.model_version import (
    V25_SOURCE_FILES,
    V25_TOKENIZER,
    detect_source_version,
    normalize_v25_config,
)

INDEXTTS_V25_SOURCE_REVISION = "d0aa86e75bb6f3437f3831e95056fa72842d89ef"
BIGVGAN_REPOSITORY = "nvidia/bigvgan_v2_22khz_80band_256x"
BIGVGAN_REVISION = "633ff708ed5b74903e86ff1298cf4a98e921c513"
BIGVGAN_WEIGHTS_SHA256 = (
    "e95ba25972d3de0628d99cd156e9315a9c018899bf739988959ebe3544080ced"
)
CONVERSION_STATE_FILENAME = ".conversion_state.json"


class WeightMappingError(ValueError):
    """Raised when converted tensors do not close against an MLX parameter tree."""


def _as_numpy(value: Any) -> np.ndarray:
    """Return a CPU NumPy view where possible, including for torch tensors."""
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            return value.numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _mlx_dtype(dtype: str | None):
    if dtype is None or dtype == "float32":
        return None if dtype is None else mx.float32
    choices = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    try:
        return choices[dtype]
    except KeyError as exc:
        raise ValueError("dtype must be float32, float16, bfloat16, or None") from exc


def _convert_array(value: np.ndarray, dtype: str | None) -> mx.array:
    array = mx.array(value)
    target = _mlx_dtype(dtype)
    if target is not None and mx.issubdtype(array.dtype, mx.floating):
        array = array.astype(target)
    return array


def _weight_norm_pairs(weights: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    weight_v: dict[str, str] = {}
    weight_g: dict[str, str] = {}
    for key in weights:
        if key.endswith(".weight_v"):
            weight_v[key[:-9]] = key
        elif key.endswith(".weight_g"):
            weight_g[key[:-9]] = key
    return weight_v, weight_g


def _fuse_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray) -> np.ndarray:
    norm_axes = tuple(range(1, weight_v.ndim))
    norm = np.sqrt(np.sum(np.square(weight_v), axis=norm_axes, keepdims=True))
    return weight_g * weight_v / np.maximum(norm, 1e-12)


def convert_codec_v25_weights(
    weights: Mapping[str, Any],
    *,
    dtype: str | None = None,
) -> dict[str, mx.array]:
    """Convert an EnhancedCodec state dict to the native MLX parameter layout."""
    _, weight_g_keys = _weight_norm_pairs(weights)
    converted: dict[str, mx.array] = {}
    for key, raw_value in weights.items():
        if key.endswith(".weight_g"):
            continue
        new_key = key
        value = _as_numpy(raw_value)
        if key.endswith(".weight_v"):
            base = key[:-9]
            if base not in weight_g_keys:
                raise WeightMappingError(f"weight-norm tensor has no weight_g pair: {key}")
            value = _fuse_weight_norm(value, _as_numpy(weights[weight_g_keys[base]]))
            new_key = f"{base}.weight"

        # Every rank-3 tensor in EnhancedCodec is a Conv1d kernel:
        # PyTorch OIK -> MLX OKI.
        if value.ndim == 3 and new_key.endswith(".weight"):
            value = value.transpose(0, 2, 1)
        converted[new_key] = _convert_array(value, dtype)
    return converted


def convert_gpt_v25_weights(
    weights: Mapping[str, Any],
    config: IndexTTSConfig,
    *,
    dtype: str | None = None,
) -> dict[str, mx.array]:
    """Convert GPT 2.5, including CampPlus and language embedding tensors."""
    numpy_weights = {key: _as_numpy(value) for key, value in weights.items()}
    converted = convert_gpt_weights(numpy_weights, config)
    if dtype is None:
        return converted
    target = _mlx_dtype(dtype)
    assert target is not None
    return {
        key: value.astype(target) if mx.issubdtype(value.dtype, mx.floating) else value
        for key, value in converted.items()
    }


def flatten_s2mel_state(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten ``net.{cfm,length_regulator,gpt_layer}`` checkpoint mappings."""
    net = checkpoint.get("net")
    if not isinstance(net, Mapping):
        raise WeightMappingError("S2Mel checkpoint must contain a nested net mapping")
    flat: dict[str, Any] = {}
    for module_name, module_state in net.items():
        if not isinstance(module_state, Mapping):
            raise WeightMappingError(
                f"S2Mel net.{module_name} must be a tensor mapping"
            )
        for key, value in module_state.items():
            if hasattr(value, "shape"):
                flat[f"{module_name}.{key}"] = value
    return flat


def convert_s2mel_v25_weights(
    checkpoint: Mapping[str, Any],
    *,
    dtype: str | None = None,
) -> dict[str, mx.array]:
    """Convert the released 2.5 S2Mel/DiT checkpoint to MLX."""
    flat = {key: _as_numpy(value) for key, value in flatten_s2mel_state(checkpoint).items()}
    converted = convert_s2mel_weights(flat)
    if dtype is None:
        return converted
    target = _mlx_dtype(dtype)
    assert target is not None
    return {
        key: value.astype(target) if mx.issubdtype(value.dtype, mx.floating) else value
        for key, value in converted.items()
    }


def convert_bigvgan_v25_weights(
    weights: Mapping[str, Any],
    *,
    dtype: str | None = None,
) -> dict[str, mx.array]:
    """Convert the pinned NVIDIA BigVGAN used by IndexTTS 2.5."""
    numpy_weights = {key: _as_numpy(value) for key, value in weights.items()}
    converted = convert_bigvgan_v2_weights(numpy_weights)
    if dtype is None:
        return converted
    target = _mlx_dtype(dtype)
    assert target is not None
    return {
        key: value.astype(target) if mx.issubdtype(value.dtype, mx.floating) else value
        for key, value in converted.items()
    }


def audit_parameter_mapping(
    *,
    expected_keys: Iterable[str],
    converted_keys: Iterable[str],
    source_tensors: int,
    ignored: Iterable[str] = (),
    strict: bool = True,
) -> dict[str, Any]:
    """Close converted tensors against a model tree and build a manifest record."""
    expected = set(expected_keys)
    converted = set(converted_keys)
    missing = sorted(expected - converted)
    unexpected = sorted(converted - expected)
    record = {
        "source_tensors": int(source_tensors),
        "mapped_tensors": len(expected & converted),
        "ignored": sorted(set(ignored)),
        "missing": missing,
        "unexpected": unexpected,
    }
    if strict and (missing or unexpected):
        raise WeightMappingError(
            f"weight mapping is incomplete: missing={missing}, unexpected={unexpected}"
        )
    return record


def model_parameter_keys(model: Any) -> set[str]:
    """Flatten an MLX module's trainable parameter names."""
    from mlx.utils import tree_flatten

    return {key for key, _ in tree_flatten(model.parameters())}


def _tensor_count(mapping: Mapping[str, Any]) -> int:
    return sum(hasattr(value, "shape") for value in mapping.values())


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    checkpoint = torch.load(
        str(path),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise WeightMappingError(f"checkpoint must contain a mapping: {path}")
    return checkpoint


def _save_safetensors(
    path: Path,
    weights: Mapping[str, mx.array],
    *,
    component: str,
) -> None:
    mx.save_safetensors(
        str(path),
        dict(weights),
        metadata={
            "format": "mlx",
            "model_family": "IndexTTS",
            "model_version": "2.5",
            "component": component,
        },
    )


def _converter_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WeightMappingError(f"cannot read conversion state: {path}") from exc
    if not isinstance(value, dict):
        raise WeightMappingError(f"conversion state must be a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    staging = path.with_name(f".{path.name}.tmp")
    staging.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(path)


def _archive_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}.backup-{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.backup-{stamp}-{suffix}")
        suffix += 1
    path.replace(candidate)
    return candidate


def prepare_conversion_staging(
    output_dir: str | Path,
    identity: Mapping[str, Any],
    *,
    resume: bool = True,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Create or validate a resumable sibling staging directory.

    ``force`` archives existing data instead of deleting it.  The caller can
    therefore recover both a previous final model and an incompatible partial
    conversion.
    """
    output = Path(output_dir)
    staging = output.with_name(f".{output.name}.staging")
    if output.exists():
        if not force:
            raise FileExistsError(f"converted model already exists: {output}")
        _archive_path(output)

    state_path = staging / CONVERSION_STATE_FILENAME
    if staging.exists():
        if not resume:
            if not force:
                raise FileExistsError(f"conversion staging already exists: {staging}")
            _archive_path(staging)
        else:
            previous = _read_json(state_path)
            previous_identity = previous.get("identity")
            if previous_identity != dict(identity):
                if not force:
                    raise WeightMappingError(
                        "existing conversion staging belongs to a different source or options"
                    )
                _archive_path(staging)
            else:
                return staging, previous

    staging.mkdir(parents=True, exist_ok=False)
    state = {
        "format_version": 1,
        "identity": dict(identity),
        "completed": [],
        "components": {},
    }
    _write_json_atomic(state_path, state)
    return staging, state


def _mark_component_complete(
    staging: Path,
    state: dict[str, Any],
    name: str,
    record: Mapping[str, Any],
) -> None:
    completed = list(state.get("completed", []))
    if name not in completed:
        completed.append(name)
    state["completed"] = completed
    state.setdefault("components", {})[name] = dict(record)
    _write_json_atomic(staging / CONVERSION_STATE_FILENAME, state)


def _component_is_resumable(
    staging: Path,
    state: Mapping[str, Any],
    component: str,
    filename: str,
) -> bool:
    path = staging / filename
    record = state.get("components", {}).get(component, {})
    return (
        component in state.get("completed", [])
        and isinstance(record, Mapping)
        and path.is_file()
        and int(record.get("bytes", -1)) == path.stat().st_size
    )


def _load_normalized_config(source_dir: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load((source_dir / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WeightMappingError(f"cannot load source config: {source_dir / 'config.yaml'}") from exc
    if not isinstance(raw, dict):
        raise WeightMappingError("source config.yaml must contain a mapping")
    return normalize_v25_config(raw)


def convert_model_v25(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    dtype: str = "float16",
    quantize_bits: int | None = None,
    group_size: int = 64,
    source_revision: str = INDEXTTS_V25_SOURCE_REVISION,
    resume: bool = True,
    force: bool = False,
    progress: Callable[[str], None] | None = print,
) -> Path:
    """Convert and atomically publish a self-contained IndexTTS 2.5 MLX model."""
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if source == output:
        raise ValueError("source_dir and output_dir must be different")
    if detect_source_version(source) != "2.5":
        raise ValueError(f"not an IndexTTS 2.5 source snapshot: {source}")
    _mlx_dtype(dtype)
    if quantize_bits not in (None, 4, 8):
        raise ValueError("quantize_bits must be 4, 8, or None")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    normalized = _load_normalized_config(source)
    identity = {
        "source": str(source),
        "source_revision": source_revision,
        "source_files": {
            name: (source / name).stat().st_size for name in sorted(V25_SOURCE_FILES)
        },
        "dtype": dtype,
        "quantize_bits": quantize_bits,
        "group_size": group_size,
        "bigvgan_revision": BIGVGAN_REVISION,
    }
    staging, state = prepare_conversion_staging(
        output,
        identity,
        resume=resume,
        force=force,
    )

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    # Config is written before component conversion so partial output is inspectable.
    (staging / "config.yaml").write_text(
        yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (staging / "config.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    from omegaconf import OmegaConf
    from mlx_indextts.models.bigvgan_v2 import BigVGANV2, BigVGANV2Config
    from mlx_indextts.models.codec_v25 import EnhancedCodecV25
    from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25
    from mlx_indextts.models.s2mel import create_s2mel_from_config

    config = IndexTTSConfig.from_omegaconf(OmegaConf.create(normalized))
    config.version = 2.5

    if not _component_is_resumable(staging, state, "gpt", "gpt.safetensors"):
        announce("[1/4] converting GPT 2.5")
        source_state = _load_checkpoint(source / "gpt.pth")
        model = UnifiedVoiceV25(config)
        weights = convert_gpt_v25_weights(source_state, config, dtype=dtype)
        ignored = [key for key in source_state if ".pos_enc.pe" in key]
        record = audit_parameter_mapping(
            expected_keys=model_parameter_keys(model),
            converted_keys=weights,
            source_tensors=_tensor_count(source_state),
            ignored=ignored,
            strict=True,
        )
        if quantize_bits is not None:
            weights = _quantize_weights(weights, config, quantize_bits, group_size)
            nn.quantize(model.gpt, bits=quantize_bits, group_size=group_size)
        _save_safetensors(staging / "gpt.safetensors", weights, component="gpt")
        model.load_weights(str(staging / "gpt.safetensors"), strict=True)
        record.update(
            {
                "file": "gpt.safetensors",
                "saved_tensors": len(weights),
                "quantized": quantize_bits is not None,
                "bytes": (staging / "gpt.safetensors").stat().st_size,
            }
        )
        _mark_component_complete(staging, state, "gpt", record)
        del weights, model, source_state
        mx.clear_cache()
    else:
        announce("[1/4] reusing completed GPT conversion")

    if not _component_is_resumable(staging, state, "codec", "codec.safetensors"):
        announce("[2/4] converting EnhancedCodec")
        checkpoint = _load_checkpoint(source / "codec.pth")
        source_state = checkpoint.get("model")
        if not isinstance(source_state, Mapping):
            raise WeightMappingError("codec.pth must contain a model mapping")
        codec_cfg = normalized["semantic_codec"]
        model = EnhancedCodecV25(
            codebook_size=int(codec_cfg.get("codebook_size", 8192)),
            hidden_size=int(codec_cfg.get("hidden_size", 1024)),
            codebook_dim=int(codec_cfg.get("codebook_dim", 8)),
            vocos_dim=int(codec_cfg.get("vocos_dim", 384)),
            vocos_intermediate_dim=int(codec_cfg.get("vocos_intermediate_dim", 2048)),
            vocos_num_layers=int(codec_cfg.get("vocos_num_layers", 12)),
        )
        weights = convert_codec_v25_weights(source_state, dtype=dtype)
        record = audit_parameter_mapping(
            expected_keys=model_parameter_keys(model),
            converted_keys=weights,
            source_tensors=_tensor_count(source_state),
            strict=True,
        )
        _save_safetensors(staging / "codec.safetensors", weights, component="codec")
        model.load_weights(str(staging / "codec.safetensors"), strict=True)
        record.update(
            {
                "file": "codec.safetensors",
                "saved_tensors": len(weights),
                "bytes": (staging / "codec.safetensors").stat().st_size,
            }
        )
        _mark_component_complete(staging, state, "codec", record)
        del weights, model, source_state, checkpoint
        mx.clear_cache()
    else:
        announce("[2/4] reusing completed EnhancedCodec conversion")

    if not _component_is_resumable(staging, state, "s2mel", "s2mel.safetensors"):
        announce("[3/4] converting S2Mel/DiT")
        checkpoint = _load_checkpoint(source / "s2mel.pth")
        flat_source = flatten_s2mel_state(checkpoint)
        model = create_s2mel_from_config(normalized["s2mel"])
        weights = convert_s2mel_v25_weights(checkpoint, dtype=dtype)
        ignored = [key for key in flat_source if "input_pos" in key]
        record = audit_parameter_mapping(
            expected_keys=model_parameter_keys(model),
            converted_keys=weights,
            source_tensors=_tensor_count(flat_source),
            ignored=ignored,
            strict=True,
        )
        _save_safetensors(staging / "s2mel.safetensors", weights, component="s2mel")
        model.load_weights(str(staging / "s2mel.safetensors"), strict=True)
        record.update(
            {
                "file": "s2mel.safetensors",
                "saved_tensors": len(weights),
                "bytes": (staging / "s2mel.safetensors").stat().st_size,
            }
        )
        _mark_component_complete(staging, state, "s2mel", record)
        del weights, model, flat_source, checkpoint
        mx.clear_cache()
    else:
        announce("[3/4] reusing completed S2Mel conversion")

    if not _component_is_resumable(staging, state, "bigvgan", "bigvgan.safetensors"):
        announce("[4/4] converting pinned BigVGAN")
        from huggingface_hub import hf_hub_download

        weights_path = Path(
            hf_hub_download(
                BIGVGAN_REPOSITORY,
                "bigvgan_generator.pt",
                revision=BIGVGAN_REVISION,
            )
        )
        config_path = Path(
            hf_hub_download(
                BIGVGAN_REPOSITORY,
                "config.json",
                revision=BIGVGAN_REVISION,
            )
        )
        checkpoint = _load_checkpoint(weights_path)
        source_state = checkpoint.get("generator", checkpoint)
        if not isinstance(source_state, Mapping):
            raise WeightMappingError("BigVGAN checkpoint must contain a generator mapping")
        bigvgan_config = BigVGANV2Config.from_dict(_read_json(config_path))
        model = BigVGANV2(bigvgan_config)
        weights = convert_bigvgan_v25_weights(source_state, dtype=dtype)
        ignored = [key for key in source_state if "filter" in key]
        record = audit_parameter_mapping(
            expected_keys=model_parameter_keys(model),
            converted_keys=weights,
            source_tensors=_tensor_count(source_state),
            ignored=ignored,
            strict=True,
        )
        _save_safetensors(staging / "bigvgan.safetensors", weights, component="bigvgan")
        model.load_weights(str(staging / "bigvgan.safetensors"), strict=True)
        record.update(
            {
                "file": "bigvgan.safetensors",
                "saved_tensors": len(weights),
                "source_repository": BIGVGAN_REPOSITORY,
                "source_revision": BIGVGAN_REVISION,
                "source_sha256": BIGVGAN_WEIGHTS_SHA256,
                "bytes": (staging / "bigvgan.safetensors").stat().st_size,
            }
        )
        _mark_component_complete(staging, state, "bigvgan", record)
        del weights, model, source_state, checkpoint
        mx.clear_cache()
    else:
        announce("[4/4] reusing completed BigVGAN conversion")

    for name in (V25_TOKENIZER, "feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt"):
        shutil.copy2(source / name, staging / name)

    components = dict(state["components"])
    quantization = (
        {"component": "gpt", "bits": quantize_bits, "group_size": group_size}
        if quantize_bits is not None
        else None
    )
    manifest = build_v25_manifest(
        source_revision=source_revision,
        source_files=collect_source_files(source, V25_SOURCE_FILES),
        converter_revision=_converter_revision(),
        dtype=dtype,
        quantization=quantization,
        components=components,
    )
    write_manifest(staging, manifest)
    report = {
        "status": "pass",
        "model_version": "2.5",
        "source_revision": source_revision,
        "bigvgan_revision": BIGVGAN_REVISION,
        "dtype": dtype,
        "quantization": quantization,
        "components": components,
        "artifacts": {
            path.name: path.stat().st_size
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != CONVERSION_STATE_FILENAME
        },
    }
    _write_json_atomic(staging / "conversion_report.json", report)
    (staging / CONVERSION_STATE_FILENAME).unlink()
    staging.replace(output)
    announce(f"conversion complete: {output}")
    return output
