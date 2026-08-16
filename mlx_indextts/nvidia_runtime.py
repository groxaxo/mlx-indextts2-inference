"""NVIDIA CUDA adapter for the official IndexTTS 2/2.5 PyTorch runtime.

The adapter deliberately keeps the official implementation as the inference
source of truth.  It adds stable configuration, validation, language routing,
emotion parsing, thread-safe model reuse, and a backend-neutral result shape.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .model_version import ModelFormatError, normalize_v25_config

UPSTREAM_REPOSITORY = "https://github.com/index-tts/index-tts"
UPSTREAM_REVISION = "4f8792ff120cd3ea470dd511e997a17c86cddd10"
MODEL_REPOSITORIES = {
    "2.5": "IndexTeam/IndexTTS-2.5",
    "2.0": "IndexTeam/IndexTTS-2",
}
MODEL_REVISIONS = {
    "2.5": "d0aa86e75bb6f3437f3831e95056fa72842d89ef",
    "2.0": None,
}
SUPPORTED_LANGUAGES = ("ZH", "EN", "JA", "ES", "AR")
EMOTION_NAMES = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)


class NvidiaBackendError(RuntimeError):
    """Base error for NVIDIA backend failures."""


class MissingNvidiaDependencies(NvidiaBackendError):
    """Raised when the pinned official PyTorch runtime is unavailable."""


class CudaUnavailableError(NvidiaBackendError):
    """Raised when a CUDA device was requested but PyTorch cannot use it."""


def normalize_version(value: str | float | int) -> str:
    """Normalize supported IndexTTS version aliases."""

    normalized = str(value).strip().lower().replace("indextts", "").replace("v", "")
    normalized = normalized.replace("_", ".").replace("-", ".").strip(" .")
    aliases = {
        "2.5": "2.5",
        "25": "2.5",
        "2.0": "2.0",
        "2": "2.0",
        "20": "2.0",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported IndexTTS version {value!r}; choose 2.5 or 2.0") from exc


def detect_language(text: str) -> str:
    """Conservatively detect the language token expected by IndexTTS 2.5.

    Latin-script input defaults to English because English and Spanish cannot
    be distinguished reliably without a language classifier.  Callers should
    pass ``--language es`` for Spanish.
    """

    if re.search(r"[\u3040-\u30ff]", text):
        return "JA"
    if re.search(r"[\u0600-\u06ff]", text):
        return "AR"
    if re.search(r"[\u3400-\u9fff]", text):
        return "ZH"
    return "EN"


def normalize_language(value: str | None, text: str = "") -> str:
    """Normalize a language name/code to the official uppercase token."""

    if value is None or str(value).strip().lower() in {"", "auto"}:
        return detect_language(text)
    aliases = {
        "zh": "ZH",
        "cn": "ZH",
        "chinese": "ZH",
        "mandarin": "ZH",
        "en": "EN",
        "english": "EN",
        "ja": "JA",
        "jp": "JA",
        "japanese": "JA",
        "es": "ES",
        "spanish": "ES",
        "ar": "AR",
        "arabic": "AR",
    }
    normalized = str(value).strip().lower()
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported language {value!r}; choose auto, zh, en, ja, es, or ar"
        ) from exc


def normalize_v25_runtime_config(model_dir: Path, config_path: Path) -> Path:
    """Write a local config for public v2.5 artifacts with stale paths."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or int(raw.get("gpt", {}).get("number_text_tokens", 0)) != 60509:
            return config_path
        normalized = normalize_v25_config(raw)
        # The MLX converter uses this metadata, but the pinned official CUDA
        # EnhancedCodec constructor rejects unknown keyword arguments.
        normalized["semantic_codec"].pop("frame_rate", None)
    except (OSError, TypeError, ValueError, yaml.YAMLError, ModelFormatError):
        return config_path

    target = model_dir / "config.nvidia-v25.yaml"
    serialized = yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False)
    if not target.is_file() or target.read_text(encoding="utf-8") != serialized:
        target.write_text(serialized, encoding="utf-8")
    return target


def _coerce_emotion_values(values: Sequence[Any]) -> list[float]:
    if len(values) != len(EMOTION_NAMES):
        raise ValueError(
            f"Emotion vectors require exactly {len(EMOTION_NAMES)} values in this order: "
            + ", ".join(EMOTION_NAMES)
        )
    vector = [float(value) for value in values]
    if any(value < 0.0 or value > 1.0 for value in vector):
        raise ValueError("Emotion vector values must be between 0.0 and 1.0")
    total = sum(vector)
    if total > 0.8:
        scale = 0.8 / total
        vector = [value * scale for value in vector]
    return vector


def parse_emotion_vector(value: str | Sequence[float] | Mapping[str, float] | None) -> list[float] | None:
    """Parse named, JSON, mapping, or numeric emotion vectors.

    Named examples: ``happy`` or ``happy:0.6,sad:0.2``.
    Numeric examples: JSON ``[0.8, 0, ...]`` or eight comma-separated values.
    The official model recommends a total emotion mass no greater than 0.8, so
    larger vectors are proportionally normalized.
    """

    if value is None:
        return None
    if isinstance(value, Mapping):
        unknown = set(value) - set(EMOTION_NAMES)
        if unknown:
            raise ValueError(f"Unknown emotions: {', '.join(sorted(unknown))}")
        return _coerce_emotion_values([value.get(name, 0.0) for name in EMOTION_NAMES])
    if not isinstance(value, str):
        return _coerce_emotion_values(value)

    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("Emotion JSON must be an array of eight numbers")
        return _coerce_emotion_values(parsed)

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if parts and all(":" not in part and part.lower() in EMOTION_NAMES for part in parts):
        weights = {part.lower(): 0.8 / len(parts) for part in parts}
        return _coerce_emotion_values([weights.get(name, 0.0) for name in EMOTION_NAMES])
    if any(":" in part for part in parts):
        weights: dict[str, float] = {}
        for part in parts:
            if ":" not in part:
                name, weight = part.lower(), 1.0
            else:
                name, raw_weight = part.split(":", 1)
                name, weight = name.strip().lower(), float(raw_weight)
            if name not in EMOTION_NAMES:
                raise ValueError(f"Unknown emotion {name!r}")
            weights[name] = weight
        return _coerce_emotion_values([weights.get(name, 0.0) for name in EMOTION_NAMES])

    try:
        return _coerce_emotion_values([float(part) for part in parts])
    except ValueError as exc:
        if raw.lower() in EMOTION_NAMES:
            return parse_emotion_vector({raw.lower(): 0.8})
        raise ValueError(
            "Emotion must be a name, named mix, JSON array, or eight comma-separated numbers"
        ) from exc


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise MissingNvidiaDependencies(
            "CUDA PyTorch is unavailable. Run ./scripts/setup_nvidia.sh or "
            "`uv sync --project nvidia`."
        ) from exc


def _validate_optional_accelerators(config: "NvidiaRuntimeConfig") -> None:
    missing: list[str] = []
    if config.use_accel and importlib.util.find_spec("flash_attn") is None:
        missing.append("GPT acceleration (--extra accel)")
    if config.use_deepspeed and importlib.util.find_spec("deepspeed") is None:
        missing.append("DeepSpeed (--extra deepspeed)")
    if missing:
        raise MissingNvidiaDependencies(
            "Missing optional NVIDIA dependencies for "
            + ", ".join(missing)
            + ". Re-run `uv sync --project nvidia` with the named extra(s)."
        )


def _load_upstream_class(version: str) -> type[Any]:
    module_name = "indextts.infer_v2_5" if version == "2.5" else "indextts.infer_v2"
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise MissingNvidiaDependencies(
            "The pinned official IndexTTS CUDA runtime is unavailable. Run "
            "./scripts/setup_nvidia.sh or `uv sync --project nvidia`."
        ) from exc
    try:
        return module.IndexTTS2
    except AttributeError as exc:
        raise MissingNvidiaDependencies(
            f"{module_name} does not expose IndexTTS2; expected upstream revision "
            f"{UPSTREAM_REVISION}."
        ) from exc


def resolve_device(torch_module: Any, requested: str | None) -> str:
    """Resolve and validate a CUDA/CPU device string."""

    value = (requested or "auto").strip().lower()
    if value == "auto":
        if torch_module.cuda.is_available():
            return "cuda:0"
        raise CudaUnavailableError(
            "No CUDA device is available. Verify the NVIDIA driver and CUDA PyTorch build, "
            "or explicitly pass --device cpu for a slow diagnostic run."
        )
    if value == "cuda":
        value = "cuda:0"
    if value == "cpu":
        return value
    match = re.fullmatch(r"cuda:(\d+)", value)
    if not match:
        raise ValueError("Device must be auto, cpu, cuda, or cuda:N")
    if not torch_module.cuda.is_available():
        raise CudaUnavailableError("CUDA was requested, but torch.cuda.is_available() is false")
    index = int(match.group(1))
    count = int(torch_module.cuda.device_count())
    if index >= count:
        raise CudaUnavailableError(f"Requested cuda:{index}, but only {count} CUDA device(s) exist")
    return value


def resolve_precision(torch_module: Any, version: str, device: str, requested: str | None) -> str:
    """Resolve the precision supported by the selected official runtime."""

    version = normalize_version(version)
    value = (requested or "auto").strip().lower()
    if value not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("Precision must be auto, bf16, fp16, or fp32")
    if device == "cpu":
        if value not in {"auto", "fp32"}:
            raise ValueError("CPU inference supports fp32 only")
        return "fp32"
    if value == "auto":
        if version == "2.5":
            supported = bool(getattr(torch_module.cuda, "is_bf16_supported", lambda: False)())
            return "bf16" if supported else "fp32"
        return "fp16"
    if version == "2.5" and value == "fp16":
        raise ValueError("The official IndexTTS 2.5 runtime exposes BF16, not FP16; use bf16")
    if version == "2.0" and value == "bf16":
        raise ValueError("The official IndexTTS 2.0 runtime exposes FP16, not BF16; use fp16")
    if value == "bf16" and not bool(
        getattr(torch_module.cuda, "is_bf16_supported", lambda: False)()
    ):
        raise ValueError("This CUDA device/PyTorch build does not report BF16 support")
    return value


@dataclass(slots=True)
class NvidiaRuntimeConfig:
    model_dir: str | Path = "checkpoints"
    version: str = "2.5"
    config_path: str | Path | None = None
    device: str = "auto"
    precision: str = "auto"
    use_cuda_kernel: bool | None = None
    use_deepspeed: bool = False
    use_accel: bool = False
    use_torch_compile: bool = False
    use_qwen_emotion: bool = False

    def normalized_version(self) -> str:
        return normalize_version(self.version)

    def resolved_model_dir(self) -> Path:
        return Path(self.model_dir).expanduser().resolve()

    def resolved_config_path(self) -> Path:
        if self.config_path:
            return Path(self.config_path).expanduser().resolve()
        return self.resolved_model_dir() / "config.yaml"


@dataclass(slots=True)
class NvidiaGenerateRequest:
    text: str
    ref_audio: str | Path
    output_path: str | Path
    language: str = "auto"
    emotion_ref_audio: str | Path | None = None
    emotion_vector: str | Sequence[float] | Mapping[str, float] | None = None
    emotion_text: str | None = None
    auto_emotion: bool = False
    use_random: bool = False
    emo_alpha: float = 0.6
    interval_silence_ms: int = 200
    max_text_tokens: int = 120
    duration_factor: float = 1.0
    text_normalization: bool = True
    max_mel_tokens: int | None = None
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    repetition_penalty: float = 10.0
    seed: int | None = None
    verbose: bool = False

    def validate(self, *, version: str, qwen_loaded: bool) -> None:
        if not self.text.strip():
            raise ValueError("Text cannot be empty")
        ref_audio = Path(self.ref_audio).expanduser()
        if not ref_audio.is_file():
            raise FileNotFoundError(f"Speaker reference audio does not exist: {ref_audio}")
        if self.emotion_ref_audio and not Path(self.emotion_ref_audio).expanduser().is_file():
            raise FileNotFoundError(
                f"Emotion reference audio does not exist: {self.emotion_ref_audio}"
            )
        emotion_modes = sum(
            bool(value)
            for value in (
                self.emotion_ref_audio,
                self.emotion_vector,
                self.emotion_text or self.auto_emotion,
            )
        )
        if emotion_modes > 1:
            raise ValueError(
                "Choose only one emotion source: reference audio, vector/named mix, or text"
            )
        if (self.emotion_text or self.auto_emotion) and not qwen_loaded:
            raise ValueError(
                "Text emotion requires a runtime created with use_qwen_emotion=True "
                "(--qwen-emotion)."
            )
        if not 0.0 <= self.emo_alpha <= 1.0:
            raise ValueError("emo_alpha must be between 0.0 and 1.0")
        if self.interval_silence_ms < 0:
            raise ValueError("interval_silence_ms cannot be negative")
        if self.max_text_tokens <= 0:
            raise ValueError("max_text_tokens must be positive")
        if self.max_mel_tokens is not None and self.max_mel_tokens <= 0:
            raise ValueError("max_mel_tokens must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        if version == "2.5" and not 0.5 <= self.duration_factor <= 2.0:
            raise ValueError("IndexTTS 2.5 duration_factor must be between 0.5 and 2.0")


@dataclass(slots=True)
class NvidiaGenerateResult:
    output_path: str
    device: str
    version: str
    precision: str
    language: str | None
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "device": self.device,
            "version": self.version,
            "precision": self.precision,
            "language": self.language,
            "elapsed_seconds": self.elapsed_seconds,
        }


class NvidiaIndexTTS:
    """Thread-safe, model-resident CUDA runtime for IndexTTS 2/2.5."""

    def __init__(
        self,
        config: NvidiaRuntimeConfig,
        *,
        torch_module: Any | None = None,
        upstream_class: type[Any] | None = None,
    ) -> None:
        self.config = config
        self.version = config.normalized_version()
        self._torch = torch_module or _import_torch()
        self.device = resolve_device(self._torch, config.device)
        if self.device.startswith("cuda"):
            self._torch.cuda.set_device(int(self.device.split(":", 1)[1]))
        self.precision = resolve_precision(
            self._torch, self.version, self.device, config.precision
        )
        self.model_dir = config.resolved_model_dir()
        self.config_path = config.resolved_config_path()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Model config does not exist: {self.config_path}")
        if self.version == "2.5":
            self.config_path = normalize_v25_runtime_config(self.model_dir, self.config_path)
        _validate_optional_accelerators(config)
        runtime_class = upstream_class or _load_upstream_class(self.version)
        kwargs: dict[str, Any] = {
            "cfg_path": str(self.config_path),
            "model_dir": str(self.model_dir),
            "device": self.device,
            "use_cuda_kernel": config.use_cuda_kernel,
            "use_deepspeed": config.use_deepspeed,
            "use_accel": config.use_accel,
            "use_torch_compile": config.use_torch_compile,
            "use_qwen_emo": config.use_qwen_emotion,
        }
        if self.version == "2.5":
            kwargs["use_bf16"] = self.precision == "bf16"
        else:
            kwargs["use_fp16"] = self.precision == "fp16"
        self._model = runtime_class(**kwargs)
        self._lock = threading.Lock()

    def _seed(self, seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        self._torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            self._torch.cuda.manual_seed_all(seed)

    def generate(self, request: NvidiaGenerateRequest) -> NvidiaGenerateResult:
        request.validate(version=self.version, qwen_loaded=self.config.use_qwen_emotion)
        output_path = Path(request.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        emotion_vector = parse_emotion_vector(request.emotion_vector)
        generation_kwargs: dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
        }
        if request.max_mel_tokens is not None:
            generation_kwargs["max_mel_tokens"] = request.max_mel_tokens
        common: dict[str, Any] = {
            "spk_audio_prompt": str(Path(request.ref_audio).expanduser().resolve()),
            "text": request.text,
            "output_path": str(output_path),
            "emo_audio_prompt": (
                str(Path(request.emotion_ref_audio).expanduser().resolve())
                if request.emotion_ref_audio
                else None
            ),
            "emo_alpha": request.emo_alpha,
            "emo_vector": emotion_vector,
            "use_emo_text": bool(request.emotion_text or request.auto_emotion),
            "emo_text": request.emotion_text,
            "use_random": request.use_random,
            "interval_silence": request.interval_silence_ms,
            "verbose": request.verbose,
            "max_text_tokens_per_segment": request.max_text_tokens,
            **generation_kwargs,
        }
        language: str | None = None
        if self.version == "2.5":
            language = normalize_language(request.language, request.text)
            common.update(
                {
                    "lang": language,
                    "duration_factor": request.duration_factor,
                    "text_normalization": request.text_normalization,
                }
            )
        started = time.perf_counter()
        with self._lock:
            self._seed(request.seed)
            if output_path.exists():
                output_path.unlink()
            self._model.infer(**common)
        elapsed = time.perf_counter() - started
        if not output_path.is_file():
            raise NvidiaBackendError(
                f"Official IndexTTS runtime returned without creating {output_path}"
            )
        return NvidiaGenerateResult(
            output_path=str(output_path),
            device=self.device,
            version=self.version,
            precision=self.precision,
            language=language,
            elapsed_seconds=elapsed,
        )

    def device_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "device": self.device,
            "version": self.version,
            "precision": self.precision,
            "model_dir": str(self.model_dir),
        }
        if self.device.startswith("cuda"):
            index = int(self.device.split(":", 1)[1])
            properties = self._torch.cuda.get_device_properties(index)
            info.update(
                {
                    "device_name": properties.name,
                    "total_vram_gb": round(properties.total_memory / 1024**3, 2),
                    "bf16_supported": bool(
                        getattr(self._torch.cuda, "is_bf16_supported", lambda: False)()
                    ),
                }
            )
        return info

    def close(self) -> None:
        self._model = None
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()
