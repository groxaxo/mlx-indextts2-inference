"""Command-line interface for MLX-IndexTTS."""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from mlx_indextts.config import default_max_mel_tokens


DEFAULT_STANDARD_MODEL = "models/mlx-indexTTS2-standard-8bit"
DEFAULT_VIETNAMESE_MODEL = "models/mlx-indexTTS2-vietnamese-8bit"
DEFAULT_V25_MODEL = "models/mlx-IndexTTS-2.5-8bit"


def looks_vietnamese(text: str) -> bool:
    """Detect Vietnamese text with tone marks for default profile routing."""
    if not text:
        return False
    return bool(re.search(r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ]", text))


def resolve_default_model(profile: str, text: str = "") -> str:
    """Resolve a local 2.0 profile or the latest 2.5 standard model."""
    standard_model = os.environ.get("MLX_INDEXTTS_STANDARD_MODEL", DEFAULT_STANDARD_MODEL)
    vietnamese_model = os.environ.get("MLX_INDEXTTS_VIETNAMESE_MODEL", DEFAULT_VIETNAMESE_MODEL)
    v25_model = os.environ.get("MLX_INDEXTTS_V25_MODEL", DEFAULT_V25_MODEL)
    if profile == "auto":
        if looks_vietnamese(text):
            profile = "vietnamese"
        else:
            profile = "v25" if Path(v25_model).is_dir() else "standard"
    if profile in {"vi", "vietnamese"}:
        return vietnamese_model
    if profile in {"v25", "2.5", "latest"}:
        return v25_model
    return standard_model


def _validate_cli_emotion_sources(
    *,
    auto_emotion: bool,
    emotion_text: str | None = None,
    emotion: str | None,
    emotion_ref_audio: str | None,
    has_row_emotion_refs: bool = False,
) -> None:
    auto = auto_emotion or bool(emotion_text) or emotion == "auto-qwen"
    manual = bool(emotion and emotion != "auto-qwen")
    emotion_ref = bool(emotion_ref_audio) or has_row_emotion_refs
    if sum(bool(value) for value in (auto, manual, emotion_ref)) > 1:
        raise SystemExit(
            "Emotion sources are mutually exclusive: use only one of "
            "--auto-emotion/--emotion-text/--emotion auto-qwen, --emotion, "
            "or --emotion-ref-audio/CSV emotion_ref_audio."
        )


def _first_row_value(row: dict[str, str], names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _parse_key_value_args(values: list[str] | None, *, option_name: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"{option_name} expects NAME=VALUE, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"{option_name} expects non-empty NAME=VALUE, got: {raw}")
        mapping[key] = value
    return mapping


def _parse_speaker_profiles(values: list[str] | None) -> dict[str, dict[str, str]]:
    profiles: dict[str, dict[str, str]] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"--speaker-profile expects SPEAKER=gender:age_band, got: {raw}")
        speaker, payload = raw.split("=", 1)
        parts = [part.strip() for part in payload.split(":", 1)]
        if len(parts) != 2 or not speaker.strip() or not parts[0] or not parts[1]:
            raise SystemExit(f"--speaker-profile expects SPEAKER=gender:age_band, got: {raw}")
        profiles[speaker.strip()] = {"gender": parts[0], "age_band": parts[1]}
    return profiles


def _emotion_vector_from_row(row: dict[str, str]) -> str:
    cn_to_en = {
        "喜": "happy",
        "怒": "angry",
        "哀": "sad",
        "惧": "afraid",
        "厌恶": "disgusted",
        "低落": "melancholic",
        "惊喜": "surprised",
        "平静": "calm",
    }
    parts = []
    for cn, en in cn_to_en.items():
        value = (row.get(cn) or "").strip()
        if not value:
            continue
        try:
            weight = float(value)
        except ValueError:
            continue
        if weight > 0:
            parts.append(f"{en}:{weight:g}")
    return ",".join(parts)


def detect_pytorch_version(model_dir: Path) -> str:
    """Detect IndexTTS version from PyTorch model directory.

    Returns '2.0' if s2mel.pth exists, otherwise '1.5'.
    """
    from mlx_indextts.model_version import ModelFormatError, detect_source_version

    try:
        return detect_source_version(model_dir)
    except ModelFormatError:
        if (model_dir / "s2mel.pth").exists():
            return "2.0"
        return "1.5"


def detect_mlx_version(model_dir: Path) -> str:
    """Detect IndexTTS version from MLX model directory.

    Returns '2.0' if s2mel.safetensors exists, otherwise '1.5'.
    """
    from mlx_indextts.model_version import ModelFormatError, detect_converted_version

    try:
        return detect_converted_version(model_dir)
    except ModelFormatError:
        if (model_dir / "s2mel.safetensors").exists():
            return "2.0"
        return "1.5"


def convert_command(args):
    """Handle the convert command (auto-detect version)."""
    model_dir = Path(args.model_dir)
    version = detect_pytorch_version(model_dir)

    # Parse quantize option
    quantize_bits = None
    if args.quantize:
        if args.quantize.lower() == "fp32":
            quantize_bits = None
        else:
            quantize_bits = int(args.quantize)

    print(f"Detected IndexTTS version: {version}")

    if version == "2.5":
        from mlx_indextts.convert_v25 import convert_model_v25

        convert_model_v25(
            source_dir=args.model_dir,
            output_dir=args.output,
            dtype=args.dtype,
            quantize_bits=quantize_bits,
            source_revision=args.source_revision,
            resume=args.resume,
            force=args.force,
        )
    elif version == "2.0":
        from mlx_indextts.convert_v2 import convert_model as convert_model_v2
        convert_model_v2(
            model_dir=args.model_dir,
            output_dir=args.output,
            config_path=args.config,
            quantize_bits=quantize_bits,
        )
    else:
        from mlx_indextts.convert import convert_model

        convert_model(
            model_dir=args.model_dir,
            output_dir=args.output,
            config_path=args.config,
            quantize_bits=quantize_bits,
        )


def generate_command(args):
    """Handle the generate command (auto-detect version)."""
    import subprocess

    _validate_cli_emotion_sources(
        auto_emotion=getattr(args, 'auto_emotion', False),
        emotion_text=getattr(args, 'emotion_text', None),
        emotion=getattr(args, 'emotion', None),
        emotion_ref_audio=getattr(args, 'emotion_ref_audio', None),
    )

    if not args.model:
        args.model = resolve_default_model(args.profile, args.text)
        print(f"Using default {args.profile} 8bit model: {args.model}")

    model_dir = Path(args.model)
    version = detect_mlx_version(model_dir)

    # Check for v2-only parameters used with v1.5 model
    v2_only_params = []
    if getattr(args, 'emotion', None) is not None:
        v2_only_params.append('--emotion')
    if getattr(args, 'auto_emotion', False):
        v2_only_params.append('--auto-emotion')
    if getattr(args, 'emotion_ref_audio', None):
        v2_only_params.append('--emotion-ref-audio')
    if getattr(args, 'emo_alpha', 0.6) != 0.6:
        v2_only_params.append('--emo-alpha')
    if getattr(args, 'diffusion_steps', 25) != 25:
        v2_only_params.append('--diffusion-steps')
    if getattr(args, 'cfg_rate', 0.7) != 0.7:
        v2_only_params.append('--cfg-rate')

    if version == "1.5" and v2_only_params:
        print(f"Error: Parameters {v2_only_params} are only available for IndexTTS 2.0 models.")
        print("Detected model version: 1.5")
        sys.exit(1)

    # Default temperature based on version
    if args.temperature is None:
        temperature = 0.8 if version in {"2.0", "2.5"} else 1.0
    else:
        temperature = args.temperature

    # v1.5: 800, v2.0: 1500, v2.5: 256 by default. Callers can explicitly
    # raise the cap for long-form work (v2.0 supports up to about 1815).
    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    else:
        max_tokens = default_max_mel_tokens(version)

    memory_limit = args.memory_limit

    # Get max_text_tokens_per_segment (default 120)
    max_text_tokens = getattr(args, 'max_text_tokens', 120)

    print(f"Using IndexTTS {version}")
    from mlx_indextts.runtime import GenerateOptions, TTSRuntime

    options = GenerateOptions(
        max_tokens=max_tokens,
        max_text_tokens=max_text_tokens,
        interval_silence=getattr(args, 'interval_silence', 200),
        temperature=temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=getattr(args, 'repetition_penalty', 10.0),
        diffusion_steps=getattr(args, 'diffusion_steps', 16),
        cfg_rate=getattr(args, 'cfg_rate', 0.7),
        emotion="auto-qwen" if getattr(args, 'auto_emotion', False) else getattr(args, 'emotion', None),
        emo_alpha=getattr(args, 'emo_alpha', 0.6),
        emotion_ref_audio=getattr(args, 'emotion_ref_audio', None),
        auto_emotion=getattr(args, 'auto_emotion', False),
        use_emo_text=bool(getattr(args, 'emotion_text', None)),
        emo_text=getattr(args, 'emotion_text', None),
        use_random=getattr(args, 'use_random', False),
        qwen_emotion_model=getattr(args, 'qwen_emotion_model', None),
        qwen_unload_after=getattr(args, 'qwen_unload_after', True),
        denoise_ref_audio=getattr(args, 'denoise_ref', False),
        denoise_emotion_ref_audio=getattr(args, 'denoise_emotion_ref', False),
        seed=getattr(args, 'seed', None),
        verbose=args.verbose,
        segment_overlap=getattr(args, 'segment_overlap', 50),
        speed=getattr(args, 'speed', 1.0),
        target_duration=getattr(args, 'target_duration', None),
        fit_duration=getattr(args, 'fit_duration', False),
        max_fit_stretch_ratio=getattr(args, 'max_fit_stretch_ratio', 2.0),
        language=getattr(args, 'language', 'auto'),
        text_normalization=getattr(args, 'text_normalization', True),
        duration_factor=getattr(args, 'duration_factor', 1.0),
        use_gpt_latent=getattr(args, 'use_gpt_latent', False),
    )
    runtime = TTSRuntime(memory_limit_gb=memory_limit, quantize=args.quantize)
    if getattr(args, "stream", False) is True:
        if version != "2.5":
            raise SystemExit("--stream currently requires an IndexTTS 2.5 model")
        import numpy as np
        import soundfile as sf

        from mlx_indextts.generate import crossfade_segments
        import mlx.core as mx

        started = time.perf_counter()
        chunks = []
        for chunk in runtime.stream(
            text=args.text,
            ref_audio=args.ref_audio,
            profile=args.profile,
            model=args.model,
            options=options,
        ):
            chunks.append(chunk)
            print(
                f"stream segment {chunk.segment_index + 1}/{chunk.segment_count} "
                f"language={chunk.resolved_language} completed={chunk.completed}"
            )
        if not chunks:
            raise RuntimeError("No streamed audio segments were generated")
        if len(chunks) > 1 and options.interval_silence <= 0 and options.segment_overlap > 0:
            audio = np.asarray(
                crossfade_segments(
                    [mx.array(chunk.audio) for chunk in chunks],
                    chunks[0].sample_rate,
                    options.segment_overlap,
                )
            )
        else:
            audio = np.concatenate([chunk.audio for chunk in chunks])
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, audio, chunks[0].sample_rate)
        elapsed = time.perf_counter() - started
        info = getattr(getattr(runtime, "_model", None), "last_generation_info", {})
        duration = len(audio) / chunks[0].sample_rate
        result = {
            "output_path": str(output),
            "model": args.model,
            "version": version,
            "model_revision": info.get("model_revision", ""),
            "language": chunks[0].resolved_language,
            "segments": len(chunks),
            "duration_s": round(duration, 3),
            "elapsed_s": round(elapsed, 3),
            "rtf": round(elapsed / duration, 4) if duration else None,
        }
    else:
        result = runtime.generate(
            text=args.text,
            ref_audio=args.ref_audio,
            output_path=args.output,
            profile=args.profile,
            model=args.model,
            options=options,
        )
    if result.get("emotion_source") == "qwen-mlx":
        print(f"Qwen emotion: {result.get('dominant_emotion')} {result.get('emotion_json')}")

    print(json.dumps(result, ensure_ascii=False))

    # Play audio if requested
    if args.play:
        if sys.platform == "darwin":
            subprocess.run(["afplay", args.output])
        elif sys.platform == "linux":
            subprocess.run(["aplay", args.output])
        else:
            print("Auto-play not supported on this platform")


def speaker_command(args):
    """Handle the speaker command - save pre-computed speaker conditioning."""
    if not args.model:
        args.model = resolve_default_model(args.profile)
        print(f"Using default {args.profile} 8bit model: {args.model}")

    version = detect_mlx_version(Path(args.model))
    print(f"Using IndexTTS {version}")
    print(f"Loading model from {args.model}...")

    if version == "2.5":
        from mlx_indextts.generate_v25 import IndexTTSv25

        tts = IndexTTSv25(model_dir=args.model, memory_limit_gb=args.memory_limit)
    elif version == "2.0":
        from mlx_indextts.generate_v2 import IndexTTSv2
        tts = IndexTTSv2(model_dir=args.model, memory_limit_gb=args.memory_limit)
    else:
        from mlx_indextts.generate import IndexTTS
        tts = IndexTTS.load_model(args.model, memory_limit_gb=args.memory_limit)

    ref_audio = args.ref_audio
    if getattr(args, "denoise_ref", True):
        from mlx_indextts.audio_denoise import maybe_denoise_reference

        ref_audio = maybe_denoise_reference(ref_audio, enabled=True, suffix="speaker") or ref_audio
        print(f"Using denoised speaker reference: {ref_audio}")
    print(f"Computing speaker conditioning from {ref_audio}...")
    start = time.perf_counter()
    tts.save_speaker(ref_audio, args.output)
    elapsed = time.perf_counter() - start
    print(f"Speaker saved to {args.output} ({elapsed:.2f}s)")


def _read_batch_items(input_path: str, text_column: str = "text") -> list[dict[str, str]]:
    """Read batch items from txt or csv."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(input_path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if text_column not in fieldnames and "文本" not in fieldnames:
                raise ValueError(f"CSV missing text column '{text_column}'. Columns: {reader.fieldnames}")
            items = []
            for idx, row in enumerate(reader, start=1):
                text = _first_row_value(row, (text_column, "文本"))
                if text:
                    speaker = _first_row_value(row, ("speaker", "role", "说话人", "角色"))
                    item_id = _first_row_value(row, ("id", "name", "speaker", "role", "说话人", "角色"), f"{idx:04d}")
                    ref_audio = _first_row_value(row, ("ref_audio", "reference_audio", "音色参考音频"))
                    emotion_ref_audio = _first_row_value(
                        row,
                        (
                            "emotion_ref_audio",
                            "emotion_reference_audio",
                            "emo_ref_audio",
                            "emotion_audio",
                            "emo_audio",
                            "上传情感参考音频",
                        ),
                    )
                    emotion = _first_row_value(row, ("emotion", "情感", "情感描述文本"))
                    vector_emotion = _emotion_vector_from_row(row)
                    if vector_emotion:
                        emotion = vector_emotion
                    emo_alpha = _first_row_value(row, ("emo_alpha", "情感权重"), "0.6")
                    item = {
                        "id": item_id,
                        "speaker": speaker,
                        "text": text,
                        "ref_audio": ref_audio,
                        "emotion_ref_audio": emotion_ref_audio,
                        "emotion": emotion,
                        "emo_alpha": emo_alpha,
                    }
                    # Preserve duration controls and token caps for the
                    # long-lived batch runtime. Dropping these fields forced
                    # subtitle pipelines back to one cold CLI process per row.
                    optional_fields = {
                        "target_duration": ("target_duration", "target_duration_s", "duration"),
                        "fit_duration": ("fit_duration",),
                        "max_tokens": ("max_tokens", "max_mel_tokens"),
                        "language": ("language", "lang", "语言"),
                        "emotion_text": ("emotion_text", "emo_text"),
                    }
                    for output_key, aliases in optional_fields.items():
                        value = _first_row_value(row, aliases)
                        if value:
                            item[output_key] = value
                    items.append(item)
            return items
    items = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if text:
            items.append({"id": f"{idx:04d}", "text": text})
    return items


def _combine_wavs(paths: list[Path], output_path: Path, silence_ms: int = 120) -> None:
    import numpy as np
    import soundfile as sf

    audio_parts = []
    sample_rate = None
    for path in paths:
        data, sr = sf.read(path, always_2d=False)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample-rate mismatch: {path} has {sr}, expected {sample_rate}")
        audio_parts.append(data)
        if silence_ms > 0:
            audio_parts.append(np.zeros(int(sr * silence_ms / 1000.0), dtype=data.dtype))
    if not audio_parts:
        raise RuntimeError("No wavs to combine")
    combined = np.concatenate(audio_parts)
    sf.write(output_path, combined, sample_rate)


def batch_command(args):
    """Generate multiple utterances while keeping the model loaded."""
    items = _read_batch_items(args.input, args.text_column)
    if not items:
        raise SystemExit("No non-empty text rows found.")
    if args.ref_audio is None and any(not item.get("ref_audio") for item in items):
        raise SystemExit("Batch requires --ref-audio unless every CSV row has ref_audio/reference_audio.")
    _validate_cli_emotion_sources(
        auto_emotion=args.auto_emotion,
        emotion_text=args.emotion_text,
        emotion=args.emotion,
        emotion_ref_audio=args.emotion_ref_audio,
        has_row_emotion_refs=any(bool(item.get("emotion_ref_audio")) for item in items),
    )

    if (args.auto_emotion or args.emotion_text) and any(item.get("emotion") for item in items):
        raise SystemExit(
            "CSV row emotion cannot be combined with --auto-emotion/--emotion-text."
        )

    all_text = "\n".join(item["text"] for item in items)
    if not args.model:
        args.model = resolve_default_model(args.profile, all_text)
        print(f"Using default {args.profile} 8bit model: {args.model}")

    model_dir = Path(args.model)
    version = detect_mlx_version(model_dir)
    memory_limit = args.memory_limit
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using IndexTTS {version}")
    print(f"Batch items: {len(items)}")
    print(f"Output dir: {output_dir}")

    from mlx_indextts.runtime import GenerateOptions, TTSRuntime

    options = GenerateOptions(
        max_tokens=args.max_tokens,
        max_text_tokens=args.max_text_tokens,
        interval_silence=args.interval_silence,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        diffusion_steps=args.diffusion_steps,
        cfg_rate=args.cfg_rate,
        emotion="auto-qwen" if args.auto_emotion else args.emotion,
        emo_alpha=args.emo_alpha,
        emotion_ref_audio=args.emotion_ref_audio,
        auto_emotion=args.auto_emotion,
        use_emo_text=bool(args.emotion_text),
        emo_text=args.emotion_text,
        use_random=args.use_random,
        qwen_emotion_model=args.qwen_emotion_model,
        qwen_unload_after=args.qwen_unload_after,
        smooth_emotion=not args.no_emotion_smoothing,
        denoise_ref_audio=args.denoise_ref,
        denoise_emotion_ref_audio=args.denoise_emotion_ref,
        seed=args.seed,
        verbose=args.verbose,
        segment_overlap=args.segment_overlap,
        speed=args.speed,
        target_duration=args.target_duration,
        fit_duration=args.fit_duration,
        dynamic_max_tokens=args.dynamic_max_tokens,
        tokens_per_char=args.tokens_per_char,
        min_max_tokens=args.min_max_tokens,
        language=args.language,
        text_normalization=args.text_normalization,
        duration_factor=args.duration_factor,
        use_gpt_latent=args.use_gpt_latent,
    )
    runtime = TTSRuntime(memory_limit_gb=memory_limit, quantize=args.quantize)
    result = runtime.batch(
        rows=items,
        ref_audio=args.ref_audio,
        output_dir=str(output_dir),
        profile=args.profile,
        model=args.model,
        options=options,
        combine=args.combine,
        combine_silence_ms=args.combine_silence_ms,
    )
    print(f"Batch complete: {result['output_dir']}")
    if result.get("combined_path"):
        print(f"Combined wav saved to {result['combined_path']}")


def plan_command(args):
    """Convert a dialogue/novel script into an editable batch CSV."""
    from mlx_indextts.novel_planner import NovelScriptPlanner

    emotion_refs_by_emotion = {}
    emotion_ref_resolver = None
    speaker_refs = _parse_key_value_args(args.speaker_ref, option_name="--speaker-ref")
    if args.emotion_library:
        from mlx_indextts.video_library import load_emotion_library_catalog

        catalog = load_emotion_library_catalog(args.emotion_library)
        if hasattr(catalog, "best_ref_for"):
            speaker_profiles = _parse_speaker_profiles(args.speaker_profile)
            default_gender = args.default_gender.strip() if args.default_gender else ""
            default_age_band = args.default_age_band.strip() if args.default_age_band else ""
            scene = args.emotion_scene.strip() if args.emotion_scene else ""
            if args.auto_duo_refs:
                speaker_refs = catalog.recommended_duo_refs(scene=scene or None)
            else:
                speaker_refs = _parse_key_value_args(args.speaker_ref, option_name="--speaker-ref")

            def emotion_ref_resolver(row):
                profile = speaker_profiles.get(row.speaker, {})
                return catalog.best_ref_for(
                    scene=scene or None,
                    emotion=row.emotion_label,
                    gender=profile.get("gender") or default_gender or None,
                    age_band=profile.get("age_band") or default_age_band or None,
                )

        else:
            emotion_refs_by_emotion = catalog.emotion_refs()

    content = Path(args.input).read_text(encoding="utf-8")
    planner = NovelScriptPlanner(max_chars_per_line=args.max_chars_per_line)
    output = planner.write_batch_csv(
        content,
        args.output,
        speaker_refs=speaker_refs,
        emotion_refs_by_emotion=emotion_refs_by_emotion,
        emotion_ref_resolver=emotion_ref_resolver,
    )
    print(f"Batch CSV saved to {output}")


def emotion2vec_command(args):
    """Build an emotion library with emotion2vec_plus_large."""
    from mlx_indextts.emotion2vec import Emotion2VecClassifier, build_emotion_catalog

    hub = {"huggingface": "hf", "modelscope": "ms"}.get(args.hub, args.hub)
    classifier = Emotion2VecClassifier(model_id=args.model, hub=hub, device=args.device)
    manifest = build_emotion_catalog(
        args.input,
        args.output_dir,
        classifier=classifier,
        copy_audio=not args.no_copy_audio,
        limit=args.limit,
    )
    output_root = Path(args.output_dir)
    print(f"Emotion library saved to {manifest}")
    print(f"Reference map saved to {output_root / 'emotion_refs_by_emotion.json'}")
    print(f"Summary saved to {output_root / 'summary.md'}")


def video_library_command(args):
    """Build a scene-emotion-gender-age library from video or audio."""
    from mlx_indextts.video_library import build_scene_emotion_library
    from mlx_indextts.emotion2vec import Emotion2VecClassifier

    hub = {"huggingface": "hf", "modelscope": "ms"}.get(args.emotion_hub, args.emotion_hub)
    classifier = Emotion2VecClassifier(
        model_id=args.emotion_model,
        hub=hub,
        device=args.emotion_device,
    )
    result = build_scene_emotion_library(
        args.source,
        args.output_dir,
        scene=args.scene,
        language=args.language,
        asr_model=args.asr_model,
        extract_vocals=not args.no_extract_vocals,
        limit=args.limit,
        clip_padding_s=args.clip_padding_s,
        emotion_classifier=classifier,
        age_gender_model=args.age_gender_model,
        age_gender_device=args.age_gender_device,
    )
    print(f"Library saved to {result['manifest_path']}")
    print(f"Emotion refs saved to {result['library_root'] / 'emotion_refs_by_emotion.json'}")
    print(f"Composite refs saved to {result['library_root'] / 'emotion_refs_by_scene_emotion_gender_age.json'}")


def denoise_command(args):
    """Run optional Demucs vocal extraction for a reference audio file."""
    from mlx_indextts.audio_denoise import denoise_audio_simple

    output = denoise_audio_simple(args.input, args.output)
    print(f"Denoised audio saved to {output}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="mlx-indextts",
        description="IndexTTS for Apple Silicon using MLX (supports v2.0 and v2.5)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert PyTorch model to MLX format (auto-detects v2.0/v2.5)",
    )
    convert_parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing PyTorch checkpoints",
    )
    convert_parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="2.5 tensor dtype (default: float16)",
    )
    convert_parser.add_argument(
        "--source-revision",
        default="d0aa86e75bb6f3437f3831e95056fa72842d89ef",
        help="Pinned IndexTTS 2.5 Hugging Face source revision",
    )
    convert_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume a compatible partial 2.5 conversion (default: true)",
    )
    convert_parser.add_argument(
        "--force",
        action="store_true",
        help="Archive an existing output/staging directory before conversion",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for MLX weights",
    )
    convert_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: model_dir/config.yaml)",
    )
    convert_parser.add_argument(
        "--quantize",
        "-q",
        type=str,
        default="fp32",
        help="Quantization bits: 4, 5, 6, 8, or fp32 (GPT only, default: fp32)",
    )
    convert_parser.set_defaults(func=convert_command)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate speech from text (auto-detects v2.0/v2.5)",
    )
    generate_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help=(
            "Path to converted MLX model directory. If omitted, uses local 8bit defaults: "
            f"{DEFAULT_V25_MODEL}, {DEFAULT_STANDARD_MODEL}, or {DEFAULT_VIETNAMESE_MODEL}."
        ),
    )
    generate_parser.add_argument(
        "--profile",
        choices=["auto", "v25", "2.5", "standard", "vietnamese", "vi"],
        default="auto",
        help="Default model profile when --model is omitted (default: auto from text)",
    )
    generate_parser.add_argument(
        "--ref-audio",
        "-r",
        type=str,
        required=True,
        help="Reference audio file for voice cloning",
    )
    generate_parser.add_argument(
        "--language",
        choices=["auto", "zh", "en", "ja", "es", "ar"],
        default="auto",
        help="2.5 text language; explicit en/es is recommended for Latin-script text",
    )
    generate_parser.add_argument(
        "--text-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the 2.5 language-specific text normalizer (default: true)",
    )
    generate_parser.add_argument(
        "--duration-factor",
        type=float,
        default=1.0,
        help="2.5 S2Mel duration multiplier (default: 1.0)",
    )
    generate_parser.add_argument(
        "--use-gpt-latent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the optional GPT latent branch before S2Mel (default: false, matching upstream)",
    )
    generate_parser.add_argument(
        "--text",
        "-t",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav)",
    )
    generate_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum semantic tokens per segment (default: 256 for v2.5; 1500 for v2.0)",
    )
    generate_parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=120,
        help="Maximum text tokens per segment for long text splitting (default: 120)",
    )
    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.8 for v2.0/v2.5)",
    )
    generate_parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-k sampling (default: 30)",
    )
    generate_parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling (default: 0.8)",
    )
    generate_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=10.0,
        help="Repetition penalty to avoid repeated tokens (default: 10.0)",
    )
    generate_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    generate_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    generate_parser.add_argument(
        "--play",
        "-p",
        action="store_true",
        help="Play audio after generation (macOS/Linux)",
    )
    generate_parser.add_argument(
        "--stream",
        action="store_true",
        help="Yield completed 2.5 text segments and assemble them into --output",
    )
    generate_parser.add_argument(
        "--memory-limit",
        type=float,
        default=None,
        help="GPU memory limit in GB (default: 24 for v2.0/v2.5)",
    )
    generate_parser.add_argument(
        "--quantize",
        "-q",
        type=str,
        default="fp32",
        help="Runtime quantization (GPT only): 4, 5, 6, 8, or fp32 (default: fp32)",
    )
    generate_parser.add_argument(
        "--segment-overlap",
        type=int,
        default=50,
        help="Overlap duration (ms) for crossfade between segments (default: 50, 0 to disable)",
    )
    # v2.0 specific options
    generate_parser.add_argument(
        "--interval-silence",
        type=int,
        default=200,
        help="Silence duration (ms) between segments (default: 200)",
    )
    generate_parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=16,
        help="[v2.0/v2.5] S2Mel diffusion/CFM sampling steps (default: 16)",
    )
    generate_parser.add_argument(
        "--cfg-rate",
        type=float,
        default=0.7,
        help="[v2.0/v2.5] Classifier-Free Guidance rate (default: 0.7)",
    )
    generate_parser.add_argument(
        "--emotion",
        type=str,
        default=None,
        help="[v2.0/v2.5] Emotion: happy/sad/angry/afraid/disgusted/melancholic/surprised/calm, weighted mix, or auto-qwen",
    )
    generate_parser.add_argument(
        "--emotion-ref-audio",
        type=str,
        default=None,
        help="[v2.0/v2.5] Separate emotion reference audio or speaker .npz",
    )
    generate_parser.add_argument(
        "--denoise-ref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run optional Demucs vocal extraction on the speaker reference before generation (default: true)",
    )
    generate_parser.add_argument(
        "--denoise-emotion-ref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run optional Demucs vocal extraction on the emotion reference before generation (default: true)",
    )
    generate_parser.add_argument(
        "--auto-emotion",
        action="store_true",
        help="[v2.0/v2.5] Use the MLX-native Qwen emotion model on the synthesis text",
    )
    generate_parser.add_argument(
        "--emotion-text",
        type=str,
        default=None,
        help="[v2.0/v2.5] Derive emotion from this separate text with Qwen",
    )
    generate_parser.add_argument(
        "--use-random",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomly select emotion prototypes (reduces voice-cloning fidelity)",
    )
    generate_parser.add_argument(
        "--qwen-emotion-model",
        type=str,
        default=None,
        help="Path to converted MLX Qwen emotion model (default: models/qwen0.6bemo4-merge-mlx-8bit)",
    )
    generate_parser.add_argument(
        "--qwen-unload-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unload Qwen emotion model after text analysis to avoid model overlap (default: true)",
    )
    generate_parser.add_argument(
        "--emo-alpha",
        type=float,
        default=0.6,
        help="[v2.0/v2.5] Emotion intensity 0.0-1.0 (default: 0.6)",
    )
    generate_parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed 0.5-2.0 (default: 1.0). Time-stretch without pitch change.",
    )
    generate_parser.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Target output duration in seconds. Used as a mel-token budget; not exact unless --fit-duration is also set.",
    )
    generate_parser.add_argument(
        "--fit-duration",
        action="store_true",
        help="After generation, time-stretch without pitch shift to match --target-duration. Disabled unless explicitly requested.",
    )
    generate_parser.add_argument(
        "--max-fit-stretch-ratio",
        type=float,
        default=2.0,
        help="Refuse destructive duration fitting beyond this speed ratio (default: 2.0).",
    )
    generate_parser.set_defaults(func=generate_command)

    speaker_parser = subparsers.add_parser(
        "speaker",
        help="Save pre-computed speaker conditioning for faster inference",
    )
    speaker_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help=(
            "Path to converted MLX model directory. If omitted, uses local 8bit defaults: "
            f"{DEFAULT_V25_MODEL}, {DEFAULT_STANDARD_MODEL}, or {DEFAULT_VIETNAMESE_MODEL}."
        ),
    )
    speaker_parser.add_argument(
        "--profile",
        choices=["v25", "2.5", "standard", "vietnamese", "vi"],
        default="v25",
        help="Default model profile when --model is omitted (default: v25)",
    )
    speaker_parser.add_argument(
        "--ref-audio",
        "-r",
        type=str,
        required=True,
        help="Reference audio file",
    )
    speaker_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output .npz file path for speaker conditioning",
    )
    speaker_parser.add_argument(
        "--memory-limit",
        type=float,
        default=None,
        help="GPU memory limit in GB (default: auto, 0 for no limit)",
    )
    speaker_parser.add_argument(
        "--denoise-ref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run optional Demucs vocal extraction on the speaker reference before saving conditioning (default: true)",
    )
    speaker_parser.set_defaults(func=speaker_command)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Generate multiple lines from a txt/csv file while keeping the model loaded",
    )
    batch_parser.add_argument("--input", "-i", required=True, help="Input .txt or .csv file")
    batch_parser.add_argument("--text-column", default="text", help="CSV text column (default: text)")
    batch_parser.add_argument("--model", "-m", default=None, help="Model directory; omitted uses local 8bit defaults")
    batch_parser.add_argument(
        "--profile",
        choices=["auto", "v25", "2.5", "standard", "vietnamese", "vi"],
        default="auto",
        help="Default model profile when --model is omitted (default: auto from batch text)",
    )
    batch_parser.add_argument(
        "--ref-audio",
        "-r",
        default=None,
        help="Reference audio or speaker .npz. Optional for CSV rows with ref_audio/reference_audio.",
    )
    batch_parser.add_argument("--output-dir", "-o", default="outputs/batch", help="Output directory")
    batch_parser.add_argument("--max-tokens", type=int, default=900)
    batch_parser.add_argument("--max-text-tokens", type=int, default=80)
    batch_parser.add_argument(
        "--language",
        choices=["auto", "zh", "en", "ja", "es", "ar"],
        default="auto",
    )
    batch_parser.add_argument(
        "--text-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    batch_parser.add_argument("--duration-factor", type=float, default=1.0)
    batch_parser.add_argument(
        "--use-gpt-latent",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    batch_parser.add_argument(
        "--dynamic-max-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate a smaller per-row mel-token cap from text length unless the CSV row sets max_tokens.",
    )
    batch_parser.add_argument("--tokens-per-char", type=float, default=14.0)
    batch_parser.add_argument("--min-max-tokens", type=int, default=320)
    batch_parser.add_argument("--temperature", type=float, default=0.8)
    batch_parser.add_argument("--top-k", type=int, default=30)
    batch_parser.add_argument("--top-p", type=float, default=0.8)
    batch_parser.add_argument("--repetition-penalty", type=float, default=10.0)
    batch_parser.add_argument("--seed", "-s", type=int, default=None)
    batch_parser.add_argument("--verbose", "-v", action="store_true")
    batch_parser.add_argument("--memory-limit", type=float, default=None)
    batch_parser.add_argument("--quantize", "-q", type=str, default="fp32")
    batch_parser.add_argument("--segment-overlap", type=int, default=50)
    batch_parser.add_argument("--interval-silence", type=int, default=0)
    batch_parser.add_argument("--diffusion-steps", type=int, default=16)
    batch_parser.add_argument("--cfg-rate", type=float, default=0.7)
    batch_parser.add_argument("--emotion", type=str, default=None)
    batch_parser.add_argument(
        "--emotion-ref-audio",
        type=str,
        default=None,
        help="Default separate emotion reference audio or speaker .npz; CSV can override with emotion_ref_audio/emo_audio.",
    )
    batch_parser.add_argument(
        "--denoise-ref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Denoise speaker reference wavs before generation (default: true)",
    )
    batch_parser.add_argument(
        "--denoise-emotion-ref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Denoise emotion reference wavs before generation (default: true)",
    )
    batch_parser.add_argument("--auto-emotion", action="store_true", help="Analyze each row with MLX-native Qwen emotion")
    batch_parser.add_argument(
        "--emotion-text",
        type=str,
        default=None,
        help="Use one separate Qwen emotion description for all rows; CSV emotion_text/emo_text overrides it",
    )
    batch_parser.add_argument(
        "--use-random",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomly select emotion prototypes",
    )
    batch_parser.add_argument(
        "--qwen-emotion-model",
        type=str,
        default=None,
        help="Path to converted MLX Qwen emotion model (default: models/qwen0.6bemo4-merge-mlx-8bit)",
    )
    batch_parser.add_argument(
        "--qwen-unload-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unload Qwen emotion model after batch emotion analysis (default: true)",
    )
    batch_parser.add_argument(
        "--no-emotion-smoothing",
        action="store_true",
        help="Disable adjacent-line emotion smoothing for --auto-emotion",
    )
    batch_parser.add_argument("--emo-alpha", type=float, default=0.6)
    batch_parser.add_argument("--speed", type=float, default=1.0)
    batch_parser.add_argument("--target-duration", type=float, default=None)
    batch_parser.add_argument("--fit-duration", action="store_true")
    batch_parser.add_argument("--combine", action="store_true", help="Also write combined.wav")
    batch_parser.add_argument("--combine-silence-ms", type=int, default=120)
    batch_parser.set_defaults(func=batch_command)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Parse a dialogue/novel script into a batch CSV with smoothed emotion hints",
    )
    plan_parser.add_argument("--input", "-i", required=True, help="Input script text file")
    plan_parser.add_argument("--output", "-o", default="outputs/batch_plan.csv", help="Output CSV path")
    plan_parser.add_argument("--max-chars-per-line", type=int, default=90)
    plan_parser.add_argument(
        "--emotion-library",
        type=str,
        default=None,
        help="Optional emotion library catalog.csv; scene libraries auto-fill emotion_ref_audio by scene/emotion/gender/age.",
    )
    plan_parser.add_argument(
        "--emotion-scene",
        default="crosstalk",
        help="Scene key for scene-emotion-gender-age libraries (default: crosstalk)",
    )
    plan_parser.add_argument(
        "--speaker-profile",
        action="append",
        default=[],
        help="Speaker demographic hint for scene library lookup, e.g. '逗哏=male:adult'. Can be repeated.",
    )
    plan_parser.add_argument(
        "--default-gender",
        default="",
        help="Fallback gender hint for scene library lookup when speaker has no --speaker-profile.",
    )
    plan_parser.add_argument(
        "--default-age-band",
        default="",
        help="Fallback age band hint for scene library lookup, e.g. adult, young_adult, middle_aged.",
    )
    plan_parser.add_argument(
        "--speaker-ref",
        action="append",
        default=[],
        help="Speaker voice reference for the generated batch CSV, e.g. '逗哏=/path/a.wav'. Can be repeated.",
    )
    plan_parser.add_argument(
        "--auto-duo-refs",
        action="store_true",
        help="Auto-pick two distinct speaker refs from the scene library for the first two roles.",
    )
    plan_parser.set_defaults(func=plan_command)

    video_library_parser = subparsers.add_parser(
        "video-library",
        help="Download a video/audio source, transcribe it, and build a scene-emotion-gender-age library",
    )
    video_library_parser.add_argument("--source", required=True, help="YouTube URL or local audio/video file")
    video_library_parser.add_argument("--output-dir", "-o", default="outputs/video_library", help="Output library directory")
    video_library_parser.add_argument("--scene", default="crosstalk", help="Scene label used in the library keys")
    video_library_parser.add_argument(
        "--language",
        default="auto",
        help="ASR language code, or auto for route selection",
    )
    video_library_parser.add_argument(
        "--asr-model",
        default=None,
        help="Override ASR model path/repo for Whisper or Qwen3-ASR",
    )
    video_library_parser.add_argument(
        "--emotion-model",
        default="emotion2vec/emotion2vec_plus_large",
        help="Emotion2vec model id or local path (default: emotion2vec_plus_large)",
    )
    video_library_parser.add_argument(
        "--emotion-hub",
        choices=["huggingface", "modelscope", "hf", "ms"],
        default="huggingface",
        help="Download source for the emotion2vec model",
    )
    video_library_parser.add_argument(
        "--emotion-device",
        default="cpu",
        help="Inference device for emotion2vec classification (default: cpu)",
    )
    video_library_parser.add_argument(
        "--age-gender-model",
        default=None,
        help="Override age/gender model path or repo (optional; falls back to heuristic labels when omitted)",
    )
    video_library_parser.add_argument(
        "--age-gender-device",
        default=None,
        help="Override age/gender inference device (default: cpu)",
    )
    video_library_parser.add_argument("--clip-padding-s", type=float, default=0.12, help="Padding on both sides of each sentence clip")
    video_library_parser.add_argument("--limit", type=int, default=None, help="Optional max number of sentence clips to process")
    video_library_parser.add_argument("--no-extract-vocals", action="store_true", help="Skip Demucs vocal extraction and keep the downloaded track")
    video_library_parser.set_defaults(func=video_library_command)

    emotion2vec_parser = subparsers.add_parser(
        "emotion2vec",
        help="Build an emotion library with emotion2vec_plus_large",
    )
    emotion2vec_parser.add_argument("--input", "-i", required=True, help="Audio directory, wav.scp, CSV, or single audio file")
    emotion2vec_parser.add_argument("--output-dir", "-o", default="outputs/emotion_library", help="Output library directory")
    emotion2vec_parser.add_argument("--model", default="emotion2vec/emotion2vec_plus_large", help="Emotion2vec model id or local path")
    emotion2vec_parser.add_argument(
        "--hub",
        choices=["hf", "huggingface", "ms", "modelscope"],
        default="hf",
        help="Download source for the emotion2vec model",
    )
    emotion2vec_parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
        help="Inference device for emotion2vec classification (default: cpu)",
    )
    emotion2vec_parser.add_argument("--limit", type=int, default=None, help="Optional max number of clips to process")
    emotion2vec_parser.add_argument(
        "--no-copy-audio",
        action="store_true",
        help="Do not copy source audio into the library output directory",
    )
    emotion2vec_parser.set_defaults(func=emotion2vec_command)

    denoise_parser = subparsers.add_parser(
        "denoise",
        help="Extract vocals from a reference audio file with Demucs",
    )
    denoise_parser.add_argument("--input", "-i", required=True, help="Input audio path")
    denoise_parser.add_argument("--output", "-o", default=None, help="Output wav path")
    denoise_parser.set_defaults(func=denoise_command)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
