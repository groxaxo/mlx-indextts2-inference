#!/usr/bin/env python3
"""Run the reproducible IndexTTS 2.5 functional acceptance matrix."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from mlx_indextts.runtime import GenerateOptions, TTSRuntime


BASIC_CASES = (
    ("basic_zh", "zh", "你好，这是 IndexTTS 二点五的中文语音验证。"),
    ("crosslingual_en", "en", "Hello, this is a cross-lingual voice cloning test."),
    ("crosslingual_ja", "ja", "こんにちは、これは日本語の音声合成テストです。"),
    ("crosslingual_es", "es", "Hola, esta es una prueba de síntesis de voz en español."),
    ("crosslingual_ar", "ar", "مرحبًا، هذا اختبار لتوليد الكلام باللغة العربية."),
)

ANNOTATION_CASES = (
    ("annotation_zh", "zh", "他在银<行|XING2>里行走。"),
    ("annotation_en", "en", "He had a <minute|M IH1 . N AH0 T> to think."),
    ("annotation_ja", "ja", "<今日|きょう>は良い天気です。"),
)


def _wav_evidence(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(audio)
    finite = bool(np.isfinite(values).all())
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
    passed = (
        info.frames > 0
        and sample_rate == 22050
        and finite
        and peak > 1e-5
        and info.duration > 0
    )
    return {
        "status": "pass" if passed else "fail",
        "path": str(path),
        "sample_rate": sample_rate,
        "frames": info.frames,
        "duration_s": round(info.duration, 4),
        "channels": info.channels,
        "finite": finite,
        "peak": round(peak, 6),
        "rms": round(rms, 6),
    }


def _base_options(args: argparse.Namespace, **updates: Any) -> GenerateOptions:
    values: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "max_text_tokens": args.max_text_tokens,
        "interval_silence": 0,
        "diffusion_steps": args.diffusion_steps,
        "cfg_rate": 0.7,
        "denoise_ref_audio": False,
        "denoise_emotion_ref_audio": False,
        "seed": args.seed,
        "verbose": args.verbose,
        "language": "auto",
        "text_normalization": True,
    }
    values.update(updates)
    return GenerateOptions(**values)


def _run_generate_case(
    runtime: TTSRuntime,
    args: argparse.Namespace,
    output_root: Path,
    *,
    name: str,
    language: str,
    text: str,
    **option_updates: Any,
) -> dict[str, Any]:
    output_path = output_root / f"{name}.wav"
    started = time.perf_counter()
    try:
        options = _base_options(args, language=language, **option_updates)
        result = runtime.generate(
            text=text,
            ref_audio=args.ref_audio,
            output_path=str(output_path),
            profile="v25",
            model=args.model,
            options=options,
        )
        evidence = _wav_evidence(output_path)
        evidence.update(
            {
                "name": name,
                "language": language,
                "resolved_language": result.get("language"),
                "language_ambiguous": result.get("language_ambiguous"),
                "text": text,
                "elapsed_s": result.get("elapsed_s"),
                "rtf": result.get("rtf"),
                "emotion_source": result.get("emotion_source"),
                "dominant_emotion": result.get("dominant_emotion"),
                "model_revision": result.get("model_revision"),
            }
        )
        if evidence["resolved_language"] != language:
            evidence["status"] = "fail"
            evidence["error"] = "resolved language did not match explicit language"
        return evidence
    except Exception as exc:  # keep the report useful after one failed case
        return {
            "name": name,
            "language": language,
            "text": text,
            "status": "fail",
            "elapsed_s": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_stream_case(
    runtime: TTSRuntime,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    text = (
        "这是第一段长文本，用来验证完成分段流式输出。"
        "这是第二段内容，模型应当保持顺序并继续使用同一个说话人缓存。"
        "这是最后一段，只有最后一个事件可以标记为完成。"
    )
    output_path = output_root / "long_text_stream.wav"
    started = time.perf_counter()
    try:
        chunks = list(
            runtime.stream(
                text=text,
                ref_audio=args.ref_audio,
                profile="v25",
                model=args.model,
                options=_base_options(
                    args,
                    language="zh",
                    max_text_tokens=min(20, args.max_text_tokens),
                    interval_silence=50,
                ),
            )
        )
        if not chunks:
            raise RuntimeError("stream produced no chunks")
        sf.write(
            output_path,
            np.concatenate([chunk.audio for chunk in chunks]),
            chunks[0].sample_rate,
        )
        evidence = _wav_evidence(output_path)
        completion = [chunk.completed for chunk in chunks]
        evidence.update(
            {
                "name": "long_text_stream",
                "language": "zh",
                "text": text,
                "segments": len(chunks),
                "segment_indexes": [chunk.segment_index for chunk in chunks],
                "completion": completion,
                "elapsed_s": round(time.perf_counter() - started, 4),
            }
        )
        if len(chunks) < 2 or completion != [False] * (len(chunks) - 1) + [True]:
            evidence["status"] = "fail"
            evidence["error"] = "stream completion contract was not satisfied"
        return evidence
    except Exception as exc:
        return {
            "name": "long_text_stream",
            "status": "fail",
            "elapsed_s": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_batch_case(
    runtime: TTSRuntime,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    batch_root = output_root / "batch"
    started = time.perf_counter()
    rows = [
        {
            "id": "zh",
            "text": "批量中文验证。",
            "language": "zh",
            "ref_audio": args.ref_audio,
        },
        {
            "id": "en",
            "text": "Batch generation in English.",
            "language": "en",
            "ref_audio": args.ref_audio,
        },
    ]
    try:
        result = runtime.batch(
            rows=rows,
            ref_audio=args.ref_audio,
            output_dir=str(batch_root),
            profile="v25",
            model=args.model,
            options=_base_options(args),
            combine=True,
            combine_silence_ms=80,
        )
        item_paths = sorted(batch_root.glob("[0-9][0-9][0-9][0-9]_*.wav"))
        evidence = {
            "name": "batch_multilingual_combined",
            "status": "pass",
            "items": result.get("items"),
            "manifest_path": result.get("manifest_path"),
            "combined": _wav_evidence(Path(result["combined_path"])),
            "outputs": [_wav_evidence(path) for path in item_paths],
            "elapsed_s": round(time.perf_counter() - started, 4),
        }
        if (
            evidence["items"] != 2
            or evidence["combined"]["status"] != "pass"
            or len(evidence["outputs"]) != 2
            or any(item["status"] != "pass" for item in evidence["outputs"])
        ):
            evidence["status"] = "fail"
        return evidence
    except Exception as exc:
        return {
            "name": "batch_multilingual_combined",
            "status": "fail",
            "elapsed_s": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/mlx-IndexTTS-2.5-8bit")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--emotion-ref-audio")
    parser.add_argument("--output-dir", default="outputs/validation/indextts25")
    parser.add_argument("--scope", choices=("basic", "full"), default="full")
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--max-text-tokens", type=int, default=80)
    parser.add_argument("--quantize", choices=("fp32", "8", "4"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = TTSRuntime(quantize=args.quantize)
    started = time.perf_counter()
    cases = [
        _run_generate_case(
            runtime,
            args,
            output_root,
            name=name,
            language=language,
            text=text,
        )
        for name, language, text in BASIC_CASES
    ]

    if args.scope == "full":
        cases.extend(
            _run_generate_case(
                runtime,
                args,
                output_root,
                name=name,
                language=language,
                text=text,
            )
            for name, language, text in ANNOTATION_CASES
        )
        cases.append(
            _run_generate_case(
                runtime,
                args,
                output_root,
                name="manual_emotion_random",
                language="zh",
                text="太好了，我们终于成功了！",
                emotion=[0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
                emo_alpha=0.8,
                use_random=True,
            )
        )
        if args.emotion_ref_audio:
            cases.append(
                _run_generate_case(
                    runtime,
                    args,
                    output_root,
                    name="separate_emotion_reference",
                    language="zh",
                    text="这是一段使用独立情感参考的语音。",
                    emotion_ref_audio=args.emotion_ref_audio,
                    emo_alpha=0.8,
                )
            )
        cases.append(_run_stream_case(runtime, args, output_root))
        cases.append(_run_batch_case(runtime, args, output_root))
        cases.append(
            _run_generate_case(
                runtime,
                args,
                output_root,
                name="qwen_separate_emotion_text",
                language="zh",
                text="快躲起来，他马上就要来了！",
                use_emo_text=True,
                emo_text="令人非常害怕和紧张的场景",
                emo_alpha=0.6,
            )
        )

    report = {
        "status": "pass" if all(case["status"] == "pass" for case in cases) else "fail",
        "model": str(Path(args.model).resolve()),
        "reference_audio": str(Path(args.ref_audio).resolve()),
        "scope": args.scope,
        "diffusion_steps": args.diffusion_steps,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "elapsed_s": round(time.perf_counter() - started, 4),
        "cases": cases,
    }
    report_path = output_root / "functional_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
