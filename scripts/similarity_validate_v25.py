#!/usr/bin/env python3
"""Measure CampPlus speaker and GPT-emotion cosine proxies for 2.5 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch.nn.functional as F

from mlx_indextts.generate_v25 import IndexTTSv25


BASIC_CASES = {
    "basic_zh",
    "crosslingual_en",
    "crosslingual_ja",
    "crosslingual_es",
    "crosslingual_ar",
}


def _torch_cosine(left, right) -> float:
    return float(F.cosine_similarity(left.float(), right.float(), dim=-1).mean().item())


def _mlx_cosine(left: mx.array, right: mx.array) -> float:
    numerator = mx.sum(left * right, axis=-1)
    denominator = mx.sqrt(mx.sum(left * left, axis=-1)) * mx.sqrt(
        mx.sum(right * right, axis=-1)
    )
    value = numerator / mx.maximum(denominator, mx.array(1e-8))
    mx.eval(value)
    return float(np.asarray(value).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/mlx-IndexTTS-2.5-8bit")
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--functional-report", required=True)
    parser.add_argument("--emotion-reference")
    parser.add_argument("--speaker-threshold", type=float, default=0.4)
    parser.add_argument("--output")
    args = parser.parse_args()

    functional_path = Path(args.functional_report)
    functional = json.loads(functional_path.read_text(encoding="utf-8"))
    tts = IndexTTSv25(args.model)
    reference = tts._process_reference_audio(args.reference_audio)
    reference_style = reference["style"]
    cases = []
    for case in functional["cases"]:
        if case.get("name") not in BASIC_CASES:
            continue
        try:
            generated = tts._process_reference_audio(case["path"])
            cosine = _torch_cosine(reference_style, generated["style"])
            cases.append(
                {
                    "name": case["name"],
                    "language": case["language"],
                    "path": case["path"],
                    "speaker_cosine": round(cosine, 6),
                    "status": "pass" if cosine >= args.speaker_threshold else "fail",
                }
            )
        except Exception as exc:
            cases.append(
                {
                    "name": case["name"],
                    "language": case["language"],
                    "path": case["path"],
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    emotion = None
    if args.emotion_reference:
        emotion_case = next(
            (
                case
                for case in functional["cases"]
                if case.get("name") == "separate_emotion_reference"
            ),
            None,
        )
        if emotion_case is not None:
            try:
                emotion_reference = tts._process_reference_audio(args.emotion_reference)
                generated = tts._process_reference_audio(emotion_case["path"])
                emotion_features = tts._mlx_reference_features(emotion_reference)
                generated_features = tts._mlx_reference_features(generated)
                speaker_features = tts._mlx_reference_features(reference)
                emotion = {
                    "status": "pass",
                    "path": emotion_case["path"],
                    "emotion_reference": args.emotion_reference,
                    "emotion_cosine_to_reference": round(
                        _mlx_cosine(
                            generated_features["emotion_vec"],
                            emotion_features["emotion_vec"],
                        ),
                        6,
                    ),
                    "emotion_cosine_to_speaker_reference": round(
                        _mlx_cosine(
                            generated_features["emotion_vec"],
                            speaker_features["emotion_vec"],
                        ),
                        6,
                    ),
                }
            except Exception as exc:
                emotion = {
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    report = {
        "status": "pass"
        if len(cases) == len(BASIC_CASES)
        and all(case["status"] == "pass" for case in cases)
        and (emotion is None or emotion["status"] == "pass")
        else "fail",
        "model": str(Path(args.model).resolve()),
        "reference_audio": str(Path(args.reference_audio).resolve()),
        "speaker_metric": "CampPlus cosine similarity",
        "speaker_threshold": args.speaker_threshold,
        "cases": cases,
        "emotion_proxy": emotion,
    }
    output_path = Path(args.output) if args.output else functional_path.with_name(
        "similarity_report.json"
    )
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {output_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
