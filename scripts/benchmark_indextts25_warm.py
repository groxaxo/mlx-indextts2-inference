#!/usr/bin/env python3
"""Measure warm IndexTTS 2.5 generation without publishing prompt contents."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--warmups", type=positive_int, default=1)
    parser.add_argument("--runs", type=positive_int, default=1)
    parser.add_argument("--diffusion-steps", type=positive_int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-mel-tokens", type=positive_int, default=256)
    parser.add_argument("--memory-limit-gb", type=positive_float, default=8.0)
    parser.add_argument("--json", type=Path, dest="json_path")
    return parser


def summarize_runs(runs: list[dict[str, float]]) -> dict[str, float | int | None]:
    """Return aggregate timing metrics without model or prompt dependencies."""
    if not runs:
        return {
            "runs": 0,
            "mean_generation_s": None,
            "median_generation_s": None,
            "mean_audio_duration_s": None,
            "aggregate_rtf": None,
        }

    generation_times = [run["generation_s"] for run in runs]
    audio_durations = [run["audio_duration_s"] for run in runs]
    total_audio = sum(audio_durations)
    return {
        "runs": len(runs),
        "mean_generation_s": statistics.fmean(generation_times),
        "median_generation_s": statistics.median(generation_times),
        "mean_audio_duration_s": statistics.fmean(audio_durations),
        "aggregate_rtf": sum(generation_times) / total_audio if total_audio > 0 else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Keep heavyweight imports out of parser/summary unit tests.
    import mlx.core as mx

    from mlx_indextts.generate_v25 import IndexTTSv25

    tts = IndexTTSv25(model_dir=str(args.model), memory_limit_gb=args.memory_limit_gb)
    mx.synchronize()
    generate_options: dict[str, Any] = {
        "language": args.language,
        "max_mel_tokens": args.max_mel_tokens,
        "diffusion_steps": args.diffusion_steps,
        "seed": args.seed,
        "verbose": False,
    }

    for _ in range(args.warmups):
        tts.generate(args.text, str(args.reference), **generate_options)
        mx.synchronize()

    run_metrics: list[dict[str, float]] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        audio = tts.generate(args.text, str(args.reference), **generate_options)
        mx.synchronize()
        generation_s = time.perf_counter() - start
        audio_duration_s = len(audio) / tts.sample_rate
        run_metrics.append(
            {
                "generation_s": generation_s,
                "audio_duration_s": audio_duration_s,
                "rtf": generation_s / audio_duration_s if audio_duration_s > 0 else 0.0,
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "device": str(mx.default_device()),
            "model_label": args.model.name,
            "reference": "redacted",
        },
        "configuration": {
            "language": args.language,
            "text_characters": len(args.text),
            "text_words": len(args.text.split()),
            "warmups": args.warmups,
            "runs": args.runs,
            "diffusion_steps": args.diffusion_steps,
            "seed": args.seed,
            "max_mel_tokens": args.max_mel_tokens,
            "memory_limit_gb": args.memory_limit_gb,
        },
        "run_metrics": run_metrics,
        "summary": summarize_runs(run_metrics),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(f"{rendered}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
