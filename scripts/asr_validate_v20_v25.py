#!/usr/bin/env python3
"""ASR-score warm 2.0/2.5 outputs from the matched benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx_whisper

from asr_validate_v25 import _score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-report", required=True)
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
    )
    parser.add_argument("--max-error-rate", type=float, default=0.5)
    parser.add_argument("--output")
    args = parser.parse_args()

    source_path = Path(args.benchmark_report)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cases = []
    for model in source["models"]:
        for case in model.get("cases", []):
            if case.get("phase") != "warm":
                continue
            language = case["language"]
            try:
                result = mlx_whisper.transcribe(
                    case["output_path"],
                    path_or_hf_repo=args.model,
                    language=language,
                    task="transcribe",
                    temperature=0.0,
                    verbose=False,
                )
                transcript = str(result.get("text", "")).strip()
                score = _score(case["text"], transcript, language)
                passed = bool(transcript) and score["error_rate"] <= args.max_error_rate
                cases.append(
                    {
                        "version": model["version"],
                        "language": language,
                        "path": case["output_path"],
                        "expected": case["text"],
                        "transcript": transcript,
                        "status": "pass" if passed else "fail",
                        **score,
                    }
                )
            except Exception as exc:
                cases.append(
                    {
                        "version": model["version"],
                        "language": language,
                        "path": case["output_path"],
                        "expected": case["text"],
                        "status": "fail",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    report = {
        "status": "pass"
        if len(cases) == 4 and all(case["status"] == "pass" for case in cases)
        else "fail",
        "asr_model": args.model,
        "max_error_rate": args.max_error_rate,
        "benchmark_report": str(source_path),
        "cases": cases,
    }
    output_path = Path(args.output) if args.output else source_path.with_name("asr_report.json")
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {output_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
