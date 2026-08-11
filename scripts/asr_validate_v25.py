#!/usr/bin/env python3
"""Transcribe the five-language 2.5 validation set with local MLX Whisper."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import mlx_whisper


VALIDATION_CASES = {
    "basic_zh",
    "crosslingual_en",
    "crosslingual_ja",
    "crosslingual_es",
    "crosslingual_ar",
}


def _strip_marks(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _symbols(value: str, language: str) -> list[str]:
    normalized = _strip_marks(value)
    if language in {"zh", "ja"}:
        return [
            char
            for char in normalized
            if unicodedata.category(char)[0] in {"L", "N"}
        ]
    words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    return words


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _score(expected: str, actual: str, language: str) -> dict[str, Any]:
    expected_symbols = _symbols(expected, language)
    actual_symbols = _symbols(actual, language)
    distance = _edit_distance(expected_symbols, actual_symbols)
    error_rate = distance / max(1, len(expected_symbols))
    return {
        "metric": "CER" if language in {"zh", "ja"} else "WER",
        "edits": distance,
        "reference_units": len(expected_symbols),
        "hypothesis_units": len(actual_symbols),
        "error_rate": round(error_rate, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functional-report", required=True)
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.5,
        help="Sanity threshold, not a publication-quality benchmark",
    )
    args = parser.parse_args()

    report_path = Path(args.functional_report)
    functional = json.loads(report_path.read_text(encoding="utf-8"))
    results = []
    for case in functional["cases"]:
        if case.get("name") not in VALIDATION_CASES:
            continue
        language = case["language"]
        try:
            transcription = mlx_whisper.transcribe(
                case["path"],
                path_or_hf_repo=args.model,
                language=language,
                task="transcribe",
                temperature=0.0,
                verbose=False,
            )
            actual = str(transcription.get("text", "")).strip()
            score = _score(case["text"], actual, language)
            passed = bool(actual) and score["error_rate"] <= args.max_error_rate
            results.append(
                {
                    "name": case["name"],
                    "language": language,
                    "path": case["path"],
                    "expected": case["text"],
                    "transcript": actual,
                    "detected_language": transcription.get("language"),
                    "status": "pass" if passed else "fail",
                    **score,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": case["name"],
                    "language": language,
                    "path": case["path"],
                    "expected": case["text"],
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "status": "pass"
        if len(results) == len(VALIDATION_CASES)
        and all(result["status"] == "pass" for result in results)
        else "fail",
        "asr_model": args.model,
        "max_error_rate": args.max_error_rate,
        "functional_report": str(report_path),
        "cases": results,
    }
    output_path = Path(args.output) if args.output else report_path.with_name("asr_report.json")
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {output_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
