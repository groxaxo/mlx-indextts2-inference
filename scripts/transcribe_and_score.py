#!/usr/bin/env python3
"""Transcribe a synthesized WAV with Parakeet and report normalized WER.

This is an intelligibility sanity check for a fixed synthesis prompt. It does
not measure speaker similarity, emotion fidelity, or naturalness.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import requests


def normalize_text(text: str) -> str:
    """Normalize text consistently for a case- and accent-insensitive WER."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^\w]+", " ", without_accents).split())


def word_error_rate(reference: str, hypothesis: str) -> tuple[int, int, float]:
    """Return edit count, reference word count, and conventional WER."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    previous = list(range(len(hyp_words) + 1))
    for row, ref_word in enumerate(ref_words, start=1):
        current = [row]
        for column, hyp_word in enumerate(hyp_words, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    errors = previous[-1]
    return errors, len(ref_words), errors / len(ref_words) if ref_words else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--expected", required=True)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:5093/v1/audio/transcriptions",
        help="OpenAI-compatible Parakeet transcription endpoint",
    )
    parser.add_argument("--model", default="parakeet-tdt-0.6b-v3")
    args = parser.parse_args()

    with args.audio.open("rb") as audio:
        response = requests.post(
            args.url,
            data={"model": args.model},
            files={"file": (args.audio.name, audio, "audio/wav")},
            timeout=120,
        )
    response.raise_for_status()
    transcript = str(response.json()["text"])
    errors, words, wer = word_error_rate(args.expected, transcript)
    print(
        json.dumps(
            {
                "audio": str(args.audio),
                "endpoint": args.url,
                "model": args.model,
                "reference": args.expected,
                "transcript": transcript,
                "normalized_reference": normalize_text(args.expected),
                "normalized_transcript": normalize_text(transcript),
                "word_errors": errors,
                "reference_words": words,
                "wer": wer,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
