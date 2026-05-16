#!/usr/bin/env python3
"""Convert the official IndexTTS2 Qwen emotion model to MLX."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mlx_indextts.qwen_emotion import (
    DEFAULT_QWEN_EMOTION_MODEL,
    OFFICIAL_QWEN_EMOTION_SOURCE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=OFFICIAL_QWEN_EMOTION_SOURCE)
    parser.add_argument("--output", default=DEFAULT_QWEN_EMOTION_MODEL)
    parser.add_argument("--q-bits", type=int, default=8)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            print(f"Qwen emotion MLX model already exists: {output}")
            return
        shutil.rmtree(output)
    if not source.exists():
        raise FileNotFoundError(
            f"Qwen emotion source checkpoint not found: {source}. "
            "Pass --source or set MLX_INDEXTTS_QWEN_EMOTION_SOURCE."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from mlx_lm import convert
    except ImportError as exc:
        raise SystemExit("Install Qwen dependencies first: uv sync --extra qwen") from exc

    convert(
        hf_path=str(source),
        mlx_path=str(output),
        quantize=True,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        dtype=args.dtype,
    )

    required = ("config.json", "tokenizer.json")
    missing = [name for name in required if not (output / name).exists()]
    weight_files = list(output.glob("*.safetensors"))
    if missing or not weight_files:
        raise RuntimeError(f"Converted model is incomplete. Missing={missing}, weights={weight_files}")
    print(f"Qwen emotion MLX model saved to: {output}")


if __name__ == "__main__":
    main()
