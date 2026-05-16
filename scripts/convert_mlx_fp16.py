#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx


WEIGHT_FILES = (
    "gpt.safetensors",
    "s2mel.safetensors",
    "bigvgan.safetensors",
    "vq2emb.safetensors",
)


def cast_weights(src_path: Path, dst_path: Path) -> tuple[int, int]:
    weights = mx.load(str(src_path))
    converted = {}
    cast_count = 0
    for key, value in weights.items():
        if value.dtype in (mx.float32, mx.bfloat16):
            converted[key] = value.astype(mx.float16)
            cast_count += 1
        else:
            converted[key] = value
    mx.save_safetensors(str(dst_path), converted)
    return len(converted), cast_count


def copy_metadata(src_dir: Path, dst_dir: Path) -> None:
    for item in src_dir.iterdir():
        if item.name in WEIGHT_FILES:
            continue
        target = dst_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def update_config(dst_dir: Path) -> None:
    config_path = dst_dir / "config.json"
    if not config_path.exists():
        return
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["precision"] = "fp16"
    data["fp16_conversion"] = {
        "floating_weights": "cast_to_float16",
        "source": "mlx fp32/fp16 safetensors",
    }
    config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_model(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_metadata(src_dir, dst_dir)

    print(f"Converting {src_dir} -> {dst_dir}")
    for name in WEIGHT_FILES:
        src_path = src_dir / name
        if not src_path.exists():
            print(f"  skip missing {name}")
            continue
        dst_path = dst_dir / name
        total, cast_count = cast_weights(src_path, dst_path)
        size_mb = dst_path.stat().st_size / 1024 / 1024
        print(f"  {name}: tensors={total}, cast={cast_count}, size={size_mb:.1f}MB")
    update_config(dst_dir)
    print("Done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an fp16 MLX IndexTTS model from converted safetensors.")
    parser.add_argument("--src", type=Path, required=True, help="Source converted MLX model directory")
    parser.add_argument("--dst", type=Path, required=True, help="Destination fp16 model directory")
    args = parser.parse_args()
    convert_model(args.src, args.dst)


if __name__ == "__main__":
    main()
