#!/usr/bin/env python3
"""Compare resident-model IndexTTS 2.0 and 2.5 generation on matched inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_indextts.runtime import GenerateOptions, TTSRuntime


CASES = (
    ("zh", "今晚的风很轻，窗外的灯慢慢亮了起来。"),
    ("en", "The room was quiet, and the morning light moved slowly across the table."),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v20-model", default="models/mlx-indexTTS2-standard-8bit")
    parser.add_argument("--v25-model", default="models/mlx-IndexTTS-2.5-8bit")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--output-dir", default="outputs/validation/v20-v25-benchmark")
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "pass",
        "reference_audio": str(Path(args.ref_audio).resolve()),
        "diffusion_steps": args.diffusion_steps,
        "max_tokens": args.max_tokens,
        "models": [],
    }
    for version, model_path in (("2.0", args.v20_model), ("2.5", args.v25_model)):
        runtime = TTSRuntime(quantize="fp32")
        model_record = {
            "version": version,
            "model": str(Path(model_path).resolve()),
            "status": "pass",
            "cases": [],
        }
        try:
            load_started = time.perf_counter()
            runtime.load(model_path)
            model_record["cold_load_s"] = round(time.perf_counter() - load_started, 4)

            speaker_cache = output_root / f"reference_v{version.replace('.', '')}.npz"
            reference_started = time.perf_counter()
            runtime.save_speaker(
                ref_audio=args.ref_audio,
                output_path=str(speaker_cache),
                model=model_path,
            )
            model_record["reference_preprocess_s"] = round(
                time.perf_counter() - reference_started,
                4,
            )

            for language, text in CASES:
                for run_index in range(2):
                    output_path = output_root / f"v{version}_{language}_run{run_index + 1}.wav"
                    wall_started = time.perf_counter()
                    result = runtime.generate(
                        text=text,
                        ref_audio=str(speaker_cache),
                        output_path=str(output_path),
                        model=model_path,
                        options=GenerateOptions(
                            max_tokens=args.max_tokens,
                            max_text_tokens=80,
                            interval_silence=0,
                            diffusion_steps=args.diffusion_steps,
                            denoise_ref_audio=False,
                            denoise_emotion_ref_audio=False,
                            seed=args.seed,
                            language=language,
                        ),
                    )
                    model_record["cases"].append(
                        {
                            "language": language,
                            "run": run_index + 1,
                            "phase": "first_resident" if run_index == 0 else "warm",
                            "text": text,
                            "status": "pass",
                            "output_path": str(output_path),
                            "duration_s": result["duration_s"],
                            "generation_elapsed_s": result["elapsed_s"],
                            "wall_elapsed_s": round(time.perf_counter() - wall_started, 4),
                            "rtf": result["rtf"],
                        }
                    )
        except Exception as exc:
            model_record["status"] = "fail"
            model_record["error"] = f"{type(exc).__name__}: {exc}"
            report["status"] = "fail"
        finally:
            runtime.unload()
        report["models"].append(model_record)

    for model_record in report["models"]:
        warm_rtfs = [
            case["rtf"]
            for case in model_record.get("cases", [])
            if case["phase"] == "warm"
        ]
        if warm_rtfs:
            model_record["mean_warm_rtf"] = round(sum(warm_rtfs) / len(warm_rtfs), 4)

    report_path = output_root / "performance_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
