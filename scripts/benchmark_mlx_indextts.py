#!/usr/bin/env python3
import argparse
import contextlib
import csv
import gc
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

from mlx_indextts.generate_v2 import IndexTTSv2


CASES = [
    {
        "case_id": "standard_zh_short",
        "profile": "standard",
        "language": "zh",
        "length": "short",
        "text": "今晚的风很轻，窗外的灯慢慢亮了起来。",
        "segment_tokens": 120,
        "max_mel_tokens": 900,
    },
    {
        "case_id": "standard_zh_long",
        "profile": "standard",
        "language": "zh",
        "length": "long",
        "text": "夜色落下之后，城市的声音渐渐变得柔和。她站在窗前，看着远处一盏一盏亮起的灯，心里忽然想起很多年前的那个夏天。那时候他们还年轻，所有告别都像只是短暂的停顿，没有人真的相信故事会走到这里。",
        "segment_tokens": 80,
        "max_mel_tokens": 900,
    },
    {
        "case_id": "standard_en_short",
        "profile": "standard",
        "language": "en",
        "length": "short",
        "text": "The room was quiet, and the morning light moved slowly across the table.",
        "segment_tokens": 32,
        "max_mel_tokens": 900,
    },
    {
        "case_id": "standard_en_long",
        "profile": "standard",
        "language": "en",
        "length": "long",
        "text": "After the rain stopped, the streets still carried the smell of summer. He walked past the old bookstore and paused for a moment, remembering the promise they made years ago. It was not a dramatic memory, only a quiet one, but it stayed with him longer than he expected.",
        "segment_tokens": 32,
        "max_mel_tokens": 900,
    },
    {
        "case_id": "vietnamese_vi_short",
        "profile": "vietnamese",
        "language": "vi",
        "length": "short",
        "text": "Đêm nay gió rất nhẹ, ánh đèn ngoài cửa sổ chậm rãi sáng lên.",
        "segment_tokens": 80,
        "max_mel_tokens": 900,
    },
    {
        "case_id": "vietnamese_vi_long",
        "profile": "vietnamese",
        "language": "vi",
        "length": "long",
        "text": "Sau cơn mưa, con phố vẫn còn mùi của mùa hè. Cô đứng trước khung cửa sổ, nhìn những ánh đèn xa xa lần lượt sáng lên, rồi bất chợt nhớ về một lời hứa rất cũ. Khi ấy họ còn trẻ, tưởng rằng mọi cuộc chia tay chỉ là một khoảng dừng ngắn ngủi.",
        "segment_tokens": 80,
        "max_mel_tokens": 900,
    },
]


TIME_PATTERNS = {
    "reported_audio_duration_s": r"Generated\s+([0-9.]+)s audio",
    "reported_total_time_s": r"Generated\s+[0-9.]+s audio in\s+([0-9.]+)s",
    "reported_rtf": r"RTF:\s+([0-9.]+)",
    "gpt_gen_time_s": r"GPT gen:\s+([0-9.]+)s",
    "s2mel_time_s": r"S2Mel:\s+([0-9.]+)s",
    "bigvgan_time_s": r"BigVGAN:\s+([0-9.]+)s",
    "total_mel_tokens": r"Total mel tokens:\s+([0-9]+)",
}


class Tee(io.StringIO):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, text):
        self.stream.write(text)
        self.stream.flush()
        return super().write(text)


def parse_metrics(log_text):
    metrics = {}
    for key, pattern in TIME_PATTERNS.items():
        match = re.search(pattern, log_text)
        if not match:
            metrics[key] = None
            continue
        value = match.group(1)
        metrics[key] = int(value) if key == "total_mel_tokens" else float(value)
    segment_lines = re.findall(r"Segment\s+(\d+):\s+(\d+)\s+tokens", log_text)
    metrics["segments"] = len(segment_lines) if segment_lines else 1
    metrics["max_segment_text_tokens_observed"] = max(
        (int(item[1]) for item in segment_lines),
        default=None,
    )
    return metrics


def audio_metrics(path):
    data, sr = sf.read(path, always_2d=True)
    duration = len(data) / float(sr)
    if data.size == 0:
        return {
            "audio_duration_s": 0.0,
            "audio_rms": 0.0,
            "audio_peak": 0.0,
            "low_energy_ratio": 1.0,
            "longest_low_energy_s": 0.0,
        }
    mono = np.abs(data).max(axis=1)
    peak = float(mono.max())
    rms = float(np.sqrt(np.mean(mono ** 2)))
    threshold = max(0.004, peak * 0.03)
    low = mono < threshold
    longest = 0
    current = 0
    for value in low:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "audio_duration_s": round(duration, 3),
        "audio_rms": round(rms, 6),
        "audio_peak": round(peak, 6),
        "low_energy_ratio": round(float(low.mean()), 4),
        "longest_low_energy_s": round(longest / float(sr), 3),
    }


def rss_mb():
    return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)


def load_baseline(path):
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            if "wall_rtf" not in row and "after_rtf" in row:
                row["wall_rtf"] = row.get("after_rtf", "")
            if "wall_time_s" not in row and "after_wall_s" in row:
                row["wall_time_s"] = row.get("after_wall_s", "")
            rows[row["case_id"]] = row
        return rows


def run_case(tts, case, speaker_path, output_dir, args):
    output_path = output_dir / f"{case['case_id']}.wav"
    log_path = output_dir / f"{case['case_id']}.log"
    print(f"\n## RUN {case['case_id']}", flush=True)
    print(
        f"text_chars={len(case['text'])}, segment_tokens={case['segment_tokens']}, "
        f"max_mel_tokens={case['max_mel_tokens']}, diffusion_steps={args.diffusion_steps}",
        flush=True,
    )

    tee = Tee(sys.stdout)
    start = time.perf_counter()
    error = ""
    with contextlib.redirect_stdout(tee):
        try:
            tts.generate(
                text=case["text"],
                reference_audio=str(speaker_path),
                output_path=str(output_path),
                max_mel_tokens=case["max_mel_tokens"],
                max_text_tokens_per_segment=case["segment_tokens"],
                interval_silence=0,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                diffusion_steps=args.diffusion_steps,
                cfg_rate=args.cfg_rate,
                emotion=args.emotion,
                emo_alpha=args.emo_alpha,
                seed=args.seed,
                verbose=True,
                segment_overlap_ms=args.segment_overlap_ms,
                speed=1.0,
            )
        except Exception as exc:
            error = repr(exc)
            print(f"CASE FAILED: {error}", flush=True)
    wall_time = time.perf_counter() - start
    log_text = tee.getvalue()
    log_path.write_text(log_text, encoding="utf-8")

    row = {
        "case_id": case["case_id"],
        "profile": case["profile"],
        "language": case["language"],
        "length": case["length"],
        "status": "failed" if error else "ok",
        "error": error,
        "text_chars": len(case["text"]),
        "segment_tokens_requested": case["segment_tokens"],
        "max_mel_tokens": case["max_mel_tokens"],
        "diffusion_steps": args.diffusion_steps,
        "wall_time_s": round(wall_time, 3),
        "wall_rtf": None,
        "rss_mb_after_case": rss_mb(),
        "output_path": str(output_path),
        "log_path": str(log_path),
    }
    row.update(parse_metrics(log_text))
    if output_path.exists():
        row.update(audio_metrics(output_path))
        duration = row.get("audio_duration_s") or 0
        row["wall_rtf"] = round(wall_time / duration, 4) if duration > 0 else None
    print(
        f"## DONE {case['case_id']}: status={row['status']} "
        f"audio={row.get('audio_duration_s', 0)}s wall={wall_time:.2f}s rtf={row['wall_rtf']}",
        flush=True,
    )
    return row


def write_report(rows, baseline, output_dir, args):
    report_path = output_dir / "mlx_vs_pytorch_report.md"
    lines = [
        "# mlx-indextts Benchmark Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- MLX standard model: `{args.standard_model}`",
        f"- MLX Vietnamese model: `{args.vietnamese_model}`",
        f"- Speaker standard: `{args.standard_speaker}`",
        f"- Speaker Vietnamese: `{args.vietnamese_speaker}`",
        f"- MLX memory limit: `{args.memory_limit_gb}GB`",
        f"- Diffusion steps: `{args.diffusion_steps}`",
        f"- Emotion: `{args.emotion}`, emo_alpha: `{args.emo_alpha}`",
        "",
        "## Summary",
        "",
        "| Case | MLX RTF | MLX wall | MLX audio | PyTorch RTF | PyTorch wall | Delta RTF | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        base = baseline.get(row["case_id"], {})
        pt_rtf = float(base["wall_rtf"]) if base.get("wall_rtf") else None
        pt_wall = float(base["wall_time_s"]) if base.get("wall_time_s") else None
        delta = row["wall_rtf"] - pt_rtf if row.get("wall_rtf") and pt_rtf else None
        lines.append(
            "| {case} | {mlx_rtf} | {mlx_wall} | {mlx_audio} | {pt_rtf} | {pt_wall} | {delta} | {status} |".format(
                case=row["case_id"],
                mlx_rtf=f"{row['wall_rtf']:.3f}" if row.get("wall_rtf") else "",
                mlx_wall=f"{row['wall_time_s']:.2f}s",
                mlx_audio=f"{row.get('audio_duration_s', 0):.2f}s",
                pt_rtf=f"{pt_rtf:.3f}" if pt_rtf is not None else "",
                pt_wall=f"{pt_wall:.2f}s" if pt_wall is not None else "",
                delta=f"{delta:+.3f}" if delta is not None else "",
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "## Audio Health",
            "",
            "| Case | RMS | Peak | Low energy ratio | Longest low energy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row.get('audio_rms', 0):.6f} | {row.get('audio_peak', 0):.6f} | "
            f"{row.get('low_energy_ratio', 0):.4f} | {row.get('longest_low_energy_s', 0):.3f}s |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- MLX results use precomputed `.npz` speaker conditioning, so they measure generation path rather than reference-audio preprocessing.",
            "- PyTorch baseline is read from the existing optimized benchmark CSV when available; reference voice may not be identical, so compare RTF as engineering signal, not as a strict acoustic A/B.",
            "- `low_energy_ratio` and longest low-energy run are included to catch the long-silence failure seen in the MPS path.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-model", type=Path, default=Path("models/mlx-indexTTS2-standard-fp32"))
    parser.add_argument("--vietnamese-model", type=Path, default=Path("models/mlx-indexTTS2-vietnamese-fp32"))
    parser.add_argument("--standard-speaker", type=Path, default=Path("speakers/ban_khoe_standard_v2.npz"))
    parser.add_argument("--vietnamese-speaker", type=Path, default=Path("speakers/ban_khoe_vietnamese_v2.npz"))
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--memory-limit-gb", type=float, default=24.0)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--cfg-rate", type=float, default=0.7)
    parser.add_argument("--emotion", type=str, default="calm")
    parser.add_argument("--emo-alpha", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--segment-overlap-ms", type=int, default=50)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or Path("outputs/benchmarks") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    rows = []
    for profile, model_path, speaker_path in [
        ("standard", args.standard_model, args.standard_speaker),
        ("vietnamese", args.vietnamese_model, args.vietnamese_speaker),
    ]:
        print(f"\n# Loading {profile}: {model_path}", flush=True)
        load_start = time.perf_counter()
        tts = IndexTTSv2(
            model_dir=str(model_path),
            memory_limit_gb=args.memory_limit_gb,
            quantize_bits=None,
        )
        load_time = time.perf_counter() - load_start
        print(f"# Loaded {profile} in {load_time:.2f}s, rss={rss_mb()}MB", flush=True)
        for case in [item for item in CASES if item["profile"] == profile]:
            row = run_case(tts, case, speaker_path, output_dir, args)
            row["model_load_time_s"] = round(load_time, 3)
            rows.append(row)
            gc.collect()
        del tts
        gc.collect()

    csv_path = output_dir / "mlx_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    report_path = write_report(rows, load_baseline(args.baseline_csv), output_dir, args)
    print(f"\nCSV: {csv_path}", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
