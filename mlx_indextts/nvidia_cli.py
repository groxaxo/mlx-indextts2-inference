"""Command line interface for NVIDIA CUDA IndexTTS inference."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import queue
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from mlx_indextts.nvidia_runtime import (
    MODEL_REPOSITORIES,
    MODEL_REVISIONS,
    UPSTREAM_REVISION,
    NvidiaGenerateRequest,
    NvidiaIndexTTS,
    NvidiaRuntimeConfig,
    _import_torch,
    normalize_version,
)


def _default_model_dir(version: str) -> str:
    return "checkpoints" if normalize_version(version) == "2.5" else "checkpoints_2"


def _add_runtime_arguments(parser: argparse.ArgumentParser, *, batch: bool = False) -> None:
    parser.add_argument("--version", default="2.5", choices=("2.5", "2.0"))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--config", dest="config_path", default=None)
    if batch:
        parser.add_argument(
            "--devices",
            default="auto",
            help="Comma-separated CUDA devices, or auto for every visible GPU",
        )
    else:
        parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--precision", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument(
        "--cuda-kernel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the official fused BigVGAN CUDA activation kernel",
    )
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--accel", action="store_true", help="Enable the official GPT acceleration engine")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--qwen-emotion",
        action="store_true",
        help="Load the Qwen text-to-emotion model (extra VRAM; required for text emotion)",
    )


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="auto")
    emotion = parser.add_mutually_exclusive_group()
    emotion.add_argument("--emotion-ref-audio")
    emotion.add_argument("--emotion-vector", help="happy:0.6,sad:0.2 or an eight-value vector")
    emotion.add_argument("--emotion-text")
    emotion.add_argument("--auto-emotion", action="store_true")
    parser.add_argument("--use-random", action="store_true")
    parser.add_argument("--emo-alpha", type=float, default=0.6)
    parser.add_argument("--interval-silence-ms", type=int, default=200)
    parser.add_argument("--max-text-tokens", type=int, default=120)
    parser.add_argument("--duration-factor", type=float, default=1.0)
    parser.add_argument(
        "--text-normalization", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-mel-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-indextts-nvidia",
        description="Official IndexTTS 2/2.5 inference on NVIDIA CUDA, wrapped for this project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Validate CUDA, PyTorch, and upstream runtime")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    download = subparsers.add_parser("download", help="Download official model checkpoints")
    download.add_argument("--version", default="2.5", choices=("2.5", "2.0"))
    download.add_argument("--output-dir", default=None)
    download.add_argument("--revision", default=None)

    generate = subparsers.add_parser("generate", help="Generate one WAV on one GPU")
    _add_runtime_arguments(generate)
    _add_generation_arguments(generate)

    batch = subparsers.add_parser(
        "batch", help="Distribute CSV/JSONL jobs across one model-resident process per GPU"
    )
    _add_runtime_arguments(batch, batch=True)
    batch.add_argument("--input", "-i", required=True)
    batch.add_argument("--output-dir", "-o", default="outputs/nvidia_batch")
    batch.add_argument("--manifest", default=None)
    batch.add_argument("--fail-fast", action="store_true")

    serve = subparsers.add_parser("serve", help="Start the NVIDIA FastAPI server")
    _add_runtime_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7863)
    serve.add_argument("--output-dir", default="outputs/nvidia_api")

    return parser


def _runtime_config(args: argparse.Namespace, *, device: str | None = None) -> NvidiaRuntimeConfig:
    model_dir = args.model_dir or _default_model_dir(args.version)
    return NvidiaRuntimeConfig(
        model_dir=model_dir,
        version=args.version,
        config_path=args.config_path,
        device=device if device is not None else args.device,
        precision=args.precision,
        use_cuda_kernel=args.cuda_kernel,
        use_deepspeed=args.deepspeed,
        use_accel=args.accel,
        use_torch_compile=args.torch_compile,
        use_qwen_emotion=args.qwen_emotion,
    )


def _request_from_args(args: argparse.Namespace) -> NvidiaGenerateRequest:
    return NvidiaGenerateRequest(
        text=args.text,
        ref_audio=args.ref_audio,
        output_path=args.output,
        language=args.language,
        emotion_ref_audio=args.emotion_ref_audio,
        emotion_vector=args.emotion_vector,
        emotion_text=args.emotion_text,
        auto_emotion=args.auto_emotion,
        use_random=args.use_random,
        emo_alpha=args.emo_alpha,
        interval_silence_ms=args.interval_silence_ms,
        max_text_tokens=args.max_text_tokens,
        duration_factor=args.duration_factor,
        text_normalization=args.text_normalization,
        max_mel_tokens=args.max_mel_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        verbose=args.verbose,
    )


def _doctor_report() -> tuple[dict[str, Any], bool]:
    report: dict[str, Any] = {
        "upstream_revision": UPSTREAM_REVISION,
        "official_runtime_installed": importlib.util.find_spec("indextts") is not None,
    }
    try:
        torch = _import_torch()
    except Exception as exc:  # diagnostic path
        report.update({"torch_available": False, "error": str(exc)})
        return report, False
    report.update(
        {
            "torch_available": True,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "torch_cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    )
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "device": f"cuda:{index}",
                    "name": properties.name,
                    "vram_gb": round(properties.total_memory / 1024**3, 2),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    report["devices"] = devices
    report["bf16_supported"] = bool(
        torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    healthy = bool(
        report["official_runtime_installed"] and report["cuda_available"] and devices
    )
    return report, healthy


def _print_doctor(as_json: bool) -> int:
    report, healthy = _doctor_report()
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Official runtime: {'OK' if report['official_runtime_installed'] else 'MISSING'}")
        print(f"PyTorch: {report.get('torch_version', 'missing')}")
        print(f"CUDA build: {report.get('torch_cuda_build') or 'none'}")
        print(f"CUDA available: {report.get('cuda_available', False)}")
        for device in report.get("devices", []):
            print(
                f"- {device['device']}: {device['name']} | {device['vram_gb']} GiB | "
                f"SM {device['compute_capability']}"
            )
        print(f"BF16 supported: {report.get('bf16_supported', False)}")
        if report.get("error"):
            print(f"Error: {report['error']}", file=sys.stderr)
    return 0 if healthy else 1


def _download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is unavailable; run ./scripts/setup_nvidia.sh first"
        ) from exc
    version = normalize_version(args.version)
    output_dir = args.output_dir or _default_model_dir(version)
    result = snapshot_download(
        repo_id=MODEL_REPOSITORIES[version],
        local_dir=output_dir,
        revision=args.revision or MODEL_REVISIONS[version],
    )
    print(result)
    return 0


def _load_jobs(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    suffix = input_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        jobs = []
        for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_number}: each JSONL row must be an object")
            jobs.append(row)
        return jobs
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("Batch input must be .csv, .jsonl, or .ndjson")


def _parse_devices(value: str, torch_module: Any | None = None) -> list[str]:
    if value.strip().lower() == "auto":
        torch = torch_module or _import_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices are available")
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    devices = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item == "cuda":
            item = "cuda:0"
        if not item.startswith("cuda:"):
            raise ValueError("Batch devices must be cuda:N values")
        if item not in devices:
            devices.append(item)
    if not devices:
        raise ValueError("At least one CUDA device is required")
    return devices


def _row_value(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return default


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _request_from_row(row: Mapping[str, Any], *, index: int, output_dir: Path) -> NvidiaGenerateRequest:
    text = str(_row_value(row, "text", default=""))
    ref_audio = _row_value(row, "ref_audio", "reference_audio", "speaker_audio")
    if not ref_audio:
        raise ValueError(f"Batch row {index} has no ref_audio/reference_audio")
    output = _row_value(row, "output", "output_path")
    if not output:
        output = output_dir / f"{index:06d}.wav"
    elif not Path(str(output)).is_absolute():
        output = output_dir / str(output)
    return NvidiaGenerateRequest(
        text=text,
        ref_audio=str(ref_audio),
        output_path=str(output),
        language=str(_row_value(row, "language", "lang", default="auto")),
        emotion_ref_audio=_row_value(row, "emotion_ref_audio", "emo_audio"),
        emotion_vector=_row_value(row, "emotion_vector", "emotion"),
        emotion_text=_row_value(row, "emotion_text", "emo_text"),
        auto_emotion=_bool_value(_row_value(row, "auto_emotion"), False),
        use_random=_bool_value(_row_value(row, "use_random"), False),
        emo_alpha=float(_row_value(row, "emo_alpha", default=0.6)),
        interval_silence_ms=int(_row_value(row, "interval_silence_ms", default=200)),
        max_text_tokens=int(_row_value(row, "max_text_tokens", default=120)),
        duration_factor=float(_row_value(row, "duration_factor", default=1.0)),
        text_normalization=_bool_value(_row_value(row, "text_normalization"), True),
        max_mel_tokens=(
            int(_row_value(row, "max_mel_tokens"))
            if _row_value(row, "max_mel_tokens") not in {None, ""}
            else None
        ),
        temperature=float(_row_value(row, "temperature", default=0.8)),
        top_p=float(_row_value(row, "top_p", default=0.8)),
        top_k=int(_row_value(row, "top_k", default=30)),
        repetition_penalty=float(_row_value(row, "repetition_penalty", default=10.0)),
        seed=(
            int(_row_value(row, "seed"))
            if _row_value(row, "seed") not in {None, ""}
            else None
        ),
        verbose=_bool_value(_row_value(row, "verbose"), False),
    )


def _worker_main(
    device: str,
    config_payload: dict[str, Any],
    jobs: list[tuple[int, dict[str, Any]]],
    output_dir: str,
    result_queue: Any,
    fail_fast: bool,
) -> None:
    try:
        config = NvidiaRuntimeConfig(**config_payload)
        config.device = device
        runtime = NvidiaIndexTTS(config)
    except Exception:
        result_queue.put(
            {
                "kind": "worker_error",
                "device": device,
                "error": traceback.format_exc(),
                "job_indexes": [index for index, _ in jobs],
            }
        )
        return
    for index, row in jobs:
        try:
            request = _request_from_row(row, index=index, output_dir=Path(output_dir))
            result = runtime.generate(request)
            result_queue.put(
                {"kind": "result", "index": index, "status": "ok", **result.as_dict()}
            )
        except Exception:
            result_queue.put(
                {
                    "kind": "result",
                    "index": index,
                    "status": "error",
                    "device": device,
                    "error": traceback.format_exc(),
                }
            )
            if fail_fast:
                break
    runtime.close()


def _write_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _batch(args: argparse.Namespace) -> int:
    jobs = _load_jobs(args.input)
    if not jobs:
        raise ValueError("Batch input contains no jobs")
    devices = _parse_devices(args.devices)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.qwen_emotion:
        for row in jobs:
            has_text_emotion = bool(_row_value(row, "emotion_text", "emo_text"))
            has_auto_emotion = _bool_value(_row_value(row, "auto_emotion"), False)
            if has_text_emotion or has_auto_emotion:
                args.qwen_emotion = True
                break
    config = _runtime_config(args, device=devices[0])
    config_payload = asdict(config)
    assignments: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in devices]
    for index, row in enumerate(jobs):
        assignments[index % len(devices)].append((index, row))

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = []
    for device, assigned in zip(devices, assignments, strict=True):
        if not assigned:
            continue
        process = context.Process(
            target=_worker_main,
            args=(
                device,
                config_payload,
                assigned,
                str(output_dir),
                result_queue,
                args.fail_fast,
            ),
            daemon=False,
        )
        process.start()
        processes.append(process)

    expected = len(jobs)
    results: list[dict[str, Any]] = []
    worker_failures: list[dict[str, Any]] = []

    def consume(message: dict[str, Any]) -> None:
        if message.get("kind") == "worker_error":
            worker_failures.append(message)
            for index in message["job_indexes"]:
                results.append(
                    {
                        "index": index,
                        "status": "error",
                        "device": message["device"],
                        "error": message["error"],
                    }
                )
        elif message.get("kind") == "result":
            results.append(message)

    while len(results) < expected and any(process.is_alive() for process in processes):
        try:
            consume(result_queue.get(timeout=0.5))
        except queue.Empty:
            continue

    for process in processes:
        process.join()
    # Multiprocessing queues may still be flushing after the worker exits.
    for _ in range(expected + len(processes)):
        try:
            consume(result_queue.get(timeout=0.2))
        except queue.Empty:
            break

    seen = {int(result["index"]) for result in results}
    for index in range(expected):
        if index not in seen:
            results.append(
                {
                    "index": index,
                    "status": "error",
                    "error": "Worker exited without returning a result",
                }
            )
    results.sort(key=lambda row: int(row["index"]))
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else output_dir / "manifest.jsonl"
    _write_manifest(manifest, results)
    failures = [result for result in results if result.get("status") != "ok"]
    print(
        json.dumps(
            {
                "jobs": expected,
                "succeeded": expected - len(failures),
                "failed": len(failures),
                "devices": devices,
                "manifest": str(manifest),
            },
            indent=2,
        )
    )
    return 2 if failures or worker_failures else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _print_doctor(args.as_json)
    if args.command == "download":
        return _download(args)
    if args.command == "generate":
        if (args.emotion_text or args.auto_emotion) and not args.qwen_emotion:
            args.qwen_emotion = True
        runtime = NvidiaIndexTTS(_runtime_config(args))
        result = runtime.generate(_request_from_args(args))
        print(json.dumps(result.as_dict(), indent=2))
        return 0
    if args.command == "batch":
        return _batch(args)
    if args.command == "serve":
        from mlx_indextts.nvidia_api import serve

        serve(
            config=_runtime_config(args),
            host=args.host,
            port=args.port,
            output_dir=args.output_dir,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
