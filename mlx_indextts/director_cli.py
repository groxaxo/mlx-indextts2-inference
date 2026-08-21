"""CLI for source-preserving IndexTTS 2.5 sentence direction and synthesis."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .directed_runtime import (
    CUDA_PROFILES,
    QUALITY_PRESETS,
    synthesize_direction_plan,
)
from .director import (
    INDEXTTS25_DIRECTOR_SYSTEM_PROMPT,
    DirectorSettings,
    HeuristicAnnotator,
    IndexTTSDirector,
    OpenAICompatibleDirector,
)


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _add_text_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Literal source text")
    source.add_argument("--input", "-i", help="UTF-8 text file, or - for stdin")
    parser.add_argument("--language", default="auto")
    parser.add_argument(
        "--style-prompt",
        default="Natural human speech; restrained emotion; preserve the speaker's identity.",
    )


def _add_director_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-base-url",
        default=_env(
            "INDEXTTS_DIRECTOR_BASE_URL",
            "LLM_BASE_URL",
            default="http://127.0.0.1:12434/v1",
        ),
    )
    parser.add_argument(
        "--llm-api-key",
        default=_env("INDEXTTS_DIRECTOR_API_KEY", "LLM_API_KEY", default="not-needed"),
    )
    parser.add_argument(
        "--llm-model",
        default=_env("INDEXTTS_DIRECTOR_MODEL", "LLM_MODEL", default="default"),
    )
    parser.add_argument("--llm-temperature", type=float, default=0.10)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--heuristic-only", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--director-batch-sentences", type=int, default=24)
    parser.add_argument("--director-batch-characters", type=int, default=8000)


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return str(args.text)
    if args.input == "-":
        return sys.stdin.read()
    return Path(args.input).expanduser().read_text(encoding="utf-8")


def _director(args: argparse.Namespace) -> IndexTTSDirector:
    if args.heuristic_only:
        annotator = HeuristicAnnotator()
    else:
        annotator = OpenAICompatibleDirector(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            temperature=args.llm_temperature,
            timeout_seconds=args.llm_timeout,
            max_retries=args.llm_retries,
        )
    return IndexTTSDirector(
        annotator,
        settings=DirectorSettings(
            batch_sentences=args.director_batch_sentences,
            batch_characters=args.director_batch_characters,
            fallback_on_llm_error=not args.no_fallback,
        ),
    )


def _write_or_print(content: str, output: str | None) -> None:
    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + ("" if content.endswith("\n") else "\n"), encoding="utf-8")
    else:
        print(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-indextts-director",
        description=(
            "Turn prose into validated native IndexTTS 2.5 controls, then optionally "
            "synthesize it on NVIDIA CUDA."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt = subparsers.add_parser("prompt", help="Print the master LLM system prompt")
    prompt.add_argument("--output", "-o")

    tag = subparsers.add_parser("tag", help="Create an audited N/N direction plan")
    _add_text_arguments(tag)
    _add_director_arguments(tag)
    tag.add_argument("--format", choices=("json", "markup"), default="json")
    tag.add_argument("--output", "-o")

    generate = subparsers.add_parser(
        "generate", help="Direct and synthesize through the model-resident NVIDIA runtime"
    )
    _add_text_arguments(generate)
    _add_director_arguments(generate)
    generate.add_argument("--ref-audio", required=True)
    generate.add_argument("--output", "-o", required=True)
    generate.add_argument("--plan-output")
    generate.add_argument("--model-dir", default="checkpoints")
    generate.add_argument("--config", dest="config_path")
    generate.add_argument("--device", default="cuda:0")
    generate.add_argument(
        "--precision", choices=("auto", "bf16", "fp32"), default="bf16"
    )
    generate.add_argument(
        "--quality-preset", choices=tuple(QUALITY_PRESETS), default="natural-hq"
    )
    generate.add_argument(
        "--cuda-profile", choices=CUDA_PROFILES, default="quality"
    )
    generate.add_argument(
        "--cuda-kernel", action=argparse.BooleanOptionalAction, default=False
    )
    generate.add_argument("--accel", action="store_true")
    generate.add_argument("--deepspeed", action="store_true")
    generate.add_argument("--torch-compile", action="store_true")
    generate.add_argument("--seed", type=int)
    generate.add_argument("--no-coalesce", action="store_true")
    generate.add_argument("--max-sentences-per-chunk", type=int, default=3)
    generate.add_argument("--max-characters-per-chunk", type=int, default=320)
    generate.add_argument("--edge-fade-ms", type=float, default=4.0)
    generate.add_argument("--keep-segments", action="store_true")
    generate.add_argument("--compact-report", action="store_true")

    return parser


def _tag(args: argparse.Namespace) -> int:
    text = _read_text(args)
    plan = _director(args).direct(
        text,
        language=args.language,
        style_prompt=args.style_prompt,
    )
    rendered = plan.to_markup() if args.format == "markup" else plan.to_json()
    _write_or_print(rendered, args.output)
    return 0


def _generate(args: argparse.Namespace) -> int:
    from .nvidia_runtime import NvidiaIndexTTS, NvidiaRuntimeConfig

    text = _read_text(args)
    plan = _director(args).direct(
        text,
        language=args.language,
        style_prompt=args.style_prompt,
    )
    if args.plan_output:
        _write_or_print(plan.to_json(), args.plan_output)

    runtime = NvidiaIndexTTS(
        NvidiaRuntimeConfig(
            model_dir=args.model_dir,
            version="2.5",
            config_path=args.config_path,
            device=args.device,
            precision=args.precision,
            use_cuda_kernel=args.cuda_kernel,
            use_deepspeed=args.deepspeed,
            use_accel=args.accel,
            use_torch_compile=args.torch_compile,
            # The external director emits the native vector directly.  Keeping
            # QwenEmotion unloaded saves VRAM and startup latency on RTX 3090.
            use_qwen_emotion=False,
        )
    )
    try:
        result = synthesize_direction_plan(
            runtime,
            plan,
            ref_audio=args.ref_audio,
            output_path=args.output,
            language=args.language,
            preset=args.quality_preset,
            cuda_profile=args.cuda_profile,
            seed=args.seed,
            coalesce=not args.no_coalesce,
            max_sentences_per_chunk=args.max_sentences_per_chunk,
            max_characters_per_chunk=args.max_characters_per_chunk,
            edge_fade_ms=args.edge_fade_ms,
            keep_segments=args.keep_segments,
        )
    finally:
        runtime.close()

    print(result.to_json(include_plan=not args.compact_report))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prompt":
        _write_or_print(INDEXTTS25_DIRECTOR_SYSTEM_PROMPT, args.output)
        return 0
    if args.command == "tag":
        return _tag(args)
    if args.command == "generate":
        return _generate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
