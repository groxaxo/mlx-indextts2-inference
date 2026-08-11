"""FastAPI server for MLX-IndexTTS."""

from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path
from typing import Any

from mlx_indextts.runtime import GenerateOptions, TTSRuntime

runtime = TTSRuntime()
runtime_lock = threading.Lock()
OUTPUTS_ROOT = (Path.cwd() / "outputs").resolve()


def _import_fastapi():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, StreamingResponse
        from pydantic import BaseModel
    except ImportError as exc:
        raise SystemExit("Install API dependencies first: uv sync --extra api") from exc
    return FastAPI, HTTPException, FileResponse, StreamingResponse, BaseModel


FastAPI, HTTPException, FileResponse, StreamingResponse, BaseModel = _import_fastapi()
app = FastAPI(title="MLX-IndexTTS API")


def _resolve_under_outputs(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    if resolved != OUTPUTS_ROOT and OUTPUTS_ROOT not in resolved.parents:
        raise HTTPException(status_code=400, detail="Path must be under outputs/")
    return resolved


def _prepare_output_path(path: str | Path) -> str:
    resolved = _resolve_under_outputs(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def _prepare_output_dir(path: str | Path) -> str:
    resolved = _resolve_under_outputs(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


class GenerateRequest(BaseModel):
    text: str
    ref_audio: str
    output_path: str = "outputs/api_output.wav"
    profile: str = "auto"
    model: str | None = None
    max_tokens: int | None = None
    max_text_tokens: int = 120
    interval_silence: int = 0
    temperature: float | None = None
    top_k: int = 30
    top_p: float = 0.8
    repetition_penalty: float = 10.0
    diffusion_steps: int = 16
    cfg_rate: float = 0.7
    emotion: str | dict[str, float] | list[float] | None = None
    emo_alpha: float = 0.6
    emotion_ref_audio: str | None = None
    auto_emotion: bool = False
    use_emo_text: bool = False
    emo_text: str | None = None
    use_random: bool = False
    qwen_emotion_model: str | None = None
    qwen_unload_after: bool = True
    smooth_emotion: bool = True
    denoise_ref_audio: bool = False
    denoise_emotion_ref_audio: bool = False
    seed: int | None = None
    segment_overlap: int = 50
    speed: float = 1.0
    target_duration: float | None = None
    fit_duration: bool = False
    max_fit_stretch_ratio: float = 2.0
    verbose: bool = False
    dynamic_max_tokens: bool = False
    tokens_per_char: float = 14.0
    min_max_tokens: int = 320
    language: str = "auto"
    text_normalization: bool = True
    duration_factor: float = 1.0
    use_gpt_latent: bool = False


class SpeakerRequest(BaseModel):
    ref_audio: str
    output_path: str
    profile: str = "auto"
    model: str | None = None
    text: str = ""


class BatchRequest(BaseModel):
    rows: list[dict[str, Any]]
    ref_audio: str | None = None
    output_dir: str = "outputs/api_batch"
    profile: str = "auto"
    model: str | None = None
    options: GenerateRequest | None = None
    combine: bool = False
    combine_silence_ms: int = 120


class PlanRequest(BaseModel):
    content: str
    output_path: str = "outputs/api_batch_plan.csv"
    max_chars_per_line: int = 90
    emotion_library: str | None = None


@app.get("/health")
def health() -> dict:
    model_obj = getattr(runtime, "_model", None)
    manifest = getattr(model_obj, "manifest", {}) if model_obj is not None else {}
    return {
        "ok": True,
        "model": runtime.model_path,
        "version": runtime.version,
        "model_revision": manifest.get("source_revision"),
        "supported_languages": manifest.get("supported_languages", []),
        "quantization": manifest.get("quantization"),
    }


@app.get("/profiles")
def profiles() -> dict:
    return {
        "v25": "models/mlx-IndexTTS-2.5-8bit",
        "standard": "models/mlx-indexTTS2-standard-8bit",
        "vietnamese": "models/mlx-indexTTS2-vietnamese-8bit",
    }


def _options_from_request(req: GenerateRequest) -> GenerateOptions:
    return GenerateOptions(
        **{
            key: getattr(req, key)
            for key in GenerateOptions.__dataclass_fields__
        }
    )


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    output_path = _prepare_output_path(req.output_path)
    options = _options_from_request(req)
    try:
        with runtime_lock:
            return runtime.generate(
                text=req.text,
                ref_audio=req.ref_audio,
                output_path=output_path,
                profile=req.profile,
                model=req.model,
                options=options,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    """Stream completed 2.5 text segments as newline-delimited WAV payloads."""
    options = _options_from_request(req)

    def events():
        try:
            with runtime_lock:
                for chunk in runtime.stream(
                    text=req.text,
                    ref_audio=req.ref_audio,
                    profile=req.profile,
                    model=req.model,
                    options=options,
                ):
                    import soundfile as sf

                    buffer = io.BytesIO()
                    sf.write(
                        buffer,
                        chunk.audio,
                        chunk.sample_rate,
                        format="WAV",
                        subtype="PCM_16",
                    )
                    event = {
                        "segment_index": chunk.segment_index,
                        "segment_count": chunk.segment_count,
                        "sample_rate": chunk.sample_rate,
                        "completed": chunk.completed,
                        "resolved_language": chunk.resolved_language,
                        "audio_wav_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    }
                    yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps(
                {
                    "error": str(exc),
                    "completed": False,
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/audio")
def audio(path: str):
    audio_path = _resolve_under_outputs(path)
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(audio_path), media_type="audio/wav", filename=audio_path.name)


@app.post("/speaker")
def speaker(req: SpeakerRequest) -> dict:
    output_path = _prepare_output_path(req.output_path)
    with runtime_lock:
        return runtime.save_speaker(
            ref_audio=req.ref_audio,
            output_path=output_path,
            profile=req.profile,
            model=req.model,
            text=req.text,
        )


@app.post("/batch")
def batch(req: BatchRequest) -> dict:
    output_dir = _prepare_output_dir(req.output_dir)
    base = req.options or GenerateRequest(text="", ref_audio=req.ref_audio or "")
    options = _options_from_request(base)
    try:
        with runtime_lock:
            return runtime.batch(
                rows=req.rows,
                ref_audio=req.ref_audio,
                output_dir=output_dir,
                profile=req.profile,
                model=req.model,
                options=options,
                combine=req.combine,
                combine_silence_ms=req.combine_silence_ms,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/plan")
def plan(req: PlanRequest) -> dict:
    from mlx_indextts.novel_planner import NovelScriptPlanner

    emotion_refs_by_emotion = {}
    if req.emotion_library:
        from mlx_indextts.video_library import load_emotion_library_catalog

        emotion_refs_by_emotion = load_emotion_library_catalog(req.emotion_library).emotion_refs()
    planner = NovelScriptPlanner(max_chars_per_line=req.max_chars_per_line)
    output_path = planner.write_batch_csv(
        req.content,
        _prepare_output_path(req.output_path),
        emotion_refs_by_emotion=emotion_refs_by_emotion,
    )
    return {
        "output_path": output_path,
        "items": len(planner.parse(req.content)),
    }


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install API dependencies first: uv sync --extra api") from exc
    uvicorn.run("mlx_indextts.api_server:app", host="127.0.0.1", port=7862)


if __name__ == "__main__":
    main()
