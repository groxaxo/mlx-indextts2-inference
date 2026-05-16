"""FastAPI server for MLX-IndexTTS."""

from __future__ import annotations

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
        from fastapi.responses import FileResponse
        from pydantic import BaseModel
    except ImportError as exc:
        raise SystemExit("Install API dependencies first: uv sync --extra api") from exc
    return FastAPI, HTTPException, FileResponse, BaseModel


FastAPI, HTTPException, FileResponse, BaseModel = _import_fastapi()
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
    emotion: str | None = None
    emo_alpha: float = 0.6
    emotion_ref_audio: str | None = None
    auto_emotion: bool = False
    qwen_emotion_model: str | None = None
    qwen_unload_after: bool = True
    smooth_emotion: bool = True
    denoise_ref_audio: bool = False
    denoise_emotion_ref_audio: bool = False
    seed: int | None = None
    segment_overlap: int = 50
    speed: float = 1.0
    verbose: bool = False


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
    return {
        "ok": True,
        "model": runtime.model_path,
        "version": runtime.version,
    }


@app.get("/profiles")
def profiles() -> dict:
    return {
        "standard": "models/mlx-indexTTS2-standard-8bit",
        "vietnamese": "models/mlx-indexTTS2-vietnamese-8bit",
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    output_path = _prepare_output_path(req.output_path)
    options = GenerateOptions(**{
        key: getattr(req, key)
        for key in GenerateOptions.__dataclass_fields__
    })
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
    options = GenerateOptions(**{
        key: getattr(base, key)
        for key in GenerateOptions.__dataclass_fields__
    })
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
