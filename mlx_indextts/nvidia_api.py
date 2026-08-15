"""FastAPI server for the NVIDIA IndexTTS runtime."""

import uuid
from pathlib import Path
from typing import Any

from mlx_indextts.nvidia_runtime import (
    NvidiaGenerateRequest,
    NvidiaIndexTTS,
    NvidiaRuntimeConfig,
)


def create_app(
    config: NvidiaRuntimeConfig | None = None,
    *,
    output_dir: str | Path = "outputs/nvidia_api",
) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI dependencies are missing; run `uv sync --project nvidia`."
        ) from exc

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    runtime_config = config or NvidiaRuntimeConfig()
    runtime_holder: dict[str, NvidiaIndexTTS | None] = {"runtime": None}

    class GenerateBody(BaseModel):
        text: str = Field(min_length=1)
        ref_audio: str
        output_name: str | None = None
        language: str = "auto"
        emotion_ref_audio: str | None = None
        emotion_vector: str | list[float] | None = None
        emotion_text: str | None = None
        auto_emotion: bool = False
        use_random: bool = False
        emo_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
        interval_silence_ms: int = Field(default=200, ge=0)
        max_text_tokens: int = Field(default=120, gt=0)
        duration_factor: float = Field(default=1.0, ge=0.5, le=2.0)
        text_normalization: bool = True
        max_mel_tokens: int | None = Field(default=None, gt=0)
        temperature: float = Field(default=0.8, gt=0.0)
        top_p: float = Field(default=0.8, gt=0.0, le=1.0)
        top_k: int = Field(default=30, gt=0)
        repetition_penalty: float = Field(default=10.0, gt=0.0)
        seed: int | None = None
        verbose: bool = False

    app = FastAPI(
        title="MLX-IndexTTS NVIDIA API",
        version="0.3.0",
        description="Model-resident IndexTTS 2/2.5 inference on NVIDIA CUDA.",
    )

    @app.on_event("startup")
    def _startup() -> None:
        runtime_holder["runtime"] = NvidiaIndexTTS(runtime_config)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        runtime = runtime_holder.get("runtime")
        if runtime is not None:
            runtime.close()

    @app.get("/health")
    def health() -> dict[str, Any]:
        runtime = runtime_holder.get("runtime")
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime is not loaded")
        return {"status": "ok", **runtime.device_info()}

    @app.post("/generate")
    def generate(body: GenerateBody) -> dict[str, Any]:
        runtime = runtime_holder.get("runtime")
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime is not loaded")
        safe_name = Path(body.output_name or f"{uuid.uuid4().hex}.wav").name
        if not safe_name.lower().endswith(".wav"):
            safe_name += ".wav"
        request = NvidiaGenerateRequest(
            text=body.text,
            ref_audio=body.ref_audio,
            output_path=destination / safe_name,
            language=body.language,
            emotion_ref_audio=body.emotion_ref_audio,
            emotion_vector=body.emotion_vector,
            emotion_text=body.emotion_text,
            auto_emotion=body.auto_emotion,
            use_random=body.use_random,
            emo_alpha=body.emo_alpha,
            interval_silence_ms=body.interval_silence_ms,
            max_text_tokens=body.max_text_tokens,
            duration_factor=body.duration_factor,
            text_normalization=body.text_normalization,
            max_mel_tokens=body.max_mel_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            repetition_penalty=body.repetition_penalty,
            seed=body.seed,
            verbose=body.verbose,
        )
        try:
            result = runtime.generate(request)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {**result.as_dict(), "audio_url": f"/audio/{safe_name}"}

    @app.get("/audio/{filename}")
    def audio(filename: str) -> FileResponse:
        path = destination / Path(filename).name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    return app


def serve(
    *,
    config: NvidiaRuntimeConfig,
    host: str = "127.0.0.1",
    port: int = 7863,
    output_dir: str | Path = "outputs/nvidia_api",
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Uvicorn is missing; run `uv sync --project nvidia`."
        ) from exc
    uvicorn.run(create_app(config, output_dir=output_dir), host=host, port=port)
