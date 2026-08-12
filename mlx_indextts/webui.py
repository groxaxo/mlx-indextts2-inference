"""Gradio WebUI for MLX-IndexTTS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from mlx_indextts.cli import detect_mlx_version
from mlx_indextts.runtime import GenerateOptions, TTSRuntime

runtime = TTSRuntime()


def _import_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Install WebUI dependencies first: uv sync --extra webui") from exc
    return gr


def _generate(
    text: str,
    ref_audio: str,
    emotion_ref_audio: str,
    profile: str,
    model: str,
    language: str,
    emotion: str,
    auto_emotion: bool,
    emotion_text: str,
    use_random: bool,
    qwen_emotion_model: str,
    max_tokens: int,
    max_text_tokens: int,
    diffusion_steps: int,
    text_normalization: bool,
    duration_factor: float,
    use_gpt_latent: bool,
    stream_segments: bool,
    denoise_ref: bool,
    denoise_emotion_ref: bool,
    seed: int,
):
    if not text.strip():
        raise ValueError("Text is required")
    if not ref_audio:
        raise ValueError("Reference audio or speaker .npz is required")
    output_dir = Path("outputs/webui")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest.wav"
    emotion_value = None if emotion == "reference" else emotion
    qwen_emotion = auto_emotion or bool(emotion_text.strip())
    if sum(bool(value) for value in (qwen_emotion, emotion_value, emotion_ref_audio)) > 1:
        raise ValueError("Choose only one emotion source: Auto Qwen, manual emotion, or emotion reference audio.")
    options = GenerateOptions(
        max_tokens=max_tokens,
        max_text_tokens=max_text_tokens,
        interval_silence=0,
        diffusion_steps=diffusion_steps,
        emotion="auto-qwen" if qwen_emotion else emotion_value,
        emotion_ref_audio=emotion_ref_audio or None,
        auto_emotion=auto_emotion,
        use_emo_text=bool(emotion_text.strip()),
        emo_text=emotion_text.strip() or None,
        use_random=use_random,
        qwen_emotion_model=qwen_emotion_model.strip() or None,
        qwen_unload_after=True,
        denoise_ref_audio=denoise_ref,
        denoise_emotion_ref_audio=denoise_emotion_ref,
        seed=None if seed < 0 else seed,
        verbose=True,
    )
    options.language = language
    options.text_normalization = text_normalization
    options.duration_factor = duration_factor
    options.use_gpt_latent = use_gpt_latent
    selected_model = model.strip() or None
    stream_requested = stream_segments
    if stream_segments:
        resolved_model = runtime.resolve_model(
            profile=profile,
            text=text,
            model=selected_model,
        )
        stream_segments = detect_mlx_version(Path(resolved_model)) == "2.5"
    if stream_segments:
        parts = []
        for chunk in runtime.stream(
            text=text,
            ref_audio=ref_audio,
            profile=profile,
            model=selected_model,
            options=options,
        ):
            parts.append(chunk.audio)
            cumulative = np.concatenate(parts)
            sf.write(output_path, cumulative, chunk.sample_rate)
            loaded = getattr(runtime, "_model", None)
            revision = getattr(loaded, "model_revision", "")
            status = (
                f"model={runtime.model_path} version={runtime.version} revision={revision}\n"
                f"language={chunk.resolved_language} "
                f"segment={chunk.segment_index + 1}/{chunk.segment_count} "
                f"completed={chunk.completed}"
            )
            yield str(output_path), status
        return

    result = runtime.generate(
        text=text,
        ref_audio=ref_audio,
        output_path=str(output_path),
        profile=profile,
        model=selected_model,
        options=options,
    )
    status = (
        f"model={result['model']} version={result['version']} revision={result.get('model_revision', '')}\n"
        f"language={result.get('language', language)} "
        f"duration={result['duration_s']}s elapsed={result['elapsed_s']}s rtf={result['rtf']}"
    )
    if stream_requested:
        status += "\nstreaming=disabled (completed-segment streaming requires IndexTTS 2.5)"
    if result.get("emotion_source") == "qwen-mlx":
        status += (
            f"\nemotion={result.get('dominant_emotion')}"
            f" qwen_elapsed={result.get('qwen_elapsed_s')}s"
            f"\n{result.get('emotion_json')}"
        )
    yield str(output_path), status


def _plan_script(script: str, max_chars_per_line: int, emotion_library: str):
    if not script.strip():
        raise ValueError("Script is required")
    from mlx_indextts.novel_planner import NovelScriptPlanner
    emotion_refs_by_emotion = {}
    if emotion_library.strip():
        from mlx_indextts.video_library import load_emotion_library_catalog

        emotion_refs_by_emotion = load_emotion_library_catalog(emotion_library.strip()).emotion_refs()

    planner = NovelScriptPlanner(max_chars_per_line=max_chars_per_line)
    output_dir = Path("outputs/webui")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "batch_plan.csv"
    rows = planner.to_batch_rows(
        planner.parse(script),
        emotion_refs_by_emotion=emotion_refs_by_emotion,
    )
    planner.write_batch_csv(script, str(output_path), emotion_refs_by_emotion=emotion_refs_by_emotion)
    preview = [
        [row["speaker"], row["text"], row["emotion"], row["emo_alpha"], row["ref_audio"], row["emotion_ref_audio"]]
        for row in rows
    ]
    return str(output_path), preview


def build_app():
    gr = _import_gradio()
    with gr.Blocks(title="MLX-IndexTTS") as demo:
        gr.Markdown("# MLX-IndexTTS")
        with gr.Row():
            profile = gr.Dropdown(
                choices=["auto", "v25", "2.5", "standard", "vietnamese", "vi"],
                value="auto",
                label="Profile",
            )
            language = gr.Dropdown(
                choices=["auto", "zh", "en", "ja", "es", "ar"],
                value="auto",
                label="Language (2.5)",
            )
            emotion = gr.Dropdown(
                choices=["reference", "calm", "happy", "sad", "angry", "melancholic", "surprised", "afraid", "disgusted", "auto-qwen"],
                value="reference",
                label="Emotion",
            )
            auto_emotion = gr.Checkbox(value=False, label="Auto Qwen emotion")
        qwen_emotion_model = gr.Textbox(
            value="",
            label="Qwen emotion model",
            placeholder="models/qwen0.6bemo4-merge-mlx-8bit",
        )
        with gr.Row():
            emotion_text = gr.Textbox(
                value="",
                label="Separate emotion text (optional)",
                placeholder="Use this description instead of the synthesis text for Qwen emotion",
            )
            use_random = gr.Checkbox(
                value=False,
                label="Random emotion prototypes",
            )
        model = gr.Textbox(
            value="",
            label="Model directory (optional)",
            placeholder="models/mlx-IndexTTS-2.5-8bit",
        )
        with gr.Tab("Generate"):
            ref_audio = gr.Audio(label="Speaker reference audio / speaker .npz", type="filepath")
            emotion_ref_audio = gr.Audio(label="Emotion reference audio / speaker .npz", type="filepath")
            with gr.Row():
                denoise_ref = gr.Checkbox(value=False, label="Denoise speaker ref")
                denoise_emotion_ref = gr.Checkbox(value=False, label="Denoise emotion ref")
            text = gr.Textbox(label="Text", lines=8)
            gr.Markdown(
                "2.5 pronunciation annotations: `<word|PRONUNCIATION>`; "
                "for example `<going|G OW1 . IH0 NG>` or `<行|XING2>`. "
                "Choose explicit `en` or `es` when Latin-script auto detection is ambiguous."
            )
            with gr.Row():
                max_tokens = gr.Slider(128, 1500, value=256, step=1, label="Max mel tokens")
                max_text_tokens = gr.Slider(40, 180, value=100, step=1, label="Max text tokens")
                diffusion_steps = gr.Slider(8, 25, value=25, step=1, label="Diffusion steps")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
            with gr.Row():
                text_normalization = gr.Checkbox(value=True, label="Text normalization")
                duration_factor = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Duration factor")
                use_gpt_latent = gr.Checkbox(value=False, label="Optional GPT latent")
                stream_segments = gr.Checkbox(value=True, label="Stream completed segments")
            button = gr.Button("Generate", variant="primary")
            audio = gr.Audio(label="Output", type="filepath")
            status = gr.Textbox(label="Status", lines=3)
            button.click(
                _generate,
                inputs=[
                    text,
                    ref_audio,
                    emotion_ref_audio,
                    profile,
                    model,
                    language,
                    emotion,
                    auto_emotion,
                    emotion_text,
                    use_random,
                    qwen_emotion_model,
                    max_tokens,
                    max_text_tokens,
                    diffusion_steps,
                    text_normalization,
                    duration_factor,
                    use_gpt_latent,
                    stream_segments,
                    denoise_ref,
                    denoise_emotion_ref,
                    seed,
                ],
                outputs=[audio, status],
            )

        with gr.Tab("Content Plan"):
            script = gr.Textbox(label="Dialogue / novel script", lines=12)
            max_chars = gr.Slider(30, 160, value=90, step=1, label="Max chars per line")
            emotion_library = gr.Textbox(
                label="Emotion library catalog.csv (optional)",
                placeholder="outputs/emotion_library/catalog.csv",
            )
            plan_button = gr.Button("Build batch CSV", variant="primary")
            plan_file = gr.File(label="Batch CSV")
            plan_preview = gr.Dataframe(
                headers=["speaker", "text", "emotion", "emo_alpha", "ref_audio", "emotion_ref_audio"],
                label="Preview",
            )
            plan_button.click(
                _plan_script,
                inputs=[script, max_chars, emotion_library],
                outputs=[plan_file, plan_preview],
            )
    return demo


def main() -> None:
    build_app().launch(server_name="127.0.0.1", server_port=7861)


if __name__ == "__main__":
    main()
