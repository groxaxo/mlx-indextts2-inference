"""Tests for public runtime and CLI routing to IndexTTS 2.5."""

import json
from pathlib import Path

import numpy as np


def test_cli_detects_v25_manifest(tmp_path: Path):
    from mlx_indextts.cli import detect_mlx_version

    (tmp_path / "model_manifest.json").write_text(
        json.dumps({"model_version": "2.5"}),
        encoding="utf-8",
    )

    assert detect_mlx_version(tmp_path) == "2.5"


def test_auto_profile_prefers_local_v25_for_non_vietnamese(tmp_path: Path, monkeypatch):
    from mlx_indextts.cli import resolve_default_model

    v25 = tmp_path / "v25"
    v25.mkdir()
    monkeypatch.setenv("MLX_INDEXTTS_V25_MODEL", str(v25))

    assert resolve_default_model("auto", "Hello") == str(v25)


def test_v25_duration_budget_uses_25_semantic_tokens_per_second():
    from mlx_indextts.runtime import estimate_mel_tokens_for_duration

    assert estimate_mel_tokens_for_duration(2.0, version="2.5") == 50


def test_v25_default_token_cap_is_256_without_changing_v20():
    from mlx_indextts.config import default_max_mel_tokens

    assert default_max_mel_tokens("2.5") == 256
    assert default_max_mel_tokens("2.0") == 1500


def test_runtime_generate_uses_v25_default_token_cap(monkeypatch, tmp_path: Path):
    import mlx_indextts.runtime as runtime_module

    class FakeTTS:
        model_revision = "revision"
        last_generation_info = {"resolved_language": "es"}
        use_gpt_latent = False

        def __init__(self):
            self.call = None

        def generate(self, **kwargs):
            self.call = kwargs
            return np.zeros(2205, dtype=np.float32)

    fake = FakeTTS()
    monkeypatch.setattr(runtime_module, "detect_mlx_version", lambda _path: "2.5")
    runtime = runtime_module.TTSRuntime(quantize="fp32")
    monkeypatch.setattr(runtime, "load", lambda _path: fake)
    monkeypatch.setattr(
        runtime,
        "_resolve_auto_emotion",
        lambda text, options: (None, {"emotion_source": "speaker_reference"}),
    )

    runtime.generate(
        text="hola",
        ref_audio="speaker.npz",
        output_path=str(tmp_path / "out.wav"),
        model="model-v25",
        options=runtime_module.GenerateOptions(
            denoise_ref_audio=False,
            denoise_emotion_ref_audio=False,
        ),
    )

    assert fake.call["max_mel_tokens"] == 256


def test_runtime_generate_routes_v25_language_and_frontend_options(monkeypatch, tmp_path: Path):
    import mlx_indextts.runtime as runtime_module

    class FakeTTS:
        model_revision = "revision"

        def __init__(self):
            self.last_generation_info = {}
            self.calls = []
            self.use_gpt_latent = False

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            self.last_generation_info = {
                "resolved_language": "ja",
                "language_ambiguous": False,
            }
            return np.zeros(2205, dtype=np.float32)

    fake = FakeTTS()
    monkeypatch.setattr(runtime_module, "detect_mlx_version", lambda path: "2.5")
    runtime = runtime_module.TTSRuntime(quantize="fp32")
    monkeypatch.setattr(runtime, "load", lambda path: fake)
    monkeypatch.setattr(
        runtime,
        "_resolve_auto_emotion",
        lambda text, options: (None, {"emotion_source": "reference"}),
    )
    output = tmp_path / "out.wav"
    options = runtime_module.GenerateOptions(
        language="ja",
        text_normalization=False,
        duration_factor=1.1,
        use_gpt_latent=True,
        use_random=True,
        denoise_ref_audio=False,
        denoise_emotion_ref_audio=False,
        diffusion_steps=2,
    )

    result = runtime.generate(
        text="こんにちは",
        ref_audio="speaker.npz",
        output_path=str(output),
        model="model-v25",
        options=options,
    )

    call = fake.calls[0]
    assert call["language"] == "ja"
    assert call["text_normalization"] is False
    assert call["duration_factor"] == 1.1
    assert call["use_random"] is True
    assert fake.use_gpt_latent is True
    assert result["version"] == "2.5"
    assert result["language"] == "ja"
    assert result["model_revision"] == "revision"


def test_runtime_creates_nested_output_directory(monkeypatch, tmp_path: Path):
    import mlx_indextts.runtime as runtime_module

    class FakeTTS:
        model_revision = "revision"
        last_generation_info = {"resolved_language": "zh"}
        use_gpt_latent = False

        def generate(self, **kwargs):
            return np.zeros(2205, dtype=np.float32)

    monkeypatch.setattr(runtime_module, "detect_mlx_version", lambda _path: "2.5")
    runtime = runtime_module.TTSRuntime(quantize="fp32")
    monkeypatch.setattr(runtime, "load", lambda _path: FakeTTS())
    monkeypatch.setattr(
        runtime,
        "_resolve_auto_emotion",
        lambda text, options: (None, {"emotion_source": "speaker_reference"}),
    )
    output = tmp_path / "nested" / "out.wav"

    runtime.generate(
        text="你好",
        ref_audio="speaker.npz",
        output_path=str(output),
        model="model-v25",
        options=runtime_module.GenerateOptions(
            denoise_ref_audio=False,
            denoise_emotion_ref_audio=False,
        ),
    )

    assert output.parent.is_dir()


def test_runtime_uses_separate_qwen_emotion_text(monkeypatch):
    import mlx_indextts.qwen_emotion as qwen_module
    import mlx_indextts.runtime as runtime_module

    prompts = []

    class FakeQwen:
        def inference(self, text):
            prompts.append(text)
            return type(
                "Result",
                (),
                {
                    "weights": {"afraid": 0.7, "calm": 0.1},
                    "source": "qwen-mlx",
                    "elapsed_s": 0.01,
                    "raw_text": "{}",
                },
            )()

    monkeypatch.setattr(qwen_module, "get_qwen_emotion", lambda _path: FakeQwen())
    monkeypatch.setattr(qwen_module, "unload_qwen_emotion", lambda _path: None)
    runtime = runtime_module.TTSRuntime()

    emotion, metadata = runtime._resolve_auto_emotion(
        "synthesis text",
        runtime_module.GenerateOptions(
            use_emo_text=True,
            emo_text="fearful description",
        ),
    )

    assert prompts == ["fearful description"]
    assert emotion == {"afraid": 0.875, "calm": 0.125}
    assert metadata["emotion_source"] == "qwen-mlx"
