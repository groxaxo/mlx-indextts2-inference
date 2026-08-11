"""WebUI routing tests."""


def test_stream_request_falls_back_to_generate_for_v2(monkeypatch):
    from mlx_indextts import webui

    called = {"generate": 0, "stream": 0}
    monkeypatch.setattr(
        webui.runtime,
        "resolve_model",
        lambda **kwargs: "models/mlx-indexTTS2-standard-8bit",
    )
    monkeypatch.setattr(webui, "detect_mlx_version", lambda _path: "2.0")

    def fake_generate(**kwargs):
        called["generate"] += 1
        return {
            "model": kwargs["model"] or "models/mlx-indexTTS2-standard-8bit",
            "version": "2.0",
            "model_revision": "",
            "language": "auto",
            "duration_s": 1.0,
            "elapsed_s": 1.0,
            "rtf": 1.0,
        }

    def fake_stream(**kwargs):
        called["stream"] += 1
        return iter(())

    monkeypatch.setattr(webui.runtime, "generate", fake_generate)
    monkeypatch.setattr(webui.runtime, "stream", fake_stream)

    results = list(
        webui._generate(
            text="hello",
            ref_audio="speaker.wav",
            emotion_ref_audio="",
            profile="standard",
            model="",
            language="auto",
            emotion="reference",
            auto_emotion=False,
            emotion_text="",
            use_random=False,
            qwen_emotion_model="",
            max_tokens=800,
            max_text_tokens=100,
            diffusion_steps=8,
            text_normalization=True,
            duration_factor=1.0,
            use_gpt_latent=False,
            stream_segments=True,
            denoise_ref=False,
            denoise_emotion_ref=False,
            seed=-1,
        )
    )

    assert called == {"generate": 1, "stream": 0}
    assert "streaming=disabled" in results[0][1]
