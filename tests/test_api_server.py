"""Tests for API server path handling."""

from pathlib import Path

import numpy as np
import pytest


def test_audio_path_must_stay_under_outputs():
    from mlx_indextts.api_server import _resolve_under_outputs

    with pytest.raises(Exception) as exc_info:
        _resolve_under_outputs("/etc/passwd")

    assert getattr(exc_info.value, "status_code", None) == 400


def test_relative_output_path_resolves_under_outputs():
    from mlx_indextts.api_server import OUTPUTS_ROOT, _resolve_under_outputs

    resolved = _resolve_under_outputs("outputs/api_test.wav")

    assert resolved == (Path.cwd() / "outputs" / "api_test.wav").resolve()
    assert OUTPUTS_ROOT in resolved.parents


def test_generate_request_exposes_v25_language_controls():
    from mlx_indextts.api_server import GenerateRequest, _options_from_request

    request = GenerateRequest(
        text="hola",
        ref_audio="speaker.npz",
        language="es",
        text_normalization=False,
        duration_factor=1.2,
        use_gpt_latent=True,
        use_emo_text=True,
        emo_text="A frightening scene",
        use_random=True,
    )
    options = _options_from_request(request)

    assert options.language == "es"
    assert options.text_normalization is False
    assert options.duration_factor == 1.2
    assert options.use_gpt_latent is True
    assert options.use_emo_text is True
    assert options.emo_text == "A frightening scene"
    assert options.use_random is True


def test_generate_request_accepts_official_eight_value_emotion_vector():
    from mlx_indextts.api_server import GenerateRequest, _options_from_request

    vector = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
    request = GenerateRequest(
        text="happy",
        ref_audio="speaker.npz",
        emotion=vector,
    )

    assert _options_from_request(request).emotion == vector


def test_stream_endpoint_emits_ndjson_completed_segment(monkeypatch):
    from fastapi.testclient import TestClient
    from mlx_indextts import api_server
    from mlx_indextts.generate_v25 import StreamChunk

    monkeypatch.setattr(
        api_server.runtime,
        "stream",
        lambda **kwargs: iter(
            [
                StreamChunk(
                    audio=np.zeros(220, dtype=np.float32),
                    sample_rate=22050,
                    segment_index=0,
                    segment_count=1,
                    completed=True,
                    resolved_language="ar",
                )
            ]
        ),
    )
    client = TestClient(api_server.app)

    response = client.post(
        "/generate/stream",
        json={
            "text": "مرحبا",
            "ref_audio": "speaker.npz",
            "language": "ar",
        },
    )

    assert response.status_code == 200
    event = response.json() if response.headers["content-type"].startswith("application/json") else None
    if event is None:
        import json

        event = json.loads(response.text.strip())
    assert event["completed"] is True
    assert event["resolved_language"] == "ar"
    assert event["audio_wav_base64"]
