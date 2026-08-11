"""Tests for IndexTTS 2.5 runtime contracts that do not need model weights."""

import numpy as np
import pytest


def test_normalize_emotion_input_accepts_ordered_vector():
    from mlx_indextts.generate_v25 import normalize_emotion_input

    weights = normalize_emotion_input([1.0, 0.2, 0, 0, 0, 0, 0, 0.1])

    assert weights == {
        "happy": 1.0,
        "angry": 0.2,
        "sad": 0.0,
        "afraid": 0.0,
        "disgusted": 0.0,
        "melancholic": 0.0,
        "surprised": 0.0,
        "calm": 0.1,
    }


def test_normalize_emotion_input_rejects_wrong_vector_length():
    from mlx_indextts.generate_v25 import normalize_emotion_input

    with pytest.raises(ValueError, match="eight"):
        normalize_emotion_input([1.0, 0.0])


@pytest.mark.parametrize(
    "values",
    [
        {"emotion_reference_audio": "emo.wav", "emotion": "happy"},
        {"emotion_reference_audio": "emo.wav", "use_emo_text": True},
        {"emotion": "happy", "use_emo_text": True},
    ],
)
def test_emotion_sources_are_mutually_exclusive(values):
    from mlx_indextts.generate_v25 import validate_emotion_sources

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_emotion_sources(**values)


def test_stream_chunk_reports_segment_completion():
    from mlx_indextts.generate_v25 import StreamChunk

    chunk = StreamChunk(
        audio=np.zeros(12, dtype=np.float32),
        sample_rate=22050,
        segment_index=1,
        segment_count=2,
        completed=False,
        resolved_language="ja",
    )

    assert chunk.audio.dtype == np.float32
    assert chunk.segment_index == 1
    assert chunk.segment_count == 2
    assert not chunk.completed
    assert chunk.resolved_language == "ja"


def test_runtime_class_is_separate_from_v20():
    from mlx_indextts.generate_v2 import IndexTTSv2
    from mlx_indextts.generate_v25 import IndexTTSv25

    assert issubclass(IndexTTSv25, IndexTTSv2)
    assert IndexTTSv25 is not IndexTTSv2


def test_resample_sequence_nearest_matches_codec_rate_change():
    import mlx.core as mx
    from mlx_indextts.generate_v25 import _resample_sequence_nearest

    values = mx.array([[[1.0], [2.0], [3.0]]])
    result = _resample_sequence_nearest(values, target_length=6)
    mx.eval(result)

    np.testing.assert_array_equal(
        np.asarray(result).reshape(-1),
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    )


def test_segment_stream_marks_only_final_chunk_complete():
    from types import SimpleNamespace

    import mlx.core as mx
    import torch

    from mlx_indextts.generate_v25 import IndexTTSv25

    runtime = IndexTTSv25.__new__(IndexTTSv25)
    runtime.model_revision = "revision"
    runtime.last_generation_info = {}
    runtime.text_frontend = SimpleNamespace(
        prepare=lambda *args, **kwargs: SimpleNamespace(
            language="en",
            language_id=0,
            language_ambiguous=True,
            token_ids=((10, 1), (11, 1)),
        )
    )
    runtime.gpt = SimpleNamespace(
        text_pos_embedding=SimpleNamespace(
            emb=SimpleNamespace(weight=mx.zeros((602, 4)))
        ),
        prepare_conditioning_latents=lambda *args, **kwargs: mx.zeros((1, 3, 4)),
    )
    reference = {"style": torch.zeros((1, 192))}
    features = {
        "style": mx.zeros((1, 192)),
        "emotion_vec": mx.zeros((1, 4)),
        "prompt_condition": mx.zeros((1, 2, 4)),
        "ref_mel": mx.zeros((1, 80, 2)),
    }
    runtime._process_reference_audio = lambda path: reference
    runtime._mlx_reference_features = lambda value: features
    runtime._emotion_vector = lambda **kwargs: (mx.zeros((1, 4)), "speaker_reference")
    runtime._generate_semantic_codes = lambda *args, **kwargs: [7, 8]
    runtime._synthesize_segment = lambda **kwargs: np.full(
        5,
        int(kwargs["text_tokens"][0, 0].item()),
        dtype=np.float32,
    )

    chunks = list(
        runtime.stream(
            "two segments",
            "speaker.npz",
            language="en",
            interval_silence=0,
            max_mel_tokens=2,
            diffusion_steps=1,
        )
    )

    assert [chunk.segment_index for chunk in chunks] == [0, 1]
    assert [chunk.completed for chunk in chunks] == [False, True]
    assert all(chunk.segment_count == 2 for chunk in chunks)
    assert runtime.last_generation_info["language_ambiguous"] is True
