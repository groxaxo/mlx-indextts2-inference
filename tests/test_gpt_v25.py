"""Tests for IndexTTS 2.5 GPT conditioning and language fusion."""

import mlx.core as mx
import numpy as np


def _tiny_config():
    from mlx_indextts.config import ConformerConfig, IndexTTSConfig

    config = IndexTTSConfig()
    config.version = 2.5
    config.gpt.model_dim = 32
    config.gpt.heads = 4
    config.gpt.layers = 1
    config.gpt.max_mel_tokens = 16
    config.gpt.max_text_tokens = 8
    config.gpt.number_text_tokens = 64
    config.gpt.number_mel_codes = 34
    config.gpt.start_mel_token = 32
    config.gpt.stop_mel_token = 33
    config.gpt.condition_num_latent = 2
    config.gpt.condition_module = ConformerConfig(
        output_size=16,
        linear_units=32,
        attention_heads=4,
        num_blocks=1,
        perceiver_mult=2,
    )
    config.gpt.emo_condition_module = ConformerConfig(
        output_size=16,
        linear_units=32,
        attention_heads=4,
        num_blocks=1,
        perceiver_mult=2,
    )
    return config


def test_v25_parameter_tree_uses_campplus_and_language_modules():
    from mlx.utils import tree_flatten
    from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25

    model = UnifiedVoiceV25(_tiny_config())
    keys = {key for key, _ in tree_flatten(model.parameters())}

    assert "spk_emb_proj.weight" in keys
    assert "lang_embedding.weight" in keys
    assert model.spk_emb_proj.weight.shape == (32, 192)
    assert model.lang_embedding.weight.shape == (107, 32)
    assert not any(key.startswith("conditioning_encoder.") for key in keys)
    assert not any(key.startswith("perceiver_encoder.") for key in keys)
    assert not any(key.startswith("speed_emb.") for key in keys)


def test_v25_conditioning_is_projected_speaker_plus_emotion_and_two_zeros():
    from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25

    model = UnifiedVoiceV25(_tiny_config())
    speaker = mx.ones((2, 192), dtype=mx.float32)
    emotion = mx.ones((2, 32), dtype=mx.float32)

    conditioning = model.prepare_conditioning_latents(speaker, emotion, batch_size=2)
    mx.eval(conditioning)

    assert conditioning.shape == (2, 3, 32)
    np.testing.assert_allclose(np.asarray(conditioning[:, 1:, :]), 0.0, atol=0.0)


def test_v25_prepare_inputs_matches_official_padding_and_language_fusion():
    from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25

    model = UnifiedVoiceV25(_tiny_config())
    model.text_embedding.weight = mx.zeros_like(model.text_embedding.weight)
    model.text_pos_embedding.emb.weight = mx.zeros_like(model.text_pos_embedding.emb.weight)
    language_weights = mx.zeros_like(model.lang_embedding.weight)
    language_weights[7] = mx.full((32,), 3.0)
    model.lang_embedding.weight = language_weights

    conditioning = mx.zeros((1, 3, 32), dtype=mx.float32)
    # The public tokenizer appends stop token 1 before this method is called.
    text_tokens = mx.array([[10, 11, 1]], dtype=mx.int32)
    embeddings, attention_mask = model.prepare_inputs(
        conditioning,
        text_tokens,
        language_ids=mx.array([7], dtype=mx.int32),
    )
    mx.eval(embeddings, attention_mask)

    assert embeddings.shape == (1, 8, 32)
    assert np.asarray(attention_mask).tolist() == [[0, 1, 1, 1, 1, 1, 1, 1]]
    np.testing.assert_allclose(np.asarray(embeddings[0, :4]), 0.0)
    np.testing.assert_allclose(np.asarray(embeddings[0, 4:]), 3.0)


def test_v25_prepare_inputs_requires_one_language_per_batch_row():
    import pytest
    from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25

    model = UnifiedVoiceV25(_tiny_config())
    conditioning = mx.zeros((2, 3, 32), dtype=mx.float32)
    text_tokens = mx.array([[2, 1], [3, 1]], dtype=mx.int32)

    with pytest.raises(ValueError, match="language ID"):
        model.prepare_inputs(conditioning, text_tokens, language_ids=mx.array([1]))
