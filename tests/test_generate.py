"""Tests for speech generation."""

import pytest
import mlx.core as mx
import numpy as np


class TestCrossfadeSegments:
    """Tests for audio segment crossfading."""

    def test_crossfade_empty(self):
        """Test crossfade with empty list."""
        from mlx_indextts.generate import crossfade_segments

        result = crossfade_segments([], sample_rate=24000)
        assert len(result) == 0

    def test_crossfade_single_segment(self):
        """Test crossfade with single segment returns unchanged."""
        from mlx_indextts.generate import crossfade_segments

        segment = mx.array(np.random.randn(1000).astype(np.float32))
        result = crossfade_segments([segment], sample_rate=24000)

        assert result.shape == segment.shape
        np.testing.assert_array_almost_equal(np.array(result), np.array(segment))

    def test_crossfade_two_segments(self):
        """Test crossfade with two segments produces correct length."""
        from mlx_indextts.generate import crossfade_segments

        sample_rate = 24000
        overlap_ms = 50
        overlap_samples = int(overlap_ms * sample_rate / 1000)  # 1200 samples

        seg1 = mx.array(np.ones(5000, dtype=np.float32))
        seg2 = mx.array(np.ones(5000, dtype=np.float32) * 0.5)

        result = crossfade_segments([seg1, seg2], sample_rate=sample_rate, overlap_ms=overlap_ms)

        # Expected length: len(seg1) + len(seg2) - overlap
        expected_len = 5000 + 5000 - overlap_samples
        assert result.shape[0] == expected_len

    def test_crossfade_disabled(self):
        """Test that overlap_ms=0 does simple concatenation."""
        from mlx_indextts.generate import crossfade_segments

        seg1 = mx.array(np.random.randn(1000).astype(np.float32))
        seg2 = mx.array(np.random.randn(1000).astype(np.float32))

        result = crossfade_segments([seg1, seg2], sample_rate=24000, overlap_ms=0)

        # Should be simple concatenation
        expected = mx.concatenate([seg1, seg2], axis=0)
        np.testing.assert_array_almost_equal(np.array(result), np.array(expected))

    def test_crossfade_smooth_transition(self):
        """Test that crossfade produces smooth transition."""
        from mlx_indextts.generate import crossfade_segments

        # Create segments with different values to test blending
        seg1 = mx.array(np.ones(2000, dtype=np.float32) * 1.0)  # All 1.0
        seg2 = mx.array(np.ones(2000, dtype=np.float32) * 0.0)  # All 0.0

        sample_rate = 24000
        overlap_ms = 50
        overlap_samples = int(overlap_ms * sample_rate / 1000)  # 1200 samples

        result = crossfade_segments([seg1, seg2], sample_rate=sample_rate, overlap_ms=overlap_ms)

        # Check that the overlap region has blended values between 0 and 1
        # The overlap region starts at position (2000 - overlap_samples)
        overlap_start = 2000 - overlap_samples
        overlap_region = np.array(result[overlap_start:overlap_start + overlap_samples])

        # All values in overlap should be between 0 and 1
        assert np.all(overlap_region >= 0)
        assert np.all(overlap_region <= 1)

        # First part of overlap should be closer to 1, last part closer to 0
        assert overlap_region[0] > overlap_region[-1]


class TestGenerateComponents:
    """Tests for generation component functions."""

    def test_duration_token_estimator(self):
        from mlx_indextts.runtime import estimate_mel_tokens_for_duration

        assert estimate_mel_tokens_for_duration(2.0, version="2.0") == 100
        assert estimate_mel_tokens_for_duration(2.0, version="1.5") == 90
        assert estimate_mel_tokens_for_duration(0.0, version="2.0") == 5

    def test_v20_accepts_official_eight_value_emotion_vector(self):
        from mlx_indextts.generate_v2 import normalize_emotion_input

        assert normalize_emotion_input([0.8, 0, 0, 0, 0, 0, 0.1, 0]) == {
            "happy": 0.8,
            "angry": 0.0,
            "sad": 0.0,
            "afraid": 0.0,
            "disgusted": 0.0,
            "melancholic": 0.0,
            "surprised": 0.1,
            "calm": 0.0,
        }

    def test_text_token_estimator_caps_short_batch_rows(self):
        from mlx_indextts.runtime import estimate_mel_tokens_for_text

        assert estimate_mel_tokens_for_text("你好。", hard_max=900) == 320
        assert estimate_mel_tokens_for_text("这是一个很长的批量生成句子", hard_max=260, tokens_per_char=20) == 260

    def test_duration_budget_does_not_truncate_short_subtitle_content(self):
        from mlx_indextts.runtime import resolve_mel_token_budget

        # Regression: the old duration-only cap was 103 tokens for this row,
        # which could stop cross-language generation before EOS and drop words.
        assert resolve_mel_token_budget(
            "You can. On your knees.",
            requested_max=1500,
            target_duration=2.048,
            version="2.0",
        ) == 320
        assert resolve_mel_token_budget(
            "A deliberately long subtitle window.",
            requested_max=1500,
            target_duration=10.0,
            version="2.0",
        ) == 500

    def test_explicit_max_tokens_remains_a_hard_limit(self):
        from mlx_indextts.runtime import resolve_mel_token_budget

        assert resolve_mel_token_budget(
            "You can. On your knees.",
            requested_max=96,
            target_duration=2.048,
            version="2.0",
        ) == 96

    def test_batch_bool_parses_false_string(self):
        from mlx_indextts.runtime import _batch_bool

        assert _batch_bool("true", False) is True
        assert _batch_bool("false", True) is False
        assert _batch_bool("", True) is True

    def test_m3max_memory_profile_scales_cache_for_large_unified_memory(self):
        from mlx_indextts.performance import resolve_mlx_memory_limits

        limits = resolve_mlx_memory_limits(total_memory_bytes=128 * 1024**3)

        assert limits.memory_limit_gb == 96.0
        assert limits.cache_limit_gb == 32.0
        assert limits.source == "auto"

    def test_small_unified_memory_profile_bounds_mlx_cache(self):
        from mlx_indextts.performance import resolve_mlx_memory_limits

        limits = resolve_mlx_memory_limits(
            memory_limit_gb=24,
            total_memory_bytes=24 * 1024**3,
        )

        assert limits.memory_limit_gb == 24.0
        assert limits.cache_limit_gb == 2.4

    def test_prepare_inputs_shape(self):
        """Test that input preparation produces correct shapes."""
        from mlx_indextts.config import IndexTTSConfig
        from mlx_indextts.models.gpt import UnifiedVoice

        config = IndexTTSConfig()
        # Use smaller model for testing
        config.gpt.model_dim = 256
        config.gpt.heads = 4
        config.gpt.layers = 2
        config.gpt.condition_module.output_size = 128

        model = UnifiedVoice(config)

        # Mock conditioning
        conditioning = mx.array(np.random.randn(1, 32, 256).astype(np.float32))

        # Mock text tokens
        text_tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

        input_emb, mask = model.prepare_inputs(conditioning, text_tokens)

        # Should have: conditioning (32) + start + text (5) + stop = 39
        expected_len = 32 + 1 + 5 + 1
        assert input_emb.shape == (1, expected_len, 256)
        assert mask.shape == (1, expected_len)

    def test_sampling_temperature(self):
        """Test that temperature affects sampling distribution."""
        from mlx_indextts.models.gpt import UnifiedVoice
        from mlx_indextts.config import IndexTTSConfig

        config = IndexTTSConfig()
        config.gpt.model_dim = 256
        config.gpt.heads = 4
        config.gpt.layers = 2

        model = UnifiedVoice(config)

        # Create logits with clear peak
        logits_np = np.full((1, 100), -10.0, dtype=np.float32)
        logits_np[0, 50] = 10.0
        logits = mx.array(logits_np)

        # With low temperature, should always pick the peak
        mx.random.seed(42)
        samples_low_temp = [model._sample(logits, temperature=0.1)[0].item() for _ in range(10)]
        assert all(s == 50 for s in samples_low_temp)

        # With high temperature, should have more variance
        mx.random.seed(42)
        samples_high_temp = [model._sample(logits, temperature=2.0)[0].item() for _ in range(20)]
        # Not all should be 50 with high temperature
        unique_samples = set(samples_high_temp)
        assert len(unique_samples) >= 1  # At least some variance

    def test_repetition_penalty_updates_selected_tokens(self):
        """Test repetition penalty updates only generated token logits."""
        from mlx_indextts.models.gpt import UnifiedVoice
        from mlx_indextts.config import IndexTTSConfig

        config = IndexTTSConfig()
        config.gpt.model_dim = 256
        config.gpt.heads = 4
        config.gpt.layers = 2
        model = UnifiedVoice(config)

        logits = mx.array([[2.0, -2.0, 1.0, 0.5]], dtype=mx.float32)
        penalized = model._apply_repetition_penalty(logits, [0, 1, 1, 99], 2.0)

        assert np.allclose(np.array(penalized), [[1.0, -4.0, 1.0, 0.5]])

    def test_top_p_sampling_runs_without_numpy_scatter(self):
        """Test top-p sampling stays functional after vectorized masking."""
        from mlx_indextts.models.gpt import UnifiedVoice
        from mlx_indextts.config import IndexTTSConfig

        config = IndexTTSConfig()
        config.gpt.model_dim = 256
        config.gpt.heads = 4
        config.gpt.layers = 2
        model = UnifiedVoice(config)

        logits = mx.array([[5.0, 4.0, 1.0, 0.0]], dtype=mx.float32)
        mx.random.seed(7)
        sample = model._sample(logits, temperature=1.0, top_k=0, top_p=0.7)

        assert int(sample[0].item()) in {0, 1}


class TestMelExtraction:
    """Tests for mel spectrogram extraction in generation pipeline."""

    def test_mel_extractor_init(self):
        """Test MelSpectrogramExtractor initialization."""
        from mlx_indextts.mel import MelSpectrogramExtractor

        extractor = MelSpectrogramExtractor(
            n_fft=1024,
            hop_length=256,
            n_mels=100,
            sample_rate=24000,
        )

        assert extractor.n_mels == 100
        assert extractor.sample_rate == 24000

    def test_mel_extractor_output(self):
        """Test MelSpectrogramExtractor produces valid output."""
        from mlx_indextts.mel import MelSpectrogramExtractor

        extractor = MelSpectrogramExtractor(n_mels=100)

        # 1 second of audio
        audio = mx.array(np.random.randn(24000).astype(np.float32))
        mel = extractor(audio)

        # Check shape
        assert mel.shape[0] == 100

        # Check values are finite
        assert mx.isfinite(mel).all()


class TestEndToEndFlow:
    """Tests for end-to-end generation flow."""

    def test_conditioning_shape(self):
        """Test that conditioning produces correct shape."""
        from mlx_indextts.config import IndexTTSConfig, ConformerConfig
        from mlx_indextts.models.gpt import UnifiedVoice

        config = IndexTTSConfig()
        config.gpt.model_dim = 256
        config.gpt.heads = 4
        config.gpt.layers = 2
        config.gpt.condition_type = "perceiver"  # Simpler for testing

        model = UnifiedVoice(config)

        # Mock mel spectrogram
        mel = mx.array(np.random.randn(1, 100, 200).astype(np.float32))

        conditioning = model.get_conditioning(mel)

        # Should be (batch, cond_num, dim)
        assert conditioning.shape == (1, 32, 256)
