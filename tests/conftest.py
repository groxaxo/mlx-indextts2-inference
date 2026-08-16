"""Pytest configuration and fixtures."""

import pytest
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


def require_mlx():
    if mx is None:
        pytest.skip("MLX is only available on Apple Silicon")
    return mx


@pytest.fixture
def sample_audio():
    """Generate sample audio data."""
    mlx = require_mlx()
    # 1 second of audio at 24kHz
    return mlx.array(np.random.randn(24000).astype(np.float32))


@pytest.fixture
def sample_mel():
    """Generate sample mel spectrogram."""
    mlx = require_mlx()
    # (batch, n_mels, time)
    return mlx.array(np.random.randn(1, 100, 200).astype(np.float32))


@pytest.fixture
def sample_text_tokens():
    """Generate sample text tokens."""
    mlx = require_mlx()
    return mlx.array([[100, 200, 300, 400, 500]], dtype=mlx.int32)


@pytest.fixture
def small_config():
    """Create a small config for testing."""
    require_mlx()
    from mlx_indextts.config import IndexTTSConfig, GPTConfig, ConformerConfig

    config = IndexTTSConfig()
    config.gpt.model_dim = 256
    config.gpt.heads = 4
    config.gpt.layers = 2
    config.gpt.max_mel_tokens = 100
    config.gpt.max_text_tokens = 50
    config.gpt.condition_module = ConformerConfig(
        output_size=128,
        attention_heads=4,
        num_blocks=2,
    )
    config.bigvgan.gpt_dim = 256
    config.bigvgan.upsample_initial_channel = 256

    return config
