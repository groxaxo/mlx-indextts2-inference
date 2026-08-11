"""Tests for the native MLX IndexTTS 2.5 semantic codec."""

import mlx.core as mx
import numpy as np


def _tiny_codec():
    from mlx_indextts.models.codec_v25 import EnhancedCodecV25

    return EnhancedCodecV25(
        codebook_size=32,
        hidden_size=16,
        codebook_dim=4,
        vocos_dim=8,
        vocos_intermediate_dim=16,
        vocos_num_layers=2,
        num_quantizers=1,
        downsample_scale=2,
    )


def test_quantize_and_decode_shapes_match_official_contract():
    codec = _tiny_codec()
    features = mx.array(np.random.default_rng(7).standard_normal((2, 10, 16)).astype(np.float32))

    codes, quantized = codec.quantize(features)
    reconstructed = codec.decode(codes)
    mx.eval(codes, quantized, reconstructed)

    assert codes.shape == (2, 5)
    assert codes.dtype == mx.int32
    assert quantized.shape == (2, 5, 16)
    assert reconstructed.shape == (2, 10, 16)
    assert bool(mx.isfinite(reconstructed).all())


def test_decode_accepts_explicit_quantizer_axis():
    codec = _tiny_codec()
    codes = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)

    plain = codec.decode(codes)
    explicit = codec.decode(codes[None, ...])
    mx.eval(plain, explicit)

    np.testing.assert_allclose(np.asarray(plain), np.asarray(explicit), rtol=1e-6, atol=1e-6)


def test_codec_parameter_tree_matches_checkpoint_names():
    from mlx.utils import tree_flatten

    codec = _tiny_codec()
    keys = {key for key, _ in tree_flatten(codec.parameters())}

    assert "down.weight" in keys
    assert "encoder.0.embed.weight" in keys
    assert "encoder.0.convnext.0.dwconv.weight" in keys
    assert "decoder.1.weight" in keys
    assert "quantizer.quantizers.0.in_project.weight" in keys
    assert "quantizer.quantizers.0.codebook.weight" in keys


def test_vector_quantizer_selects_nearest_l2_normalized_code():
    from mlx_indextts.models.codec_v25 import FactorizedVectorQuantizer

    quantizer = FactorizedVectorQuantizer(
        input_dim=2,
        codebook_size=3,
        codebook_dim=2,
        use_l2_normalize=True,
    )
    quantizer.codebook.weight = mx.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    indices, vectors = quantizer.nearest_code(mx.array([[[3.0, 0.1], [-0.2, 4.0]]]))
    mx.eval(indices, vectors)

    assert np.asarray(indices).tolist() == [[0, 1]]
    np.testing.assert_allclose(np.asarray(vectors), [[[1.0, 0.0], [0.0, 1.0]]])
