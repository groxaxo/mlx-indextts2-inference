"""Regression tests for the remaining MLX inference hot paths."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_indextts.models.sampling import (
    RepetitionPenaltyState,
    RepetitionStateCache,
    resolve_repetition_state,
    schedule_decode_outputs,
)


def test_plain_list_repetition_history_is_mirrored_incrementally():
    class Owner:
        pass

    owner = Owner()
    history = [2, 3]
    first = resolve_repetition_state(owner, history, vocab_size=8)

    assert isinstance(first, RepetitionPenaltyState)
    assert list(first) == [2, 3]

    history.append(5)
    second = resolve_repetition_state(owner, history, vocab_size=8)
    mx.eval(second.seen)

    assert second is first
    assert list(second) == [2, 3, 5]
    np.testing.assert_array_equal(
        np.asarray(second.seen),
        [[False, False, True, True, False, True, False, False]],
    )


def test_repetition_cache_resets_when_history_shrinks():
    cache = RepetitionStateCache(vocab_size=6)
    history = [1, 2, 3]
    first = cache.resolve(history)

    history.clear()
    history.append(4)
    second = cache.resolve(history)
    mx.eval(second.seen)

    assert second is not first
    assert list(second) == [4]
    np.testing.assert_array_equal(
        np.asarray(second.seen),
        [[False, False, False, False, True, False]],
    )


def test_decode_output_scheduler_accepts_token_and_cache_tree():
    token = mx.array([3], dtype=mx.int32)
    cache = [(mx.ones((1, 2, 4)), mx.zeros((1, 2, 4)))]

    schedule_decode_outputs(token, cache)
    assert int(token[0].item()) == 3
    assert np.asarray(cache[0][0]).shape == (1, 2, 4)


def test_v25_single_query_mask_is_only_the_padding_mask():
    from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25

    padding = mx.array([[0, 1, 1, 1, 1]], dtype=mx.int32)
    mask = UnifiedVoiceV25.generation_attention_mask(
        padding,
        query_len=1,
        key_len=5,
    )
    mx.eval(mask)

    assert mask.shape == (1, 1, 1, 5)
    values = np.asarray(mask)[0, 0, 0]
    assert values[0] < -1e8
    np.testing.assert_array_equal(values[1:], 0.0)


def test_repeat_batch_twice_preserves_cfg_order():
    from mlx_indextts.models.s2mel.cfm import _repeat_batch_twice

    values = mx.array([[1.0], [2.0]], dtype=mx.float32)
    repeated = _repeat_batch_twice(values)
    mx.eval(repeated)

    np.testing.assert_array_equal(
        np.asarray(repeated),
        [[1.0], [2.0], [1.0], [2.0]],
    )


def test_cfm_estimator_eager_fallback_remains_callable():
    from mlx_indextts.models.s2mel.cfm import CFM

    calls = []

    def estimator(*args):
        calls.append(tuple(arg.shape for arg in args))
        return mx.zeros_like(args[0])

    runtime = CFM.__new__(CFM)
    runtime.estimator = estimator
    runtime._compiled_estimator = None
    runtime._compile_attempted = True

    x = mx.zeros((1, 4, 6))
    result = runtime._call_estimator(
        x,
        x,
        mx.array([6]),
        mx.array([0.5]),
        mx.zeros((1, 3)),
        mx.zeros((1, 6, 8)),
    )
    mx.eval(result)

    assert result.shape == x.shape
    assert len(calls) == 1


def test_nlc_antialias_activation_matches_historical_ncl_layout():
    from mlx_indextts.models.activations import Activation1d, SnakeBeta

    ncl_activation = Activation1d(
        SnakeBeta(3, alpha_logscale=True, channel_axis=1),
        channel_axis=1,
    )
    nlc_activation = Activation1d(
        SnakeBeta(3, alpha_logscale=True, channel_axis=-1),
        channel_axis=-1,
    )

    alpha = mx.array([0.1, -0.2, 0.3], dtype=mx.float32)
    beta = mx.array([-0.1, 0.2, -0.3], dtype=mx.float32)
    ncl_activation.act.alpha = alpha
    ncl_activation.act.beta = beta
    nlc_activation.act.alpha = alpha
    nlc_activation.act.beta = beta

    values = mx.arange(2 * 3 * 8, dtype=mx.float32).reshape(2, 3, 8)
    values = values / 10.0
    ncl_result = ncl_activation(values)
    nlc_result = nlc_activation(values.transpose(0, 2, 1))
    mx.eval(ncl_result, nlc_result)

    np.testing.assert_allclose(
        np.asarray(ncl_result),
        np.asarray(nlc_result.transpose(0, 2, 1)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_tiny_bigvgan_keeps_public_ncl_contract():
    from mlx_indextts.models.bigvgan_v2 import (
        BigVGANV2,
        BigVGANV2Config,
    )

    config = BigVGANV2Config(
        num_mels=4,
        upsample_rates=[2],
        upsample_kernel_sizes=[4],
        upsample_initial_channel=8,
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1]],
        activation="snakebeta",
    )
    model = BigVGANV2(config)
    output = model(mx.zeros((1, 4, 3), dtype=mx.float32))
    mx.eval(output)

    assert output.shape == (1, 1, 6)
    assert np.isfinite(np.asarray(output)).all()


def test_public_amp_block_preserves_historical_ncl_contract():
    from mlx_indextts.models.bigvgan_v2 import AMPBlock1

    block = AMPBlock1(
        channels=4,
        kernel_size=3,
        dilations=[1],
        activation="snakebeta",
    )
    values = mx.zeros((1, 4, 8), dtype=mx.float32)
    output = block(values)
    mx.eval(output)

    assert output.shape == values.shape
    assert np.isfinite(np.asarray(output)).all()
