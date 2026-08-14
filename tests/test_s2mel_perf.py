"""Regression tests for S2Mel inference hot-path optimizations."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np


def test_dit_uses_broadcastable_key_mask_instead_of_quadratic_mask():
    from mlx_indextts.models.s2mel.dit import DiT

    captured = {}
    model = DiT(
        hidden_dim=16,
        num_heads=2,
        depth=1,
        in_channels=4,
        content_dim=8,
        style_dim=4,
        long_skip_connection=False,
        uvit_skip_connection=False,
        style_condition=False,
        final_layer_type="mlp",
    )

    class CaptureTransformer:
        def __call__(self, x, c, input_pos, mask):
            captured["mask_shape"] = tuple(mask.shape)
            captured["mask"] = mask
            return x

    model.transformer = CaptureTransformer()
    x = mx.zeros((1, 4, 4), dtype=mx.float32)
    prompt = mx.zeros_like(x)
    cond = mx.zeros((1, 4, 8), dtype=mx.float32)
    style = mx.zeros((1, 4), dtype=mx.float32)
    output = model(
        x,
        prompt,
        mx.array([3], dtype=mx.int32),
        mx.array([0.5], dtype=mx.float32),
        style,
        cond,
    )
    mx.eval(output, captured["mask"])

    assert captured["mask_shape"] == (1, 1, 1, 4)
    mask = np.asarray(captured["mask"])[0, 0, 0]
    np.testing.assert_array_equal(mask[:3], 0.0)
    assert np.isneginf(mask[3])


def test_cfm_cfg_stacks_lengths_and_timesteps_for_full_batch():
    from mlx_indextts.models.s2mel.cfm import CFM

    calls = []

    def estimator(x, prompt_x, x_lens, t, style, cond):
        batch = x.shape[0]
        assert prompt_x.shape[0] == batch
        assert x_lens.shape[0] == batch
        assert t.shape[0] == batch
        assert style.shape[0] == batch
        assert cond.shape[0] == batch
        calls.append(batch)
        return mx.zeros_like(x)

    runtime = SimpleNamespace(
        zero_prompt_speech_token=False,
        estimator=estimator,
    )
    x = mx.arange(2 * 4 * 6, dtype=mx.float32).reshape(2, 4, 6)
    original = np.asarray(x)
    prompt = mx.zeros((2, 4, 2), dtype=mx.float32)
    mu = mx.zeros((2, 6, 8), dtype=mx.float32)
    style = mx.zeros((2, 3), dtype=mx.float32)
    result = CFM.solve_euler(
        runtime,
        x,
        mx.array([6, 6], dtype=mx.int32),
        prompt,
        mu,
        style,
        None,
        mx.linspace(0.0, 1.0, 3),
        inference_cfg_rate=0.7,
    )
    mx.eval(result)

    assert calls == [4, 4]
    output = np.asarray(result)
    np.testing.assert_array_equal(output[:, :, :2], 0.0)
    np.testing.assert_array_equal(output[:, :, 2:], original[:, :, 2:])
