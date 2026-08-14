"""Conditional Flow Matching (CFM) for S2Mel."""

from __future__ import annotations

import os
import mlx.core as mx
import mlx.nn as nn

from mlx_indextts.models.s2mel.dit import DiT
from mlx_indextts.performance import schedule_mlx_eval


def _enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _repeat_batch_twice(value: mx.array) -> mx.array:
    """Return ``[value, value]`` as a broadcast/reshape view when possible."""
    expanded = mx.broadcast_to(
        value[None, ...],
        (2, *value.shape),
    )
    return expanded.reshape(
        (2 * value.shape[0], *value.shape[1:]),
    )


class CFM(nn.Module):
    """Conditional Flow Matching model using a DiT Euler estimator."""

    def __init__(
        self,
        in_channels: int = 80,
        hidden_dim: int = 512,
        num_heads: int = 8,
        depth: int = 13,
        content_dim: int = 512,
        style_dim: int = 192,
        class_dropout_prob: float = 0.1,
        long_skip_connection: bool = True,
        uvit_skip_connection: bool = True,
        time_as_token: bool = False,
        style_as_token: bool = False,
        style_condition: bool = True,
        final_layer_type: str = "wavenet",
        wavenet_hidden_dim: int = 512,
        wavenet_num_layers: int = 8,
        wavenet_kernel_size: int = 5,
        wavenet_dilation_rate: int = 1,
        wavenet_p_dropout: float = 0.2,
        zero_prompt_speech_token: bool = False,
    ):
        super().__init__()

        self.sigma_min = 1e-6
        self.in_channels = in_channels
        self.zero_prompt_speech_token = zero_prompt_speech_token
        self.estimator = DiT(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=depth,
            in_channels=in_channels,
            content_dim=content_dim,
            style_dim=style_dim,
            class_dropout_prob=class_dropout_prob,
            long_skip_connection=long_skip_connection,
            uvit_skip_connection=uvit_skip_connection,
            time_as_token=time_as_token,
            style_as_token=style_as_token,
            style_condition=style_condition,
            final_layer_type=final_layer_type,
            wavenet_hidden_dim=wavenet_hidden_dim,
            wavenet_num_layers=wavenet_num_layers,
            wavenet_kernel_size=wavenet_kernel_size,
            wavenet_dilation_rate=wavenet_dilation_rate,
            wavenet_p_dropout=wavenet_p_dropout,
        )
        self._compiled_estimator = None
        self._compile_attempted = False

    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_length: int,
    ):
        """Setup KV caches for the estimator."""
        self.estimator.setup_caches(max_batch_size, max_seq_length)

    def _call_estimator(
        self,
        x: mx.array,
        prompt_x: mx.array,
        x_lens: mx.array,
        timestep: mx.array,
        style: mx.array,
        condition: mx.array,
    ) -> mx.array:
        """Run a lazily compiled DiT estimator with an eager fallback."""
        if not self._compile_attempted:
            self._compile_attempted = True
            compiler = getattr(mx, "compile", None)
            if (
                compiler is not None
                and _enabled("MLX_INDEXTTS_COMPILE_CFM", True)
            ):
                try:
                    self._compiled_estimator = compiler(
                        self.estimator,
                        shapeless=True,
                    )
                except TypeError:
                    # Older MLX releases expose compile without shapeless.
                    try:
                        self._compiled_estimator = compiler(self.estimator)
                    except Exception:
                        self._compiled_estimator = None
                except Exception:
                    self._compiled_estimator = None

        if self._compiled_estimator is not None:
            try:
                return self._compiled_estimator(
                    x,
                    prompt_x,
                    x_lens,
                    timestep,
                    style,
                    condition,
                )
            except Exception:
                # Compilation support is shape/version dependent. Disable it
                # after the first failure and preserve the eager inference path.
                self._compiled_estimator = None

        return self.estimator(
            x,
            prompt_x,
            x_lens,
            timestep,
            style,
            condition,
        )

    def inference(
        self,
        mu: mx.array,
        x_lens: mx.array,
        prompt: mx.array,
        style: mx.array,
        f0: mx.array | None,
        n_timesteps: int,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.5,
    ) -> mx.array:
        """Generate a mel spectrogram with fixed-step Euler integration."""
        batch_size, sequence_length, _ = mu.shape
        z = (
            mx.random.normal(
                (batch_size, self.in_channels, sequence_length)
            )
            * temperature
        )
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        return self.solve_euler(
            z,
            x_lens,
            prompt,
            mu,
            style,
            f0,
            t_span,
            inference_cfg_rate,
        )

    def solve_euler(
        self,
        x: mx.array,
        x_lens: mx.array,
        prompt: mx.array,
        mu: mx.array,
        style: mx.array,
        f0: mx.array | None,
        t_span: mx.array,
        inference_cfg_rate: float = 0.5,
    ) -> mx.array:
        """Solve the CFM ODE while bounding graph and allocation growth."""
        del f0
        sequence_length = x.shape[2]
        prompt_len = prompt.shape[-1]

        if prompt_len > sequence_length:
            raise ValueError("prompt length cannot exceed CFM sequence length")

        prompt_tail_zeros = mx.zeros(
            (
                x.shape[0],
                x.shape[1],
                sequence_length - prompt_len,
            ),
            dtype=prompt.dtype,
        )
        prompt_x = mx.concatenate(
            [prompt[:, :, :prompt_len], prompt_tail_zeros],
            axis=2,
        )

        # A broadcastable mask is cheaper than rebuilding the zero prompt prefix
        # with a full concatenate after every Euler update. The elementwise
        # multiply can fuse into the surrounding lazy graph.
        state_mask = mx.concatenate(
            [
                mx.zeros((prompt_len,), dtype=x.dtype),
                mx.ones(
                    (sequence_length - prompt_len,),
                    dtype=x.dtype,
                ),
            ]
        )[None, None, :]
        x = x * state_mask

        if self.zero_prompt_speech_token:
            content_mask = state_mask.transpose(0, 2, 1)
            mu = mu * content_mask

        estimator_call = getattr(self, "_call_estimator", None)
        if estimator_call is None:
            estimator_call = self.estimator

        if inference_cfg_rate > 0:
            stacked_prompt_x = mx.concatenate(
                [prompt_x, mx.zeros_like(prompt_x)],
                axis=0,
            )
            stacked_style = mx.concatenate(
                [style, mx.zeros_like(style)],
                axis=0,
            )
            stacked_mu = mx.concatenate(
                [mu, mx.zeros_like(mu)],
                axis=0,
            )
            stacked_x_lens = _repeat_batch_twice(x_lens)
            schedule_mlx_eval(
                stacked_prompt_x,
                stacked_style,
                stacked_mu,
                stacked_x_lens,
            )
        else:
            schedule_mlx_eval(prompt_x)

        for step in range(1, len(t_span)):
            timestep_value = t_span[step - 1]
            dt = t_span[step] - timestep_value

            if inference_cfg_rate > 0:
                stacked_x = _repeat_batch_twice(x)
                stacked_t = mx.broadcast_to(
                    timestep_value,
                    (stacked_x.shape[0],),
                )
                stacked_dphi_dt = estimator_call(
                    stacked_x,
                    stacked_prompt_x,
                    stacked_x_lens,
                    stacked_t,
                    stacked_style,
                    stacked_mu,
                )
                dphi_dt, cfg_dphi_dt = mx.split(
                    stacked_dphi_dt,
                    2,
                    axis=0,
                )
                dphi_dt = (
                    (1.0 + inference_cfg_rate) * dphi_dt
                    - inference_cfg_rate * cfg_dphi_dt
                )
            else:
                timestep = mx.broadcast_to(
                    timestep_value,
                    (x.shape[0],),
                )
                dphi_dt = estimator_call(
                    x,
                    prompt_x,
                    x_lens,
                    timestep,
                    style,
                    mu,
                )

            x = (x + dt * dphi_dt) * state_mask

            # Submit each step without a host barrier. Materializing the current
            # state prevents the lazy graph from growing across all Euler steps;
            # the next dependent step is naturally ordered by MLX.
            schedule_mlx_eval(x)

        return x

    def __call__(
        self,
        x1: mx.array,
        x_lens: mx.array,
        prompt_lens: mx.array,
        mu: mx.array,
        style: mx.array,
    ):
        """Compute training loss (not implemented for inference)."""
        raise NotImplementedError("Training not implemented in MLX version")


def create_cfm_from_config(config) -> CFM:
    """Create a CFM model from the S2Mel configuration section."""
    dit_config = config.get("DiT", {})
    wavenet_config = config.get("wavenet", {})
    style_config = config.get("style_encoder", {})

    return CFM(
        in_channels=dit_config.get("in_channels", 80),
        hidden_dim=dit_config.get("hidden_dim", 512),
        num_heads=dit_config.get("num_heads", 8),
        depth=dit_config.get("depth", 13),
        content_dim=dit_config.get("content_dim", 512),
        style_dim=style_config.get("dim", 192),
        class_dropout_prob=dit_config.get(
            "class_dropout_prob",
            0.1,
        ),
        long_skip_connection=dit_config.get(
            "long_skip_connection",
            True,
        ),
        uvit_skip_connection=dit_config.get(
            "uvit_skip_connection",
            True,
        ),
        time_as_token=dit_config.get("time_as_token", False),
        style_as_token=dit_config.get("style_as_token", False),
        style_condition=dit_config.get("style_condition", True),
        final_layer_type=dit_config.get(
            "final_layer_type",
            "wavenet",
        ),
        wavenet_hidden_dim=wavenet_config.get(
            "hidden_dim",
            512,
        ),
        wavenet_num_layers=wavenet_config.get(
            "num_layers",
            8,
        ),
        wavenet_kernel_size=wavenet_config.get(
            "kernel_size",
            5,
        ),
        wavenet_dilation_rate=wavenet_config.get(
            "dilation_rate",
            1,
        ),
        wavenet_p_dropout=wavenet_config.get(
            "p_dropout",
            0.2,
        ),
        zero_prompt_speech_token=dit_config.get(
            "zero_prompt_speech_token",
            False,
        ),
    )
