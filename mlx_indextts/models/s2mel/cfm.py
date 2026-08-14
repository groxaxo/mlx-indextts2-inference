"""Conditional Flow Matching (CFM) for S2Mel.

This implements the flow matching diffusion model used for mel generation.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_indextts.models.s2mel.dit import DiT


class CFM(nn.Module):
    """Conditional Flow Matching model.

    Uses DiT as the estimator to denoise mel spectrograms
    via an ODE solver (Euler method).
    """

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
        final_layer_type: str = 'wavenet',
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

        # Build DiT estimator
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

    def setup_caches(self, max_batch_size: int, max_seq_length: int):
        """Setup KV caches for the estimator."""
        self.estimator.setup_caches(max_batch_size, max_seq_length)

    def inference(
        self,
        mu: mx.array,
        x_lens: mx.array,
        prompt: mx.array,
        style: mx.array,
        f0: Optional[mx.array],
        n_timesteps: int,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.5,
    ) -> mx.array:
        """Forward diffusion inference.

        Args:
            mu: Semantic conditioning (batch, seq_len, content_dim)
            x_lens: Target lengths (batch,) or single value
            prompt: Reference mel spectrogram (batch, in_channels, prompt_len)
            style: Style embedding (batch, style_dim)
            f0: F0 conditioning (not used)
            n_timesteps: Number of diffusion steps
            temperature: Noise temperature
            inference_cfg_rate: Classifier-free guidance rate

        Returns:
            Generated mel spectrogram (batch, in_channels, seq_len)
        """
        B, T, _ = mu.shape

        # Initialize with random noise
        z = mx.random.normal((B, self.in_channels, T)) * temperature

        # Create timestep schedule
        t_span = mx.linspace(0, 1, n_timesteps + 1)

        # Solve ODE with Euler method
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(
        self,
        x: mx.array,
        x_lens: mx.array,
        prompt: mx.array,
        mu: mx.array,
        style: mx.array,
        f0: Optional[mx.array],
        t_span: mx.array,
        inference_cfg_rate: float = 0.5,
    ) -> mx.array:
        """Fixed Euler solver for ODE.

        Args:
            x: Initial noise (batch, in_channels, seq_len)
            x_lens: Target lengths
            prompt: Reference mel (batch, in_channels, prompt_len)
            mu: Semantic conditioning (batch, seq_len, content_dim)
            style: Style embedding (batch, style_dim)
            f0: F0 conditioning (not used)
            t_span: Timestep schedule (n_timesteps + 1,)
            inference_cfg_rate: CFG rate

        Returns:
            Denoised mel spectrogram (batch, in_channels, seq_len)
        """
        T = x.shape[2]
        prompt_len = prompt.shape[-1]

        # These prompt-zero tensors are invariant across every Euler step. Reuse
        # them instead of allocating a new prefix on each diffusion iteration.
        zero_prompt_region = mx.zeros(
            (x.shape[0], x.shape[1], prompt_len),
            dtype=x.dtype,
        )
        prompt_tail_zeros = mx.zeros(
            (x.shape[0], x.shape[1], T - prompt_len),
            dtype=prompt.dtype,
        )
        prompt_x = mx.concatenate(
            [prompt[:, :, :prompt_len], prompt_tail_zeros],
            axis=2,
        )
        x = mx.concatenate([zero_prompt_region, x[:, :, prompt_len:]], axis=2)

        if self.zero_prompt_speech_token:
            zero_mu_prefix = mx.zeros(
                (mu.shape[0], prompt_len, mu.shape[2]),
                dtype=mu.dtype,
            )
            mu = mx.concatenate([zero_mu_prefix, mu[:, prompt_len:, :]], axis=1)

        # CFG null inputs are invariant across Euler steps. Constructing these
        # large tensors inside every step adds avoidable unified-memory traffic.
        if inference_cfg_rate > 0:
            null_prompt_x = mx.zeros_like(prompt_x)
            null_style = mx.zeros_like(style)
            null_mu = mx.zeros_like(mu)
            stacked_prompt_x = mx.concatenate([prompt_x, null_prompt_x], axis=0)
            stacked_style = mx.concatenate([style, null_style], axis=0)
            stacked_mu = mx.concatenate([mu, null_mu], axis=0)
            stacked_x_lens = mx.concatenate([x_lens, x_lens], axis=0)
            mx.eval(stacked_prompt_x, stacked_style, stacked_mu, stacked_x_lens)
        else:
            mx.eval(prompt_x)

        for step in range(1, len(t_span)):
            # Use the schedule directly rather than carrying a scalar dependency
            # chain from one Euler iteration to the next.
            t = t_span[step - 1]
            dt = t_span[step] - t

            if inference_cfg_rate > 0:
                # CFG: stack original and null inputs
                stacked_x = mx.concatenate([x, x], axis=0)
                stacked_t = mx.broadcast_to(t, (stacked_x.shape[0],))

                # Single forward pass for both
                stacked_dphi_dt = self.estimator(
                    stacked_x,
                    stacked_prompt_x,
                    stacked_x_lens,
                    stacked_t,
                    stacked_style,
                    stacked_mu,
                )

                # Split and apply CFG
                dphi_dt, cfg_dphi_dt = mx.split(stacked_dphi_dt, 2, axis=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                timestep = mx.broadcast_to(t, (x.shape[0],))
                dphi_dt = self.estimator(
                    x,
                    prompt_x,
                    x_lens,
                    timestep,
                    style,
                    mu,
                )

            # Euler step
            x = x + dt * dphi_dt

            # Keep prompt region zero with the single reusable zero prefix.
            x = mx.concatenate([zero_prompt_region, x[:, :, prompt_len:]], axis=2)

            # Evaluate for MLX lazy execution
            mx.eval(x)

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
    """Create CFM model from config dictionary.

    Args:
        config: S2Mel config section from IndexTTS config

    Returns:
        CFM model
    """
    dit_config = config.get('DiT', {})
    wavenet_config = config.get('wavenet', {})
    style_config = config.get('style_encoder', {})

    return CFM(
        in_channels=dit_config.get('in_channels', 80),
        hidden_dim=dit_config.get('hidden_dim', 512),
        num_heads=dit_config.get('num_heads', 8),
        depth=dit_config.get('depth', 13),
        content_dim=dit_config.get('content_dim', 512),
        style_dim=style_config.get('dim', 192),
        class_dropout_prob=dit_config.get('class_dropout_prob', 0.1),
        long_skip_connection=dit_config.get('long_skip_connection', True),
        uvit_skip_connection=dit_config.get('uvit_skip_connection', True),
        time_as_token=dit_config.get('time_as_token', False),
        style_as_token=dit_config.get('style_as_token', False),
        style_condition=dit_config.get('style_condition', True),
        final_layer_type=dit_config.get('final_layer_type', 'wavenet'),
        wavenet_hidden_dim=wavenet_config.get('hidden_dim', 512),
        wavenet_num_layers=wavenet_config.get('num_layers', 8),
        wavenet_kernel_size=wavenet_config.get('kernel_size', 5),
        wavenet_dilation_rate=wavenet_config.get('dilation_rate', 1),
        wavenet_p_dropout=wavenet_config.get('p_dropout', 0.2),
        zero_prompt_speech_token=dit_config.get('zero_prompt_speech_token', False),
    )
