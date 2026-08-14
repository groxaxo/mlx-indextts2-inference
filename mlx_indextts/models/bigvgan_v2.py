"""BigVGAN v2 vocoder for IndexTTS 2.x.

The public API remains NCL, while the full convolutional hot path stays NLC to
match MLX Conv1d/ConvTranspose1d and avoid repeated layout transposes.
"""

from __future__ import annotations

from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn

from mlx_indextts.models.activations import (
    Activation1d,
    Snake,
    SnakeBeta,
)


@dataclass
class BigVGANV2Config:
    """Configuration matching nvidia/bigvgan_v2_22khz_80band_256x."""

    num_mels: int = 80
    upsample_rates: list[int] | None = None
    upsample_kernel_sizes: list[int] | None = None
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: list[int] | None = None
    resblock_dilation_sizes: list[list[int]] | None = None
    activation: str = "snakebeta"
    snake_logscale: bool = True
    use_tanh_at_final: bool = False
    use_bias_at_final: bool = False
    resblock: str = "1"

    def __post_init__(self):
        if self.upsample_rates is None:
            self.upsample_rates = [4, 4, 2, 2, 2, 2]
        if self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [8, 8, 4, 4, 4, 4]
        if self.resblock_kernel_sizes is None:
            self.resblock_kernel_sizes = [3, 7, 11]
        if self.resblock_dilation_sizes is None:
            self.resblock_dilation_sizes = [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5],
            ]

    @classmethod
    def from_dict(cls, values: dict) -> "BigVGANV2Config":
        return cls(
            num_mels=values.get("num_mels", 80),
            upsample_rates=values.get("upsample_rates"),
            upsample_kernel_sizes=values.get(
                "upsample_kernel_sizes"
            ),
            upsample_initial_channel=values.get(
                "upsample_initial_channel",
                1536,
            ),
            resblock_kernel_sizes=values.get(
                "resblock_kernel_sizes"
            ),
            resblock_dilation_sizes=values.get(
                "resblock_dilation_sizes"
            ),
            activation=values.get("activation", "snakebeta"),
            snake_logscale=values.get("snake_logscale", True),
            use_tanh_at_final=values.get(
                "use_tanh_at_final",
                False,
            ),
            use_bias_at_final=values.get(
                "use_bias_at_final",
                False,
            ),
            resblock=values.get("resblock", "1"),
        )


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate same-length convolution padding."""
    return int((kernel_size * dilation - dilation) / 2)


def _periodic_activation(
    name: str,
    channels: int,
    alpha_logscale: bool,
) -> Activation1d:
    if name == "snakebeta":
        activation = SnakeBeta(
            channels,
            alpha_logscale,
            channel_axis=-1,
        )
    elif name == "snake":
        activation = Snake(
            channels,
            alpha_logscale,
            channel_axis=-1,
        )
    else:
        raise ValueError(f"Unknown activation: {name}")
    return Activation1d(activation, channel_axis=-1)


class _AMPBlock1NLC(nn.Module):
    """Type-1 anti-aliased residual block operating entirely in NLC."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        activation: str = "snakebeta",
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.channels = channels
        dilation_values = [1, 3, 5] if dilations is None else dilations

        self.convs1 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                padding=get_padding(kernel_size, dilation),
            )
            for dilation in dilation_values
        ]
        self.convs2 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
            for _ in dilation_values
        ]

        self.activations = [
            _periodic_activation(
                activation,
                channels,
                alpha_logscale,
            )
            for _ in range(len(self.convs1) + len(self.convs2))
        ]

    def __call__(self, x: mx.array) -> mx.array:
        acts1 = self.activations[::2]
        acts2 = self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(
            self.convs1,
            self.convs2,
            acts1,
            acts2,
        ):
            residual = conv2(act2(conv1(act1(x))))
            x = x + residual
        return x


class AMPBlock1(_AMPBlock1NLC):
    """Compatibility wrapper preserving the historical NCL block contract."""

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.transpose(0, 2, 1)).transpose(0, 2, 1)


class _AMPBlock2NLC(nn.Module):
    """Type-2 anti-aliased residual block operating entirely in NLC."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        activation: str = "snakebeta",
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.channels = channels
        dilation_values = [1, 3, 5] if dilations is None else dilations

        self.convs = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                padding=get_padding(kernel_size, dilation),
            )
            for dilation in dilation_values
        ]
        self.activations = [
            _periodic_activation(
                activation,
                channels,
                alpha_logscale,
            )
            for _ in self.convs
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for conv, activation in zip(
            self.convs,
            self.activations,
        ):
            x = x + conv(activation(x))
        return x


class AMPBlock2(_AMPBlock2NLC):
    """Compatibility wrapper preserving the historical NCL block contract."""

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.transpose(0, 2, 1)).transpose(0, 2, 1)


class BigVGANV2(nn.Module):
    """Mel-to-waveform BigVGAN v2 with an NLC internal hot path."""

    def __init__(self, config: BigVGANV2Config):
        super().__init__()
        self.config = config

        if config.upsample_rates is None:
            raise ValueError("upsample_rates must be initialized")
        if config.upsample_kernel_sizes is None:
            raise ValueError("upsample_kernel_sizes must be initialized")
        if config.resblock_kernel_sizes is None:
            raise ValueError("resblock_kernel_sizes must be initialized")
        if config.resblock_dilation_sizes is None:
            raise ValueError("resblock_dilation_sizes must be initialized")

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        if config.resblock == "1":
            resblock_class = _AMPBlock1NLC
        elif config.resblock == "2":
            resblock_class = _AMPBlock2NLC
        else:
            raise ValueError(f"Unknown resblock type: {config.resblock}")

        self.conv_pre = nn.Conv1d(
            config.num_mels,
            config.upsample_initial_channel,
            kernel_size=7,
            padding=3,
        )

        self.ups = []
        channels = config.upsample_initial_channel
        for rate, kernel in zip(
            config.upsample_rates,
            config.upsample_kernel_sizes,
        ):
            out_channels = channels // 2
            padding = (kernel - rate) // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    channels,
                    out_channels,
                    kernel,
                    stride=rate,
                    padding=padding,
                )
            )
            channels = out_channels

        self.resblocks = []
        channels = config.upsample_initial_channel
        for _ in range(self.num_upsamples):
            channels //= 2
            for kernel, dilations in zip(
                config.resblock_kernel_sizes,
                config.resblock_dilation_sizes,
            ):
                self.resblocks.append(
                    resblock_class(
                        channels,
                        kernel_size=kernel,
                        dilations=dilations,
                        activation=config.activation,
                        alpha_logscale=config.snake_logscale,
                    )
                )

        self.activation_post = _periodic_activation(
            config.activation,
            channels,
            config.snake_logscale,
        )
        self.conv_post = nn.Conv1d(
            channels,
            1,
            kernel_size=7,
            padding=3,
            bias=config.use_bias_at_final,
        )
        self.use_tanh = config.use_tanh_at_final

    def __call__(self, x: mx.array) -> mx.array:
        """Generate NCL waveform output from an NCL mel spectrogram."""
        # Enter MLX's native convolution layout once.
        x = self.conv_pre(x.transpose(0, 2, 1))

        for stage in range(self.num_upsamples):
            x = self.ups[stage](x)
            residual_sum = None
            for branch in range(self.num_kernels):
                block = self.resblocks[
                    stage * self.num_kernels + branch
                ]
                branch_output = block(x)
                residual_sum = (
                    branch_output
                    if residual_sum is None
                    else residual_sum + branch_output
                )
            if residual_sum is None:
                raise RuntimeError("BigVGAN stage has no residual branches")
            x = residual_sum / self.num_kernels

        x = self.conv_post(self.activation_post(x))
        x = mx.tanh(x) if self.use_tanh else mx.clip(x, -1.0, 1.0)

        # Preserve the public NCL contract.
        return x.transpose(0, 2, 1)
