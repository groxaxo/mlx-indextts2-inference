"""Activation functions for IndexTTS."""

from __future__ import annotations

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def kaiser_sinc_filter1d(
    cutoff: float,
    half_width: float,
    kernel_size: int,
) -> np.ndarray:
    """Create a Kaiser-windowed sinc low-pass filter."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    attenuation = (
        2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    )
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = (
            0.5842 * (attenuation - 21) ** 0.4
            + 0.07886 * (attenuation - 21.0)
        )
    else:
        beta = 0.0

    window = np.kaiser(kernel_size, beta)
    if even:
        time = np.arange(-half_size, half_size) + 0.5
    else:
        time = np.arange(kernel_size) - half_size

    def sinc(value):
        return np.where(
            value == 0,
            1.0,
            np.sin(np.pi * value) / (np.pi * value),
        )

    if cutoff == 0:
        filter_ = np.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        filter_ /= filter_.sum()

    return filter_.reshape(1, 1, kernel_size).astype(np.float32)


def _normalized_axis(axis: int, ndim: int) -> int:
    normalized = axis if axis >= 0 else ndim + axis
    if normalized <= 0 or normalized >= ndim:
        raise ValueError("channel_axis must select a non-batch tensor dimension")
    return normalized


class Snake(nn.Module):
    """Periodic Snake activation supporting NCL and NLC layouts."""

    def __init__(
        self,
        channels: int,
        alpha_logscale: bool = True,
        channel_axis: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale
        self.channel_axis = channel_axis
        self.alpha = (
            mx.zeros((channels,))
            if alpha_logscale
            else mx.ones((channels,))
        )

    def __call__(self, x: mx.array) -> mx.array:
        axis = _normalized_axis(self.channel_axis, x.ndim)
        shape = [1] * x.ndim
        shape[axis] = self.channels
        alpha = self.alpha.reshape(shape)
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        sine = mx.sin(alpha * x)
        return x + (sine * sine) / (alpha + 1e-9)


class SnakeBeta(nn.Module):
    """Periodic SnakeBeta activation supporting NCL and NLC layouts."""

    def __init__(
        self,
        channels: int,
        alpha_logscale: bool = True,
        channel_axis: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale
        self.channel_axis = channel_axis
        if alpha_logscale:
            self.alpha = mx.zeros((channels,))
            self.beta = mx.zeros((channels,))
        else:
            self.alpha = mx.ones((channels,))
            self.beta = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        axis = _normalized_axis(self.channel_axis, x.ndim)
        shape = [1] * x.ndim
        shape[axis] = self.channels
        alpha = self.alpha.reshape(shape)
        beta = self.beta.reshape(shape)
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)
        sine = mx.sin(alpha * x)
        return x + (sine * sine) / (beta + 1e-9)


class UpSample1d(nn.Module):
    """Kaiser-sinc depthwise upsampling in NCL or NLC layout."""

    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        channel_axis: int = 1,
    ):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2
            if kernel_size is None
            else kernel_size
        )
        self.channel_axis = channel_axis
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = (
            self.pad * self.stride
            + (self.kernel_size - self.stride) // 2
        )
        self.pad_right = (
            self.pad * self.stride
            + (self.kernel_size - self.stride + 1) // 2
        )

        filter_np = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )
        self._filter = mx.array(
            filter_np.reshape(1, self.kernel_size, 1)
        )

    def __call__(self, x: mx.array) -> mx.array:
        axis = _normalized_axis(self.channel_axis, x.ndim)
        if x.ndim != 3 or axis not in (1, 2):
            raise ValueError("UpSample1d expects 3D NCL or NLC input")

        if axis == 1:
            channels = x.shape[1]
            x_nlc = mx.pad(
                x,
                ((0, 0), (0, 0), (self.pad, self.pad)),
                mode="edge",
            ).transpose(0, 2, 1)
        else:
            channels = x.shape[2]
            x_nlc = mx.pad(
                x,
                ((0, 0), (self.pad, self.pad), (0, 0)),
                mode="edge",
            )

        filter_expanded = mx.broadcast_to(
            self._filter,
            (channels, self.kernel_size, 1),
        )
        out = mx.conv_transpose1d(
            x_nlc,
            filter_expanded,
            stride=self.stride,
            groups=channels,
        )
        out = out * self.ratio

        if self.pad_right > 0:
            out = out[:, self.pad_left : -self.pad_right, :]
        else:
            out = out[:, self.pad_left :, :]
        return out.transpose(0, 2, 1) if axis == 1 else out


class DownSample1d(nn.Module):
    """Kaiser-sinc depthwise downsampling in NCL or NLC layout."""

    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        channel_axis: int = 1,
    ):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2
            if kernel_size is None
            else kernel_size
        )
        self.channel_axis = channel_axis

        filter_np = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )
        self._filter = mx.array(
            filter_np.reshape(1, self.kernel_size, 1)
        )

        even = self.kernel_size % 2 == 0
        self.pad_left = self.kernel_size // 2 - int(even)
        self.pad_right = self.kernel_size // 2

    def __call__(self, x: mx.array) -> mx.array:
        axis = _normalized_axis(self.channel_axis, x.ndim)
        if x.ndim != 3 or axis not in (1, 2):
            raise ValueError("DownSample1d expects 3D NCL or NLC input")

        if axis == 1:
            channels = x.shape[1]
            x_nlc = mx.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (self.pad_left, self.pad_right),
                ),
                mode="edge",
            ).transpose(0, 2, 1)
        else:
            channels = x.shape[2]
            x_nlc = mx.pad(
                x,
                (
                    (0, 0),
                    (self.pad_left, self.pad_right),
                    (0, 0),
                ),
                mode="edge",
            )

        filter_expanded = mx.broadcast_to(
            self._filter,
            (channels, self.kernel_size, 1),
        )
        out = mx.conv1d(
            x_nlc,
            filter_expanded,
            stride=self.ratio,
            groups=channels,
        )
        return out.transpose(0, 2, 1) if axis == 1 else out


class Activation1d(nn.Module):
    """Anti-aliased activation supporting NCL and NLC layouts."""

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        channel_axis: int | None = None,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.channel_axis = (
            getattr(activation, "channel_axis", 1)
            if channel_axis is None
            else channel_axis
        )
        self.upsample = UpSample1d(
            up_ratio,
            up_kernel_size,
            channel_axis=self.channel_axis,
        )
        self.downsample = DownSample1d(
            down_ratio,
            down_kernel_size,
            channel_axis=self.channel_axis,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


def get_activation(
    name: str,
    channels: int,
    alpha_logscale: bool = True,
) -> nn.Module:
    """Get a periodic activation by name in the historical NCL layout."""
    if name == "snake":
        return Snake(channels, alpha_logscale)
    if name == "snakebeta":
        return SnakeBeta(channels, alpha_logscale)
    raise ValueError(f"Unknown activation: {name}")
