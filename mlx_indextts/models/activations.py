"""Activation functions for IndexTTS."""

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> np.ndarray:
    """Create Kaiser-windowed sinc filter (matching PyTorch implementation).

    Args:
        cutoff: Cutoff frequency (0-0.5)
        half_width: Transition band half-width
        kernel_size: Filter kernel size

    Returns:
        Filter of shape (1, 1, kernel_size)
    """
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    # Kaiser window beta calculation
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.

    window = np.kaiser(kernel_size, beta)

    # Time array
    if even:
        time = np.arange(-half_size, half_size) + 0.5
    else:
        time = np.arange(kernel_size) - half_size

    # Sinc function: sin(pi*x) / (pi*x)
    def sinc(x):
        return np.where(x == 0, 1.0, np.sin(np.pi * x) / (np.pi * x))

    if cutoff == 0:
        filter_ = np.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        filter_ /= filter_.sum()

    return filter_.reshape(1, 1, kernel_size).astype(np.float32)


class Snake(nn.Module):
    """Snake activation function.

    Snake(x) = x + (1/a) * sin^2(a*x)

    This is a periodic activation function that helps with audio generation.
    Expects input in NCL format (batch, channels, length).
    """

    def __init__(self, channels: int, alpha_logscale: bool = True):
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale

        if alpha_logscale:
            self.alpha = mx.zeros((channels,))
        else:
            self.alpha = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # Reshape alpha to broadcast: (channels,) -> (1, channels, 1)
        alpha = self.alpha[None, :, None]

        if self.alpha_logscale:
            alpha = mx.exp(alpha)

        return x + (1.0 / (alpha + 1e-9)) * mx.power(mx.sin(alpha * x), 2)


class SnakeBeta(nn.Module):
    """Snake Beta activation function.

    SnakeBeta(x) = x + (1/b) * sin^2(a*x)
    Expects input in NCL format (batch, channels, length).
    """

    def __init__(self, channels: int, alpha_logscale: bool = True):
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale

        if alpha_logscale:
            self.alpha = mx.zeros((channels,))
            self.beta = mx.zeros((channels,))
        else:
            self.alpha = mx.ones((channels,))
            self.beta = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha[None, :, None]
        beta = self.beta[None, :, None]

        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)

        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(alpha * x), 2)


class UpSample1d(nn.Module):
    """1D upsampling with Kaiser-sinc low-pass filter.

    Matches PyTorch alias_free_torch.UpSample1d.
    Input: NCL format (batch, channels, length).
    Uses depthwise transposed convolution (groups=C) for GPU acceleration.
    """

    def __init__(self, ratio: int = 2, kernel_size: int = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2

        # Create filter: shape (kernel_size,) for now, will be expanded in __call__
        filter_np = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size
        )
        # Store as (1, kernel_size, 1) for depthwise conv_transpose1d
        self._filter = mx.array(filter_np.reshape(1, kernel_size, 1))

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample via transposed convolution with anti-aliasing filter.

        Args:
            x: Input (batch, channels, length) NCL format

        Returns:
            Upsampled (batch, channels, length * ratio)
        """
        batch, channels, length = x.shape

        # Keep padding on MLX to avoid CPU round-trips in the vocoder path.
        x_padded = mx.pad(x, ((0, 0), (0, 0), (self.pad, self.pad)), mode="edge")

        # NCL -> NLC for conv_transpose1d
        x_nlc = x_padded.transpose(0, 2, 1)  # (batch, padded_length, channels)

        # Expand filter for depthwise: (1, kernel_size, 1) -> (channels, kernel_size, 1)
        # conv_transpose1d weight shape: (C_out, K, C_in) with groups=C means C_in=1
        filter_expanded = mx.broadcast_to(self._filter, (channels, self.kernel_size, 1))

        # Depthwise transposed convolution
        out = mx.conv_transpose1d(x_nlc, filter_expanded, stride=self.stride, groups=channels)

        # Scale by ratio
        out = out * self.ratio

        # NLC -> NCL
        out = out.transpose(0, 2, 1)

        # Crop
        if self.pad_right > 0:
            out = out[:, :, self.pad_left:-self.pad_right]
        else:
            out = out[:, :, self.pad_left:]

        return out


class DownSample1d(nn.Module):
    """1D downsampling with Kaiser-sinc low-pass filter.

    Matches PyTorch alias_free_torch.DownSample1d.
    Input: NCL format (batch, channels, length).
    Uses depthwise convolution (groups=C) for GPU acceleration.
    """

    def __init__(self, ratio: int = 2, kernel_size: int = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size

        # Create lowpass filter: shape (1, kernel_size, 1) for depthwise conv
        filter_np = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size
        )
        self._filter = mx.array(filter_np.reshape(1, kernel_size, 1))

        # Padding (same as PyTorch LowPassFilter1d)
        even = (self.kernel_size % 2 == 0)
        self.pad_left = self.kernel_size // 2 - int(even)
        self.pad_right = self.kernel_size // 2

    def __call__(self, x: mx.array) -> mx.array:
        """Downsample via strided convolution with anti-aliasing filter.

        Args:
            x: Input (batch, channels, length) NCL format

        Returns:
            Downsampled (batch, channels, length // ratio)
        """
        batch, channels, length = x.shape

        # Keep padding on MLX to avoid CPU round-trips in the vocoder path.
        x_padded = mx.pad(x, ((0, 0), (0, 0), (self.pad_left, self.pad_right)), mode="edge")

        # NCL -> NLC for conv1d
        x_nlc = x_padded.transpose(0, 2, 1)  # (batch, padded_length, channels)

        # Expand filter for depthwise: (1, kernel_size, 1) -> (channels, kernel_size, 1)
        filter_expanded = mx.broadcast_to(self._filter, (channels, self.kernel_size, 1))

        # Depthwise strided convolution
        out = mx.conv1d(x_nlc, filter_expanded, stride=self.ratio, groups=channels)

        # NLC -> NCL
        out = out.transpose(0, 2, 1)

        return out


class Activation1d(nn.Module):
    """1D anti-aliased activation.

    Applies: upsample -> activation -> downsample
    This prevents aliasing artifacts in audio generation.
    Matches PyTorch alias_free_torch.Activation1d.

    Expects input in NCL format (batch, channels, length).
    """

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply anti-aliased activation.

        Args:
            x: Input (batch, channels, length) NCL format

        Returns:
            Activated (batch, channels, length)
        """
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


def get_activation(name: str, channels: int, alpha_logscale: bool = True) -> nn.Module:
    """Get activation function by name."""
    if name == "snake":
        return Snake(channels, alpha_logscale)
    elif name == "snakebeta":
        return SnakeBeta(channels, alpha_logscale)
    else:
        raise ValueError(f"Unknown activation: {name}")
