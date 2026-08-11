"""Native MLX implementation of the IndexTTS 2.5 EnhancedCodec.

The public 2.5 checkpoint uses two Vocos/ConvNeXt backbones around a single
factorized residual vector quantizer.  This module keeps the PyTorch checkpoint
module names so conversion can be strict and auditable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class ConvNeXtBlock1D(nn.Module):
    """The 1-D ConvNeXt block used by the official semantic codec."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = mx.full((dim,), layer_scale_init_value)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        return residual + x * self.gamma


class VocosBackbone(nn.Module):
    """Resolution-preserving Vocos backbone operating on ``(B, T, C)``."""

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        scale = layer_scale_init_value or 1.0 / num_layers
        self.convnext = [
            ConvNeXtBlock1D(dim, intermediate_dim, scale)
            for _ in range(num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(self.embed(x))
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x)


class FactorizedVectorQuantizer(nn.Module):
    """Single factorized codebook with the 2.5 checkpoint parameter layout."""

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        use_l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim
        self.use_l2_normalize = use_l2_normalize
        self.in_project = nn.Conv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_project = nn.Conv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    @staticmethod
    def _normalize(x: mx.array) -> mx.array:
        return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

    def nearest_code(self, latents: mx.array) -> Tuple[mx.array, mx.array]:
        """Return nearest code indices and raw codebook vectors for NLC latents."""
        encodings = latents.reshape(-1, latents.shape[-1])
        codebook = self.codebook.weight
        if self.use_l2_normalize:
            encodings = self._normalize(encodings)
            codebook = self._normalize(codebook)
        distances = (
            mx.sum(encodings * encodings, axis=1, keepdims=True)
            - 2.0 * (encodings @ codebook.T)
            + mx.sum(codebook * codebook, axis=1)[None, :]
        )
        indices = mx.argmin(distances, axis=1).reshape(latents.shape[:-1]).astype(mx.int32)
        return indices, self.codebook(indices)

    def encode(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        projected = self.in_project(x)
        indices, vectors = self.nearest_code(projected)
        return indices, self.out_project(vectors)

    def decode(self, codes: mx.array) -> mx.array:
        return self.out_project(self.codebook(codes.astype(mx.int32)))


class ResidualVectorQuantizer(nn.Module):
    """Residual wrapper; the released 2.5 checkpoint contains one quantizer."""

    def __init__(
        self,
        input_dim: int,
        num_quantizers: int,
        codebook_size: int,
        codebook_dim: int,
    ) -> None:
        super().__init__()
        if num_quantizers < 1:
            raise ValueError("num_quantizers must be positive")
        self.quantizers = [
            FactorizedVectorQuantizer(
                input_dim=input_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                use_l2_normalize=True,
            )
            for _ in range(num_quantizers)
        ]

    def encode(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        residual = x
        quantized = mx.zeros_like(x)
        all_indices = []
        for quantizer in self.quantizers:
            indices, contribution = quantizer.encode(residual)
            all_indices.append(indices)
            quantized = quantized + contribution
            residual = residual - contribution
        return mx.stack(all_indices, axis=0), quantized

    def vq2emb(self, codes: mx.array) -> mx.array:
        if codes.ndim == 2:
            codes = codes[None, ...]
        if codes.ndim != 3:
            raise ValueError("codes must have shape (B, T) or (N, B, T)")
        if codes.shape[0] > len(self.quantizers):
            raise ValueError("codes contain more quantizer streams than the model")
        result = None
        for stream, quantizer in zip(codes, self.quantizers):
            contribution = quantizer.decode(stream)
            result = contribution if result is None else result + contribution
        if result is None:
            raise ValueError("codes must contain at least one quantizer stream")
        return result


class EnhancedCodecV25(nn.Module):
    """IndexTTS 2.5 semantic codec with the official inference API."""

    def __init__(
        self,
        codebook_size: int = 8192,
        hidden_size: int = 1024,
        codebook_dim: int = 8,
        vocos_dim: int = 384,
        vocos_intermediate_dim: int = 2048,
        vocos_num_layers: int = 12,
        num_quantizers: int = 1,
        downsample_scale: int = 2,
    ) -> None:
        super().__init__()
        if downsample_scale not in (1, 2):
            raise ValueError("the released EnhancedCodec supports downsample_scale 1 or 2")
        self.downsample_scale = downsample_scale
        if downsample_scale > 1:
            self.down = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
            self.up = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.encoder = [
            VocosBackbone(
                input_channels=hidden_size,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
            ),
            nn.Linear(vocos_dim, hidden_size),
        ]
        self.decoder = [
            VocosBackbone(
                input_channels=hidden_size,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
            ),
            nn.Linear(vocos_dim, hidden_size),
        ]
        self.quantizer = ResidualVectorQuantizer(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

    @staticmethod
    def _run_stack(stack: list[nn.Module], x: mx.array) -> mx.array:
        for layer in stack:
            x = layer(x)
        return x

    def quantize(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Encode Qwen features into semantic codes and quantized embeddings."""
        if x.ndim != 3:
            raise ValueError("codec input must have shape (B, T, hidden_size)")
        if self.downsample_scale > 1:
            x = nn.gelu(self.down(x))
        encoded = self._run_stack(self.encoder, x)
        all_indices, quantized = self.quantizer.encode(encoded)
        if all_indices.shape[0] == 1:
            return all_indices[0], quantized
        return all_indices, quantized

    def decode(self, codes: mx.array) -> mx.array:
        """Decode semantic code IDs into S2Mel content features."""
        quantized = self.quantizer.vq2emb(codes)
        x = self._run_stack(self.decoder, quantized)
        if self.downsample_scale > 1:
            x = mx.repeat(x, self.downsample_scale, axis=1)
            x = self.up(x)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """Inference reconstruction path used for parity diagnostics."""
        if x.shape[1] % 2:
            x = x[:, :-1, :]
        codes, _ = self.quantize(x)
        return self.decode(codes)


# Keep the upstream class name available to conversion/runtime code.
EnhancedCodec = EnhancedCodecV25
