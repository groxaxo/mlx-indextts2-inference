"""Diffusion Transformer (DiT) for S2Mel.

This implements the DiT model used for mel spectrogram generation in IndexTTS 2.0.
Based on gpt_fast architecture with adaptive layer normalization.
"""

import math
from dataclasses import dataclass
from typing import Optional, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_indextts.models.s2mel.wavenet import WN, SConv1d


@dataclass
class ModelArgs:
    """Configuration for the Transformer model."""
    block_size: int = 16384
    vocab_size: int = 1024
    n_layer: int = 13
    n_head: int = 8
    dim: int = 512
    intermediate_size: Optional[int] = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    uvit_skip_connection: bool = True
    time_as_token: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            # Round up to multiple of 256
            self.intermediate_size = ((n_hidden + 255) // 256) * 256
        self.head_dim = self.dim // self.n_head


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Uses sinusoidal position embeddings followed by MLP.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 10000
        self.scale = 1000

        # MLP: freq_dim -> hidden -> hidden
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        # Precompute frequency terms
        half = frequency_embedding_size // 2
        freqs = mx.exp(
            -math.log(self.max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        self.freqs = freqs

    def timestep_embedding(self, t: mx.array) -> mx.array:
        """Create sinusoidal timestep embeddings.

        Args:
            t: Timestep tensor (batch,)

        Returns:
            Embeddings (batch, frequency_embedding_size)
        """
        args = self.scale * t[:, None] * self.freqs[None, :]  # (batch, half)
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        if self.frequency_embedding_size % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros((embedding.shape[0], 1))],
                axis=-1
            )

        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        """Forward pass.

        Args:
            t: Timestep tensor (batch,)

        Returns:
            Timestep embeddings (batch, hidden_size)
        """
        t_freq = self.timestep_embedding(t)
        t_emb = self.linear1(t_freq)
        t_emb = nn.silu(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb


class FinalLayer(nn.Module):
    """Final layer of DiT with adaptive layer normalization."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        # PyTorch uses nn.LayerNorm(hidden_size, elementwise_affine=False) - no learnable params
        # Just store hidden_size for the non-affine normalization
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        # Match PyTorch Sequential structure: [SiLU(), Linear()]
        # Weights will be at adaLN_modulation.layers.1.weight/bias
        self.adaLN_modulation = AdaLNModulation(hidden_size)

    def _layer_norm(self, x: mx.array) -> mx.array:
        """Layer normalization without learnable parameters."""
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) / mx.sqrt(var + 1e-6)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        """Forward pass with adaptive layer normalization.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            c: Conditioning tensor (batch, hidden_size)

        Returns:
            Output tensor (batch, seq_len, out_channels)
        """
        # Get modulation parameters
        modulation = self.adaLN_modulation(c)  # (batch, 2*hidden)
        shift = modulation[:, :self.hidden_size]
        scale = modulation[:, self.hidden_size:]

        # Apply adaptive layer norm: norm(x) * (1 + scale) + shift
        x = self._layer_norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]

        # Final linear
        x = self.linear(x)
        return x


class AdaLNModulation(nn.Module):
    """Adaptive LayerNorm modulation (SiLU + Linear)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        # Use ModuleList-like structure to match weight names
        # layers.0 = SiLU (no weights)
        # layers.1 = Linear
        self.layers = [
            SiLUWrapper(),  # Index 0, no weights
            nn.Linear(hidden_size, 2 * hidden_size),  # Index 1
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layers[0](x)  # SiLU
        x = self.layers[1](x)  # Linear
        return x


class SiLUWrapper(nn.Module):
    """SiLU activation wrapped as a module for Sequential compatibility."""

    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (..., dims)

        Returns:
            Normalized tensor (..., dims)
        """
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization for conditioning."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = RMSNorm(d_model, eps=eps)

    def __call__(self, x: mx.array, embedding: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            embedding: Optional conditioning (batch, d_model) or (batch, 1, d_model)

        Returns:
            Normalized tensor (batch, seq_len, d_model)
        """
        if embedding is None:
            return self.norm(x)

        # Handle 2D embedding
        if embedding.ndim == 2:
            embedding = embedding[:, None, :]  # (batch, 1, d_model)

        # Project to get weight and bias
        proj = self.project_layer(embedding)  # (batch, 1, 2*d_model)
        weight = proj[..., :self.d_model]
        bias = proj[..., self.d_model:]

        # Apply adaptive norm: weight * norm(x) + bias
        return weight * self.norm(x) + bias


class RotaryPositionEmbedding:
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 16384, base: float = 10000):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        self.freqs_cis = self._precompute_freqs_cis()

    def _precompute_freqs_cis(self) -> mx.array:
        """Precompute the frequency tensor for complex exponentials."""
        n_elem = self.dim
        freqs = 1.0 / (self.base ** (mx.arange(0, n_elem, 2, dtype=mx.float32)[:n_elem // 2] / n_elem))
        t = mx.arange(self.max_seq_len, dtype=mx.float32)
        freqs = mx.outer(t, freqs)

        # Store as [cos, sin] for efficiency
        cos_freqs = mx.cos(freqs)
        sin_freqs = mx.sin(freqs)
        return mx.stack([cos_freqs, sin_freqs], axis=-1)  # (max_seq, dim/2, 2)


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Apply rotary position embedding.

    Args:
        x: Input tensor (batch, seq_len, n_heads, head_dim)
        freqs_cis: Precomputed freqs (seq_len, head_dim/2, 2)

    Returns:
        Tensor with position encoding applied
    """
    # Reshape x for rotation
    x_shape = x.shape
    x = x.reshape(*x_shape[:-1], -1, 2)  # (batch, seq, heads, dim/2, 2)

    # Get cos and sin
    cos = freqs_cis[..., 0]  # (seq, dim/2)
    sin = freqs_cis[..., 1]  # (seq, dim/2)

    # Reshape for broadcasting
    cos = cos[None, :, None, :]  # (1, seq, 1, dim/2)
    sin = sin[None, :, None, :]  # (1, seq, 1, dim/2)

    # Apply rotation
    x_real = x[..., 0]  # (batch, seq, heads, dim/2)
    x_imag = x[..., 1]  # (batch, seq, heads, dim/2)

    x_out_real = x_real * cos - x_imag * sin
    x_out_imag = x_imag * cos + x_real * sin

    # Interleave back
    x_out = mx.stack([x_out_real, x_out_imag], axis=-1)
    x_out = x_out.reshape(x_shape)

    return x_out


class Attention(nn.Module):
    """Multi-head attention with rotary position embeddings."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim
        self.dim = config.dim

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dim)
            freqs_cis: Rotary embeddings (seq_len, head_dim/2, 2)
            mask: Optional broadcastable additive attention mask

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim

        # Project to q, k, v
        qkv = self.wqkv(x)
        q = qkv[:, :, :kv_size]
        k = qkv[:, :, kv_size:2*kv_size]
        v = qkv[:, :, 2*kv_size:]

        # Reshape
        q = q.reshape(bsz, seqlen, self.n_head, self.head_dim)
        k = k.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Transpose for attention: (batch, heads, seq, dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Repeat k, v for grouped query attention
        if self.n_head != self.n_local_heads:
            k = mx.repeat(k, self.n_head // self.n_local_heads, axis=1)
            v = mx.repeat(v, self.n_head // self.n_local_heads, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        fast = getattr(mx, "fast", None)
        sdpa = getattr(fast, "scaled_dot_product_attention", None)
        if sdpa is not None:
            y = sdpa(q, k, v, scale=scale, mask=mask)
        else:
            # Compatibility fallback for older MLX builds.
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale
            if mask is not None:
                scores = scores + mask
            attn = mx.softmax(scores, axis=-1)
            y = attn @ v

        # Reshape back
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        y = self.wo(y)

        return y


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Transformer block with adaptive layer normalization."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = AdaptiveLayerNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = AdaptiveLayerNorm(config.dim, eps=config.norm_eps)

        self.uvit_skip_connection = config.uvit_skip_connection
        if self.uvit_skip_connection:
            self.skip_in_linear = nn.Linear(config.dim * 2, config.dim)

        self.time_as_token = config.time_as_token

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        freqs_cis: mx.array,
        mask: Optional[mx.array] = None,
        skip_in_x: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dim)
            c: Conditioning tensor (batch, 1, dim)
            freqs_cis: Rotary embeddings
            mask: Attention mask
            skip_in_x: Skip connection from earlier layer

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Use conditioning if not time_as_token mode
        c_use = None if self.time_as_token else c

        # U-ViT skip connection
        if self.uvit_skip_connection and skip_in_x is not None:
            x = self.skip_in_linear(mx.concatenate([x, skip_in_x], axis=-1))

        # Self attention with adaptive norm
        h = x + self.attention(self.attention_norm(x, c_use), freqs_cis, mask)

        # Feed forward with adaptive norm
        out = h + self.feed_forward(self.ffn_norm(h, c_use))

        return out


class Transformer(nn.Module):
    """Transformer backbone for DiT."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.norm = AdaptiveLayerNorm(config.dim, eps=config.norm_eps)

        # Rotary embeddings
        self.rope = RotaryPositionEmbedding(config.head_dim, config.block_size, config.rope_base)

        # U-ViT skip connections
        if config.uvit_skip_connection:
            self.layers_emit_skip = [i for i in range(config.n_layer) if i < config.n_layer // 2]
            self.layers_receive_skip = [i for i in range(config.n_layer) if i > config.n_layer // 2]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        input_pos: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dim)
            c: Conditioning (batch, 1, dim)
            input_pos: Position indices (seq_len,)
            mask: Broadcastable additive attention mask or None

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Get rotary embeddings for positions
        freqs_cis = self.rope.freqs_cis[input_pos]

        skip_in_x_list = []

        for i, layer in enumerate(self.layers):
            # Get skip connection if receiving
            if self.config.uvit_skip_connection and i in self.layers_receive_skip:
                skip_in_x = skip_in_x_list.pop(-1)
            else:
                skip_in_x = None

            x = layer(x, c, freqs_cis, mask, skip_in_x)

            # Store for skip connection if emitting
            if self.config.uvit_skip_connection and i in self.layers_emit_skip:
                skip_in_x_list.append(x)

        x = self.norm(x, c)
        return x


class DiT(nn.Module):
    """Diffusion Transformer for mel spectrogram generation.

    This is the main estimator used in CFM for denoising.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        depth: int = 13,
        in_channels: int = 80,
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
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.time_as_token = time_as_token
        self.style_as_token = style_as_token
        self.transformer_style_condition = style_condition
        self.class_dropout_prob = class_dropout_prob
        self.long_skip_connection = long_skip_connection
        self.final_layer_type = final_layer_type

        # Build transformer config
        config = ModelArgs(
            block_size=16384,
            n_layer=depth,
            n_head=num_heads,
            dim=hidden_dim,
            head_dim=hidden_dim // num_heads,
            vocab_size=1024,
            uvit_skip_connection=uvit_skip_connection,
            time_as_token=time_as_token,
        )

        self.transformer = Transformer(config)

        # Input projection
        self.x_embedder = nn.Linear(in_channels, hidden_dim)

        # Content projection (continuous)
        self.cond_projection = nn.Linear(content_dim, hidden_dim)

        # Discrete content embedding (exists in weights but not used in inference)
        self.cond_embedder = nn.Embedding(1024, hidden_dim)  # codebook_size=1024
        # Content mask embedding (exists in weights but not used in inference)
        self.content_mask_embedder = nn.Embedding(1, hidden_dim)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # Input merge: x + prompt_x + cond [+ style]
        merge_dim = hidden_dim + in_channels * 2
        if style_condition and not style_as_token:
            merge_dim += style_dim
        self.cond_x_merge_linear = nn.Linear(merge_dim, hidden_dim)

        # Long skip connection
        if long_skip_connection:
            self.skip_linear = nn.Linear(hidden_dim + in_channels, hidden_dim)

        # Style token projection
        if style_as_token:
            self.style_in = nn.Linear(style_dim, hidden_dim)

        # Final layer (wavenet or mlp)
        if final_layer_type == 'wavenet':
            self.t_embedder2 = TimestepEmbedder(wavenet_hidden_dim)
            self.conv1 = nn.Linear(hidden_dim, wavenet_hidden_dim)
            self.conv2 = nn.Conv1d(wavenet_hidden_dim, in_channels, kernel_size=1)
            self.wavenet = WN(
                hidden_channels=wavenet_hidden_dim,
                kernel_size=wavenet_kernel_size,
                dilation_rate=wavenet_dilation_rate,
                n_layers=wavenet_num_layers,
                gin_channels=wavenet_hidden_dim,
                p_dropout=wavenet_p_dropout,
                causal=False,
            )
            self.final_layer = FinalLayer(wavenet_hidden_dim, 1, wavenet_hidden_dim)
            self.res_projection = nn.Linear(hidden_dim, wavenet_hidden_dim)
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, in_channels),
            )

    def setup_caches(self, max_batch_size: int, max_seq_length: int):
        """Setup KV caches (not used in this implementation)."""
        pass  # KV cache not implemented yet

    def __call__(
        self,
        x: mx.array,
        prompt_x: mx.array,
        x_lens: mx.array,
        t: mx.array,
        style: mx.array,
        cond: mx.array,
        mask_content: bool = False,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Noisy mel (batch, in_channels, seq_len) - NCL format
            prompt_x: Prompt mel (batch, in_channels, seq_len) - NCL format
            x_lens: Sequence lengths (batch,)
            t: Timesteps (batch,)
            style: Style embedding (batch, style_dim)
            cond: Content conditioning (batch, seq_len, content_dim)
            mask_content: Whether to mask content for CFG

        Returns:
            Predicted noise (batch, in_channels, seq_len) - NCL format
        """
        B, _, T = x.shape

        # Timestep embedding
        t1 = self.t_embedder(t)  # (batch, hidden_dim)

        # Content projection
        cond = self.cond_projection(cond)  # (batch, seq_len, hidden_dim)

        # Transpose to NLC format
        x_t = x.transpose(0, 2, 1)  # (batch, seq_len, in_channels)
        prompt_x_t = prompt_x.transpose(0, 2, 1)

        # Merge inputs
        x_in = mx.concatenate([x_t, prompt_x_t, cond], axis=-1)

        # Add style conditioning without materializing a repeated copy first.
        if self.transformer_style_condition and not self.style_as_token:
            style_expanded = mx.broadcast_to(
                style[:, None, :],
                (B, T, style.shape[-1]),
            )
            x_in = mx.concatenate([x_in, style_expanded], axis=-1)

        # Apply content masking for CFG
        if mask_content:
            x_in = mx.concatenate([
                x_in[..., :self.in_channels],
                mx.zeros_like(x_in[..., self.in_channels:])
            ], axis=-1)

        # Merge projection
        x_in = self.cond_x_merge_linear(x_in)

        # Add style token if needed
        if self.style_as_token:
            style_token = self.style_in(style)
            if mask_content:
                style_token = mx.zeros_like(style_token)
            x_in = mx.concatenate([style_token[:, None, :], x_in], axis=1)

        # Add time token if needed
        if self.time_as_token:
            x_in = mx.concatenate([t1[:, None, :], x_in], axis=1)

        seq_len = x_in.shape[1]
        input_pos = mx.arange(seq_len)

        # Key validity is identical for every query. Keep the mask in its
        # broadcastable (B, 1, 1, T) form instead of materializing a T x T
        # matrix on every diffusion step.
        valid_keys = _sequence_mask(
            x_lens + self.style_as_token + self.time_as_token,
            seq_len,
        )
        attn_mask = mx.where(
            valid_keys[:, None, None, :] > 0,
            0.0,
            float('-inf'),
        )

        # Transformer forward
        x_res = self.transformer(x_in, t1[:, None, :], input_pos, attn_mask)

        # Remove style/time tokens if added
        if self.time_as_token:
            x_res = x_res[:, 1:]
        if self.style_as_token:
            x_res = x_res[:, 1:]

        # Long skip connection
        if self.long_skip_connection:
            x_res = self.skip_linear(mx.concatenate([x_res, x_t], axis=-1))

        # Final layer
        if self.final_layer_type == 'wavenet':
            x_out = self.conv1(x_res)
            x_out = x_out.transpose(0, 2, 1)  # NLC -> NCL
            t2 = self.t_embedder2(t)
            x_mask_wn = mx.ones((B, 1, T))
            wn_out = self.wavenet(x_out, x_mask_wn, g=t2[:, :, None])
            wn_out = wn_out.transpose(0, 2, 1)  # NCL -> NLC
            x_out = wn_out + self.res_projection(x_res)  # Long residual
            x_out = self.final_layer(x_out, t1)
            x_out = x_out.transpose(0, 2, 1)  # NLC -> NCL
            # Final conv
            x_out = x_out.transpose(0, 2, 1)  # NCL -> NLC for Conv1d
            x_out = self.conv2(x_out)
            x_out = x_out.transpose(0, 2, 1)  # NLC -> NCL
        else:
            x_out = self.final_mlp(x_res)
            x_out = x_out.transpose(0, 2, 1)

        return x_out


def _sequence_mask(lengths: mx.array, max_length: int) -> mx.array:
    """Create sequence mask.

    Args:
        lengths: Sequence lengths (batch,)
        max_length: Maximum length

    Returns:
        Mask (batch, max_length) where True indicates valid positions
    """
    seq_range = mx.arange(max_length)[None, :]
    return (seq_range < lengths[:, None]).astype(mx.float32)
