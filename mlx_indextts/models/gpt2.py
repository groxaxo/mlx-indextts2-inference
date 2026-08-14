"""GPT-2 model implementation for IndexTTS."""

from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class KVCache(list):
    """Chunked KV cache that avoids reallocating the full history every token.

    The cache intentionally stores pre-head K/V tensors with shape ``(B, T, D)``
    so it remains compatible with the historical ``cache[layer][0/1]`` contract
    used by IndexTTS. Capacity grows in coarse chunks and new tokens are written
    into the existing MLX arrays in place, matching the strategy used by MLX-LM.
    """

    step = 256

    def __init__(self, step: int = 256):
        super().__init__([None, None])
        self.step = max(1, int(step))
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset = 0

    @property
    def capacity(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[1])

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError("KVCache expects K/V tensors with shape (B, T, D)")
        if keys.shape[:2] != values.shape[:2]:
            raise ValueError("KVCache key/value batch and sequence dimensions must match")

        previous = self.offset
        new_tokens = int(keys.shape[1])
        required = previous + new_tokens

        if self.keys is None or required > self.capacity:
            batch_size = int(keys.shape[0])
            key_dim = int(keys.shape[2])
            value_dim = int(values.shape[2])
            growth = ((new_tokens + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((batch_size, growth, key_dim), dtype=keys.dtype)
            new_v = mx.zeros((batch_size, growth, value_dim), dtype=values.dtype)

            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                # Keep only the logical prefix before growing. This prevents a
                # partially used capacity block from becoming part of the next
                # logical cache and mirrors MLX-LM's chunked KVCache behavior.
                if previous % self.step != 0:
                    self.keys = self.keys[:, :previous, :]
                    self.values = self.values[:, :previous, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=1)
                self.values = mx.concatenate([self.values, new_v], axis=1)

        self.offset = required
        self.keys[:, previous:self.offset, :] = keys
        self.values[:, previous:self.offset, :] = values

        active_keys = self.keys[:, :self.offset, :]
        active_values = self.values[:, :self.offset, :]
        # Keep list semantics so mx.eval(cache) and cache[layer][0/1] continue
        # to work exactly as they did with the historical tuple cache.
        self[0] = active_keys
        self[1] = active_values
        return active_keys, active_values

    @classmethod
    def from_legacy(
        cls,
        state: Tuple[mx.array, mx.array],
        *,
        step: int = 256,
    ) -> "KVCache":
        cache = cls(step=step)
        cache.update_and_fetch(state[0], state[1])
        return cache


CacheState = Union[KVCache, Tuple[mx.array, mx.array]]


class GPT2Attention(nn.Module):
    """GPT-2 style multi-head attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection (HuggingFace GPT2 uses bias)
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[CacheState] = None,
    ) -> Tuple[mx.array, KVCache]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Attention mask
            cache: Chunked KV cache, or a legacy ``(k, v)`` tuple

        Returns:
            Output tensor and updated cache
        """
        batch_size, _, _ = x.shape

        # Combined QKV projection
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Preserve backwards compatibility with callers that may still hand in
        # a tuple cache while using the chunked representation for all new work.
        if cache is None:
            cache = KVCache()
        elif not isinstance(cache, KVCache):
            cache = KVCache.from_legacy(cache)
        k, v = cache.update_and_fetch(k, v)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Prefer MLX's fused attention kernel when available. The fallback keeps
        # compatibility with older MLX versions allowed by pyproject.toml.
        fast = getattr(mx, "fast", None)
        sdpa = getattr(fast, "scaled_dot_product_attention", None)
        if sdpa is not None:
            out = sdpa(q, k, v, scale=self.scale, mask=mask)
        else:
            scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
            if mask is not None:
                scores = scores + mask
            attn = mx.softmax(scores, axis=-1)
            out = mx.matmul(attn, v)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.dim)

        # Output projection
        out = self.c_proj(out)

        return out, cache


class GPT2MLP(nn.Module):
    """GPT-2 style MLP."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.c_fc = nn.Linear(dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        x = self.c_fc(x)
        x = nn.gelu_approx(x)  # GPT-2 uses GELU
        x = self.c_proj(x)
        return x


class GPT2Block(nn.Module):
    """GPT-2 transformer block."""

    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()

        self.ln_1 = nn.LayerNorm(dim)
        self.attn = GPT2Attention(dim, num_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = GPT2MLP(dim)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[CacheState] = None,
    ) -> Tuple[mx.array, KVCache]:
        """Forward pass."""
        # Self-attention with residual
        residual = x
        x = self.ln_1(x)
        x, new_cache = self.attn(x, mask, cache)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_cache


class GPT2Model(nn.Module):
    """GPT-2 model backbone.

    This is the transformer backbone without embeddings or output heads,
    as those are handled by the UnifiedVoice model.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int = 2048,
    ):
        """Initialize GPT-2 model.

        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Transformer blocks
        self.h = [
            GPT2Block(dim, num_heads, max_seq_len)
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.ln_f = nn.LayerNorm(dim)

    def __call__(
        self,
        inputs_embeds: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[CacheState]] = None,
    ) -> Tuple[mx.array, List[KVCache]]:
        """Forward pass.

        Args:
            inputs_embeds: Input embeddings (batch, seq_len, dim)
            mask: Attention mask (if None, uses causal mask)
            cache: List of KV caches per layer

        Returns:
            Output hidden states and updated cache
        """
        x = inputs_embeds
        query_len = x.shape[1]
        new_cache = []

        # Calculate key length (includes cached tokens if any)
        if cache is not None and cache[0] is not None:
            cache_len = cache[0][0].shape[1]
            key_len = cache_len + query_len
        else:
            key_len = query_len

        # Create causal mask if not provided (GPT-2 is autoregressive)
        if mask is None:
            mask = self.create_causal_mask(query_len, key_len)

        for i, block in enumerate(self.h):
            layer_cache = cache[i] if cache is not None else None
            x, updated_cache = block(x, mask, layer_cache)
            new_cache.append(updated_cache)

        x = self.ln_f(x)

        return x, new_cache

    def create_causal_mask(self, query_len: int, key_len: int) -> mx.array:
        """Create causal attention mask.

        For autoregressive generation with KV cache:
        - query_len: length of current input (1 during generation)
        - key_len: total length including cached keys (cache_len + query_len)

        Args:
            query_len: Query sequence length
            key_len: Key sequence length (may be larger due to cache)

        Returns:
            Causal mask of shape (1, 1, query_len, key_len)
            Where positions that should NOT be attended have -inf
        """
        # Create mask where each query position can only attend to
        # key positions up to and including itself
        # For incremental generation: query at position i attends to keys 0..i
        #
        # With cache, if we're at step N (cache has N tokens, query_len=1):
        # - key_len = N + 1
        # - We want mask shape (1, key_len) = (1, N+1) all zeros (can attend to all)
        #
        # Without cache (initial forward, query_len = key_len = S):
        # - Standard lower triangular mask
        if query_len == key_len:
            # Standard causal mask for initial forward pass
            mask = mx.triu(mx.ones((query_len, key_len)), k=1)
        else:
            # Incremental decoding: query can attend to all keys
            # (causal constraint is already satisfied by only having past keys in cache)
            mask = mx.zeros((query_len, key_len))

        # Convert to additive mask: 0 -> 0, 1 -> -inf
        mask = mx.where(mask > 0, float("-inf"), 0.0)
        return mask[None, None, :, :]
