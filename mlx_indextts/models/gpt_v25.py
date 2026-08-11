"""IndexTTS 2.5 GPT model for native MLX inference.

IndexTTS 2.5 keeps the 2.0 emotion and autoregressive GPT branches, but replaces
the speaker Conformer/Perceiver with a 192-dimensional CampPlus projection and
adds a learned language embedding to every text position during generation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_indextts.config import IndexTTSConfig
from mlx_indextts.models.gpt2 import GPT2Model
from mlx_indextts.models.gpt_v2 import UnifiedVoiceV2
from mlx_indextts.tokenizer_v25 import LANGUAGE_CODES


class UnifiedVoiceV25(UnifiedVoiceV2):
    """Released 2.5 GPT architecture with CampPlus and language conditioning."""

    speaker_embedding_dim = 192
    num_language_embeddings = len(LANGUAGE_CODES) + 1

    def __init__(self, config: IndexTTSConfig) -> None:
        super().__init__(config)

        # These 2.0-only modules do not exist in the public 2.5 checkpoint.
        for name in ("conditioning_encoder", "perceiver_encoder", "speed_emb"):
            if hasattr(self, name):
                delattr(self, name)

        self.spk_emb_proj = nn.Linear(self.speaker_embedding_dim, self.model_dim)
        self.lang_embedding = nn.Embedding(
            self.num_language_embeddings,
            self.model_dim,
        )

        # Match the context size constructed by the official GPT builder:
        # (max_mel + 2 + max_conditioning_inputs) + (max_text + 2).
        self.gpt = GPT2Model(
            dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=self.max_mel_tokens + self.max_text_tokens + 5,
        )

    def project_speaker_embedding(self, speaker_embedding: mx.array) -> mx.array:
        """Project a raw CampPlus embedding to one GPT conditioning token."""
        if speaker_embedding.ndim == 1:
            speaker_embedding = speaker_embedding[None, :]
        if speaker_embedding.ndim == 2:
            if speaker_embedding.shape[-1] != self.speaker_embedding_dim:
                raise ValueError(
                    "raw CampPlus speaker embedding must have 192 features"
                )
            return self.spk_emb_proj(speaker_embedding)[:, None, :]
        if speaker_embedding.ndim == 3:
            if speaker_embedding.shape[1] != 1:
                raise ValueError("2.5 speaker conditioning must contain one token")
            if speaker_embedding.shape[-1] == self.model_dim:
                return speaker_embedding
            if speaker_embedding.shape[-1] == self.speaker_embedding_dim:
                return self.spk_emb_proj(speaker_embedding)
        raise ValueError(
            "speaker embedding must have shape (192,), (B, 192), "
            "(B, 1, 192), or (B, 1, model_dim)"
        )

    def prepare_conditioning_latents(
        self,
        speech_conditioning: mx.array,
        emo_vec: mx.array,
        batch_size: int,
    ) -> mx.array:
        """Build ``[speaker + emotion, zero, zero]`` as in official 2.5."""
        speaker = self.project_speaker_embedding(speech_conditioning)
        if speaker.shape[0] == 1 and batch_size > 1:
            speaker = mx.broadcast_to(speaker, (batch_size, 1, self.model_dim))
        if speaker.shape[0] != batch_size:
            raise ValueError("speaker conditioning batch size mismatch")

        if emo_vec.ndim == 1:
            emo_vec = emo_vec[None, :]
        if emo_vec.shape[0] == 1 and batch_size > 1:
            emo_vec = mx.broadcast_to(emo_vec, (batch_size, self.model_dim))
        if emo_vec.shape != (batch_size, self.model_dim):
            raise ValueError("emotion vector must have shape (B, model_dim)")

        conditioned_speaker = speaker + emo_vec[:, None, :]
        zero_tokens = mx.zeros(
            (batch_size, 2, self.model_dim),
            dtype=conditioned_speaker.dtype,
        )
        return mx.concatenate([conditioned_speaker, zero_tokens], axis=1)

    def _language_rows(self, language_ids: mx.array | int, batch_size: int) -> mx.array:
        ids = mx.array(language_ids, dtype=mx.int32)
        if ids.ndim == 0:
            ids = mx.broadcast_to(ids[None], (batch_size,))
        if ids.ndim != 1 or ids.shape[0] != batch_size:
            raise ValueError("one language ID is required for each text batch row")
        return ids

    def prepare_inputs(
        self,
        conditioning: mx.array,
        text_tokens: mx.array,
        language_ids: mx.array | int,
    ) -> Tuple[mx.array, mx.array]:
        """Prepare left-padded condition/text embeddings with language fusion.

        The public tokenizer has already appended stop ID 1.  The official
        generator removes start/stop IDs, adds one canonical pair, then pads on
        the left to ``conditioning_len + original_text_len + 2``.
        """
        if text_tokens.ndim != 2:
            raise ValueError("text_tokens must have shape (B, T)")
        batch_size, original_text_len = text_tokens.shape
        if conditioning.ndim != 3:
            raise ValueError("conditioning must have shape (B, C, model_dim)")
        if conditioning.shape[0] not in (1, batch_size):
            raise ValueError("conditioning batch size mismatch")
        language_rows = self._language_rows(language_ids, batch_size)
        single_condition = conditioning.shape[0] == 1
        target_len = conditioning.shape[1] + original_text_len + 2

        batch_embeddings = []
        attention_masks = []
        for row in range(batch_size):
            token_values = [
                int(value)
                for value in text_tokens[row].tolist()
                if int(value) not in (self.start_text_token, self.stop_text_token)
            ]
            canonical = mx.array(
                [self.start_text_token, *token_values, self.stop_text_token],
                dtype=mx.int32,
            )
            positions = mx.arange(canonical.shape[0], dtype=mx.int32)
            text_emb = self.text_embedding(canonical)
            text_emb = text_emb + self.text_pos_embedding.emb(positions)
            text_emb = text_emb + self.lang_embedding(language_rows[row])

            row_condition = conditioning[0] if single_condition else conditioning[row]
            padding = target_len - row_condition.shape[0] - text_emb.shape[0]
            if padding < 0:
                raise ValueError("prepared text exceeds the configured position budget")
            pieces = []
            mask = mx.ones((target_len,), dtype=mx.int32)
            if padding:
                pieces.append(mx.zeros((padding, self.model_dim), dtype=text_emb.dtype))
                mask = mx.concatenate(
                    [
                        mx.zeros((padding,), dtype=mx.int32),
                        mx.ones((target_len - padding,), dtype=mx.int32),
                    ]
                )
            pieces.extend([row_condition, text_emb])
            batch_embeddings.append(mx.concatenate(pieces, axis=0))
            attention_masks.append(mask)

        return mx.stack(batch_embeddings, axis=0), mx.stack(attention_masks, axis=0)

    @staticmethod
    def generation_attention_mask(
        padding_mask: mx.array,
        *,
        query_len: int,
        key_len: int,
    ) -> mx.array:
        """Combine the official left-padding mask with autoregressive causality."""
        if padding_mask.ndim != 2 or padding_mask.shape[1] != key_len:
            raise ValueError("padding mask length must match the GPT key length")
        if query_len > key_len:
            raise ValueError("query length cannot exceed key length")
        masked_value = mx.array(-1e9, dtype=mx.float32)
        if query_len == key_len:
            causal = mx.triu(mx.ones((query_len, key_len)), k=1)
        else:
            cache_len = key_len - query_len
            query_positions = mx.arange(query_len)[:, None] + cache_len
            key_positions = mx.arange(key_len)[None, :]
            causal = key_positions > query_positions
        causal = mx.where(causal, masked_value, 0.0)[None, None, :, :]
        padding = mx.where(
            padding_mask[:, None, None, :] > 0,
            0.0,
            masked_value,
        )
        return causal + padding

    def generate_step(
        self,
        input_emb: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        temperature: float = 1.0,
        top_k: int = 30,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        generated_tokens: Optional[List[int]] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, List[Tuple[mx.array, mx.array]]]:
        """Generate one semantic token while preserving 2.5 left-padding."""
        mask = None
        if attention_mask is not None:
            cache_len = 0 if cache is None else cache[0][0].shape[1]
            key_len = cache_len + input_emb.shape[1]
            mask = self.generation_attention_mask(
                attention_mask,
                query_len=input_emb.shape[1],
                key_len=key_len,
            )
        hidden, new_cache = self.gpt(input_emb, mask=mask, cache=cache)
        hidden = self.final_norm(hidden[:, -1:, :])
        logits = self.mel_head(hidden)
        next_token = self._sample(
            logits[:, 0, :],
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            generated_tokens,
        )
        return next_token, logits, new_cache


# Naming parity with the official source.
UnifiedVoice = UnifiedVoiceV25
