"""UnifiedVoice v2 GPT model for IndexTTS 2.0.

This extends the 1.5 UnifiedVoice with emotion conditioning modules:
- emo_conditioning_encoder (ConformerEncoder)
- emo_perceiver_encoder (PerceiverResampler with num_latents=1)
- speed_emb, emo_layer, emovec_layer
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_indextts.config import ConformerConfig, IndexTTSConfig
from mlx_indextts.models.attention import AttentionBlock, LearnedPositionEmbedding
from mlx_indextts.models.conformer import ConformerEncoder
from mlx_indextts.models.gpt2 import GPT2Model
from mlx_indextts.models.perceiver import PerceiverResampler
from mlx_indextts.models.sampling import (
    RepetitionPenaltyState,
    apply_repetition_penalty,
    resolve_repetition_state,
    sample_logits,
    schedule_decode_outputs,
)


class ConditioningEncoder(nn.Module):
    """Simple conditioning encoder with attention blocks."""

    def __init__(
        self,
        spec_dim: int,
        embedding_dim: int,
        num_attn_heads: int = 4,
        num_blocks: int = 6,
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        self.attn = [
            AttentionBlock(embedding_dim, num_attn_heads)
            for _ in range(num_blocks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Encode NCL conditioning features and return NCL features."""
        x = x.transpose(0, 2, 1)
        x = self.init_conv(x)
        x = x.transpose(0, 2, 1)
        for attn in self.attn:
            x = attn(x)
        return x


class UnifiedVoiceV2(nn.Module):
    """UnifiedVoice v2 model for IndexTTS 2.0."""

    def __init__(self, config: IndexTTSConfig):
        super().__init__()
        self.config = config
        gpt_config = config.gpt

        self.model_dim = gpt_config.model_dim
        self.num_heads = gpt_config.heads
        self.num_layers = gpt_config.layers
        self.max_mel_tokens = gpt_config.max_mel_tokens
        self.max_text_tokens = gpt_config.max_text_tokens
        self.mel_length_compression = gpt_config.mel_length_compression

        self.number_text_tokens = gpt_config.number_text_tokens
        self.number_mel_codes = gpt_config.number_mel_codes
        self.start_text_token = gpt_config.start_text_token
        self.stop_text_token = gpt_config.stop_text_token
        self.start_mel_token = gpt_config.start_mel_token
        self.stop_mel_token = gpt_config.stop_mel_token

        self.condition_type = gpt_config.condition_type
        self.cond_num = gpt_config.condition_num_latent

        if gpt_config.condition_type == "conformer_perceiver":
            cond_config = gpt_config.condition_module or ConformerConfig()
            cond_config_v2 = ConformerConfig(
                input_size=1024,
                output_size=cond_config.output_size,
                linear_units=cond_config.linear_units,
                attention_heads=cond_config.attention_heads,
                num_blocks=cond_config.num_blocks,
                dropout_rate=cond_config.dropout_rate,
                input_layer=cond_config.input_layer,
                pos_enc_layer_type=cond_config.pos_enc_layer_type,
                normalize_before=cond_config.normalize_before,
                use_cnn_module=cond_config.use_cnn_module,
                cnn_module_kernel=cond_config.cnn_module_kernel,
                perceiver_mult=cond_config.perceiver_mult,
            )
            self.conditioning_encoder = ConformerEncoder(cond_config_v2)
            self.perceiver_encoder = PerceiverResampler(
                dim=self.model_dim,
                n_dim_context=cond_config.output_size,
                n_latents=self.cond_num,
                n_heads=cond_config.attention_heads,
                n_ff_mult=cond_config.perceiver_mult,
            )
        elif gpt_config.condition_type == "perceiver":
            self.conditioning_encoder = ConditioningEncoder(
                1024,
                self.model_dim,
                num_attn_heads=self.num_heads,
            )
            self.perceiver_encoder = PerceiverResampler(
                dim=self.model_dim,
                n_latents=self.cond_num,
            )
        else:
            self.conditioning_encoder = ConditioningEncoder(
                1024,
                self.model_dim,
                num_attn_heads=self.num_heads,
            )
            self.perceiver_encoder = None

        emo_cond_config = gpt_config.emo_condition_module
        if emo_cond_config is None:
            raise ValueError("emo_condition_module is required for UnifiedVoice v2")

        emo_cond_config_v2 = ConformerConfig(
            input_size=1024,
            output_size=emo_cond_config.output_size,
            linear_units=emo_cond_config.linear_units,
            attention_heads=emo_cond_config.attention_heads,
            num_blocks=emo_cond_config.num_blocks,
            dropout_rate=emo_cond_config.dropout_rate,
            input_layer=emo_cond_config.input_layer,
            pos_enc_layer_type=emo_cond_config.pos_enc_layer_type,
            normalize_before=emo_cond_config.normalize_before,
            use_cnn_module=emo_cond_config.use_cnn_module,
            cnn_module_kernel=emo_cond_config.cnn_module_kernel,
            perceiver_mult=emo_cond_config.perceiver_mult,
        )
        self.emo_conditioning_encoder = ConformerEncoder(emo_cond_config_v2)
        self.emo_perceiver_encoder = PerceiverResampler(
            dim=1024,
            n_dim_context=emo_cond_config.output_size,
            n_latents=1,
            n_heads=emo_cond_config.attention_heads,
            n_ff_mult=emo_cond_config.perceiver_mult,
        )

        self.emo_layer = nn.Linear(self.model_dim, self.model_dim)
        self.emovec_layer = nn.Linear(1024, self.model_dim)
        self.speed_emb = nn.Embedding(2, self.model_dim)

        self.text_embedding = nn.Embedding(
            self.number_text_tokens + 1,
            self.model_dim,
        )
        self.mel_embedding = nn.Embedding(
            self.number_mel_codes,
            self.model_dim,
        )

        self.mel_pos_embedding = LearnedPositionEmbedding(
            self.max_mel_tokens + 3,
            self.model_dim,
        )
        self.text_pos_embedding = LearnedPositionEmbedding(
            self.max_text_tokens + 2,
            self.model_dim,
        )

        max_seq_len = (
            self.max_mel_tokens
            + self.max_text_tokens
            + self.cond_num
            + 6
        )
        self.gpt = GPT2Model(
            dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=max_seq_len,
        )

        self.final_norm = nn.LayerNorm(self.model_dim)
        self.text_head = nn.Linear(
            self.model_dim,
            self.number_text_tokens + 1,
        )
        self.mel_head = nn.Linear(
            self.model_dim,
            self.number_mel_codes,
        )

    def new_repetition_state(self) -> RepetitionPenaltyState:
        """Create list-compatible token history with an incremental seen mask."""
        return RepetitionPenaltyState(self.number_mel_codes)

    def get_conditioning(
        self,
        speech_conditioning_input: mx.array,
        cond_mel_lengths: Optional[mx.array] = None,
    ) -> mx.array:
        """Extract speaker conditioning from semantic reference features."""
        if self.condition_type == "conformer_perceiver":
            x = speech_conditioning_input.transpose(0, 2, 1)
            x, _ = self.conditioning_encoder(x, cond_mel_lengths)
            return self.perceiver_encoder(x)
        if self.condition_type == "perceiver":
            x = self.conditioning_encoder(speech_conditioning_input)
            return self.perceiver_encoder(x.transpose(0, 2, 1))

        x = self.conditioning_encoder(speech_conditioning_input)
        return x.mean(axis=-1, keepdims=True).transpose(0, 2, 1)

    def get_emo_conditioning(
        self,
        emo_conditioning_input: mx.array,
        emo_cond_lengths: Optional[mx.array] = None,
    ) -> mx.array:
        """Extract one emotion latent from semantic reference features."""
        x = emo_conditioning_input.transpose(0, 2, 1)
        x, _ = self.emo_conditioning_encoder(x, emo_cond_lengths)
        return self.emo_perceiver_encoder(x).squeeze(1)

    def get_emovec(
        self,
        emo_conditioning_input: mx.array,
        emo_cond_lengths: Optional[mx.array] = None,
    ) -> mx.array:
        """Project the reference emotion latent into GPT model space."""
        emo_vec_raw = self.get_emo_conditioning(
            emo_conditioning_input,
            emo_cond_lengths,
        )
        return self.emo_layer(self.emovec_layer(emo_vec_raw))

    def merge_emovec(
        self,
        speech_conditioning_input: mx.array,
        emo_conditioning_input: mx.array,
        cond_lengths: Optional[mx.array] = None,
        emo_cond_lengths: Optional[mx.array] = None,
        alpha: float = 1.0,
    ) -> mx.array:
        """Blend speaker and emotion reference vectors."""
        base_vec = self.get_emovec(
            speech_conditioning_input,
            cond_lengths,
        )
        emo_vec = self.get_emovec(
            emo_conditioning_input,
            emo_cond_lengths,
        )
        return base_vec + alpha * (emo_vec - base_vec)

    def prepare_conditioning_latents(
        self,
        speech_conditioning: mx.array,
        emo_vec: mx.array,
        batch_size: int,
    ) -> mx.array:
        """Prepare speaker, emotion, and speed conditioning tokens."""
        conds_with_emo = speech_conditioning + emo_vec[:, None, :]
        zeros = mx.zeros((batch_size,), dtype=mx.int32)
        ones = mx.ones((batch_size,), dtype=mx.int32)
        duration_emb = self.speed_emb(zeros)[:, None, :]
        duration_emb_half = self.speed_emb(ones)[:, None, :]
        return mx.concatenate(
            [conds_with_emo, duration_emb_half, duration_emb],
            axis=1,
        )

    def prepare_inputs(
        self,
        conditioning: mx.array,
        text_tokens: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Prepare conditioning and text embeddings for semantic generation."""
        batch_size = text_tokens.shape[0]
        start_tokens = mx.full(
            (batch_size, 1),
            self.start_text_token,
            dtype=mx.int32,
        )
        stop_tokens = mx.full(
            (batch_size, 1),
            self.stop_text_token,
            dtype=mx.int32,
        )
        text_tokens = mx.concatenate(
            [start_tokens, text_tokens, stop_tokens],
            axis=1,
        )

        text_emb = self.text_embedding(text_tokens)
        text_emb = text_emb + self.text_pos_embedding(text_emb)
        emb = mx.concatenate([conditioning, text_emb], axis=1)
        mask = mx.ones((batch_size, emb.shape[1]))
        return emb, mask

    def generate_step(
        self,
        input_emb: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        temperature: float = 1.0,
        top_k: int = 30,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        generated_tokens: Optional[List[int]] = None,
    ) -> Tuple[mx.array, mx.array, List[Tuple[mx.array, mx.array]]]:
        """Generate one semantic token and schedule token/cache together."""
        hidden, new_cache = self.gpt(input_emb, cache=cache)
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
        schedule_decode_outputs(next_token, new_cache)
        return next_token, logits, new_cache

    def _apply_repetition_penalty(
        self,
        logits: mx.array,
        generated_tokens: List[int],
        penalty: float,
    ) -> mx.array:
        """Apply a sign-aware penalty to tokens already generated."""
        history = resolve_repetition_state(
            self,
            generated_tokens,
            self.number_mel_codes,
        )
        return apply_repetition_penalty(logits, history, penalty)

    def _sample(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: int = 30,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        generated_tokens: Optional[List[int]] = None,
    ) -> mx.array:
        """Sample with bounded candidates and incremental repetition state."""
        history = resolve_repetition_state(
            self,
            generated_tokens,
            self.number_mel_codes,
        )
        return sample_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=history,
        )

    def forward_latent(
        self,
        conditioning: mx.array,
        text_tokens: mx.array,
        mel_codes: mx.array,
    ) -> mx.array:
        """Return GPT latents aligned to generated semantic codes."""
        batch_size = text_tokens.shape[0]
        mel_len = mel_codes.shape[1]

        start_tokens = mx.full(
            (batch_size, 1),
            self.start_text_token,
            dtype=mx.int32,
        )
        stop_tokens = mx.full(
            (batch_size, 1),
            self.stop_text_token,
            dtype=mx.int32,
        )
        text_tokens = mx.concatenate(
            [start_tokens, text_tokens, stop_tokens],
            axis=1,
        )
        text_emb = self.text_embedding(text_tokens)
        text_emb = text_emb + self.text_pos_embedding(text_emb)

        mel_start = mx.full(
            (batch_size, 1),
            self.start_mel_token,
            dtype=mx.int32,
        )
        mel_stop = mx.full(
            (batch_size, 1),
            self.stop_mel_token,
            dtype=mx.int32,
        )
        mel_tokens = mx.concatenate(
            [mel_start, mel_codes, mel_stop],
            axis=1,
        )
        mel_emb = self.mel_embedding(mel_tokens)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_emb)

        emb = mx.concatenate([conditioning, text_emb, mel_emb], axis=1)
        hidden, _ = self.gpt(emb)

        cond_len = conditioning.shape[1]
        enc = self.final_norm(hidden[:, cond_len:, :])
        text_len_with_tokens = text_emb.shape[1]
        return enc[
            :,
            text_len_with_tokens : text_len_with_tokens + mel_len,
            :,
        ]
