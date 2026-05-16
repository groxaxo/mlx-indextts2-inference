"""UnifiedVoice v2 GPT model for IndexTTS 2.0.

This extends the 1.5 UnifiedVoice with emotion conditioning modules:
- emo_conditioning_encoder (ConformerEncoder)
- emo_perceiver_encoder (PerceiverResampler with num_latents=1)
- speed_emb, emo_layer, emovec_layer
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_indextts.config import IndexTTSConfig, ConformerConfig
from mlx_indextts.models.gpt2 import GPT2Model
from mlx_indextts.models.conformer import ConformerEncoder
from mlx_indextts.models.perceiver import PerceiverResampler
from mlx_indextts.models.attention import LearnedPositionEmbedding, AttentionBlock


class ConditioningEncoder(nn.Module):
    """Simple conditioning encoder with attention blocks."""

    def __init__(self, spec_dim: int, embedding_dim: int, num_attn_heads: int = 4, num_blocks: int = 6):
        super().__init__()
        self.init_conv = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        self.attn = [AttentionBlock(embedding_dim, num_attn_heads) for _ in range(num_blocks)]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input mel spectrogram (batch, spec_dim, time) - NCL format

        Returns:
            Encoded features (batch, embedding_dim, time) - NCL format
        """
        x = x.transpose(0, 2, 1)  # NCL -> NLC
        x = self.init_conv(x)
        x = x.transpose(0, 2, 1)  # NLC -> NCL
        for attn in self.attn:
            x = attn(x)
        return x


class UnifiedVoiceV2(nn.Module):
    """UnifiedVoice v2 model for IndexTTS 2.0.

    Extends 1.5 with emotion conditioning:
    - emo_conditioning_encoder: ConformerEncoder for emotion feature extraction
    - emo_perceiver_encoder: PerceiverResampler with 1 latent for emotion
    - speed_emb: Speed control embedding
    - emo_layer, emovec_layer: Emotion projection layers
    """

    def __init__(self, config: IndexTTSConfig):
        """Initialize UnifiedVoice v2.

        Args:
            config: Model configuration (must have emo_condition_module)
        """
        super().__init__()
        self.config = config
        gpt_config = config.gpt

        # Store important params
        self.model_dim = gpt_config.model_dim
        self.num_heads = gpt_config.heads
        self.num_layers = gpt_config.layers
        self.max_mel_tokens = gpt_config.max_mel_tokens
        self.max_text_tokens = gpt_config.max_text_tokens
        self.mel_length_compression = gpt_config.mel_length_compression

        # Token config
        self.number_text_tokens = gpt_config.number_text_tokens
        self.number_mel_codes = gpt_config.number_mel_codes
        self.start_text_token = gpt_config.start_text_token
        self.stop_text_token = gpt_config.stop_text_token
        self.start_mel_token = gpt_config.start_mel_token
        self.stop_mel_token = gpt_config.stop_mel_token

        # Conditioning config
        self.condition_type = gpt_config.condition_type
        self.cond_num = gpt_config.condition_num_latent

        # === Speaker conditioning encoder (same as 1.5) ===
        if gpt_config.condition_type == "conformer_perceiver":
            cond_config = gpt_config.condition_module or ConformerConfig()
            # Override input_size to 1024 for v2 (W2V-BERT semantic features)
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
                1024, self.model_dim, num_attn_heads=self.num_heads
            )
            self.perceiver_encoder = PerceiverResampler(
                dim=self.model_dim,
                n_latents=self.cond_num,
            )
        else:
            self.conditioning_encoder = ConditioningEncoder(
                1024, self.model_dim, num_attn_heads=self.num_heads
            )
            self.perceiver_encoder = None

        # === Emotion conditioning encoder (NEW in 2.0) ===
        emo_cond_config = gpt_config.emo_condition_module
        if emo_cond_config is None:
            raise ValueError("emo_condition_module is required for UnifiedVoice v2")

        # Override input_size to 1024 for v2 (W2V-BERT semantic features)
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
            dim=1024,  # Output dim is 1024 (matches emovec_layer input)
            n_dim_context=emo_cond_config.output_size,
            n_latents=1,  # Single emotion latent
            n_heads=emo_cond_config.attention_heads,
            n_ff_mult=emo_cond_config.perceiver_mult,
        )

        # Emotion projection layers
        self.emo_layer = nn.Linear(self.model_dim, self.model_dim)
        self.emovec_layer = nn.Linear(1024, self.model_dim)

        # Speed embedding (2 speeds: normal=0, half=1)
        self.speed_emb = nn.Embedding(2, self.model_dim)

        # Embeddings
        self.text_embedding = nn.Embedding(self.number_text_tokens + 1, self.model_dim)
        self.mel_embedding = nn.Embedding(self.number_mel_codes, self.model_dim)

        # Position embeddings
        self.mel_pos_embedding = LearnedPositionEmbedding(
            self.max_mel_tokens + 2 + 1, self.model_dim
        )
        self.text_pos_embedding = LearnedPositionEmbedding(
            self.max_text_tokens + 2, self.model_dim
        )

        # GPT backbone
        # v2 has additional conditioning tokens: cond_num + 2 (speed embeddings)
        max_seq_len = self.max_mel_tokens + self.max_text_tokens + self.cond_num + 6
        self.gpt = GPT2Model(
            dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=max_seq_len,
        )

        # Output heads
        self.final_norm = nn.LayerNorm(self.model_dim)
        self.text_head = nn.Linear(self.model_dim, self.number_text_tokens + 1)
        self.mel_head = nn.Linear(self.model_dim, self.number_mel_codes)

    def get_conditioning(
        self,
        speech_conditioning_input: mx.array,
        cond_mel_lengths: Optional[mx.array] = None,
    ) -> mx.array:
        """Extract speaker conditioning from reference audio.

        Args:
            speech_conditioning_input: Semantic features (batch, 1024, time) - NCL format
            cond_mel_lengths: Optional mel lengths

        Returns:
            Conditioning latents (batch, cond_num, model_dim)
        """
        if self.condition_type == "conformer_perceiver":
            # Transpose NCL -> NLC for Conformer
            x = speech_conditioning_input.transpose(0, 2, 1)
            x, mask = self.conditioning_encoder(x, cond_mel_lengths)
            conds = self.perceiver_encoder(x)
        elif self.condition_type == "perceiver":
            # ConditioningEncoder expects NCL format
            x = self.conditioning_encoder(speech_conditioning_input)
            x = x.transpose(0, 2, 1)  # NCL -> NLC for Perceiver
            conds = self.perceiver_encoder(x)
        else:
            x = self.conditioning_encoder(speech_conditioning_input)
            conds = x.mean(axis=-1, keepdims=True)
            conds = conds.transpose(0, 2, 1)

        return conds

    def get_emo_conditioning(
        self,
        emo_conditioning_input: mx.array,
        emo_cond_lengths: Optional[mx.array] = None,
    ) -> mx.array:
        """Extract emotion conditioning from reference audio.

        Args:
            emo_conditioning_input: Semantic features (batch, 1024, time) - NCL format
            emo_cond_lengths: Optional mel lengths

        Returns:
            Emotion vector (batch, 1024)
        """
        # Transpose NCL -> NLC for Conformer
        x = emo_conditioning_input.transpose(0, 2, 1)
        x, mask = self.emo_conditioning_encoder(x, emo_cond_lengths)
        # Perceiver outputs (batch, 1, 1024)
        conds = self.emo_perceiver_encoder(x)
        # Squeeze to (batch, 1024)
        return conds.squeeze(1)

    def get_emovec(
        self,
        emo_conditioning_input: mx.array,
        emo_cond_lengths: Optional[mx.array] = None,
    ) -> mx.array:
        """Get emotion vector for conditioning.

        Args:
            emo_conditioning_input: Semantic features (batch, 1024, time) - NCL format
            emo_cond_lengths: Optional lengths

        Returns:
            Emotion vector (batch, model_dim)
        """
        emo_vec_raw = self.get_emo_conditioning(emo_conditioning_input, emo_cond_lengths)
        emo_vec = self.emovec_layer(emo_vec_raw)
        emo_vec = self.emo_layer(emo_vec)
        return emo_vec

    def merge_emovec(
        self,
        speech_conditioning_input: mx.array,
        emo_conditioning_input: mx.array,
        cond_lengths: Optional[mx.array] = None,
        emo_cond_lengths: Optional[mx.array] = None,
        alpha: float = 1.0,
    ) -> mx.array:
        """Blend speaker-reference and emotion-reference emotion vectors.

        Mirrors the official PyTorch IndexTTS2 formula:
        base_vec + alpha * (emo_vec - base_vec).
        """
        base_vec = self.get_emovec(speech_conditioning_input, cond_lengths)
        emo_vec = self.get_emovec(emo_conditioning_input, emo_cond_lengths)
        return base_vec + alpha * (emo_vec - base_vec)

    def prepare_conditioning_latents(
        self,
        speech_conditioning: mx.array,
        emo_vec: mx.array,
        batch_size: int,
    ) -> mx.array:
        """Prepare full conditioning latents including emotion and speed.

        Args:
            speech_conditioning: Speaker conditioning (batch, cond_num, model_dim)
            emo_vec: Emotion vector (batch, model_dim)
            batch_size: Batch size

        Returns:
            Full conditioning (batch, cond_num + 2, model_dim)
        """
        # Add emotion to speaker conditioning
        # speech_conditioning: (batch, cond_num, model_dim)
        # emo_vec: (batch, model_dim) -> need to expand to (batch, cond_num, model_dim)
        conds_with_emo = speech_conditioning + emo_vec[:, None, :]

        # Speed embeddings
        # duration_emb = speed_emb(0) for normal speed
        # duration_emb_half = speed_emb(1) for half speed
        zeros = mx.zeros((batch_size,), dtype=mx.int32)
        ones = mx.ones((batch_size,), dtype=mx.int32)
        duration_emb = self.speed_emb(zeros)[:, None, :]  # (batch, 1, model_dim)
        duration_emb_half = self.speed_emb(ones)[:, None, :]  # (batch, 1, model_dim)

        # Concatenate: [conds_with_emo, duration_emb_half, duration_emb]
        conds = mx.concatenate([conds_with_emo, duration_emb_half, duration_emb], axis=1)

        return conds

    def prepare_inputs(
        self,
        conditioning: mx.array,
        text_tokens: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Prepare inputs for generation.

        Args:
            conditioning: Full conditioning latents (batch, cond_num + 2, dim)
            text_tokens: Text token IDs (batch, text_len)

        Returns:
            Tuple of (input_embeddings, attention_mask)
        """
        batch_size, text_len = text_tokens.shape

        # Add start/stop tokens to text
        start_tokens = mx.full((batch_size, 1), self.start_text_token, dtype=mx.int32)
        stop_tokens = mx.full((batch_size, 1), self.stop_text_token, dtype=mx.int32)
        text_tokens = mx.concatenate([start_tokens, text_tokens, stop_tokens], axis=1)

        # Get text embeddings with position
        text_emb = self.text_embedding(text_tokens)
        text_emb = text_emb + self.text_pos_embedding(text_emb)

        # Concatenate conditioning and text
        emb = mx.concatenate([conditioning, text_emb], axis=1)

        # Create attention mask (all ones)
        seq_len = emb.shape[1]
        mask = mx.ones((batch_size, seq_len))

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
        """Generate one mel token.

        Args:
            input_emb: Input embeddings (batch, seq_len, dim)
            cache: KV cache
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            generated_tokens: List of previously generated token IDs for repetition penalty

        Returns:
            Tuple of (next_token, logits, updated_cache)
        """
        # Forward through GPT
        hidden, new_cache = self.gpt(input_emb, cache=cache)

        # Get logits for last position
        hidden = self.final_norm(hidden[:, -1:, :])
        logits = self.mel_head(hidden)

        # Sample
        next_token = self._sample(
            logits[:, 0, :], temperature, top_k, top_p,
            repetition_penalty, generated_tokens
        )

        return next_token, logits, new_cache

    def _apply_repetition_penalty(
        self,
        logits: mx.array,
        generated_tokens: List[int],
        penalty: float,
    ) -> mx.array:
        """Apply repetition penalty to logits.

        For tokens that have been generated before:
        - If logits > 0: divide by penalty (reduce probability)
        - If logits < 0: multiply by penalty (reduce probability)

        Args:
            logits: Logits (batch, vocab_size)
            generated_tokens: List of previously generated token IDs
            penalty: Repetition penalty (1.0 = no penalty, >1.0 = penalize)

        Returns:
            Modified logits
        """
        if penalty == 1.0 or not generated_tokens:
            return logits

        vocab_size = logits.shape[-1]
        unique_tokens = sorted({token for token in generated_tokens if 0 <= token < vocab_size})
        if not unique_tokens:
            return logits

        token_ids = mx.array([unique_tokens], dtype=mx.int32)
        token_logits = mx.take_along_axis(logits, token_ids, axis=-1)
        penalized = mx.where(
            token_logits > 0,
            token_logits / penalty,
            token_logits * penalty,
        )
        return mx.put_along_axis(logits, token_ids, penalized, axis=-1)

    def _sample(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: int = 30,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        generated_tokens: Optional[List[int]] = None,
    ) -> mx.array:
        """Sample from logits.

        Args:
            logits: Logits (batch, vocab_size)
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            repetition_penalty: Penalty for repeating tokens
            generated_tokens: List of previously generated token IDs

        Returns:
            Sampled tokens (batch,)
        """
        # Apply repetition penalty first (before temperature)
        if repetition_penalty != 1.0 and generated_tokens:
            logits = self._apply_repetition_penalty(logits, generated_tokens, repetition_penalty)

        if temperature == 0:
            return mx.argmax(logits, axis=-1)

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            top_k_values = mx.topk(logits, top_k)
            threshold = top_k_values[:, :1]
            indices_to_remove = logits < threshold
            logits = mx.where(indices_to_remove, float("-inf"), logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
            sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            first_col = mx.zeros((sorted_indices_to_remove.shape[0], 1), dtype=mx.bool_)
            sorted_indices_to_remove = mx.concatenate([
                first_col,
                sorted_indices_to_remove[:, :-1]
            ], axis=-1)

            indices_to_remove = mx.zeros_like(sorted_indices_to_remove)
            indices_to_remove = mx.put_along_axis(
                indices_to_remove,
                sorted_indices,
                sorted_indices_to_remove,
                axis=-1,
            )
            logits = mx.where(indices_to_remove, float("-inf"), logits)

        # Sample
        probs = mx.softmax(logits, axis=-1)
        return mx.random.categorical(mx.log(probs + 1e-10))

    def forward_latent(
        self,
        conditioning: mx.array,
        text_tokens: mx.array,
        mel_codes: mx.array,
    ) -> mx.array:
        """Forward pass to get latents for S2Mel.

        This is used after generation to get the final latent
        representations for the S2Mel diffusion model.

        Args:
            conditioning: Full conditioning (batch, cond_num + 2, dim)
            text_tokens: Text tokens (batch, text_len)
            mel_codes: Generated mel codes (batch, mel_len)

        Returns:
            Latent features (batch, mel_len, dim)
        """
        batch_size = text_tokens.shape[0]
        mel_len = mel_codes.shape[1]

        # Prepare text: [start, text..., stop]
        start_tokens = mx.full((batch_size, 1), self.start_text_token, dtype=mx.int32)
        stop_tokens = mx.full((batch_size, 1), self.stop_text_token, dtype=mx.int32)
        text_tokens = mx.concatenate([start_tokens, text_tokens, stop_tokens], axis=1)

        # Text embeddings
        text_emb = self.text_embedding(text_tokens)
        text_emb = text_emb + self.text_pos_embedding(text_emb)

        # Prepare mel: [start, mel..., stop]
        mel_start = mx.full((batch_size, 1), self.start_mel_token, dtype=mx.int32)
        mel_stop = mx.full((batch_size, 1), self.stop_mel_token, dtype=mx.int32)
        mel_tokens = mx.concatenate([mel_start, mel_codes, mel_stop], axis=1)

        # Mel embeddings
        mel_emb = self.mel_embedding(mel_tokens)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_emb)

        # Concatenate all: [conditioning, text_emb, mel_emb]
        emb = mx.concatenate([conditioning, text_emb, mel_emb], axis=1)

        # Forward through GPT
        hidden, _ = self.gpt(emb)

        # Apply final norm to non-conditioning part
        cond_len = conditioning.shape[1]
        enc = self.final_norm(hidden[:, cond_len:, :])

        # Extract mel latents
        text_len_with_tokens = text_emb.shape[1]  # text_len + 2
        mel_latent = enc[:, text_len_with_tokens:text_len_with_tokens + mel_len, :]

        return mel_latent
