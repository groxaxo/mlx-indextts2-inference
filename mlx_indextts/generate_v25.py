"""Native MLX inference pipeline for the public IndexTTS 2.5 release."""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

from mlx_indextts.generate import compress_silence, crossfade_segments, time_stretch_wsola
from mlx_indextts.generate_v2 import (
    EMOTION_CATEGORIES,
    IndexTTSv2,
    parse_emotion,
)
from mlx_indextts.model_manifest import load_manifest
from mlx_indextts.normalizer_v25 import IndexTTS25TextFrontend
from mlx_indextts.performance import configure_mlx_runtime, configure_torch_threads
from mlx_indextts.qwen_emotion import DEFAULT_QWEN_EMOTION_MODEL, get_qwen_emotion
from mlx_indextts.speaker_cache_v25 import load_speaker_cache, save_speaker_cache


@dataclass(frozen=True)
class StreamChunk:
    """One completed text-segment waveform from segment-level streaming."""

    audio: np.ndarray
    sample_rate: int
    segment_index: int
    segment_count: int
    completed: bool
    resolved_language: str


def validate_emotion_sources(
    *,
    emotion_reference_audio: str | None = None,
    emotion: Any = None,
    use_emo_text: bool = False,
) -> None:
    selected = sum(
        (
            bool(emotion_reference_audio),
            emotion is not None,
            bool(use_emo_text),
        )
    )
    if selected > 1:
        raise ValueError(
            "emotion reference, explicit emotion, and Qwen emotion text are mutually exclusive"
        )


def normalize_emotion_input(
    emotion: str | Mapping[str, float] | Sequence[float],
) -> dict[str, float]:
    """Normalize named, mapped, or official ordered-vector emotion controls."""
    if isinstance(emotion, str):
        return parse_emotion(emotion)
    if isinstance(emotion, Mapping):
        normalized = {
            str(key).lower(): max(0.0, min(1.2, float(value)))
            for key, value in emotion.items()
            if str(key).lower() in EMOTION_CATEGORIES
        }
        if not normalized:
            raise ValueError("explicit emotion mapping contains no supported emotion")
        return normalized
    if isinstance(emotion, Sequence) and not isinstance(emotion, (bytes, bytearray)):
        if len(emotion) != len(EMOTION_CATEGORIES):
            raise ValueError("emotion vector must contain exactly eight values")
        return {
            name: max(0.0, min(1.2, float(value)))
            for name, value in zip(EMOTION_CATEGORIES, emotion)
        }
    raise TypeError("emotion must be a name, mapping, or eight-value sequence")


def _resample_sequence_nearest(sequence: mx.array, target_length: int) -> mx.array:
    """Nearest-neighbor sequence resize for optional GPT-latent rate alignment."""
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    source_length = sequence.shape[1]
    if source_length == target_length:
        return sequence
    positions = mx.floor(mx.arange(target_length) * source_length / target_length)
    positions = mx.minimum(positions.astype(mx.int32), source_length - 1)
    return sequence[:, positions, :]


class IndexTTSv25(IndexTTSv2):
    """IndexTTS 2.5 with MLX GPT, EnhancedCodec, S2Mel, and BigVGAN."""

    model_version = "2.5"
    sample_rate = 22050

    def __init__(
        self,
        model_dir: str,
        config_path: str | None = None,
        device: str = "mps",
        memory_limit_gb: float | None = None,
        quantize_bits: int | None = None,
        qwen_model_path: str = DEFAULT_QWEN_EMOTION_MODEL,
        use_gpt_latent: bool = False,
    ) -> None:
        if memory_limit_gb is None:
            configure_mlx_runtime()
        elif memory_limit_gb > 0:
            configure_mlx_runtime(memory_limit_gb=memory_limit_gb)
        configure_torch_threads()

        self.model_dir = Path(model_dir).resolve()
        self.mlx_model_dir = self.model_dir
        self.device = self._resolve_torch_device(device)
        self.quantize_bits = quantize_bits
        self.qwen_model_path = qwen_model_path
        self.use_gpt_latent = use_gpt_latent
        self.config_path = str(
            Path(config_path).resolve() if config_path else self.model_dir / "config.yaml"
        )
        if not Path(self.config_path).is_file():
            raise FileNotFoundError(f"IndexTTS 2.5 config not found: {self.config_path}")

        self.manifest = load_manifest(self.model_dir)
        if str(self.manifest["model_version"]) != "2.5":
            raise ValueError("IndexTTSv25 requires a converted 2.5 model manifest")
        self.model_revision = str(self.manifest["source_revision"])
        self.cfg = OmegaConf.load(self.config_path)
        self.stop_mel_token = int(self.cfg.gpt.stop_mel_token)

        self._preprocessing_initialized = False
        self.semantic_model = None
        self.campplus = None
        self.semantic_mean = None
        self.semantic_std = None
        self.extract_features = None
        self.cache: dict[str, Any] = {}
        self._reference_cache: dict[str, dict[str, Any]] = {}
        self.last_generation_info: dict[str, Any] = {}

        self._load_emotion_matrices()
        self._init_mlx_models_v25()
        self.text_frontend = IndexTTS25TextFrontend(self.model_dir)
        glossary = self.model_dir / "glossary.yaml"
        if glossary.is_file():
            self.text_frontend.normalizer.load_glossary(glossary)
        self._init_mel_config()

    @staticmethod
    def _resolve_torch_device(device: str) -> str:
        requested = str(device).lower()
        if requested.startswith("mps") and not torch.backends.mps.is_available():
            return "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested

    def _init_mlx_models_v25(self) -> None:
        from mlx_indextts.config import IndexTTSConfig
        from mlx_indextts.models.bigvgan_v2 import BigVGANV2, BigVGANV2Config
        from mlx_indextts.models.codec_v25 import EnhancedCodecV25
        from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25
        from mlx_indextts.models.s2mel import create_s2mel_from_config

        config = IndexTTSConfig.from_omegaconf(self.cfg)
        config.version = 2.5
        self.gpt = UnifiedVoiceV25(config)

        saved_quantization = self.manifest.get("quantization")
        if saved_quantization:
            bits = int(saved_quantization["bits"])
            group_size = int(saved_quantization.get("group_size", 64))
            nn.quantize(self.gpt.gpt, bits=bits, group_size=group_size)
        self.gpt.load_weights(str(self.model_dir / "gpt.safetensors"), strict=True)
        if self.quantize_bits and not saved_quantization:
            if self.quantize_bits not in (4, 5, 6, 8):
                raise ValueError("quantize_bits must be 4, 5, 6, 8, or None")
            nn.quantize(self.gpt.gpt, bits=self.quantize_bits, group_size=64)

        codec_cfg = self.cfg.semantic_codec
        self.semantic_codec_mlx = EnhancedCodecV25(
            codebook_size=int(codec_cfg.get("codebook_size", 8192)),
            hidden_size=int(codec_cfg.get("hidden_size", 1024)),
            codebook_dim=int(codec_cfg.get("codebook_dim", 8)),
            vocos_dim=int(codec_cfg.get("vocos_dim", 384)),
            vocos_intermediate_dim=int(codec_cfg.get("vocos_intermediate_dim", 2048)),
            vocos_num_layers=int(codec_cfg.get("vocos_num_layers", 12)),
        )
        self.semantic_codec_mlx.load_weights(
            str(self.model_dir / "codec.safetensors"),
            strict=True,
        )

        s2mel_config = OmegaConf.to_container(self.cfg.s2mel, resolve=True)
        self.s2mel_mlx = create_s2mel_from_config(s2mel_config)
        self.s2mel_mlx.load_weights(
            str(self.model_dir / "s2mel.safetensors"),
            strict=True,
        )
        self.s2mel_mlx.eval()

        self.bigvgan_mlx = BigVGANV2(BigVGANV2Config())
        self.bigvgan_mlx.load_weights(
            str(self.model_dir / "bigvgan.safetensors"),
            strict=True,
        )

    def _ensure_pytorch_modules(self) -> None:
        if self._preprocessing_initialized:
            return
        local_w2v = self.model_dir.parent / "facebook-w2v-bert-2.0"
        if local_w2v.is_dir() and not os.environ.get("INDEXTTS_W2V_BERT_DIR"):
            os.environ["INDEXTTS_W2V_BERT_DIR"] = str(local_w2v)
        self._init_pytorch_modules()
        self._preprocessing_initialized = True

    def _load_speaker_v25(self, cache_path: str) -> dict[str, Any]:
        cached = load_speaker_cache(cache_path, model_revision=self.model_revision)
        return {
            "audio_path": cache_path,
            **{
                name: torch.from_numpy(value).to(self.device)
                for name, value in cached.features.items()
            },
            "cache_metadata": cached.metadata,
        }

    def save_speaker(self, audio_path: str, output_path: str) -> None:
        reference = self._process_reference_audio(audio_path)
        features = {
            name: reference[name].detach().cpu().numpy()
            for name in ("spk_cond_emb", "ref_mel", "style", "prompt_condition")
        }
        save_speaker_cache(
            output_path,
            features,
            model_revision=self.model_revision,
            source_audio=audio_path,
            preprocessing={
                "max_seconds": 15,
                "semantic_sample_rate": 16000,
                "mel_sample_rate": self.sample_rate,
                "campplus_features": 80,
            },
        )

    @torch.no_grad()
    def _process_reference_audio(self, audio_path: str) -> dict[str, Any]:
        cache_key = str(Path(audio_path).resolve())
        cached = self._reference_cache.get(cache_key)
        if cached is not None:
            self.cache = cached
            return cached
        if str(audio_path).lower().endswith(".npz"):
            reference = self._load_speaker_v25(audio_path)
            self._reference_cache[cache_key] = reference
            self.cache = reference
            return reference

        self._ensure_pytorch_modules()
        audio_np, source_rate = librosa.load(audio_path, sr=None, mono=True)
        max_samples = int(source_rate * 15)
        audio = torch.from_numpy(audio_np[:max_samples]).float()[None, :]
        audio_22k = torchaudio.functional.resample(audio, source_rate, self.sample_rate)
        audio_16k = torchaudio.functional.resample(audio, source_rate, 16000)
        spk_cond_emb = self._get_semantic_embedding(audio_16k)
        ref_mel = self.mel_fn(audio_22k.to(self.device).float())
        target_lengths = mx.array([ref_mel.shape[2]], dtype=mx.int32)

        fbank = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(self.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        style = self.campplus(fbank[None, ...])

        semantic_mx = mx.array(spk_cond_emb.detach().cpu().numpy())
        prompt_condition, _, _, _, _ = self.s2mel_mlx.length_regulator(
            semantic_mx,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None,
        )
        mx.eval(prompt_condition)
        reference = {
            "audio_path": audio_path,
            "spk_cond_emb": spk_cond_emb,
            "ref_mel": ref_mel,
            "style": style,
            "prompt_condition": torch.from_numpy(np.asarray(prompt_condition)).to(self.device),
        }
        self._reference_cache[cache_key] = reference
        self.cache = reference
        return reference

    def _mlx_reference_features(self, reference: dict[str, Any]) -> dict[str, mx.array]:
        cached = reference.get("_mlx_features_v25")
        if cached is not None:
            return cached
        semantic = mx.array(reference["spk_cond_emb"].detach().cpu().numpy())
        semantic_ncl = semantic.transpose(0, 2, 1)
        lengths = mx.array([semantic.shape[1]], dtype=mx.int32)
        features = {
            "semantic_ncl": semantic_ncl,
            "lengths": lengths,
            "emotion_vec": self.gpt.get_emovec(semantic_ncl, lengths),
            "prompt_condition": mx.array(
                reference["prompt_condition"].detach().cpu().numpy()
            ),
            "ref_mel": mx.array(reference["ref_mel"].detach().cpu().numpy()),
            "style": mx.array(reference["style"].detach().cpu().numpy()),
        }
        mx.eval(*features.values())
        reference["_mlx_features_v25"] = features
        return features

    def _emotion_vector(
        self,
        *,
        text: str,
        speaker_reference: dict[str, Any],
        speaker_features: dict[str, mx.array],
        emotion_reference_audio: str | None,
        emotion: str | Mapping[str, float] | Sequence[float] | None,
        use_emo_text: bool,
        emo_text: str | None,
        emo_alpha: float,
        use_random: bool,
    ) -> tuple[mx.array, str]:
        validate_emotion_sources(
            emotion_reference_audio=emotion_reference_audio,
            emotion=emotion,
            use_emo_text=use_emo_text,
        )
        alpha = max(0.0, min(1.0, float(emo_alpha)))
        base = speaker_features["emotion_vec"]
        source = "speaker_reference"
        if emotion_reference_audio:
            emotion_reference = self._process_reference_audio(emotion_reference_audio)
            emotion_features = self._mlx_reference_features(emotion_reference)
            base = base + alpha * (emotion_features["emotion_vec"] - base)
            source = "emotion_reference"

        explicit = emotion
        if use_emo_text:
            result = get_qwen_emotion(self.qwen_model_path).inference(emo_text or text)
            explicit = result.weights
            source = "qwen_text"
        if explicit is None:
            return base, source

        weights = normalize_emotion_input(explicit)
        weights = {key: value * alpha for key, value in weights.items()}
        style = speaker_reference["style"]
        matrix_vector = self._compute_emotion_vector(
            weights,
            style,
            use_random=use_random,
        )
        matrix_vector_mx = mx.array(matrix_vector.detach().cpu().numpy())
        weight_sum = sum(weights.get(name, 0.0) for name in EMOTION_CATEGORIES)
        return matrix_vector_mx + (1.0 - weight_sum) * base, source if use_emo_text else "explicit"

    def _generate_semantic_codes(
        self,
        conditioning: mx.array,
        text_tokens: mx.array,
        language_id: int,
        *,
        max_mel_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> list[int]:
        input_emb, padding_mask = self.gpt.prepare_inputs(
            conditioning,
            text_tokens,
            language_ids=language_id,
        )
        mel_start = mx.array([[self.gpt.start_mel_token]], dtype=mx.int32)
        mel_start_emb = self.gpt.mel_embedding(mel_start)
        mel_start_emb = mel_start_emb + self.gpt.mel_pos_embedding.get_fixed_embedding(0)
        current_emb = mx.concatenate([input_emb, mel_start_emb], axis=1)
        current_mask = mx.concatenate(
            [padding_mask, mx.ones((padding_mask.shape[0], 1), dtype=mx.int32)],
            axis=1,
        )
        generated: list[int] = []
        cache = None
        for _ in range(max_mel_tokens):
            next_token, _, cache = self.gpt.generate_step(
                current_emb,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated,
                attention_mask=current_mask,
            )
            token = int(next_token[0].item())
            if token == self.gpt.stop_mel_token:
                break
            generated.append(token)
            mx.eval(cache)
            last_token = mx.array([[token]], dtype=mx.int32)
            current_emb = self.gpt.mel_embedding(last_token)
            current_emb = current_emb + self.gpt.mel_pos_embedding.get_fixed_embedding(
                len(generated)
            )
            current_mask = mx.concatenate(
                [current_mask, mx.ones((current_mask.shape[0], 1), dtype=mx.int32)],
                axis=1,
            )
        if len(generated) >= max_mel_tokens:
            warnings.warn(
                f"generation reached max_mel_tokens={max_mel_tokens}",
                RuntimeWarning,
            )
        return compress_silence(generated)

    def _synthesize_segment(
        self,
        *,
        text_tokens: mx.array,
        codes: list[int],
        conditioning: mx.array,
        prompt_condition: mx.array,
        ref_mel: mx.array,
        style: mx.array,
        diffusion_steps: int,
        cfg_rate: float,
        duration_factor: float,
    ) -> np.ndarray:
        codes_mx = mx.array([codes], dtype=mx.int32)
        semantic = self.semantic_codec_mlx.decode(codes_mx)
        if self.use_gpt_latent:
            latent = self.gpt.forward_latent(conditioning, text_tokens, codes_mx)
            latent = self.s2mel_mlx.gpt_layer(latent)
            latent = _resample_sequence_nearest(latent, semantic.shape[1])
            semantic = semantic + latent

        target_length = max(1, int(semantic.shape[1] * 1.72 * duration_factor))
        cond, _, _, _, _ = self.s2mel_mlx.length_regulator(
            semantic,
            mx.array([target_length], dtype=mx.int32),
            n_quantizers=3,
            f0=None,
        )
        combined = mx.concatenate([prompt_condition, cond], axis=1)
        mel = self.s2mel_mlx.cfm.inference(
            mu=combined,
            x_lens=mx.array([combined.shape[1]], dtype=mx.int32),
            prompt=ref_mel,
            style=style,
            f0=None,
            n_timesteps=diffusion_steps,
            temperature=1.0,
            inference_cfg_rate=cfg_rate,
        )
        mel = mel[:, :, ref_mel.shape[-1] :]
        audio = self.bigvgan_mlx(mel)
        mx.eval(audio)
        waveform = np.asarray(audio[0, 0]).astype(np.float32, copy=False)
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1.0:
            waveform = waveform / peak
        return np.clip(waveform, -0.99, 0.99).astype(np.float32, copy=False)

    def stream(
        self,
        text: str,
        reference_audio: str,
        *,
        language: str = "auto",
        emotion_reference_audio: str | None = None,
        emotion: str | Mapping[str, float] | Sequence[float] | None = None,
        use_emo_text: bool = False,
        emo_text: str | None = None,
        emo_alpha: float = 1.0,
        use_random: bool = False,
        max_mel_tokens: int = 1500,
        max_text_tokens_per_segment: int = 120,
        interval_silence: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        repetition_penalty: float = 10.0,
        diffusion_steps: int = 25,
        cfg_rate: float = 0.7,
        duration_factor: float = 1.0,
        text_normalization: bool = True,
        seed: int | None = None,
        speed: float = 1.0,
    ) -> Iterator[StreamChunk]:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        if max_mel_tokens < 1 or max_text_tokens_per_segment < 1:
            raise ValueError("token limits must be positive")
        if diffusion_steps < 1:
            raise ValueError("diffusion_steps must be positive")
        if duration_factor <= 0 or speed <= 0:
            raise ValueError("duration_factor and speed must be positive")
        validate_emotion_sources(
            emotion_reference_audio=emotion_reference_audio,
            emotion=emotion,
            use_emo_text=use_emo_text,
        )
        if seed is not None:
            import random

            mx.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

        prepared = self.text_frontend.prepare(
            text,
            language=language,
            text_normalization=text_normalization,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            text_position_capacity=self.gpt.text_pos_embedding.emb.weight.shape[0],
        )
        speaker_reference = self._process_reference_audio(reference_audio)
        speaker_features = self._mlx_reference_features(speaker_reference)
        emotion_vector, emotion_source = self._emotion_vector(
            text=text,
            speaker_reference=speaker_reference,
            speaker_features=speaker_features,
            emotion_reference_audio=emotion_reference_audio,
            emotion=emotion,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            emo_alpha=emo_alpha,
            use_random=use_random,
        )
        conditioning = self.gpt.prepare_conditioning_latents(
            speaker_features["style"],
            emotion_vector,
            batch_size=1,
        )
        mx.eval(conditioning)

        segment_count = len(prepared.token_ids)
        self.last_generation_info = {
            "model_version": "2.5",
            "model_revision": self.model_revision,
            "resolved_language": prepared.language,
            "language_ambiguous": prepared.language_ambiguous,
            "segments": segment_count,
            "emotion_source": emotion_source,
        }
        silence = np.zeros(
            int(self.sample_rate * max(0, interval_silence) / 1000),
            dtype=np.float32,
        )
        for index, token_ids in enumerate(prepared.token_ids):
            text_tokens = mx.array([token_ids], dtype=mx.int32)
            codes = self._generate_semantic_codes(
                conditioning,
                text_tokens,
                prepared.language_id,
                max_mel_tokens=max_mel_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            if not codes:
                raise RuntimeError(f"no semantic codes generated for segment {index + 1}")
            audio = self._synthesize_segment(
                text_tokens=text_tokens,
                codes=codes,
                conditioning=conditioning,
                prompt_condition=speaker_features["prompt_condition"],
                ref_mel=speaker_features["ref_mel"],
                style=speaker_features["style"],
                diffusion_steps=diffusion_steps,
                cfg_rate=cfg_rate,
                duration_factor=duration_factor,
            )
            if speed != 1.0:
                audio = time_stretch_wsola(audio, rate=speed, sample_rate=self.sample_rate)
            if silence.size and index < segment_count - 1:
                audio = np.concatenate([audio, silence])
            yield StreamChunk(
                audio=audio,
                sample_rate=self.sample_rate,
                segment_index=index,
                segment_count=segment_count,
                completed=index == segment_count - 1,
                resolved_language=prepared.language,
            )

    def generate(
        self,
        text: str,
        reference_audio: str,
        output_path: str | None = None,
        *,
        segment_overlap_ms: int = 50,
        verbose: bool = False,
        **options: Any,
    ) -> np.ndarray:
        start = time.perf_counter()
        chunks = list(self.stream(text, reference_audio, **options))
        if not chunks:
            raise RuntimeError("no audio generated")
        interval_silence = int(options.get("interval_silence", 200))
        if len(chunks) == 1:
            audio = chunks[0].audio
        elif interval_silence <= 0 and segment_overlap_ms > 0:
            audio = np.asarray(
                crossfade_segments(
                    [mx.array(chunk.audio) for chunk in chunks],
                    self.sample_rate,
                    segment_overlap_ms,
                )
            )
        else:
            audio = np.concatenate([chunk.audio for chunk in chunks])
        audio = np.asarray(audio, dtype=np.float32)
        if output_path:
            import soundfile as sf

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio, self.sample_rate)
        if verbose:
            duration = len(audio) / self.sample_rate
            elapsed = time.perf_counter() - start
            rtf = elapsed / duration if duration else float("inf")
            print(
                f"IndexTTS 2.5 generated {duration:.2f}s in {elapsed:.2f}s "
                f"(RTF {rtf:.3f}, language={chunks[0].resolved_language})"
            )
        return audio


# Naming parity with the upstream 2.5 runtime.
IndexTTS2 = IndexTTSv25
