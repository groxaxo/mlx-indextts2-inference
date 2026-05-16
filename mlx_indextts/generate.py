"""Speech generation with IndexTTS."""

import time
import warnings
from pathlib import Path
from typing import Generator, List, Optional, Union

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_indextts.config import IndexTTSConfig
from mlx_indextts.mel import MelSpectrogramExtractor
from mlx_indextts.tokenizer import TextTokenizer
from mlx_indextts.normalize import TextNormalizer
from mlx_indextts.models.gpt import UnifiedVoice
from mlx_indextts.models.bigvgan import BigVGAN
from mlx_indextts.performance import configure_mlx_runtime


def compress_silence(
    mel_codes: List[int],
    silent_token: int = 52,
    max_consecutive: int = 30,
    keep: int = 10,
) -> List[int]:
    """Compress consecutive silence tokens to prevent unnatural long pauses.

    Matches PyTorch IndexTTS remove_long_silence logic: if more than
    max_consecutive silent tokens exist, shrink runs to at most `keep`.
    """
    count = sum(1 for c in mel_codes if c == silent_token)
    if count <= max_consecutive:
        return mel_codes

    result = []
    consecutive = 0
    for code in mel_codes:
        if code != silent_token:
            result.append(code)
            consecutive = 0
        elif consecutive < keep:
            result.append(code)
            consecutive += 1
        # else: skip excess silence token
    return result


def time_stretch_wsola(
    audio: np.ndarray,
    rate: float,
    frame_ms: int = 30,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Time-stretch audio using WSOLA (Waveform Similarity Overlap-Add).

    Much better than phase vocoder for speech — preserves timbre and naturalness.
    rate > 1.0 speeds up, rate < 1.0 slows down. No pitch change.
    """
    if abs(rate - 1.0) < 1e-3:
        return audio

    frame_len = int(sample_rate * frame_ms / 1000)
    half = frame_len // 2
    # How far the output pointer advances per frame
    syn_hop = half
    # How far the input pointer advances per frame
    ana_hop = int(syn_hop * rate)
    search = half // 2  # search range for best overlap

    n = len(audio)
    # Pre-allocate output buffer
    out_len = int(n / rate) + frame_len
    output = np.zeros(out_len, dtype=np.float32)
    window = np.hanning(frame_len).astype(np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    in_pos = 0
    out_pos = 0

    while in_pos + frame_len < n and out_pos + frame_len < out_len:
        if out_pos == 0:
            best_offset = 0
        else:
            # Search for best match in a region around the expected position
            lo = max(0, in_pos - search)
            hi = min(n - frame_len, in_pos + search)
            if lo >= hi:
                best_offset = 0
            else:
                # Cross-correlation to find best overlap
                ref = output[out_pos:out_pos + half]
                best_corr = -np.inf
                best_offset = 0
                for offset in range(lo, hi):
                    candidate = audio[offset:offset + half]
                    corr = np.dot(ref, candidate)
                    if corr > best_corr:
                        best_corr = corr
                        best_offset = offset - in_pos

        src_pos = in_pos + best_offset
        if src_pos < 0:
            src_pos = 0
        if src_pos + frame_len > n:
            break

        frame = audio[src_pos:src_pos + frame_len] * window
        output[out_pos:out_pos + frame_len] += frame
        norm[out_pos:out_pos + frame_len] += window

        in_pos += ana_hop
        out_pos += syn_hop

    # Normalize overlapping regions
    mask = norm > 1e-8
    output[mask] /= norm[mask]

    # Trim to actual content
    last_nonzero = np.max(np.where(mask)[0]) + 1 if np.any(mask) else out_pos
    return output[:last_nonzero]


def crossfade_segments(
    audio_segments: List[mx.array],
    sample_rate: int,
    overlap_ms: int = 50,
) -> mx.array:
    """Crossfade audio segments for smooth transitions.

    Uses linear crossfade to eliminate boundary discontinuities between segments.

    Args:
        audio_segments: List of audio segments (1D mx.arrays)
        sample_rate: Audio sample rate in Hz
        overlap_ms: Overlap duration in milliseconds (default: 50ms)

    Returns:
        Concatenated audio with smooth transitions
    """
    if len(audio_segments) == 0:
        return mx.array([])
    if len(audio_segments) == 1:
        return audio_segments[0]
    if overlap_ms <= 0:
        return mx.concatenate(audio_segments, axis=0)

    overlap_samples = int(overlap_ms * sample_rate / 1000)

    result = audio_segments[0]

    for i in range(1, len(audio_segments)):
        current_segment = audio_segments[i]

        # If either segment is too short, just concatenate
        if len(result) < overlap_samples or len(current_segment) < overlap_samples:
            result = mx.concatenate([result, current_segment], axis=0)
            continue

        # Get overlap regions
        prev_overlap = result[-overlap_samples:]
        curr_overlap = current_segment[:overlap_samples]

        # Create linear fade weights
        fade_out = mx.linspace(1.0, 0.0, overlap_samples)
        fade_in = mx.linspace(0.0, 1.0, overlap_samples)

        # Mix overlap region
        overlap_mix = prev_overlap * fade_out + curr_overlap * fade_in

        # Concatenate: prev (without overlap) + mixed overlap + current (without overlap)
        result = mx.concatenate([
            result[:-overlap_samples],
            overlap_mix,
            current_segment[overlap_samples:]
        ], axis=0)

    return result


def save_speaker(
    conditioning: mx.array,
    ref_mel: mx.array,
    output_path: Union[str, Path],
) -> None:
    """Save speaker conditioning to file for reuse.

    Args:
        conditioning: Conditioning tensor from get_conditioning()
        ref_mel: Reference mel spectrogram
        output_path: Output .npz file path
    """
    np.savez(
        str(output_path),
        version=np.array([1.5]),  # Version identifier
        conditioning=np.array(conditioning),
        ref_mel=np.array(ref_mel),
    )


def load_speaker(path: Union[str, Path]) -> tuple:
    """Load speaker conditioning from file.

    Args:
        path: Path to .npz file saved by save_speaker()

    Returns:
        Tuple of (conditioning, ref_mel) as mx.arrays

    Raises:
        ValueError: If the file is not a v1.5 speaker file
    """
    data = np.load(str(path))

    # Check version
    if 'version' in data:
        version = float(data['version'][0])
        if version >= 2.0:
            raise ValueError(
                f"Speaker file is v{version:.1f} format, but this is IndexTTS 1.5. "
                f"Please use the correct model version."
            )

    conditioning = mx.array(data["conditioning"])
    ref_mel = mx.array(data["ref_mel"])
    return conditioning, ref_mel


class IndexTTS:
    """IndexTTS model for speech synthesis.

    This class provides a high-level interface for text-to-speech
    generation using the IndexTTS model on Apple Silicon.
    """

    def __init__(
        self,
        config: IndexTTSConfig,
        gpt: UnifiedVoice,
        bigvgan: BigVGAN,
        tokenizer: TextTokenizer,
    ):
        """Initialize IndexTTS.

        Args:
            config: Model configuration
            gpt: GPT model
            bigvgan: BigVGAN vocoder
            tokenizer: Text tokenizer
        """
        self.config = config
        self.gpt = gpt
        self.bigvgan = bigvgan
        self.tokenizer = tokenizer
        self.sample_rate = config.sample_rate

        # Mel extractor
        self.mel_extractor = MelSpectrogramExtractor(
            n_fft=config.mel.n_fft,
            hop_length=config.mel.hop_length,
            win_length=config.mel.win_length,
            n_mels=config.mel.n_mels,
            sample_rate=config.mel.sample_rate,
            f_min=config.mel.mel_fmin,
            f_max=config.mel.mel_fmax,
        )

        # Cache for reference audio
        self._cached_ref_path = None
        self._cached_conditioning = None
        self._cached_ref_mel = None

    @classmethod
    def load_model(
        cls,
        model_dir: Union[str, Path],
        memory_limit_gb: float | None = None,
        quantize_bits: Optional[int] = None,
    ) -> "IndexTTS":
        """Load model from directory.

        Args:
            model_dir: Directory containing converted MLX model
            memory_limit_gb: GPU memory limit in GB (None = auto, 0 for no limit)
            quantize_bits: Runtime quantization bits (4 or 8), None for no quantization

        Returns:
            IndexTTS instance
        """
        import mlx.nn as nn

        # Configure MLX memory/cache before loading model. Auto mode scales up
        # on large Apple unified-memory hosts while respecting explicit limits.
        if memory_limit_gb is None:
            configure_mlx_runtime()
        elif memory_limit_gb > 0:
            configure_mlx_runtime(memory_limit_gb=memory_limit_gb)

        from mlx_indextts.convert import load_mlx_model

        model_dir = Path(model_dir)

        # Load config and weights (returns quantize_bits from saved model)
        config, gpt_weights, bigvgan_weights, saved_quantize_bits = load_mlx_model(model_dir)

        # Create models
        gpt = UnifiedVoice(config)
        bigvgan = BigVGAN(config.bigvgan)

        # Determine quantization: use saved if model was pre-quantized, otherwise use runtime option
        effective_quantize = saved_quantize_bits or quantize_bits

        # If model was saved with quantization, quantize before loading weights
        if saved_quantize_bits:
            nn.quantize(gpt.gpt, bits=saved_quantize_bits, group_size=64)

        # Load weights
        gpt.load_weights(list(gpt_weights.items()))
        bigvgan.load_weights(list(bigvgan_weights.items()))

        # If runtime quantization requested (and model wasn't pre-quantized)
        if quantize_bits and not saved_quantize_bits:
            nn.quantize(gpt.gpt, bits=quantize_bits, group_size=64)

        # Set to eval mode for inference (critical for BatchNorm)
        gpt = gpt.eval()
        bigvgan = bigvgan.eval()

        # Load tokenizer
        tokenizer_path = model_dir / "tokenizer.model"
        normalizer = TextNormalizer()
        normalizer.load()
        tokenizer = TextTokenizer(tokenizer_path, normalizer)

        return cls(config, gpt, bigvgan, tokenizer)

    def load_audio(self, audio_path: Union[str, Path]) -> mx.array:
        """Load and preprocess audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio tensor (samples,)
        """
        audio, sr = sf.read(str(audio_path))

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            # Simple linear interpolation resampling
            duration = len(audio) / sr
            new_length = int(duration * self.sample_rate)
            x_old = np.linspace(0, 1, len(audio))
            x_new = np.linspace(0, 1, new_length)
            audio = np.interp(x_new, x_old, audio)

        # Ensure float32
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        return mx.array(audio.astype(np.float32))

    def get_conditioning(
        self,
        ref_audio: Union[str, Path, mx.array],
    ) -> tuple:
        """Get conditioning from reference audio or pre-computed speaker file.

        Args:
            ref_audio: Reference audio path, tensor, or .npz speaker file

        Returns:
            Tuple of (conditioning, ref_mel)
        """
        # Check if it's a pre-computed speaker file
        if isinstance(ref_audio, (str, Path)):
            ref_path = str(ref_audio)

            # Load from .npz speaker file
            if ref_path.endswith(".npz"):
                conditioning, ref_mel = load_speaker(ref_path)
                self._cached_ref_path = ref_path
                self._cached_conditioning = conditioning
                self._cached_ref_mel = ref_mel
                return conditioning, ref_mel

            # Check memory cache
            if ref_path == self._cached_ref_path and self._cached_conditioning is not None:
                return self._cached_conditioning, self._cached_ref_mel

            audio = self.load_audio(ref_audio)
            self._cached_ref_path = ref_path
        else:
            audio = ref_audio
            self._cached_ref_path = None

        # Compute mel spectrogram
        ref_mel = self.mel_extractor(audio)
        if ref_mel.ndim == 2:
            ref_mel = ref_mel[None, :, :]  # Add batch dim

        # Get conditioning
        conditioning = self.gpt.get_conditioning(ref_mel)

        # Cache
        self._cached_conditioning = conditioning
        self._cached_ref_mel = ref_mel

        return conditioning, ref_mel

    def save_speaker(
        self,
        ref_audio: Union[str, Path],
        output_path: Union[str, Path],
    ) -> None:
        """Pre-compute and save speaker conditioning for faster inference.

        Args:
            ref_audio: Reference audio path
            output_path: Output .npz file path
        """
        conditioning, ref_mel = self.get_conditioning(ref_audio)
        save_speaker(conditioning, ref_mel, output_path)

    def generate(
        self,
        text: str,
        ref_audio: Union[str, Path, mx.array],
        max_mel_tokens: int = 600,
        max_text_tokens_per_segment: int = 120,
        interval_silence: int = 200,
        temperature: float = 1.0,
        top_k: int = 30,
        top_p: float = 0.8,
        repetition_penalty: float = 10.0,
        seed: Optional[int] = None,
        verbose: bool = False,
        segment_overlap_ms: int = 50,
        speed: float = 1.0,
    ) -> mx.array:
        """Generate speech from text.

        Args:
            text: Input text
            ref_audio: Reference audio for voice cloning
            max_mel_tokens: Maximum mel tokens to generate per segment
            max_text_tokens_per_segment: Maximum text tokens per segment (for long text splitting)
            interval_silence: Silence duration (ms) to insert between segments (default: 200)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens (default: 10.0)
            seed: Random seed for reproducible generation
            verbose: Whether to print progress
            segment_overlap_ms: Overlap duration in ms for crossfade (only used when interval_silence=0)
            speed: Playback speed multiplier (0.5-2.0, default 1.0). Uses time-stretch without pitch change.

        Returns:
            Generated audio waveform (samples,)
        """
        start_time = time.perf_counter()

        # Set random seed if provided
        if seed is not None:
            mx.random.seed(seed)

        # Get conditioning
        conditioning, ref_mel = self.get_conditioning(ref_audio)

        # Tokenize text and split into segments
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(
            text_tokens_list,
            max_tokens_per_segment=max_text_tokens_per_segment,
        )

        if verbose:
            total_tokens = len(text_tokens_list)
            print(f"Text tokens: {total_tokens}, Segments: {len(segments)}")
            if len(segments) > 1:
                for i, seg in enumerate(segments):
                    print(f"  Segment {i+1}: {len(seg)} tokens")

        # Generate audio for each segment
        all_audio = []
        total_mel_tokens = 0
        total_gpt_time = 0.0
        total_latent_time = 0.0
        total_bigvgan_time = 0.0

        # Create silence for interval (same logic as v2)
        if interval_silence > 0 and len(segments) > 1:
            silence_samples = int(self.sample_rate * interval_silence / 1000.0)
            silence = mx.zeros(silence_samples, dtype=mx.float32)
        else:
            silence = None

        for seg_idx, segment_tokens in enumerate(segments):
            if verbose and len(segments) > 1:
                print(f"Processing segment {seg_idx + 1}/{len(segments)}...")

            # Convert tokens to IDs
            token_ids = self.tokenizer.convert_tokens_to_ids(segment_tokens)
            text_tokens = mx.array(token_ids, dtype=mx.int32)[None, :]  # Add batch dim

            # Prepare inputs
            input_emb, _ = self.gpt.prepare_inputs(conditioning, text_tokens)

            # Add start mel token
            mel_start = mx.array([[self.gpt.start_mel_token]], dtype=mx.int32)
            mel_start_emb = self.gpt.mel_embedding(mel_start)
            mel_start_emb = mel_start_emb + self.gpt.mel_pos_embedding.get_fixed_embedding(0)
            input_emb = mx.concatenate([input_emb, mel_start_emb], axis=1)

            # Generate mel codes autoregressively
            mel_codes = []
            cache = None
            gpt_start = time.perf_counter()

            for i in range(max_mel_tokens):
                if cache is None:
                    next_token, _, cache = self.gpt.generate_step(
                        input_emb, cache, temperature, top_k, top_p,
                        repetition_penalty, mel_codes
                    )
                else:
                    # Only feed the last token
                    last_token = mx.array([[mel_codes[-1]]], dtype=mx.int32)
                    last_emb = self.gpt.mel_embedding(last_token)
                    mel_pos = len(mel_codes) + 1
                    last_emb = last_emb + self.gpt.mel_pos_embedding.get_fixed_embedding(mel_pos)
                    next_token, _, cache = self.gpt.generate_step(
                        last_emb, cache, temperature, top_k, top_p,
                        repetition_penalty, mel_codes
                    )

                token_id = next_token[0].item()

                # Check for stop token
                if token_id == self.gpt.stop_mel_token:
                    break

                mel_codes.append(token_id)
                mx.eval(cache)  # Ensure cache is computed

                if verbose and (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1} mel tokens...")

            # Warn if generation stopped due to max tokens
            if len(mel_codes) >= max_mel_tokens - 1:
                warnings.warn(
                    f"Generation stopped due to exceeding max_mel_tokens ({max_mel_tokens}). "
                    f"Consider reducing max_text_tokens_per_segment ({max_text_tokens_per_segment}) "
                    f"or increasing max_mel_tokens.",
                    RuntimeWarning,
                )

            gpt_end = time.perf_counter()
            total_gpt_time += gpt_end - gpt_start

            # Compress long silence runs
            orig_len = len(mel_codes)
            mel_codes = compress_silence(mel_codes)
            if verbose and len(mel_codes) < orig_len:
                print(f"  Silence compression: {orig_len} -> {len(mel_codes)} tokens")

            total_mel_tokens += len(mel_codes)

            if verbose:
                print(f"  Segment {seg_idx + 1}: {len(mel_codes)} mel tokens")

            if len(mel_codes) == 0:
                warnings.warn(f"No mel tokens generated for segment {seg_idx + 1}")
                continue

            # Convert to tensor
            mel_codes_tensor = mx.array(mel_codes, dtype=mx.int32)[None, :]

            # Get latents for BigVGAN
            latent_start = time.perf_counter()
            latents = self.gpt.forward_latent(conditioning, text_tokens, mel_codes_tensor)
            mx.eval(latents)
            latent_end = time.perf_counter()
            total_latent_time += latent_end - latent_start

            # Generate audio with BigVGAN
            bigvgan_start = time.perf_counter()
            audio = self.bigvgan(latents, ref_mel)
            audio = audio.squeeze()  # Remove batch and channel dims

            # Peak normalization + clamp with headroom
            peak = mx.abs(audio).max()
            if peak > 1.0:
                audio = audio / mx.maximum(peak, mx.array(1e-6))
            audio = mx.clip(audio, -0.99, 0.99)
            mx.eval(audio)
            bigvgan_end = time.perf_counter()
            total_bigvgan_time += bigvgan_end - bigvgan_start

            all_audio.append(audio)

            # Add silence between segments (not after the last one)
            if silence is not None and seg_idx < len(segments) - 1:
                all_audio.append(silence)

        if len(all_audio) == 0:
            raise RuntimeError("No audio generated")

        # Concatenate all segments (same logic as v2)
        # Note: silence segments (if any) are not crossfaded
        if len(all_audio) == 1:
            final_audio = all_audio[0]
        elif segment_overlap_ms > 0 and silence is None:
            # Apply crossfade when no silence insertion
            final_audio = crossfade_segments(
                all_audio,
                sample_rate=self.sample_rate,
                overlap_ms=segment_overlap_ms,
            )
        else:
            # Simple concatenation when silence is used
            final_audio = mx.concatenate(all_audio, axis=0)

        if verbose:
            elapsed = time.perf_counter() - start_time
            audio_duration = len(final_audio) / self.sample_rate
            rtf = elapsed / audio_duration
            print(f"Generated {audio_duration:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.3f})")
            print(f"Total mel tokens: {total_mel_tokens}")
            print(f"  GPT gen: {total_gpt_time:.2f}s")
            print(f"  Latent: {total_latent_time:.2f}s")
            print(f"  BigVGAN: {total_bigvgan_time:.2f}s")

        # Apply speed control via WSOLA time-stretch (no pitch change)
        if speed != 1.0:
            audio_np = np.array(final_audio)
            audio_np = time_stretch_wsola(audio_np, rate=speed, sample_rate=self.sample_rate)
            final_audio = mx.array(audio_np)
            if verbose:
                print(f"Speed: {speed:.2f}x")

        return final_audio

    def generate_stream(
        self,
        text: str,
        ref_audio: Union[str, Path, mx.array],
        chunk_size: int = 100,
        **kwargs,
    ) -> Generator[mx.array, None, None]:
        """Generate speech in streaming mode.

        Args:
            text: Input text
            ref_audio: Reference audio
            chunk_size: Number of mel tokens per chunk
            **kwargs: Additional arguments for generate()

        Yields:
            Audio chunks
        """
        # For now, just generate full audio and yield
        # TODO: Implement true streaming with chunked generation
        audio = self.generate(text, ref_audio, **kwargs)
        yield audio

    def save_audio(
        self,
        audio: mx.array,
        output_path: Union[str, Path],
    ) -> None:
        """Save audio to file.

        Args:
            audio: Audio tensor
            output_path: Output file path
        """
        # Convert to numpy
        audio_np = np.array(audio)

        # Scale to int16 range
        audio_np = (audio_np * 32767).astype(np.int16)

        # Save
        sf.write(str(output_path), audio_np, self.sample_rate)

    def infer(
        self,
        audio_prompt: Union[str, Path],
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        max_text_tokens_per_segment: int = 120,
        **kwargs,
    ) -> Union[str, tuple]:
        """Inference interface compatible with original IndexTTS.

        Args:
            audio_prompt: Reference audio path
            text: Input text
            output_path: Output audio path (optional)
            verbose: Whether to print progress
            max_text_tokens_per_segment: Maximum text tokens per segment
            **kwargs: Additional generation parameters

        Returns:
            Output path if provided, else (sample_rate, audio_data) tuple
        """
        audio = self.generate(
            text, audio_prompt, verbose=verbose,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            **kwargs
        )

        if output_path:
            self.save_audio(audio, output_path)
            return str(output_path)
        else:
            audio_np = np.array(audio)
            audio_np = (audio_np * 32767).astype(np.int16)
            return (self.sample_rate, audio_np)
