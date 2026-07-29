"""YouTube/video emotion library builder for scene-emotion-gender-age lookup."""

from __future__ import annotations

import csv
import functools
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mlx_indextts.emotion2vec import (
    Emotion2VecClassifier,
    INDEXTTS_EMOTION_ORDER,
    clamp_score,
)

DEFAULT_ASR_MODEL = os.environ.get(
    "MLX_INDEXTTS_ASR_MODEL",
    "mlx-community/whisper-large-v3-turbo",
)
DEFAULT_ASR_BIN = os.environ.get(
    "MLX_INDEXTTS_ASR_BIN",
    "/Users/vanch/.codex/envs/bluestone-mlx-audio/bin/mlx_whisper",
)
DEFAULT_QWEN_ASR_MODEL = os.environ.get(
    "MLX_INDEXTTS_QWEN_ASR_MODEL",
    "/Users/vanch/.cache/huggingface/hub/mlx-audio/mlx-community_Qwen3-ASR-1.7B-4bit",
)
DEFAULT_AGE_GENDER_MODEL = os.environ.get(
    "MLX_INDEXTTS_AGE_GENDER_MODEL",
    "",
)
DEFAULT_AGE_GENDER_DEVICE = os.environ.get("MLX_INDEXTTS_AGE_GENDER_DEVICE", "cpu")
SUPPORTED_QWEN_ASR_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "yue": "Cantonese",
}
SENTENCE_ENDING_RE = re.compile(r"[^。！？!?；;…\n]+[。！？!?；;…]*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;…])(?=[^\s。！？!?；;…])")
DOWNLOAD_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm"}


def safe_component(value: str, fallback: str = "item") -> str:
    """Convert a label into a filesystem-safe component."""
    clean = re.sub(r"[\\/:*?\"<>|]+", "_", str(value).strip())
    clean = re.sub(r"\s+", "_", clean)
    clean = clean.strip("._")
    return clean or fallback


def age_band_from_years(age_years: float) -> str:
    if age_years < 14:
        return "child"
    if age_years < 20:
        return "teen"
    if age_years < 30:
        return "young_adult"
    if age_years < 45:
        return "adult"
    if age_years < 60:
        return "middle_aged"
    return "senior"


def looks_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def normalize_language(language: str | None) -> str:
    if not language:
        return "auto"
    value = language.strip().lower()
    aliases = {
        "zh-cn": "zh",
        "zh-hans": "zh",
        "mandarin": "zh",
        "chinese": "zh",
        "cn": "zh",
        "vi-vn": "vi",
        "vietnamese": "vi",
        "english": "en",
    }
    return aliases.get(value, value)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def _find_first_file(root: Path, suffixes: Iterable[str]) -> Path | None:
    for suffix in suffixes:
        matches = sorted(root.glob(f"*{suffix}"))
        if matches:
            return matches[0]
    return None


def _audio_command_path(name: str, fallback: str) -> str:
    return shutil.which(name) or fallback


def fetch_youtube_metadata(url: str) -> dict[str, Any]:
    yt_dlp = _audio_command_path("yt-dlp", "yt-dlp")
    result = _run([yt_dlp, "-J", "--no-warnings", "--no-playlist", url])
    return json.loads(result.stdout)


def download_youtube_audio(url: str, output_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Download a YouTube video as a wav audio file."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = fetch_youtube_metadata(url)

    yt_dlp = _audio_command_path("yt-dlp", "yt-dlp")
    output_template = str(output_root / "%(id)s.%(ext)s")
    cmd = [
        yt_dlp,
        "--no-warnings",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        output_template,
        url,
    ]
    _run(cmd)

    audio_path = _find_first_file(output_root, (".wav", ".m4a", ".mp3", ".webm", ".opus"))
    if audio_path is None:
        raise FileNotFoundError("yt-dlp finished but no audio file was found")
    return audio_path, metadata


def normalize_audio_file(source_path: str | Path, output_path: str | Path) -> Path:
    """Convert arbitrary audio/video to 16k mono wav."""
    ffmpeg = _audio_command_path("ffmpeg", "ffmpeg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    _run(cmd)
    return output_path


def maybe_extract_vocals(
    source_audio: str | Path,
    output_dir: str | Path,
    *,
    enabled: bool = True,
) -> tuple[Path, str]:
    """Extract vocals with Demucs when available, otherwise keep the input audio."""
    source_path = Path(source_audio)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if not enabled:
        return source_path, "original"
    cached_vocals = output_root / f"{source_path.stem}_vocals.wav"
    if cached_vocals.exists():
        return cached_vocals, "demucs"
    cached_fallback = output_root / f"{source_path.stem}_vocals_fallback.wav"
    if cached_fallback.exists():
        return cached_fallback, "fallback"
    try:
        from mlx_indextts.audio_denoise import denoise_audio_simple

        return Path(denoise_audio_simple(str(source_path), str(cached_vocals))), "demucs"
    except Exception:
        return normalize_audio_file(source_path, cached_fallback), "fallback"


def _split_sentence_text(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(line) if piece.strip()]
        if pieces:
            parts.extend(pieces)
    if not parts:
        parts = [match.group().strip() for match in SENTENCE_ENDING_RE.finditer(raw)]
    if not parts:
        return [raw]
    return [part for part in parts if part]


def _segment_length(text: str) -> int:
    clean = re.sub(r"\s+", "", text)
    return max(1, len(clean))


@dataclass
class SentenceSegment:
    index: int
    start_s: float
    end_s: float
    text: str
    source_segment_index: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def split_asr_segments_into_sentences(segments: list[dict[str, Any]]) -> list[SentenceSegment]:
    """Split ASR segment-level timestamps into sentence-like units."""
    results: list[SentenceSegment] = []
    sentence_index = 1
    for source_index, segment in enumerate(segments, start=1):
        text = str(segment.get("text") or segment.get("sentence") or "").strip()
        if not text:
            continue
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        if end <= start:
            end = start + 0.01
        parts = _split_sentence_text(text)
        if len(parts) <= 1:
            results.append(
                SentenceSegment(
                    index=sentence_index,
                    start_s=start,
                    end_s=end,
                    text=parts[0] if parts else text,
                    source_segment_index=source_index,
                )
            )
            sentence_index += 1
            continue
        lengths = [_segment_length(part) for part in parts]
        total = sum(lengths) or len(parts)
        cursor = start
        for idx, part in enumerate(parts):
            if idx == len(parts) - 1:
                part_end = end
            else:
                part_end = cursor + (end - start) * (lengths[idx] / total)
            clean_part = re.sub(r"\s+", "", part)
            if results and len(clean_part) <= 6 and not re.search(r"[。！？!?；;…]$", clean_part):
                prev = results[-1]
                prev.end_s = round(part_end, 3)
                prev.text = f"{prev.text}{part}"
                cursor = part_end
                continue
            results.append(
                SentenceSegment(
                    index=sentence_index,
                    start_s=round(cursor, 3),
                    end_s=round(part_end, 3),
                    text=part,
                    source_segment_index=source_index,
                )
            )
            sentence_index += 1
            cursor = part_end
    return results


def _parse_transcript_json(data: dict[str, Any]) -> list[dict[str, Any]]:
    if "sentences" in data and isinstance(data["sentences"], list):
        segments = []
        for item in data["sentences"]:
            segments.append(
                {
                    "text": item.get("text", ""),
                    "start": item.get("start", 0.0),
                    "end": item.get("end", 0.0),
                    "speaker_id": item.get("speaker_id"),
                }
            )
        return segments
    if "segments" in data and isinstance(data["segments"], list):
        segments = []
        for item in data["segments"]:
            segments.append(
                {
                    "text": item.get("text", ""),
                    "start": item.get("start", 0.0),
                    "end": item.get("end", 0.0),
                    "speaker_id": item.get("speaker_id"),
                    "words": item.get("words"),
                }
            )
        return segments
    return []


def _asr_output_path(output_dir: Path) -> Path:
    candidate = output_dir / "transcript.json"
    if candidate.exists():
        return candidate
    candidates = sorted(output_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No ASR JSON output found in {output_dir}")
    return candidates[0]


def _choose_asr_route(language: str, metadata: dict[str, Any]) -> str:
    if language in SUPPORTED_QWEN_ASR_LANGUAGES:
        return "qwen"
    title = str(metadata.get("title") or "")
    if looks_chinese(title):
        return "qwen"
    if language in {"zh", "yue", "en", "ja", "ko", "de", "es", "fr", "it", "pt", "ru"}:
        return "qwen"
    return "whisper"


def _qwen_language_name(language: str) -> str | None:
    return SUPPORTED_QWEN_ASR_LANGUAGES.get(language)


def transcribe_audio(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    language: str = "auto",
    asr_model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run local ASR and return normalized segments plus metadata."""
    metadata = metadata or {}
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        transcript_path = _asr_output_path(output_root)
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        segments = _parse_transcript_json(data)
        if segments:
            return segments, data
    except FileNotFoundError:
        pass
    language = normalize_language(language)
    route = _choose_asr_route(language, metadata)

    if route == "qwen":
        from_path = os.environ.get("MLX_INDEXTTS_QWEN_ASR_MODEL", DEFAULT_QWEN_ASR_MODEL)
        model_path = asr_model or from_path
        binary = "/Users/vanch/.codex/envs/bluestone-mlx-audio/bin/mlx_audio.stt.generate"
        cmd = [
            binary,
            "--model",
            model_path,
            "--audio",
            str(audio_path),
            "--output-path",
            str(output_root / "transcript"),
            "--format",
            "json",
        ]
        qwen_language = _qwen_language_name(language) or ("Chinese" if looks_chinese(str(metadata.get("title", ""))) else "English")
        cmd.extend(["--language", qwen_language])
        _run(cmd)
    else:
        binary = _audio_command_path("mlx_whisper", DEFAULT_ASR_BIN)
        model_path = asr_model or DEFAULT_ASR_MODEL
        cmd = [
            binary,
            "--model",
            model_path,
            "--output-format",
            "json",
            "--output-dir",
            str(output_root),
            "--output-name",
            "transcript",
            str(audio_path),
        ]
        if language != "auto":
            cmd.extend(["--language", language])
        _run(cmd)

    transcript_path = _asr_output_path(output_root)
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = _parse_transcript_json(data)
    return segments, data


class _AgeGenderModelResult:
    def __init__(self, age_score: float, gender_probs: dict[str, float], hidden_state: Any):
        self.age_score = age_score
        self.gender_probs = gender_probs
        self.hidden_state = hidden_state


class _AgeGenderModel:
    def __init__(self, model_name: str, device: str = DEFAULT_AGE_GENDER_DEVICE):
        self.model_name = model_name
        self.device = device
        self._processor = None
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import numpy as np  # noqa: F401
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class ModelHead(nn.Module):
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, features, **kwargs):
                x = self.dropout(features)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                return self.out_proj(x)

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.init_weights()

            def forward(self, input_values):
                outputs = self.wav2vec2(input_values)
                hidden_states = outputs[0]
                hidden_states = torch.mean(hidden_states, dim=1)
                logits_age = self.age(hidden_states)
                logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
                return hidden_states, logits_age, logits_gender

        self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self._model = AgeGenderModel.from_pretrained(self.model_name)
        self._model.eval().to(self.device)

    def unload(self) -> None:
        self._processor = None
        self._model = None

    def classify(self, audio_path: str | Path) -> dict[str, Any]:
        self.load()
        import numpy as np
        import torch
        import torchaudio
        import soundfile as sf

        waveform, sample_rate = sf.read(str(audio_path), always_2d=True)
        waveform = torch.from_numpy(waveform.T.astype(np.float32))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        signal = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
        inputs = self._processor(signal, sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        with torch.no_grad():
            hidden, age_logits, gender_probs = self._model(input_values)
        age_score = float(age_logits.squeeze().detach().cpu().item())
        age_years = round(max(0.0, min(1.0, age_score)) * 100.0, 1)
        # The model card example labels the 3-way gender vector as female / male / child.
        gender_names = ("female", "male", "child")
        gender_values = gender_probs.squeeze(0).detach().cpu().tolist()
        gender_map = {
            name: round(float(value), 4)
            for name, value in zip(gender_names, gender_values)
        }
        gender_label = max(gender_map.items(), key=lambda item: item[1])[0]
        return {
            "age_score": round(age_score, 4),
            "age_years": age_years,
            "age_band": age_band_from_years(age_years),
            "gender_label": gender_label,
            "gender_confidence": round(gender_map[gender_label], 4),
            "gender_probs": gender_map,
        }


def _heuristic_age_gender(audio_path: str | Path) -> dict[str, Any]:
    """Cheap fallback when the age/gender model is unavailable.

    This is intentionally conservative: it prefers stable, low-variance buckets
    over pretending to know more than the signal supports.
    """
    import numpy as np
    import librosa
    import soundfile as sf

    waveform, sample_rate = sf.read(str(audio_path), always_2d=True)
    waveform = waveform.mean(axis=1).astype(np.float32)
    if waveform.size == 0:
        return {
            "age_score": 0.35,
            "age_years": 35.0,
            "age_band": "adult",
            "gender_label": "unknown",
            "gender_confidence": 0.0,
            "gender_probs": {"female": 0.0, "male": 0.0, "child": 0.0},
            "age_gender_source": "heuristic-empty",
        }

    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    pitch = None
    voiced_ratio = 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            waveform,
            fmin=65.0,
            fmax=350.0,
            sr=sample_rate,
            frame_length=2048,
            hop_length=160,
        )
        voiced = f0[np.isfinite(f0)]
        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None and len(voiced_flag) else 0.0
        if voiced.size:
            pitch = float(np.median(voiced))
    except Exception:
        pitch = None

    duration_s = max(0.0, float(len(waveform)) / float(sample_rate or 16000))
    # Prefer stable defaults over speculative values when pitch cannot be estimated.
    if pitch is None:
        age_band = "adult" if duration_s >= 1.0 else "young_adult"
        gender_label = "unknown"
        gender_probs = {"female": 0.34, "male": 0.34, "child": 0.32}
        age_years = 34.0 if duration_s >= 1.0 else 24.0
        age_score = age_years / 100.0
        gender_confidence = 0.0
    else:
        if pitch >= 250.0:
            age_band = "child"
            gender_label = "female"
        elif pitch >= 200.0:
            age_band = "teen"
            gender_label = "female"
        elif pitch >= 165.0:
            age_band = "young_adult"
            gender_label = "female"
        elif pitch >= 125.0:
            age_band = "adult"
            gender_label = "male"
        elif pitch >= 95.0:
            age_band = "middle_aged"
            gender_label = "male"
        else:
            age_band = "senior"
            gender_label = "male"

        if gender_label == "female":
            gender_probs = {"female": 0.68, "male": 0.24, "child": 0.08}
        elif gender_label == "male":
            gender_probs = {"female": 0.24, "male": 0.66, "child": 0.10}
        else:
            gender_probs = {"female": 0.33, "male": 0.33, "child": 0.34}

        age_years_by_band = {
            "child": 10.0,
            "teen": 16.0,
            "young_adult": 24.0,
            "adult": 35.0,
            "middle_aged": 50.0,
            "senior": 68.0,
        }
        age_years = age_years_by_band.get(age_band, 35.0)
        age_score = age_years / 100.0
        pitch_distance = abs(pitch - (200.0 if gender_label == "female" else 130.0))
        confidence = 0.52 + min(0.23, max(0.0, 0.23 - pitch_distance / 1000.0))
        confidence += min(0.1, voiced_ratio * 0.1)
        gender_confidence = round(min(0.92, max(0.35, confidence)), 4)

    return {
        "age_score": round(age_score, 4),
        "age_years": round(age_years, 1),
        "age_band": age_band,
        "gender_label": gender_label,
        "gender_confidence": round(gender_confidence, 4),
        "gender_probs": {key: round(float(value), 4) for key, value in gender_probs.items()},
        "age_gender_source": "heuristic",
    }


def _clip_score(row: dict[str, Any]) -> float:
    emotion_conf = clamp_score(row.get("emotion_confidence") or row.get("confidence") or 0.0)
    gender_conf = clamp_score(row.get("gender_confidence") or 0.0)
    duration = float(row.get("duration_s") or 0.0)
    duration_bonus = 0.15 if 1.5 <= duration <= 8.0 else 0.05 if duration >= 0.8 else -0.05
    return round(emotion_conf * 0.72 + gender_conf * 0.18 + duration_bonus, 4)


def _fallback_emotions(emotion: str | None) -> tuple[str, ...]:
    mapping = {
        "melancholic": ("sad", "calm"),
        "afraid": ("surprised", "sad", "calm"),
        "disgusted": ("angry", "calm"),
        "sad": ("calm",),
        "surprised": ("happy", "calm"),
    }
    return mapping.get(str(emotion or "").strip(), ("calm",) if emotion and emotion != "calm" else ())


@functools.lru_cache(maxsize=256)
def _estimate_median_f0(audio_path: str) -> float | None:
    import librosa
    import numpy as np
    import soundfile as sf

    waveform, sample_rate = sf.read(audio_path, always_2d=True)
    waveform = waveform.mean(axis=1).astype(np.float32)
    if waveform.size == 0:
        return None
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    try:
        f0, _, _ = librosa.pyin(
            waveform,
            fmin=65.0,
            fmax=350.0,
            sr=sample_rate,
            frame_length=2048,
            hop_length=160,
        )
    except Exception:
        return None
    voiced = f0[np.isfinite(f0)]
    if not voiced.size:
        return None
    return float(np.median(voiced))


@dataclass
class SceneEmotionCatalog:
    rows: list[dict[str, Any]]
    root: Path

    @classmethod
    def load(cls, path: str | Path) -> "SceneEmotionCatalog":
        csv_path = Path(path)
        rows: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)
        return cls(rows=rows, root=csv_path.parent)

    def _resolve_path(self, row: dict[str, Any]) -> str:
        source = (
            row.get("clip_path")
            or row.get("copied_path")
            or row.get("source_path")
            or row.get("path")
            or ""
        )
        if not source:
            return ""
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = (self.root / source_path).resolve()
        return str(source_path)

    def best_for(
        self,
        *,
        scene: str | None = None,
        emotion: str | None = None,
        gender: str | None = None,
        age_band: str | None = None,
    ) -> dict[str, Any] | None:
        candidates = self.rows
        if scene:
            candidates = [row for row in candidates if str(row.get("scene", "")).strip() == scene]
        if emotion:
            candidates = [
                row
                for row in candidates
                if str(row.get("emotion_label", row.get("dominant_emotion", ""))).strip() == emotion
            ]
        if gender:
            candidates = [row for row in candidates if str(row.get("gender_label", "")).strip() == gender]
        if age_band:
            candidates = [row for row in candidates if str(row.get("age_band", "")).strip() == age_band]
        if not candidates:
            return None
        return max(candidates, key=_clip_score)

    def best_ref_for(
        self,
        *,
        scene: str | None = None,
        emotion: str | None = None,
        gender: str | None = None,
        age_band: str | None = None,
    ) -> str:
        """Return the best clip path with relaxed scene/emotion/gender/age fallback."""
        attempts = [
            {"scene": scene, "emotion": emotion, "gender": gender, "age_band": age_band},
            {"scene": scene, "emotion": emotion, "gender": gender, "age_band": None},
            {"scene": scene, "emotion": emotion, "gender": None, "age_band": age_band},
            {"scene": scene, "emotion": emotion, "gender": None, "age_band": None},
            {"scene": None, "emotion": emotion, "gender": gender, "age_band": age_band},
            {"scene": None, "emotion": emotion, "gender": gender, "age_band": None},
            {"scene": None, "emotion": emotion, "gender": None, "age_band": None},
        ]
        for filters in attempts:
            row = self.best_for(**filters)
            if row:
                return self._resolve_path(row)
        for fallback_emotion in _fallback_emotions(emotion):
            source = self.best_ref_for(
                scene=scene,
                emotion=fallback_emotion,
                gender=gender,
                age_band=age_band,
            )
            if source:
                return source
        return ""

    def emotion_refs(self) -> dict[str, str]:
        refs: dict[str, str] = {}
        for emotion in INDEXTTS_EMOTION_ORDER:
            source = self.best_ref_for(emotion=emotion)
            if source:
                refs[emotion] = source
        return refs

    def composite_refs(self) -> dict[str, str]:
        refs: dict[str, str] = {}
        for row in sorted(self.rows, key=_clip_score, reverse=True):
            scene = str(row.get("scene", "")).strip() or "scene"
            emotion = str(row.get("emotion_label", row.get("dominant_emotion", ""))).strip() or "emotion"
            gender = str(row.get("gender_label", "")).strip() or "gender"
            age_band = str(row.get("age_band", "")).strip() or "age"
            key = f"{scene}|{emotion}|{gender}|{age_band}"
            if key not in refs:
                refs[key] = self._resolve_path(row)
        return refs

    def recommended_duo_refs(self, *, scene: str | None = None) -> dict[str, str]:
        """Pick two acoustically distinct refs for a crosstalk duo."""
        candidates = self.rows
        if scene:
            candidates = [row for row in candidates if str(row.get("scene", "")).strip() == scene]
        scored: list[tuple[float, float, dict[str, Any], str]] = []
        for row in candidates:
            path = self._resolve_path(row)
            if not path:
                continue
            pitch = _estimate_median_f0(path)
            if pitch is None:
                gender = str(row.get("gender_label", "")).strip()
                pitch = 120.0 if gender == "male" else 220.0 if gender == "female" else 170.0
            scored.append((pitch, _clip_score(row), row, path))
        if not scored:
            return {}
        scored.sort(key=lambda item: (item[0], item[1]), reverse=False)
        low_candidates = scored[:18]
        high_candidates = scored[-18:] if len(scored) > 18 else scored
        best_pair: tuple[tuple[float, float, dict[str, Any], str], tuple[float, float, dict[str, Any], str]] | None = None
        best_score = float("-inf")
        for low in low_candidates:
            for high in high_candidates:
                if low[3] == high[3]:
                    continue
                low_gender = str(low[2].get("gender_label", "")).strip()
                high_gender = str(high[2].get("gender_label", "")).strip()
                gender_bonus = 0.5 if low_gender != high_gender else 0.0
                score = (high[0] - low[0]) + (low[1] + high[1]) * 0.15 + gender_bonus
                if score > best_score:
                    best_score = score
                    best_pair = (low, high)
        if best_pair is None:
            best_pair = (scored[0], scored[-1])
        low, high = best_pair
        return {
            "逗哏": low[3],
            "捧哏": high[3],
        }


def load_emotion_library_catalog(path: str | Path) -> SceneEmotionCatalog | Any:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
    if {"scene", "gender_label", "age_band"} & fieldnames:
        return SceneEmotionCatalog.load(csv_path)
    from mlx_indextts.emotion2vec import Emotion2VecCatalog

    return Emotion2VecCatalog.load(csv_path)


def _safe_ratio_score(duration: float, total_duration: float) -> float:
    if total_duration <= 0:
        return 1.0
    return max(0.0, min(1.0, duration / total_duration))


def _extract_clip(source_audio: Path, start_s: float, end_s: float, output_path: Path) -> Path:
    ffmpeg = _audio_command_path("ffmpeg", "ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_s):.3f}",
        "-to",
        f"{max(start_s + 0.01, end_s):.3f}",
        "-i",
        str(source_audio),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    _run(cmd)
    return output_path


def _write_summary(output_root: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    emotion_counts: dict[str, int] = {}
    gender_counts: dict[str, int] = {}
    age_band_counts: dict[str, int] = {}
    for row in rows:
        emotion_counts[row["emotion_label"]] = emotion_counts.get(row["emotion_label"], 0) + 1
        gender_counts[row["gender_label"]] = gender_counts.get(row["gender_label"], 0) + 1
        age_band_counts[row["age_band"]] = age_band_counts.get(row["age_band"], 0) + 1
    lines = [
        "# scene-emotion-gender-age library",
        "",
        f"- items: {len(rows)}",
        f"- scene: {metadata.get('scene', '')}",
        f"- source_title: {metadata.get('title', '')}",
        f"- source_url: {metadata.get('webpage_url', '')}",
        "",
        "## emotion counts",
        "",
    ]
    for emotion, count in sorted(emotion_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {emotion}: {count}")
    lines.extend(["", "## gender counts", ""])
    for gender, count in sorted(gender_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {gender}: {count}")
    lines.extend(["", "## age band counts", ""])
    for band, count in sorted(age_band_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {band}: {count}")
    lines.extend([
        "",
        "## notes",
        "",
        "- emotion2vec is used for audio emotion tagging.",
        "- age/gender use audEERING wav2vec2 heads when available, otherwise a lightweight heuristic fallback.",
        "- sentence boundaries are derived from ASR timestamps plus punctuation splitting.",
    ])
    return "\n".join(lines)


def build_scene_emotion_library(
    source: str | Path,
    output_dir: str | Path,
    *,
    scene: str = "crosstalk",
    language: str = "auto",
    asr_model: str | None = None,
    extract_vocals: bool = True,
    limit: int | None = None,
    clip_padding_s: float = 0.12,
    emotion_classifier: Emotion2VecClassifier | None = None,
    age_gender_model: str | None = None,
    age_gender_device: str | None = None,
) -> dict[str, Any]:
    """Build a scene/emotion/gender/age library from a YouTube URL or local media."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    source_str = str(source)
    source_path = Path(source_str)
    metadata: dict[str, Any] = {
        "scene": scene,
        "source": source_str,
    }
    if source_str.startswith("http://") or source_str.startswith("https://"):
        source_audio, youtube_meta = download_youtube_audio(source_str, work_root / "download")
        metadata.update(youtube_meta)
    else:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        metadata["title"] = source_path.stem
        metadata["webpage_url"] = ""
        source_audio = normalize_audio_file(source_path, work_root / f"{source_path.stem}_source.wav")

    vocal_audio, vocal_source = maybe_extract_vocals(source_audio, work_root / "vocals", enabled=extract_vocals)
    if vocal_audio.suffix.lower() != ".wav":
        vocal_audio = normalize_audio_file(vocal_audio, work_root / f"{vocal_audio.stem}_normalized.wav")

    transcript_segments, asr_meta = transcribe_audio(
        vocal_audio,
        work_root / "asr",
        language=language,
        asr_model=asr_model,
        metadata=metadata,
    )
    sentence_segments = split_asr_segments_into_sentences(transcript_segments)
    if limit is not None:
        sentence_segments = sentence_segments[:limit]

    emotion_classifier = emotion_classifier or Emotion2VecClassifier()
    emotion_model_id = getattr(emotion_classifier, "model_id", "")
    gender_classifier = None
    if age_gender_model or DEFAULT_AGE_GENDER_MODEL:
        gender_classifier = _AgeGenderModel(
            age_gender_model or DEFAULT_AGE_GENDER_MODEL,
            device=age_gender_device or DEFAULT_AGE_GENDER_DEVICE,
        )

    clips_root = output_root / "clips"
    rows: list[dict[str, Any]] = []
    total_duration = float(metadata.get("duration") or 0.0)

    for segment in sentence_segments:
        clip_start = max(0.0, segment.start_s - clip_padding_s)
        clip_end = segment.end_s + clip_padding_s
        clip_name = (
            f"{segment.index:04d}_"
            f"{safe_component(scene)}_"
            f"{int(round(segment.start_s * 1000)):07d}-"
            f"{int(round(segment.end_s * 1000)):07d}.wav"
        )
        temp_clip = work_root / "clips" / clip_name
        clip_path = _extract_clip(vocal_audio, clip_start, clip_end, temp_clip)

        emotion = emotion_classifier.classify(clip_path)
        if gender_classifier is not None:
            try:
                gender_age = gender_classifier.classify(clip_path)
            except Exception:
                gender_age = _heuristic_age_gender(clip_path)
        else:
            gender_age = _heuristic_age_gender(clip_path)

        row = {
            "scene": scene,
            "video_id": metadata.get("id", ""),
            "video_title": metadata.get("title", ""),
            "source_url": metadata.get("webpage_url", ""),
            "source_path": str(Path(source_audio).resolve()),
            "vocal_path": str(Path(vocal_audio).resolve()),
            "vocal_source": vocal_source,
            "segment_index": segment.index,
            "source_segment_index": segment.source_segment_index,
            "start_s": f"{segment.start_s:.3f}",
            "end_s": f"{segment.end_s:.3f}",
            "duration_s": f"{segment.duration_s:.3f}",
            "sentence": segment.text,
            "emotion_label": emotion.dominant_emotion,
            "emotion_confidence": f"{emotion.confidence:.4f}",
            "emotion_json": json.dumps(emotion.indextts_scores, ensure_ascii=False),
            "emotion_raw_json": json.dumps(emotion.raw_scores, ensure_ascii=False),
            "melancholic_hint": f"{emotion.melancholic_hint:.4f}",
            "gender_label": gender_age["gender_label"],
            "gender_confidence": f"{gender_age['gender_confidence']:.4f}",
            "gender_probs_json": json.dumps(gender_age["gender_probs"], ensure_ascii=False),
            "age_score": f"{gender_age['age_score']:.4f}",
            "age_years": f"{gender_age['age_years']:.1f}",
            "age_band": gender_age["age_band"],
            "clip_path": str(clip_path.relative_to(output_root)),
            "age_gender_source": gender_age.get("age_gender_source", "model" if gender_classifier else "heuristic"),
        }
        library_score = (
            clamp_score(row["emotion_confidence"]) * 0.72
            + clamp_score(row["gender_confidence"]) * 0.18
            + (0.15 if 1.5 <= segment.duration_s <= 8.0 else 0.05 if segment.duration_s >= 0.8 else -0.05)
        )
        row["library_score"] = f"{library_score:.4f}"
        row["composite_key"] = (
            f"{scene}|{row['emotion_label']}|{row['gender_label']}|{row['age_band']}"
        )
        row["duration_ratio"] = f"{_safe_ratio_score(segment.duration_s, total_duration):.4f}"
        rows.append(row)

    fieldnames = [
        "scene",
        "video_id",
        "video_title",
        "source_url",
        "source_path",
        "vocal_path",
        "vocal_source",
        "segment_index",
        "source_segment_index",
        "start_s",
        "end_s",
        "duration_s",
        "sentence",
        "emotion_label",
        "emotion_confidence",
        "emotion_json",
        "emotion_raw_json",
        "melancholic_hint",
        "gender_label",
        "gender_confidence",
        "gender_probs_json",
        "age_score",
        "age_years",
        "age_band",
        "age_gender_source",
        "clip_path",
        "composite_key",
        "library_score",
        "duration_ratio",
    ]
    manifest_path = output_root / "catalog.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    catalog = SceneEmotionCatalog(rows=rows, root=output_root)
    emotion_refs = catalog.emotion_refs()
    composite_refs = catalog.composite_refs()
    (output_root / "emotion_refs_by_emotion.json").write_text(
        json.dumps(emotion_refs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "emotion_refs_by_scene_emotion_gender_age.json").write_text(
        json.dumps(composite_refs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "summary.md").write_text(
        _write_summary(output_root, rows, metadata),
        encoding="utf-8",
    )
    (output_root / "metadata.json").write_text(
        json.dumps(
            {
                **metadata,
                "scene": scene,
                "asr_model": asr_model or (DEFAULT_QWEN_ASR_MODEL if looks_chinese(str(metadata.get("title", ""))) else DEFAULT_ASR_MODEL),
                "emotion_model": emotion_model_id,
                "age_gender_model": age_gender_model or DEFAULT_AGE_GENDER_MODEL or "",
                "clip_padding_s": clip_padding_s,
                "extract_vocals": extract_vocals,
                "asr_route": "qwen" if looks_chinese(str(metadata.get("title", ""))) or normalize_language(language) in SUPPORTED_QWEN_ASR_LANGUAGES else "whisper",
                "age_gender_mode": "model" if (age_gender_model or DEFAULT_AGE_GENDER_MODEL) else "heuristic",
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "items": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "manifest_path": manifest_path,
        "library_root": output_root,
        "catalog": catalog,
        "rows": rows,
        "emotion_refs": emotion_refs,
        "composite_refs": composite_refs,
        "metadata": metadata,
    }
