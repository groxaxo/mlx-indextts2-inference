"""emotion2vec_plus_large integration for audio emotion libraries."""

from __future__ import annotations

import csv
import gc
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

EMOTION2VEC_MODEL_ID = "emotion2vec/emotion2vec_plus_large"
EMOTION2VEC_HUB = "hf"
EMOTION2VEC_RAW_LABELS = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)
INDEXTTS_EMOTION_ORDER = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)

RAW_TO_INDEXTTS = {
    "happy": "happy",
    "angry": "angry",
    "sad": "sad",
    "disgusted": "disgusted",
    "fearful": "afraid",
    "surprised": "surprised",
    "neutral": "calm",
    "other": "calm",
    "unknown": "calm",
}

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}


def clamp_score(value: Any, min_score: float = 0.0, max_score: float = 1.2) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(min_score, min(max_score, number))


def normalize_scores(labels: Iterable[str], scores: Iterable[float]) -> dict[str, float]:
    raw = {label: 0.0 for label in EMOTION2VEC_RAW_LABELS}
    for label, score in zip(labels, scores):
        key = str(label).strip().lower()
        if "/" in key:
            key = key.rsplit("/", 1)[-1].strip()
        key = {
            "<unk>": "unknown",
            "unk": "unknown",
            "fear": "fearful",
            "fearfulness": "fearful",
        }.get(key, key)
        if key in raw:
            raw[key] = clamp_score(score)
    return raw


def map_emotion2vec_to_indextts(raw_scores: dict[str, float]) -> dict[str, float]:
    mapped = {key: 0.0 for key in INDEXTTS_EMOTION_ORDER}
    for raw_label, target in RAW_TO_INDEXTTS.items():
        mapped[target] += clamp_score(raw_scores.get(raw_label, 0.0))
    return mapped


def normalize_total(weights: dict[str, float], target_total: float = 1.0) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0.0:
        normalized = {key: 0.0 for key in weights}
        if "calm" in normalized:
            normalized["calm"] = target_total
        return normalized
    scale = target_total / total
    return {key: round(max(0.0, float(value)) * scale, 4) for key, value in weights.items()}


def estimate_melancholic_hint(mapped_scores: dict[str, float]) -> float:
    sad = clamp_score(mapped_scores.get("sad", 0.0))
    calm = clamp_score(mapped_scores.get("calm", 0.0))
    angry = clamp_score(mapped_scores.get("angry", 0.0))
    happy = clamp_score(mapped_scores.get("happy", 0.0))
    afraid = clamp_score(mapped_scores.get("afraid", 0.0))
    base = sad * 0.72 + calm * 0.18
    penalty = max(angry, happy, afraid) * 0.18
    return round(max(0.0, min(1.2, base - penalty)), 4)


def audio_duration_seconds(path: Path) -> float:
    try:
        with sf.SoundFile(str(path)) as handle:
            frames = len(handle)
            samplerate = handle.samplerate
        return round(frames / float(samplerate), 3) if samplerate else 0.0
    except Exception:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        frames = waveform.shape[-1]
        return round(frames / float(sample_rate), 3) if sample_rate else 0.0


def iter_audio_paths(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        return sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS
        )
    if path.suffix.lower() == ".csv":
        rows: list[Path] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = next(
                    (
                        row.get(column)
                        for column in ("audio_path", "path", "file", "ref_audio", "wav", "audio")
                        if row.get(column)
                    ),
                    None,
                )
                if value:
                    candidate = Path(str(value).strip())
                    if not candidate.is_absolute():
                        candidate = (path.parent / candidate).resolve()
                    rows.append(candidate)
        return rows
    if path.suffix.lower() == ".scp":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if not clean:
                continue
            parts = clean.split(maxsplit=1)
            if len(parts) == 2:
                candidate = Path(parts[1].strip())
                if not candidate.is_absolute():
                    candidate = (path.parent / candidate).resolve()
                rows.append(candidate)
        return rows
    return [path]


@dataclass
class Emotion2VecResult:
    source_path: str
    raw_scores: dict[str, float]
    indextts_scores: dict[str, float]
    dominant_emotion: str
    confidence: float
    melancholic_hint: float
    elapsed_s: float
    model: str
    hub: str
    raw_output: Any | None = None

    def to_row(self) -> dict[str, str]:
        return {
            "source_path": self.source_path,
            "raw_scores_json": json.dumps(self.raw_scores, ensure_ascii=False),
            "indextts_scores_json": json.dumps(self.indextts_scores, ensure_ascii=False),
            "dominant_emotion": self.dominant_emotion,
            "confidence": f"{self.confidence:.4f}",
            "melancholic_hint": f"{self.melancholic_hint:.4f}",
            "elapsed_s": f"{self.elapsed_s:.3f}",
            "model": self.model,
            "hub": self.hub,
        }


class Emotion2VecClassifier:
    """Lazy FunASR wrapper for emotion2vec_plus_large."""

    def __init__(
        self,
        model_id: str = EMOTION2VEC_MODEL_ID,
        hub: str = EMOTION2VEC_HUB,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.hub = hub
        self.device = device
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError("Install emotion2vec dependencies first: uv sync --extra emotion2vec") from exc

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "hub": self.hub,
            "disable_update": True,
        }
        if self.device:
            kwargs["device"] = self.device
        self._model = AutoModel(**kwargs)

    def unload(self) -> None:
        self._model = None
        gc.collect()

    def _prepare_audio(self, source_path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
        import torch
        import torchaudio

        waveform, sample_rate = sf.read(str(source_path), always_2d=True)
        waveform = torch.from_numpy(waveform.T.astype("float32"))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        tmpdir = tempfile.TemporaryDirectory(prefix="emotion2vec_")
        tmp_path = Path(tmpdir.name) / f"{source_path.stem}.wav"
        sf.write(str(tmp_path), waveform.squeeze(0).detach().cpu().numpy(), 16000)
        return tmp_path, tmpdir

    def classify(self, source_path: str | Path) -> Emotion2VecResult:
        self.load()
        path = Path(source_path)
        prepared_path, tmpdir = self._prepare_audio(path)
        start = time.perf_counter()
        try:
            output_dir = Path(tempfile.mkdtemp(prefix="emotion2vec_out_"))
            try:
                result = self._model.generate(
                    str(prepared_path),
                    output_dir=str(output_dir),
                    granularity="utterance",
                    extract_embedding=False,
                )
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()

        labels, scores = _extract_labels_and_scores(result)
        raw_scores = normalize_scores(labels, scores)
        indextts_scores = normalize_total(map_emotion2vec_to_indextts(raw_scores))
        dominant_emotion = max(indextts_scores.items(), key=lambda item: item[1])[0]
        confidence = max(indextts_scores.values()) if indextts_scores else 0.0
        elapsed = time.perf_counter() - start
        return Emotion2VecResult(
            source_path=str(path),
            raw_scores=raw_scores,
            indextts_scores=indextts_scores,
            dominant_emotion=dominant_emotion,
            confidence=round(confidence, 4),
            melancholic_hint=estimate_melancholic_hint(indextts_scores),
            elapsed_s=round(elapsed, 3),
            model=self.model_id,
            hub=self.hub,
            raw_output=result,
        )


def _extract_labels_and_scores(result: Any) -> tuple[list[str], list[float]]:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            labels = first.get("labels") or first.get("label") or []
            scores = first.get("scores") or first.get("score") or []
            return list(labels), list(scores)
    if isinstance(result, dict):
        labels = result.get("labels") or result.get("label") or []
        scores = result.get("scores") or result.get("score") or []
        return list(labels), list(scores)
    return [], []


def _catalog_score(row: dict[str, Any], emotion: str) -> float:
    mapped = json.loads(row.get("indextts_scores_json") or "{}")
    target = clamp_score(mapped.get(emotion, 0.0))
    confidence = clamp_score(row.get("confidence", 0.0))
    duration = float(row.get("duration_s") or 0.0)
    duration_bonus = 0.12 if 2.0 <= duration <= 8.0 else -0.08
    if emotion == "melancholic":
        duration_bonus += 0.08 if float(row.get("melancholic_hint") or 0.0) >= 0.25 else -0.05
    return round(target * 2.0 + confidence * 0.4 + duration_bonus, 4)


@dataclass
class Emotion2VecCatalog:
    rows: list[dict[str, Any]]
    root: Path

    @classmethod
    def load(cls, path: str | Path) -> "Emotion2VecCatalog":
        csv_path = Path(path)
        rows: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)
        return cls(rows=rows, root=csv_path.parent)

    def best_for(self, emotion: str) -> dict[str, Any] | None:
        if not self.rows:
            return None
        ranked = sorted(
            self.rows,
            key=lambda row: _catalog_score(row, emotion),
            reverse=True,
        )
        return ranked[0]

    def emotion_refs(self) -> dict[str, str]:
        refs: dict[str, str] = {}
        for emotion in INDEXTTS_EMOTION_ORDER:
            row = self.best_for(emotion)
            if not row:
                continue
            source = row.get("copied_path") or row.get("source_path") or ""
            if source and not Path(source).is_absolute():
                source = str((self.root / source).resolve())
            refs[emotion] = source
        return refs


def build_emotion_catalog(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    classifier: Emotion2VecClassifier | None = None,
    copy_audio: bool = True,
    limit: int | None = None,
) -> Path:
    classifier = classifier or Emotion2VecClassifier()
    source_paths = iter_audio_paths(input_path)
    if limit is not None:
        source_paths = source_paths[:limit]
    output_root = Path(output_dir)
    clips_root = output_root / "clips"
    output_root.mkdir(parents=True, exist_ok=True)
    if copy_audio:
        clips_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, source_path in enumerate(source_paths, start=1):
        result = classifier.classify(source_path)
        duration = audio_duration_seconds(source_path)
        copied_path = ""
        if copy_audio:
            copied_name = f"{index:04d}_{source_path.stem}{source_path.suffix.lower()}"
            copied_target = clips_root / copied_name
            shutil.copy2(source_path, copied_target)
            copied_path = str(copied_target.relative_to(output_root))
        row = {
            "index": index,
            "source_path": str(source_path.resolve()),
            "copied_path": copied_path,
            "duration_s": f"{duration:.3f}",
            "dominant_emotion": result.dominant_emotion,
            "confidence": f"{result.confidence:.4f}",
            "melancholic_hint": f"{result.melancholic_hint:.4f}",
            "raw_scores_json": json.dumps(result.raw_scores, ensure_ascii=False),
            "indextts_scores_json": json.dumps(result.indextts_scores, ensure_ascii=False),
            "elapsed_s": f"{result.elapsed_s:.3f}",
            "model": result.model,
            "hub": result.hub,
        }
        rows.append(row)

    fieldnames = [
        "index",
        "source_path",
        "copied_path",
        "duration_s",
        "dominant_emotion",
        "confidence",
        "melancholic_hint",
        "raw_scores_json",
        "indextts_scores_json",
        "elapsed_s",
        "model",
        "hub",
    ]
    manifest_path = output_root / "catalog.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = _write_catalog_summary(output_root, rows)
    refs_path = output_root / "emotion_refs_by_emotion.json"
    refs = Emotion2VecCatalog(rows=rows, root=output_root).emotion_refs()
    refs_path.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "summary.md").write_text(summary, encoding="utf-8")
    return manifest_path


def _write_catalog_summary(output_root: Path, rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["dominant_emotion"]] = counts.get(row["dominant_emotion"], 0) + 1
    lines = [
        "# emotion2vec library summary",
        "",
        f"- items: {len(rows)}",
        f"- output_dir: {output_root}",
        "",
        "## dominant emotion counts",
        "",
    ]
    for emotion, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {emotion}: {count}")
    lines.extend(["", "## note", "", "- `melancholic` remains a derived hint, not a native emotion2vec label."])
    return "\n".join(lines)
