"""Novel/dialogue planning helpers for MLX-IndexTTS batch generation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


EMOTION_VECTOR_ORDER = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)


@dataclass
class NovelLine:
    speaker: str
    text: str
    emotion_label: str
    emotion: str
    emo_alpha: float


class NovelScriptPlanner:
    """Parse voice-novel or dialogue scripts into stable TTS batch rows."""

    label_aliases = {
        "自然": "calm",
        "中性": "calm",
        "旁白": "calm",
        "叙述": "calm",
        "冷静": "calm",
        "平静": "calm",
        "温柔": "calm",
        "轻声": "calm",
        "开心": "happy",
        "高兴": "happy",
        "愉快": "happy",
        "喜悦": "happy",
        "悲伤": "sad",
        "伤感": "sad",
        "难过": "sad",
        "低落": "melancholic",
        "沮丧": "melancholic",
        "疲惫": "melancholic",
        "愤怒": "angry",
        "生气": "angry",
        "恼怒": "angry",
        "恐惧": "afraid",
        "害怕": "afraid",
        "惊恐": "afraid",
        "紧张": "afraid",
        "惊讶": "surprised",
        "惊喜": "surprised",
        "震惊": "surprised",
        "厌恶": "disgusted",
        "反感": "disgusted",
        "happy": "happy",
        "angry": "angry",
        "sad": "sad",
        "afraid": "afraid",
        "disgusted": "disgusted",
        "melancholic": "melancholic",
        "surprised": "surprised",
        "calm": "calm",
    }

    keyword_rules = (
        ("angry", ("怒", "吼", "喊", "骂", "恨", "混蛋", "该死", "闭嘴")),
        ("afraid", ("怕", "恐", "惊恐", "颤抖", "危险", "救命", "别过来")),
        ("sad", ("哭", "泪", "痛苦", "悲", "绝望", "心碎", "失去")),
        ("melancholic", ("累", "疲惫", "低落", "沉默", "无力", "算了")),
        ("surprised", ("突然", "竟然", "怎么会", "什么", "真的吗", "不可能")),
        ("happy", ("笑", "开心", "高兴", "太好了", "终于", "喜欢")),
        ("afraid", ("紧张", "屏住", "小心", "压低", "急促")),
    )

    default_weights = {
        "calm": 0.55,
        "happy": 0.72,
        "angry": 0.78,
        "sad": 0.68,
        "afraid": 0.65,
        "disgusted": 0.62,
        "melancholic": 0.66,
        "surprised": 0.68,
    }

    def __init__(self, max_chars_per_line: int = 90):
        self.max_chars_per_line = max_chars_per_line

    def parse(self, content: str) -> list[NovelLine]:
        rows: list[NovelLine] = []
        previous: NovelLine | None = None
        for raw_line in self._iter_source_lines(content):
            speaker, payload = self._split_speaker(raw_line)
            explicit_label, explicit_weight, text = self._parse_payload(payload, speaker)
            for segment in self._split_long_text(text):
                emotion = self.normalize_label(explicit_label or self.detect_emotion(segment, speaker))
                weight = self._normalize_weight(explicit_weight, emotion, speaker)
                if previous is not None:
                    weight = self._smooth_weight(previous.emo_alpha, weight, previous.emotion, emotion)
                row = NovelLine(
                    speaker=speaker,
                    text=segment,
                    emotion_label=emotion,
                    emotion=f"{emotion}:{weight:.3f}",
                    emo_alpha=weight,
                )
                rows.append(row)
                previous = row
        return rows

    def to_batch_rows(
        self,
        rows: Iterable[NovelLine],
        *,
        speaker_refs: dict[str, str] | None = None,
        emotion_refs: dict[str, str] | None = None,
        emotion_refs_by_emotion: dict[str, str] | None = None,
        emotion_ref_resolver: Callable[[NovelLine], str] | None = None,
    ) -> list[dict[str, str]]:
        speaker_refs = speaker_refs or {}
        emotion_refs = emotion_refs or {}
        emotion_refs_by_emotion = emotion_refs_by_emotion or {}
        output = []
        for idx, row in enumerate(rows, start=1):
            emotion_ref_audio = emotion_ref_resolver(row) if emotion_ref_resolver else ""
            if not emotion_ref_audio:
                emotion_ref_audio = emotion_refs_by_emotion.get(row.emotion_label, "")
            if not emotion_ref_audio:
                emotion_ref_audio = emotion_refs.get(row.speaker, "")
            output.append(
                {
                    "id": f"{idx:04d}_{self._safe_id(row.speaker)}",
                    "speaker": row.speaker,
                    "text": row.text,
                    "ref_audio": speaker_refs.get(row.speaker, ""),
                    "emotion": row.emotion,
                    "emo_alpha": f"{row.emo_alpha:.3f}",
                    "emotion_ref_audio": emotion_ref_audio,
                }
            )
        return output

    def write_batch_csv(
        self,
        content: str,
        output_path: str,
        *,
        speaker_refs: dict[str, str] | None = None,
        emotion_refs: dict[str, str] | None = None,
        emotion_refs_by_emotion: dict[str, str] | None = None,
        emotion_ref_resolver: Callable[[NovelLine], str] | None = None,
    ) -> str:
        rows = self.to_batch_rows(
            self.parse(content),
            speaker_refs=speaker_refs,
            emotion_refs=emotion_refs,
            emotion_refs_by_emotion=emotion_refs_by_emotion,
            emotion_ref_resolver=emotion_ref_resolver,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["id", "speaker", "text", "ref_audio", "emotion", "emo_alpha", "emotion_ref_audio"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    def normalize_label(self, label: str) -> str:
        clean = re.sub(r"[地的得\s，。,.!！?？:：]+", "", str(label or "").strip())
        if clean in self.label_aliases:
            return self.label_aliases[clean]
        lower = clean.lower()
        if lower in self.label_aliases:
            return self.label_aliases[lower]
        for key, value in self.label_aliases.items():
            if key and key in clean:
                return value
        return "calm"

    def detect_emotion(self, text: str, speaker: str = "") -> str:
        if speaker.strip() == "旁白":
            for label, keywords in self.keyword_rules:
                if label in {"angry", "disgusted", "happy"}:
                    continue
                if any(keyword in text for keyword in keywords):
                    return label
            return "calm"
        for label, keywords in self.keyword_rules:
            if any(keyword in text for keyword in keywords):
                return label
        if "！" in text or "!" in text:
            return "surprised"
        if "？" in text or "?" in text:
            return "afraid"
        return "calm"

    def _iter_source_lines(self, content: str) -> Iterable[str]:
        for line in str(content or "").splitlines():
            clean = line.strip()
            if not clean:
                continue
            clean = re.sub(r"^\s*[-*•\d一二三四五六七八九十、.）)]+", "", clean).strip()
            if clean:
                yield clean

    def _split_speaker(self, line: str) -> tuple[str, str]:
        match = re.match(r"^([^:：]{1,24})[:：]\s*(.+)$", line)
        if match:
            speaker = match.group(1).strip().strip("[]【】")
            payload = match.group(2).strip()
            return speaker or "旁白", payload
        return "旁白", line

    def _parse_payload(self, payload: str, speaker: str) -> tuple[str, float | None, str]:
        text = payload.strip().strip("[]【】").strip()
        if "|" in text:
            parts = [part.strip() for part in text.split("|", 2)]
            if len(parts) == 3:
                return parts[0], self._safe_float(parts[1]), self._clean_text(parts[2])
        return self.detect_emotion(text, speaker), None, self._clean_text(text)

    def _split_long_text(self, text: str) -> Iterable[str]:
        text = self._clean_text(text)
        if len(text) <= self.max_chars_per_line:
            yield text
            return

        pieces = re.split(r"(?<=[。！？!?；;])", text)
        current = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if current and len(current) + len(piece) > self.max_chars_per_line:
                yield current
                current = piece
            else:
                current += piece
            while len(current) > self.max_chars_per_line:
                yield current[: self.max_chars_per_line]
                current = current[self.max_chars_per_line :]
        if current:
            yield current

    def _normalize_weight(self, weight: float | None, emotion: str, speaker: str) -> float:
        if weight is not None:
            return max(0.1, min(1.0, weight))
        if speaker.strip() == "旁白":
            return min(self.default_weights.get(emotion, 0.55), 0.62)
        return self.default_weights.get(emotion, 0.6)

    def _smooth_weight(self, previous: float, target: float, previous_emotion: str, emotion: str) -> float:
        max_step = 0.22 if previous_emotion != emotion else 0.16
        delta = target - previous
        if abs(delta) > max_step:
            target = previous + max_step if delta > 0 else previous - max_step
        return round(max(0.1, min(1.0, target)), 3)

    def _safe_float(self, value: str) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _clean_text(self, text: str) -> str:
        return str(text or "").strip().strip("[]【】").strip()

    def _safe_id(self, value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
        return safe or "speaker"
