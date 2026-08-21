from __future__ import annotations

import json

import pytest

from mlx_indextts.director import (
    DirectorSettings,
    IndexTTSDirector,
    OpenAICompatibleDirector,
    segment_text,
    validate_plan,
)


class PartialAnnotator:
    model = "test-director"

    def annotate(self, units, **kwargs):
        del kwargs
        rows = []
        for unit in units:
            if unit.index == 0:
                rows.append(
                    {
                        "index": unit.index,
                        "text": "rewritten and ignored",
                        "emotion": "warm calm",
                        "emotion_vector": {
                            "happy": 1,
                            "calm": 3,
                        },
                        "alpha": 99,
                        "speed": -4,
                        "pause_after_ms": 9999,
                    }
                )
        return rows


class ExplodingAnnotator:
    model = "broken"

    def annotate(self, units, **kwargs):
        del units, kwargs
        raise RuntimeError("endpoint unavailable")


def test_segmenter_preserves_exact_source_offsets_and_abbreviations():
    text = '  Dr. Smith arrived at 3 p.m.  Really?\n\nSí. Todo bien.  '
    units = segment_text(text)

    assert [unit.text for unit in units] == [
        "Dr. Smith arrived at 3 p.m.",
        "  Really?",
        "Sí.",
        " Todo bien.",
    ]
    cursor = 0
    rebuilt = []
    for unit in units:
        rebuilt.append(text[cursor : unit.start])
        rebuilt.append(unit.text)
        cursor = unit.end
    rebuilt.append(text[cursor:])
    assert "".join(rebuilt) == text


def test_director_repairs_missing_rows_ignores_model_text_and_clamps_controls():
    text = "Good morning. Are you ready?"
    plan = IndexTTSDirector(PartialAnnotator()).direct(text, language="en")

    assert validate_plan(plan) == []
    assert plan.original_text == text
    assert plan.sentence_count == plan.directed_sentence_count == 2
    assert plan.undirected_sentence_indexes == []
    assert plan.directions[0].text == "Good morning."
    assert plan.directions[0].source == "llm-repair"
    assert "model-supplied text was ignored" in (plan.directions[0].warning or "")
    assert "alpha is outside" in (plan.directions[0].warning or "")
    assert plan.directions[0].alpha == 0.70
    assert plan.directions[0].speed == 0.90
    assert plan.directions[0].pause_after_ms == 450
    assert sum(plan.directions[0].emotion_vector) == pytest.approx(0.8)
    assert plan.directions[1].source == "heuristic"
    assert "Missing direction" in (plan.directions[1].warning or "")


def test_director_falls_back_on_model_failure_and_preserves_n_over_n_invariant():
    plan = IndexTTSDirector(ExplodingAnnotator()).direct("One. Two! Three?")

    assert plan.sentence_count == 3
    assert all(item.source == "heuristic" for item in plan.directions)
    assert validate_plan(plan) == []
    assert any("endpoint unavailable" in warning for warning in plan.warnings)


def test_director_can_fail_closed_when_fallback_is_disabled():
    director = IndexTTSDirector(
        ExplodingAnnotator(),
        settings=DirectorSettings(fallback_on_llm_error=False),
    )
    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        director.direct("One.")


def test_document_continuity_limits_parameter_jumps_and_marks_paragraph_pause():
    class JumpingAnnotator:
        model = "jump"

        def annotate(self, units, **kwargs):
            del kwargs
            return [
                {
                    "index": unit.index,
                    "emotion": "angry" if unit.index else "calm",
                    "emotion_vector": {"angry": 0.8} if unit.index else {"calm": 0.8},
                    "alpha": 0.70 if unit.index else 0.30,
                    "speed": 1.10 if unit.index else 0.90,
                    "pause_after_ms": 100,
                }
                for unit in units
            ]

    plan = IndexTTSDirector(JumpingAnnotator()).direct("First.\n\nSecond.")

    assert plan.directions[0].pause_after_ms == 260
    assert plan.directions[1].alpha == pytest.approx(0.42)
    assert plan.directions[1].speed == pytest.approx(0.95)
    assert plan.directions[1].pause_after_ms == 180


def test_markup_is_a_human_readable_intermediate_format_only():
    plan = IndexTTSDirector().direct("Hello.")
    markup = plan.to_markup()

    assert '[SEG index="0"' in markup
    assert "Hello." in markup
    assert "[/SEG]" in markup


def test_openai_payload_extractor_accepts_clean_and_fenced_json():
    clean = OpenAICompatibleDirector._extract_payload('{"sentences": []}')
    fenced = OpenAICompatibleDirector._extract_payload(
        '```json\n{"sentences": [{"index": 0}]}\n```'
    )

    assert clean == {"sentences": []}
    assert fenced["sentences"][0]["index"] == 0


def test_system_output_remains_json_serializable():
    plan = IndexTTSDirector().direct("Hola. ¿Todo bien?", language="es")
    payload = json.loads(plan.to_json())

    assert payload["original_text"] == "Hola. ¿Todo bien?"
    assert payload["sentence_count"] == 2


def test_duplicate_rows_are_repaired_instead_of_silently_accepted():
    class DuplicateAnnotator:
        model = "duplicate"

        def annotate(self, units, **kwargs):
            del kwargs
            row = {
                "index": units[0].index,
                "emotion": "calm",
                "emotion_vector": {name: (0.8 if name == "calm" else 0.0) for name in (
                    "happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"
                )},
                "alpha": 0.4,
                "speed": 1.0,
                "pause_after_ms": 150,
            }
            return [row, dict(row)]

    plan = IndexTTSDirector(DuplicateAnnotator()).direct("One.")

    assert plan.directions[0].source == "heuristic"
    assert "Duplicate directions" in (plan.directions[0].warning or "")
    assert validate_plan(plan) == []


def test_no_fallback_rejects_contract_violations_not_only_transport_errors():
    director = IndexTTSDirector(
        PartialAnnotator(),
        settings=DirectorSettings(fallback_on_llm_error=False),
    )

    with pytest.raises(RuntimeError, match="Invalid direction for sentence 0"):
        director.direct("Good morning.")


def test_segmenter_keeps_pronunciation_urls_and_decimals_atomic():
    text = (
        "Say <minute|M IH1 . N AH0 T> clearly. "
        "Visit https://example.com/v2.5/docs. Version 2.5 works."
    )
    units = segment_text(text)

    assert [unit.text for unit in units] == [
        "Say <minute|M IH1 . N AH0 T> clearly.",
        " Visit https://example.com/v2.5/docs.",
        " Version 2.5 works.",
    ]
    assert "".join(
        text[units[index - 1].end : unit.start] + unit.text
        if index
        else text[: unit.start] + unit.text
        for index, unit in enumerate(units)
    ) + text[units[-1].end :] == text
