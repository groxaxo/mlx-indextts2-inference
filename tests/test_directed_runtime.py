from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from mlx_indextts.directed_runtime import (
    apply_cuda_profile,
    build_synthesis_chunks,
    get_quality_preset,
    synthesize_direction_plan,
)
from mlx_indextts.director import IndexTTSDirector


class FakeRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeRuntime:
    version = "2.5"
    device = "cuda:0"
    precision = "bf16"

    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        path = Path(request.output_path)
        # 100 ms constant waveform at the upstream 22.05 kHz output rate.
        audio = np.full((2205, 1), 0.1 + 0.01 * len(self.requests), dtype=np.float32)
        sf.write(path, audio, 22050, subtype="PCM_16")
        return SimpleNamespace(
            as_dict=lambda: {
                "output_path": str(path),
                "device": self.device,
                "precision": self.precision,
            }
        )


@pytest.fixture(autouse=True)
def stub_nvidia_runtime(monkeypatch):
    module = types.ModuleType("mlx_indextts.nvidia_runtime")
    module.NvidiaGenerateRequest = FakeRequest
    monkeypatch.setitem(sys.modules, "mlx_indextts.nvidia_runtime", module)


def test_natural_hq_is_the_default_production_preset():
    preset = get_quality_preset(None)

    assert preset.name == "natural-hq"
    assert preset.temperature == pytest.approx(0.72)
    assert preset.top_p == pytest.approx(0.80)
    assert preset.top_k == 30
    assert preset.max_text_tokens == 90
    assert preset.max_mel_tokens == 1500


def test_compatible_sentences_coalesce_but_paragraphs_do_not():
    plan = IndexTTSDirector().direct("Hello. Nice to meet you.\n\nHow are you?")
    chunks = build_synthesis_chunks(plan)

    assert chunks[0].sentence_indexes == (0, 1)
    assert chunks[0].text == "Hello. Nice to meet you."
    assert chunks[1].sentence_indexes == (2,)
    assert chunks[1].text == "How are you?"


def test_no_coalesce_produces_one_chunk_per_sentence():
    plan = IndexTTSDirector().direct("One. Two. Three.")
    chunks = build_synthesis_chunks(plan, coalesce=False)

    assert [chunk.sentence_indexes for chunk in chunks] == [(0,), (1,), (2,)]


def test_directed_synthesis_uses_native_vector_controls_and_concatenates_audio(tmp_path):
    ref = tmp_path / "reference.wav"
    sf.write(ref, np.zeros(2205, dtype=np.float32), 22050, subtype="PCM_16")
    output = tmp_path / "result.wav"
    plan = IndexTTSDirector().direct("Hello. Are you there?")
    runtime = FakeRuntime()

    result = synthesize_direction_plan(
        runtime,
        plan,
        ref_audio=ref,
        output_path=output,
        language="en",
        coalesce=False,
        seed=100,
        cuda_profile="unchanged",
    )

    assert output.is_file()
    assert len(runtime.requests) == 2
    first = runtime.requests[0]
    assert first.emotion_vector == list(plan.directions[0].emotion_vector)
    assert first.emo_alpha == plan.directions[0].alpha
    assert first.use_random is False
    assert first.interval_silence_ms == 0
    assert first.duration_factor == pytest.approx(1.0 / plan.directions[0].speed)
    assert first.temperature == pytest.approx(0.72)
    assert first.seed == 100
    assert runtime.requests[1].seed == 101

    audio, sample_rate = sf.read(output, always_2d=True)
    expected_pause = round(22050 * plan.directions[0].pause_after_ms / 1000)
    assert sample_rate == 22050
    assert len(audio) == 2205 + expected_pause + 2205
    assert result.chunk_count == 2
    assert result.sentence_count == 2
    assert result.peak < 1.0
    assert result.cuda_tuning["profile"] == "unchanged"
    assert all("cuda_tuning" not in item for item in result.segment_results)
    assert all(item["retained_output_path"] is None for item in result.segment_results)


def test_synthesis_rejects_non_25_runtime(tmp_path):
    runtime = FakeRuntime()
    runtime.version = "2.0"
    plan = IndexTTSDirector().direct("Hello.")
    ref = tmp_path / "ref.wav"
    sf.write(ref, np.zeros(100, dtype=np.float32), 22050)

    with pytest.raises(ValueError, match="requires IndexTTS 2.5"):
        synthesize_direction_plan(
            runtime,
            plan,
            ref_audio=ref,
            output_path=tmp_path / "out.wav",
        )


class FakeMatmul:
    allow_tf32 = None


class FakeCudnn:
    allow_tf32 = None
    benchmark = None
    deterministic = None


class FakeTorch:
    precision = None
    backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=FakeMatmul()),
        cudnn=FakeCudnn(),
    )

    @classmethod
    def set_float32_matmul_precision(cls, value):
        cls.precision = value


def test_quality_cuda_profile_disables_tf32_substitution():
    report = apply_cuda_profile("quality", FakeTorch)

    assert report["allow_tf32"] is False
    assert FakeTorch.precision == "highest"
    assert FakeTorch.backends.cuda.matmul.allow_tf32 is False
    assert FakeTorch.backends.cudnn.allow_tf32 is False
    assert FakeTorch.backends.cudnn.benchmark is False


def test_balanced_cuda_profile_allows_tf32_without_shape_benchmarking():
    report = apply_cuda_profile("balanced", FakeTorch)

    assert report["allow_tf32"] is True
    assert FakeTorch.precision == "high"
    assert FakeTorch.backends.cudnn.benchmark is False


def test_throughput_cuda_profile_is_explicitly_opt_in():
    report = apply_cuda_profile("throughput", FakeTorch)

    assert report["allow_tf32"] is True
    assert FakeTorch.precision == "high"
    assert FakeTorch.backends.cudnn.benchmark is True
