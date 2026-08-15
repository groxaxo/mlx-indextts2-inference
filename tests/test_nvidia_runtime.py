from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_indextts.nvidia_runtime import (
    NvidiaGenerateRequest,
    NvidiaIndexTTS,
    NvidiaRuntimeConfig,
    detect_language,
    normalize_language,
    normalize_version,
    parse_emotion_vector,
    resolve_device,
    resolve_precision,
)


class FakeCuda:
    def __init__(self, available=True, count=3, bf16=True):
        self._available = available
        self._count = count
        self._bf16 = bf16
        self.seed = None
        self.selected_device = None

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def is_bf16_supported(self):
        return self._bf16

    def set_device(self, index):
        self.selected_device = index

    def manual_seed_all(self, seed):
        self.seed = seed

    def empty_cache(self):
        return None

    def get_device_properties(self, index):
        return SimpleNamespace(name=f"GPU {index}", total_memory=24 * 1024**3, major=8, minor=6)


class FakeTorch:
    def __init__(self, cuda=None):
        self.cuda = cuda or FakeCuda()
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed


class FakeOfficialRuntime:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.infer_kwargs = None

    def infer(self, **kwargs):
        self.infer_kwargs = kwargs
        Path(kwargs["output_path"]).write_bytes(b"RIFFfresh")
        return kwargs["output_path"]


@pytest.mark.parametrize(
    ("raw", "expected"), [("v2.5", "2.5"), ("25", "2.5"), (2, "2.0"), ("2_0", "2.0")]
)
def test_normalize_version(raw, expected):
    assert normalize_version(raw) == expected


def test_language_detection_and_normalization():
    assert detect_language("こんにちは") == "JA"
    assert detect_language("مرحبا") == "AR"
    assert detect_language("你好") == "ZH"
    assert detect_language("hola") == "EN"
    assert normalize_language("es", "hola") == "ES"


def test_named_emotion_vector_is_normalized_to_safe_mass():
    vector = parse_emotion_vector("happy:0.8,sad:0.4")
    assert vector is not None
    assert len(vector) == 8
    assert sum(vector) == pytest.approx(0.8)
    assert vector[0] > vector[2]


def test_device_and_precision_resolution():
    torch = FakeTorch()
    assert resolve_device(torch, "auto") == "cuda:0"
    assert resolve_device(torch, "cuda:2") == "cuda:2"
    assert resolve_precision(torch, "2.5", "cuda:0", "auto") == "bf16"
    assert resolve_precision(torch, "2.0", "cuda:0", "auto") == "fp16"


def test_runtime_dispatches_25_arguments(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("version: 2.5", encoding="utf-8")
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"wav")
    output = tmp_path / "out.wav"
    runtime = NvidiaIndexTTS(
        NvidiaRuntimeConfig(model_dir=model_dir, device="cuda:1", use_qwen_emotion=True),
        torch_module=FakeTorch(),
        upstream_class=FakeOfficialRuntime,
    )
    result = runtime.generate(
        NvidiaGenerateRequest(
            text="Hola mundo",
            ref_audio=ref,
            output_path=output,
            language="es",
            emotion_text="alegre",
            duration_factor=0.9,
            seed=42,
        )
    )
    assert result.language == "ES"
    assert result.device == "cuda:1"
    assert runtime._torch.cuda.selected_device == 1
    assert runtime._model.init_kwargs["use_bf16"] is True
    assert runtime._model.infer_kwargs["lang"] == "ES"
    assert runtime._model.infer_kwargs["use_emo_text"] is True
    assert runtime._model.infer_kwargs["duration_factor"] == 0.9


def test_runtime_dispatches_20_without_language(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("version: 2", encoding="utf-8")
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"wav")
    runtime = NvidiaIndexTTS(
        NvidiaRuntimeConfig(model_dir=model_dir, version="2.0", device="cuda:0"),
        torch_module=FakeTorch(),
        upstream_class=FakeOfficialRuntime,
    )
    runtime.generate(
        NvidiaGenerateRequest(text="hello", ref_audio=ref, output_path=tmp_path / "out.wav")
    )
    assert runtime._model.init_kwargs["use_fp16"] is True
    assert "lang" not in runtime._model.infer_kwargs
    assert "duration_factor" not in runtime._model.infer_kwargs


def test_runtime_replaces_stale_output(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("version: 2.5", encoding="utf-8")
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"wav")
    output = tmp_path / "out.wav"
    output.write_bytes(b"stale")
    runtime = NvidiaIndexTTS(
        NvidiaRuntimeConfig(model_dir=model_dir, device="cuda:0"),
        torch_module=FakeTorch(),
        upstream_class=FakeOfficialRuntime,
    )
    runtime.generate(NvidiaGenerateRequest(text="hello", ref_audio=ref, output_path=output))
    assert output.read_bytes() == b"RIFFfresh"
