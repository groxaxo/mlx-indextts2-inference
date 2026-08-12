"""Tests for strict IndexTTS 2.5 PyTorch-to-MLX weight conversion."""

import mlx.core as mx
import numpy as np
import pytest


def test_supported_persistent_gpt_quantization_bits_include_requested_variants():
    from mlx_indextts.convert_v25 import SUPPORTED_GPT_QUANTIZATION_BITS

    assert SUPPORTED_GPT_QUANTIZATION_BITS == frozenset({3, 4, 5, 6, 8})


def test_mlx_three_bit_group_quantization_round_trip():
    import mlx.core as mx

    weights = mx.ones((128, 128), dtype=mx.float16)
    packed, scales, biases = mx.quantize(weights, bits=3, group_size=64)
    restored = mx.dequantize(packed, scales, biases, bits=3, group_size=64)

    assert restored.shape == weights.shape


def test_convert_codec_fuses_weight_norm_and_transposes_conv1d():
    from mlx_indextts.convert_v25 import convert_codec_v25_weights

    source = {
        "down.weight": np.arange(18, dtype=np.float32).reshape(2, 3, 3),
        "down.bias": np.zeros(2, dtype=np.float32),
        "quantizer.quantizers.0.out_project.weight_v": np.array(
            [[[3.0], [4.0]]], dtype=np.float32
        ),
        "quantizer.quantizers.0.out_project.weight_g": np.array(
            [[[10.0]]], dtype=np.float32
        ),
        "quantizer.quantizers.0.out_project.bias": np.zeros(1, dtype=np.float32),
    }

    converted = convert_codec_v25_weights(source)

    assert converted["down.weight"].shape == (2, 3, 3)
    np.testing.assert_array_equal(
        np.asarray(converted["down.weight"]),
        source["down.weight"].transpose(0, 2, 1),
    )
    np.testing.assert_allclose(
        np.asarray(converted["quantizer.quantizers.0.out_project.weight"]),
        [[[6.0, 8.0]]],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not any(key.endswith(("weight_g", "weight_v")) for key in converted)


def test_convert_codec_can_cast_floating_weights():
    from mlx_indextts.convert_v25 import convert_codec_v25_weights

    converted = convert_codec_v25_weights(
        {"decoder.1.weight": np.ones((4, 3), dtype=np.float32)},
        dtype="float16",
    )

    assert converted["decoder.1.weight"].dtype == mx.float16


def test_convert_gpt_v25_preserves_new_modules_and_transposes_hf_conv1d():
    from mlx_indextts.config import IndexTTSConfig
    from mlx_indextts.convert_v25 import convert_gpt_v25_weights

    config = IndexTTSConfig()
    config.gpt.layers = 1
    source = {
        "spk_emb_proj.weight": np.ones((8, 192), dtype=np.float32),
        "spk_emb_proj.bias": np.zeros(8, dtype=np.float32),
        "lang_embedding.weight": np.ones((107, 8), dtype=np.float32),
        "gpt.h.0.attn.c_attn.weight": np.arange(192, dtype=np.float32).reshape(8, 24),
    }

    converted = convert_gpt_v25_weights(source, config)

    assert converted["spk_emb_proj.weight"].shape == (8, 192)
    assert converted["lang_embedding.weight"].shape == (107, 8)
    assert converted["gpt.h.0.attn.c_attn.weight"].shape == (24, 8)


def test_flatten_and_convert_s2mel_nested_checkpoint():
    from mlx_indextts.convert_v25 import convert_s2mel_v25_weights

    source = {
        "net": {
            "gpt_layer": {
                "0.weight": np.ones((4, 3), dtype=np.float32),
                "0.bias": np.zeros(4, dtype=np.float32),
            },
            "length_regulator": {
                "model.0.weight": np.ones((2, 2, 3), dtype=np.float32),
            },
        }
    }

    converted = convert_s2mel_v25_weights(source, dtype="float16")

    assert converted["gpt_layer.layers.0.weight"].shape == (4, 3)
    assert converted["length_regulator.model.0.weight"].shape == (2, 3, 2)
    assert all(value.dtype == mx.float16 for value in converted.values())


def test_flatten_s2mel_rejects_invalid_checkpoint_shape():
    from mlx_indextts.convert_v25 import WeightMappingError, flatten_s2mel_state

    with pytest.raises(WeightMappingError, match="net"):
        flatten_s2mel_state({"wrong": {}})


def test_strict_parameter_audit_reports_missing_and_unexpected():
    from mlx_indextts.convert_v25 import WeightMappingError, audit_parameter_mapping

    with pytest.raises(WeightMappingError, match="missing=.*bias.*unexpected=.*extra"):
        audit_parameter_mapping(
            expected_keys={"layer.weight", "layer.bias"},
            converted_keys={"layer.weight", "extra"},
            source_tensors=3,
            ignored=["buffer"],
            strict=True,
        )


def test_complete_parameter_audit_builds_manifest_component_record():
    from mlx_indextts.convert_v25 import audit_parameter_mapping

    record = audit_parameter_mapping(
        expected_keys={"a", "b"},
        converted_keys={"a", "b"},
        source_tensors=3,
        ignored=["runtime_buffer"],
        strict=True,
    )

    assert record == {
        "source_tensors": 3,
        "mapped_tensors": 2,
        "ignored": ["runtime_buffer"],
        "missing": [],
        "unexpected": [],
    }


def test_conversion_staging_resumes_only_matching_identity_and_artifact_size(tmp_path):
    from mlx_indextts.convert_v25 import (
        _component_is_resumable,
        _mark_component_complete,
        prepare_conversion_staging,
    )

    output = tmp_path / "mlx-v25"
    identity = {"source_revision": "pinned", "dtype": "float16"}
    staging, state = prepare_conversion_staging(output, identity)
    artifact = staging / "codec.safetensors"
    artifact.write_bytes(b"safe")
    _mark_component_complete(
        staging,
        state,
        "codec",
        {"bytes": artifact.stat().st_size, "missing": []},
    )

    resumed_staging, resumed_state = prepare_conversion_staging(output, identity)

    assert resumed_staging == staging
    assert _component_is_resumable(
        resumed_staging,
        resumed_state,
        "codec",
        "codec.safetensors",
    )
    artifact.write_bytes(b"truncated")
    assert not _component_is_resumable(
        resumed_staging,
        resumed_state,
        "codec",
        "codec.safetensors",
    )


def test_conversion_staging_rejects_mismatched_resume_without_force(tmp_path):
    from mlx_indextts.convert_v25 import WeightMappingError, prepare_conversion_staging

    output = tmp_path / "mlx-v25"
    prepare_conversion_staging(output, {"dtype": "float16"})

    with pytest.raises(WeightMappingError, match="different source or options"):
        prepare_conversion_staging(output, {"dtype": "float32"})


def test_conversion_force_archives_existing_output(tmp_path):
    from mlx_indextts.convert_v25 import prepare_conversion_staging

    output = tmp_path / "mlx-v25"
    output.mkdir()
    (output / "keep.txt").write_text("old", encoding="utf-8")

    staging, _ = prepare_conversion_staging(
        output,
        {"dtype": "float16"},
        force=True,
    )

    backups = list(tmp_path.glob("mlx-v25.backup-*"))
    assert staging.is_dir()
    assert len(backups) == 1
    assert (backups[0] / "keep.txt").read_text(encoding="utf-8") == "old"
