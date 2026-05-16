"""Tests for API server path handling."""

from pathlib import Path

import pytest


def test_audio_path_must_stay_under_outputs():
    from mlx_indextts.api_server import _resolve_under_outputs

    with pytest.raises(Exception) as exc_info:
        _resolve_under_outputs("/etc/passwd")

    assert getattr(exc_info.value, "status_code", None) == 400


def test_relative_output_path_resolves_under_outputs():
    from mlx_indextts.api_server import OUTPUTS_ROOT, _resolve_under_outputs

    resolved = _resolve_under_outputs("outputs/api_test.wav")

    assert resolved == (Path.cwd() / "outputs" / "api_test.wav").resolve()
    assert OUTPUTS_ROOT in resolved.parents
