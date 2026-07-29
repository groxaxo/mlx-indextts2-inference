from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from mlx_indextts import cli


def test_generate_command_prints_structured_runtime_metrics(capsys):
    args = MagicMock()
    args.model = "models/test"
    args.profile = "standard"
    args.text = "hello"
    args.ref_audio = "reference.wav"
    args.output = "output.wav"
    args.temperature = 0.8
    args.max_tokens = 128
    args.memory_limit = None
    args.quantize = None
    args.top_k = 30
    args.top_p = 0.8
    args.verbose = False
    args.play = False
    args.seed = 42
    args.auto_emotion = False
    args.emotion = None
    args.emotion_ref_audio = None
    args.denoise_ref = False
    args.denoise_emotion_ref = False

    runtime = MagicMock()
    runtime.generate.return_value = {
        "output_path": "output.wav",
        "duration_s": 2.0,
        "elapsed_s": 1.0,
        "rtf": 0.5,
    }
    with (
        patch.object(cli, "detect_mlx_version", return_value="2.0"),
        patch("mlx_indextts.runtime.TTSRuntime", return_value=runtime),
    ):
        cli.generate_command(args)

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["rtf"] == 0.5
    assert payload["duration_s"] == 2.0
