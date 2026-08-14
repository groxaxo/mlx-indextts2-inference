import pytest

from scripts.benchmark_indextts25_warm import build_parser, summarize_runs


def test_parser_uses_fast_warm_defaults():
    args = build_parser().parse_args(
        ["--model", "model", "--reference", "voice.wav", "--text", "hello"]
    )

    assert args.warmups == 1
    assert args.runs == 1
    assert args.diffusion_steps == 8
    assert args.seed == 42
    assert args.max_mel_tokens == 256
    assert args.memory_limit_gb == 8.0


def test_summary_reports_aggregate_rtf():
    summary = summarize_runs(
        [
            {"generation_s": 4.0, "audio_duration_s": 2.0},
            {"generation_s": 8.0, "audio_duration_s": 4.0},
        ]
    )

    assert summary["runs"] == 2
    assert summary["mean_generation_s"] == 6.0
    assert summary["median_generation_s"] == 6.0
    assert summary["mean_audio_duration_s"] == 3.0
    assert summary["aggregate_rtf"] == 2.0


def test_summary_avoids_dividing_by_zero_audio():
    summary = summarize_runs([{"generation_s": 1.0, "audio_duration_s": 0.0}])

    assert summary["aggregate_rtf"] is None


@pytest.mark.parametrize("flag", ["--warmups", "--runs", "--diffusion-steps"])
def test_parser_rejects_zero_counts(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--model", "model", "--reference", "voice.wav", "--text", "hello", flag, "0"]
        )
