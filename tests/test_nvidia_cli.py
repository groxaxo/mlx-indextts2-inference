import json

from mlx_indextts.nvidia_cli import (
    _load_jobs,
    _parse_devices,
    _request_from_row,
    build_parser,
)


class FakeCuda:
    def is_available(self):
        return True

    def device_count(self):
        return 3


class FakeTorch:
    cuda = FakeCuda()


def test_parser_generate_defaults():
    args = build_parser().parse_args(
        ["generate", "--text", "hello", "--ref-audio", "ref.wav", "--output", "out.wav"]
    )
    assert args.version == "2.5"
    assert args.precision == "auto"
    assert args.device == "auto"


def test_parse_devices_auto_uses_every_visible_gpu():
    assert _parse_devices("auto", FakeTorch()) == ["cuda:0", "cuda:1", "cuda:2"]


def test_load_jsonl_and_create_request(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps({"text": "hola", "ref_audio": "voice.wav", "language": "es"}) + "\n")
    jobs = _load_jobs(path)
    request = _request_from_row(jobs[0], index=0, output_dir=tmp_path / "outputs")
    assert request.language == "es"
    assert request.output_path.endswith("000000.wav")


def test_request_accepts_list_emotion_vector(tmp_path):
    row = {
        "text": "hello",
        "ref_audio": "voice.wav",
        "emotion_vector": [0.8, 0, 0, 0, 0, 0, 0, 0],
    }
    request = _request_from_row(row, index=1, output_dir=tmp_path)
    assert request.emotion_vector[0] == 0.8
