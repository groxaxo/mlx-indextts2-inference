"""Optional Demucs-based vocal extraction for reference audio cleanup."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _ensure_demucs() -> None:
    if importlib.util.find_spec("demucs") is None:
        raise RuntimeError(
            "Demucs is not installed. Install it with: uv sync --extra denoise"
        )


def _cache_name(input_audio_path: str, suffix: str = "denoised") -> str:
    path = Path(input_audio_path)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}_{suffix}.wav"


def denoise_audio_simple(input_audio_path: str, output_audio_path: str | None = None) -> str:
    """Extract vocals from an audio file using Demucs CLI.

    This intentionally does not auto-install dependencies. Auto-installing from
    inside a WebUI request caused broken environments in the PyTorch project.
    """
    _ensure_demucs()
    input_path = Path(input_audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio does not exist: {input_audio_path}")

    if output_audio_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_denoised.wav")
    else:
        output_path = Path(output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            "htdemucs",
            "--two-stems=vocals",
            "-o",
            temp_dir,
            str(input_path),
        ]
        subprocess.run(cmd, check=True)
        vocals_dir = Path(temp_dir) / "htdemucs" / input_path.stem
        vocals_path = vocals_dir / "vocals.wav"
        if not vocals_path.exists():
            wavs = sorted(vocals_dir.glob("*.wav"))
            if not wavs:
                raise FileNotFoundError("Demucs completed but no vocals wav was found")
            vocals_path = wavs[0]
        shutil.copy2(vocals_path, output_path)

    return str(output_path)


def maybe_denoise_reference(
    audio_path: str | None,
    *,
    enabled: bool,
    cache_dir: str = "outputs/denoise_cache",
    suffix: str = "denoised",
) -> str | None:
    """Return a denoised wav path when enabled, otherwise the original path."""
    if not audio_path or not enabled:
        return audio_path
    path = Path(audio_path)
    if path.suffix.lower() == ".npz":
        return audio_path

    output_path = Path(cache_dir) / _cache_name(audio_path, suffix=suffix)
    if output_path.exists():
        return str(output_path)
    return denoise_audio_simple(audio_path, str(output_path))
