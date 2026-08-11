"""MLX-IndexTTS: IndexTTS 2.0 and 2.5 for Apple Silicon using MLX."""

from mlx_indextts.generate import IndexTTS

__version__ = "0.2.0"
__all__ = ["IndexTTS", "IndexTTSv2", "IndexTTSv25"]


def __getattr__(name: str):
    """Lazily expose versioned runtimes without making base imports Torch-heavy."""
    if name == "IndexTTSv2":
        from mlx_indextts.generate_v2 import IndexTTSv2

        return IndexTTSv2
    if name == "IndexTTSv25":
        from mlx_indextts.generate_v25 import IndexTTSv25

        return IndexTTSv25
    raise AttributeError(name)
