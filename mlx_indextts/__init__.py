"""IndexTTS inference for Apple MLX and NVIDIA CUDA."""

__version__ = "0.3.0"
__all__ = [
    "IndexTTS",
    "IndexTTSv2",
    "IndexTTSv25",
    "NvidiaIndexTTS",
    "NvidiaRuntimeConfig",
    "NvidiaGenerateRequest",
]


def __getattr__(name: str):
    """Lazily import platform-specific backends.

    Avoiding an eager MLX import is what lets the package and NVIDIA utilities
    import cleanly on Linux/Windows hosts where Apple's MLX package is absent.
    """

    if name == "IndexTTS":
        from mlx_indextts.generate import IndexTTS

        return IndexTTS
    if name == "IndexTTSv2":
        from mlx_indextts.generate_v2 import IndexTTSv2

        return IndexTTSv2
    if name == "IndexTTSv25":
        from mlx_indextts.generate_v25 import IndexTTSv25

        return IndexTTSv25
    if name in {"NvidiaIndexTTS", "NvidiaRuntimeConfig", "NvidiaGenerateRequest"}:
        from mlx_indextts.nvidia_runtime import (
            NvidiaGenerateRequest,
            NvidiaIndexTTS,
            NvidiaRuntimeConfig,
        )

        return {
            "NvidiaIndexTTS": NvidiaIndexTTS,
            "NvidiaRuntimeConfig": NvidiaRuntimeConfig,
            "NvidiaGenerateRequest": NvidiaGenerateRequest,
        }[name]
    raise AttributeError(name)
