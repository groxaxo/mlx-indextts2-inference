from __future__ import annotations

import os
from dataclasses import dataclass

BYTES_PER_GB = 1024 ** 3
DEFAULT_MEMORY_CAP_GB = 96.0
DEFAULT_CACHE_CAP_GB = 32.0
LARGE_UNIFIED_MEMORY_GB = 64.0


@dataclass(frozen=True)
class MlxMemoryLimits:
    memory_limit_gb: float | None
    cache_limit_gb: float | None
    source: str


def _system_memory_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            return page_size * pages
        except (OSError, ValueError):
            return None
    return None


def _env_float(*names: str) -> float | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            return max(float(raw), 0.0)
        except ValueError:
            continue
    return None


def _normalize_limit(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return round(float(value), 3)


def resolve_mlx_memory_limits(
    *,
    memory_limit_gb: float | None = None,
    cache_limit_gb: float | None = None,
    total_memory_bytes: int | None = None,
) -> MlxMemoryLimits:
    """Resolve MLX memory/cache limits for Apple unified-memory TTS workloads.

    Explicit arguments win, then environment variables, then an automatic profile.
    The automatic profile is intentionally conservative: it only changes defaults
    on large unified-memory systems and caps M3 Max 128GB hosts at 96GB memory /
    32GB cache, leaving RAM for PyTorch preprocessing, audio codecs, and the OS.
    """
    source = "auto"
    env_memory = _env_float("MLX_INDEXTTS_MEMORY_LIMIT_GB", "MLX_TTS_MEMORY_LIMIT_GB", "MLX_MEMORY_LIMIT_GB")
    env_cache = _env_float("MLX_INDEXTTS_CACHE_LIMIT_GB", "MLX_TTS_CACHE_LIMIT_GB", "MLX_CACHE_LIMIT_GB")

    if memory_limit_gb is None:
        memory_limit_gb = env_memory
        if env_memory is not None:
            source = "env"
    else:
        source = "explicit"

    if cache_limit_gb is None:
        cache_limit_gb = env_cache
        if env_cache is not None and source == "auto":
            source = "env"
    elif source == "auto":
        source = "explicit"

    total_bytes = total_memory_bytes if total_memory_bytes is not None else _system_memory_bytes()
    total_gb = (float(total_bytes) / BYTES_PER_GB) if total_bytes else 0.0

    if memory_limit_gb is None and total_gb >= LARGE_UNIFIED_MEMORY_GB:
        memory_limit_gb = min(DEFAULT_MEMORY_CAP_GB, total_gb * 0.75)
    if cache_limit_gb is None and total_gb >= LARGE_UNIFIED_MEMORY_GB:
        cache_limit_gb = min(DEFAULT_CACHE_CAP_GB, total_gb * 0.25)

    return MlxMemoryLimits(
        memory_limit_gb=_normalize_limit(memory_limit_gb),
        cache_limit_gb=_normalize_limit(cache_limit_gb),
        source=source,
    )


def configure_mlx_runtime(
    *,
    memory_limit_gb: float | None = None,
    cache_limit_gb: float | None = None,
) -> MlxMemoryLimits:
    """Apply resolved MLX limits and return the selected profile."""
    limits = resolve_mlx_memory_limits(
        memory_limit_gb=memory_limit_gb,
        cache_limit_gb=cache_limit_gb,
    )
    try:
        import mlx.core as mx

        if limits.memory_limit_gb is not None and hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(int(limits.memory_limit_gb * BYTES_PER_GB))
        if limits.cache_limit_gb is not None and hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(int(limits.cache_limit_gb * BYTES_PER_GB))
    except Exception:
        # Runtime configuration must never block generation on older MLX builds.
        pass
    return limits


def configure_torch_threads(default_threads: int | None = None) -> int | None:
    """Set a bounded PyTorch CPU preprocessing thread count when torch is loaded."""
    raw = os.environ.get("MLX_INDEXTTS_TORCH_THREADS") or os.environ.get("MLX_TTS_TORCH_THREADS")
    if raw:
        try:
            threads = max(1, int(raw))
        except ValueError:
            threads = None
    else:
        threads = default_threads or min(12, max(1, os.cpu_count() or 1))
    if threads is None:
        return None
    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, threads // 2 or 1)))
        except RuntimeError:
            pass
    except Exception:
        return None
    return threads
