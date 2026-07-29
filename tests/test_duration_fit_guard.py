from __future__ import annotations

from mlx_indextts.runtime import duration_fit_allowed


def test_duration_fit_allows_moderate_adjustment():
    assert duration_fit_allowed(8.0, 10.0, 2.0)
    assert duration_fit_allowed(12.0, 10.0, 2.0)


def test_duration_fit_rejects_extreme_stretch_that_damages_audio():
    assert not duration_fit_allowed(3.0, 10.0, 2.0)
    assert not duration_fit_allowed(3.0, 20.0, 2.0)
