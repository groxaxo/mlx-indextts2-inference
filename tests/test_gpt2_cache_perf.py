"""Regression tests for GPT-2 inference hot-path optimizations."""

import mlx.core as mx

from mlx_indextts.models.gpt2 import GPT2Model, KVCache


def test_kv_cache_grows_in_chunks_and_reuses_storage():
    cache = KVCache(step=8)
    first = mx.zeros((1, 3, 16))
    keys, values = cache.update_and_fetch(first, first)

    assert cache.offset == 3
    assert cache.capacity == 8
    assert keys.shape == (1, 3, 16)
    assert values.shape == (1, 3, 16)

    backing_keys = cache.keys
    next_token = mx.ones((1, 1, 16))
    keys, _ = cache.update_and_fetch(next_token, next_token)

    assert cache.offset == 4
    assert cache.capacity == 8
    assert cache.keys is backing_keys
    assert keys.shape == (1, 4, 16)


def test_gpt2_incremental_cache_preserves_legacy_indexing_contract():
    model = GPT2Model(dim=32, num_heads=4, num_layers=2)
    prefix = mx.zeros((1, 5, 32))
    _, cache = model(prefix)
    mx.eval(cache)

    assert isinstance(cache[0], KVCache)
    assert cache[0][0].shape[1] == 5
    first_layer_cache = cache[0]

    token = mx.zeros((1, 1, 32))
    output, updated = model(token, cache=cache)
    mx.eval(output, updated)

    assert output.shape == (1, 1, 32)
    assert updated[0] is first_layer_cache
    assert updated[0][0].shape[1] == 6
    assert updated[0].capacity >= 6
