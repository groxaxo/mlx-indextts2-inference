"""Sampling helpers shared by the IndexTTS GPT runtimes."""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mlx_indextts.performance import schedule_mlx_eval


class RepetitionPenaltyState(list[int]):
    """List-compatible token history with an incremental device-side seen mask.

    IndexTTS generation still needs the ordered Python token list for EOS handling,
    silence compression, and downstream decoding. The additional MLX boolean mask
    avoids reconstructing and sorting a Python set on every autoregressive step.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = max(1, int(vocab_size))
        self.seen = mx.zeros((1, self.vocab_size), dtype=mx.bool_)

    def append(self, token: int) -> None:
        value = int(token)
        super().append(value)
        if 0 <= value < self.vocab_size:
            self.seen[:, value] = True

    def extend(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            self.append(token)


@dataclass
class _CachedHistory:
    source: list[int]
    state: RepetitionPenaltyState
    synced_length: int
    last_token: int | None


class RepetitionStateCache:
    """Incrementally mirror append-only Python histories into MLX seen masks.

    The legacy v1.5/v2.0 generation loops pass ordinary lists. Keeping a small,
    bounded cache lets those loops obtain the same O(1)-per-token repetition state
    as v2.5 without changing their public list contract. Entries retain their list
    briefly and are evicted in LRU order, so long-lived model servers do not grow
    unbounded bookkeeping.
    """

    def __init__(self, vocab_size: int, max_entries: int = 16):
        self.vocab_size = max(1, int(vocab_size))
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[int, _CachedHistory] = OrderedDict()

    def _new_entry(self, tokens: list[int]) -> _CachedHistory:
        state = RepetitionPenaltyState(self.vocab_size)
        state.extend(tokens)
        return _CachedHistory(
            source=tokens,
            state=state,
            synced_length=len(tokens),
            last_token=int(tokens[-1]) if tokens else None,
        )

    def resolve(
        self,
        tokens: Sequence[int] | None,
    ) -> Sequence[int] | None:
        if tokens is None or isinstance(tokens, RepetitionPenaltyState):
            return tokens
        if not isinstance(tokens, list):
            # Non-list callers are uncommon and not append-tracked. Preserve
            # semantics with a one-shot state instead of holding their lifetime.
            state = RepetitionPenaltyState(self.vocab_size)
            state.extend(tokens)
            return state

        key = id(tokens)
        entry = self._entries.get(key)
        append_only = (
            entry is not None
            and entry.source is tokens
            and len(tokens) >= entry.synced_length
            and (
                entry.synced_length == 0
                or int(tokens[entry.synced_length - 1]) == entry.last_token
            )
        )
        if not append_only:
            entry = self._new_entry(tokens)
            self._entries[key] = entry
        elif len(tokens) > entry.synced_length:
            entry.state.extend(tokens[entry.synced_length :])
            entry.synced_length = len(tokens)
            entry.last_token = int(tokens[-1])

        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return entry.state

    def clear(self) -> None:
        self._entries.clear()


_MODEL_HISTORY_CACHES: weakref.WeakKeyDictionary[Any, RepetitionStateCache] = (
    weakref.WeakKeyDictionary()
)
_MODEL_HISTORY_LOCK = threading.Lock()


def resolve_repetition_state(
    owner: Any,
    generated_tokens: Sequence[int] | None,
    vocab_size: int,
) -> Sequence[int] | None:
    """Return an incremental repetition state for one resident GPT model."""
    if generated_tokens is None or isinstance(
        generated_tokens,
        RepetitionPenaltyState,
    ):
        return generated_tokens
    with _MODEL_HISTORY_LOCK:
        try:
            cache = _MODEL_HISTORY_CACHES.get(owner)
        except TypeError:
            # Preserve semantics for exotic non-weak-referenceable callers.
            state = RepetitionPenaltyState(vocab_size)
            state.extend(generated_tokens)
            return state
        if cache is None or cache.vocab_size != int(vocab_size):
            cache = RepetitionStateCache(vocab_size)
            try:
                _MODEL_HISTORY_CACHES[owner] = cache
            except TypeError:
                state = RepetitionPenaltyState(vocab_size)
                state.extend(generated_tokens)
                return state
        return cache.resolve(generated_tokens)


def schedule_decode_outputs(next_token: mx.array, cache: Any) -> None:
    """Submit sampled token and KV cache together before host EOS inspection."""
    schedule_mlx_eval(next_token, cache)


def apply_repetition_penalty(
    logits: mx.array,
    generated_tokens: Sequence[int] | None,
    penalty: float,
) -> mx.array:
    """Apply the sign-aware repetition penalty without Python set sorting."""
    if penalty == 1.0 or not generated_tokens:
        return logits

    vocab_size = int(logits.shape[-1])
    if isinstance(generated_tokens, RepetitionPenaltyState):
        if generated_tokens.vocab_size != vocab_size:
            raise ValueError(
                "repetition state vocabulary does not match the logits vocabulary"
            )
        penalized = mx.where(
            logits > 0,
            logits / penalty,
            logits * penalty,
        )
        return mx.where(generated_tokens.seen, penalized, logits)

    valid_tokens = [
        int(token)
        for token in generated_tokens
        if 0 <= int(token) < vocab_size
    ]
    if not valid_tokens:
        return logits

    token_ids = mx.array([valid_tokens], dtype=mx.int32)
    token_logits = mx.take_along_axis(logits, token_ids, axis=-1)
    penalized = mx.where(
        token_logits > 0,
        token_logits / penalty,
        token_logits * penalty,
    )
    return mx.put_along_axis(logits, token_ids, penalized, axis=-1)


def _select_top_k_candidates(
    logits: mx.array,
    top_k: int,
) -> tuple[mx.array, mx.array | None]:
    """Return only the top-k logits and their original vocabulary indices."""
    vocab_size = int(logits.shape[-1])
    candidate_count = min(max(int(top_k), 0), vocab_size)
    if candidate_count == 0 or candidate_count == vocab_size:
        return logits, None

    # Partitioning is O(V) and avoids relying on mx.topk ordering. MLX documents
    # topk values as not necessarily sorted, so its first value is not a valid
    # threshold. Negating puts the largest logits in the first partition.
    candidate_indices = mx.argpartition(
        -logits,
        kth=candidate_count - 1,
        axis=-1,
    )[..., :candidate_count].astype(mx.int32)
    candidate_logits = mx.take_along_axis(
        logits,
        candidate_indices,
        axis=-1,
    )
    return candidate_logits, candidate_indices


def _sample_from_candidates(
    candidate_logits: mx.array,
    candidate_indices: mx.array | None,
    top_p: float,
) -> mx.array:
    """Apply nucleus filtering to the candidate set and map back to token IDs."""
    token_indices = candidate_indices
    if top_p < 1.0:
        sort_order = mx.argsort(candidate_logits, axis=-1)[..., ::-1]
        candidate_logits = mx.take_along_axis(
            candidate_logits,
            sort_order,
            axis=-1,
        )
        if token_indices is None:
            token_indices = sort_order
        else:
            token_indices = mx.take_along_axis(
                token_indices,
                sort_order,
                axis=-1,
            )

        cumulative_probs = mx.cumsum(
            mx.softmax(candidate_logits, axis=-1),
            axis=-1,
        )
        remove = cumulative_probs > top_p
        keep_first = mx.zeros((*remove.shape[:-1], 1), dtype=mx.bool_)
        remove = mx.concatenate([keep_first, remove[..., :-1]], axis=-1)
        candidate_logits = mx.where(remove, -float("inf"), candidate_logits)

    sampled_position = mx.random.categorical(candidate_logits)
    if token_indices is None:
        return sampled_position
    return mx.take_along_axis(
        token_indices,
        sampled_position[..., None],
        axis=-1,
    ).squeeze(-1)


def sample_logits(
    logits: mx.array,
    *,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.8,
    repetition_penalty: float = 1.0,
    generated_tokens: Sequence[int] | None = None,
) -> mx.array:
    """Sample semantic tokens with bounded top-k/top-p work.

    Top-k is applied before top-p to preserve this project's historical sampling
    order. When top-k is active, nucleus sorting is bounded to that candidate set
    instead of sorting the complete semantic vocabulary. MLX categorical consumes
    unnormalized logits directly, so no final softmax/log round trip is required.
    """
    logits = apply_repetition_penalty(
        logits,
        generated_tokens,
        repetition_penalty,
    )
    if temperature == 0:
        return mx.argmax(logits, axis=-1)

    scaled_logits = logits / temperature
    candidate_logits, candidate_indices = _select_top_k_candidates(
        scaled_logits,
        top_k,
    )
    return _sample_from_candidates(
        candidate_logits,
        candidate_indices,
        top_p,
    )
