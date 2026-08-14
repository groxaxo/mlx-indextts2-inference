"""Sampling helpers shared by the IndexTTS 2.x GPT runtimes."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx


class RepetitionPenaltyState(list[int]):
    """List-compatible token history with an incremental device-side seen mask.

    IndexTTS generation still needs the ordered Python token list for EOS handling,
    silence compression, and downstream decoding.  The additional MLX boolean mask
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

    # The legacy v2.0 loop still supplies a normal list. Duplicate indices are
    # intentionally retained: they gather identical source logits and scatter
    # identical penalized values, avoiding the old set construction and sort.
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
