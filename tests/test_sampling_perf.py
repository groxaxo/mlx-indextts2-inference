"""Regression tests for bounded IndexTTS 2.x sampling hot paths."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_indextts.models.sampling import (
    RepetitionPenaltyState,
    _select_top_k_candidates,
    apply_repetition_penalty,
    sample_logits,
)


def test_repetition_state_tracks_seen_tokens_and_penalizes_once():
    state = RepetitionPenaltyState(vocab_size=4)
    state.extend([0, 1, 1, 99])
    mx.eval(state.seen)

    assert list(state) == [0, 1, 1, 99]
    np.testing.assert_array_equal(
        np.asarray(state.seen),
        [[True, True, False, False]],
    )

    logits = mx.array([[2.0, -2.0, 1.0, 0.5]], dtype=mx.float32)
    penalized = apply_repetition_penalty(logits, state, penalty=2.0)
    mx.eval(penalized)

    np.testing.assert_allclose(
        np.asarray(penalized),
        [[1.0, -4.0, 1.0, 0.5]],
    )


def test_legacy_repetition_history_ignores_invalid_tokens_and_duplicates():
    logits = mx.array([[2.0, -2.0, 1.0, 0.5]], dtype=mx.float32)
    penalized = apply_repetition_penalty(
        logits,
        [0, 1, 1, 99],
        penalty=2.0,
    )
    mx.eval(penalized)

    np.testing.assert_allclose(
        np.asarray(penalized),
        [[1.0, -4.0, 1.0, 0.5]],
    )


def test_top_k_candidate_selection_is_bounded_and_preserves_vocab_ids():
    logits = mx.array([[1.0, 7.0, 5.0, 3.0, 9.0]], dtype=mx.float32)
    candidate_logits, candidate_indices = _select_top_k_candidates(logits, top_k=3)
    mx.eval(candidate_logits, candidate_indices)

    assert candidate_logits.shape == (1, 3)
    assert candidate_indices is not None
    assert set(np.asarray(candidate_indices).reshape(-1).tolist()) == {1, 2, 4}
    assert set(np.asarray(candidate_logits).reshape(-1).tolist()) == {5.0, 7.0, 9.0}


def test_top_k_sampling_never_leaks_filtered_vocabulary_tokens():
    logits = mx.array([[8.0, 7.0, 1.0, 0.0]], dtype=mx.float32)
    mx.random.seed(41)

    samples = {
        int(
            sample_logits(
                logits,
                temperature=1.0,
                top_k=2,
                top_p=1.0,
            )[0].item()
        )
        for _ in range(64)
    }

    assert samples <= {0, 1}
    assert samples


def test_top_p_runs_on_top_k_candidates_and_maps_back_to_vocab():
    logits = mx.array([[10.0, 9.0, 1.0, 0.0]], dtype=mx.float32)
    mx.random.seed(7)

    samples = [
        int(
            sample_logits(
                logits,
                temperature=1.0,
                top_k=2,
                top_p=0.6,
            )[0].item()
        )
        for _ in range(16)
    ]

    assert samples == [0] * 16


def test_v25_decode_preallocates_mask_and_uses_incremental_history():
    from mlx_indextts.generate_v25 import IndexTTSv25

    class FakeEmbedding:
        def __call__(self, _tokens):
            return mx.zeros((1, 1, 4))

    class FakeGPT:
        start_mel_token = 1
        stop_mel_token = 99

        def __init__(self):
            self.mel_embedding = FakeEmbedding()
            self.mel_pos_embedding = SimpleNamespace(
                get_fixed_embedding=lambda _index: mx.zeros((1, 1, 4))
            )
            self.tokens = iter((7, 8, self.stop_mel_token))
            self.mask_lengths = []
            self.histories = []
            self.history_types = []

        def prepare_inputs(self, *_args, **_kwargs):
            return (
                mx.zeros((1, 3, 4)),
                mx.array([[0, 1, 1]], dtype=mx.int32),
            )

        def new_repetition_state(self):
            return RepetitionPenaltyState(vocab_size=128)

        def generate_step(self, _current_emb, **kwargs):
            history = kwargs["generated_tokens"]
            self.mask_lengths.append(int(kwargs["attention_mask"].shape[1]))
            self.histories.append(list(history))
            self.history_types.append(type(history))
            token = next(self.tokens)
            return (
                mx.array([token], dtype=mx.int32),
                mx.zeros((1, 1, 1)),
                [mx.zeros((1, 1))],
            )

    runtime = IndexTTSv25.__new__(IndexTTSv25)
    runtime.gpt = FakeGPT()

    codes = runtime._generate_semantic_codes(
        mx.zeros((1, 3, 4)),
        mx.array([[10, 1]], dtype=mx.int32),
        0,
        max_mel_tokens=3,
        temperature=0.8,
        top_p=0.8,
        top_k=30,
        repetition_penalty=10.0,
    )

    assert codes == [7, 8]
    assert runtime.gpt.mask_lengths == [4, 5, 6]
    assert runtime.gpt.histories == [[], [7], [7, 8]]
    assert runtime.gpt.history_types == [RepetitionPenaltyState] * 3
