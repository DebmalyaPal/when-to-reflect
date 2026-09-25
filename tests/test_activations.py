import pytest
import torch

from when_to_reflect.activations import (
    capture_prefix_hidden_states,
    prefix_fingerprint,
    state_id,
    tensor_is_finite,
)

HIDDEN_SIZE = 4
NUM_LAYERS = 3


class FakeConfig:
    num_hidden_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE


class FakeOutputs:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...], logits: torch.Tensor) -> None:
        self.hidden_states = hidden_states
        self.last_hidden_state = hidden_states[-1]
        self.logits = logits


class FakeModel:
    """Returns one distinct hidden state per position, per layer."""

    config = FakeConfig()

    def __init__(self) -> None:
        self.seen_input_ids: list[list[int]] = []
        self.seen_use_cache: list[bool] = []

    def get_input_embeddings(self):
        def embed(input_ids: torch.Tensor) -> torch.Tensor:
            return self._layer(input_ids, layer=0)

        return embed

    def _layer(self, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
        positions = input_ids[0].to(torch.float32)
        base = positions[:, None] + layer * 100.0
        return (base + torch.arange(HIDDEN_SIZE, dtype=torch.float32))[None, :, :]

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 2.0

    def __call__(self, input_ids: torch.Tensor, output_hidden_states: bool, use_cache: bool):
        self.seen_input_ids.append(input_ids[0].tolist())
        self.seen_use_cache.append(use_cache)
        assert output_hidden_states
        hidden_states = tuple(self._layer(input_ids, layer) for layer in range(NUM_LAYERS + 1))
        return FakeOutputs(hidden_states, self.lm_head(hidden_states[-1]))


def test_state_id_is_stable_and_orders_by_boundary() -> None:
    assert state_id("math_train_00003", 76) == "math_train_00003#b00076"
    assert state_id("math_train_00003", 76) == state_id("math_train_00003", 76)
    assert state_id("math_train_00003", 76) != state_id("math_train_00003", 77)


def test_prefix_fingerprint_distinguishes_token_sequences() -> None:
    assert prefix_fingerprint([1, 2, 3]) == prefix_fingerprint([1, 2, 3])
    assert prefix_fingerprint([1, 2, 3]) != prefix_fingerprint([1, 2, 4])
    # Identity is over token IDs, so a different split is a different state.
    assert prefix_fingerprint([1, 23]) != prefix_fingerprint([12, 3])


def test_capture_reads_the_final_prefix_token_of_every_layer() -> None:
    model = FakeModel()
    result = capture_prefix_hidden_states(model, [7, 8, 9], torch.device("cpu"))

    hidden_states = result["hidden_states"]
    assert hidden_states.shape == (NUM_LAYERS + 1, HIDDEN_SIZE)
    assert hidden_states.dtype == torch.float32
    # Final prefix token is 9, and layer l adds 100 * l.
    assert hidden_states[0][0] == pytest.approx(9.0)
    assert hidden_states[NUM_LAYERS][0] == pytest.approx(9.0 + 100.0 * NUM_LAYERS)


def test_capture_runs_on_the_exact_prefix_without_a_cache() -> None:
    model = FakeModel()
    result = capture_prefix_hidden_states(model, [7, 8, 9], torch.device("cpu"))

    assert model.seen_input_ids == [[7, 8, 9]]
    assert model.seen_use_cache == [False]
    assert result["captured_token_ids"] == [7, 8, 9]


def test_capture_reports_verified_hidden_state_indexing() -> None:
    indexing = capture_prefix_hidden_states(FakeModel(), [7, 8], torch.device("cpu"))["indexing"]

    assert indexing["num_hidden_states"] == NUM_LAYERS + 1
    assert indexing["includes_embedding_output"] is True
    assert indexing["first_matches_embedding_output"] is True
    assert indexing["last_matches_last_hidden_state"] is True
    assert indexing["final_matches_lm_head_input"] is True
    assert indexing["hidden_size"] == HIDDEN_SIZE
    assert indexing["layer_labels"][0] == "embedding_output"
    assert indexing["layer_labels"][-1] == f"layer_{NUM_LAYERS}_output"


def test_capture_rejects_an_empty_prefix() -> None:
    with pytest.raises(ValueError, match="empty prefix"):
        capture_prefix_hidden_states(FakeModel(), [], torch.device("cpu"))


def test_tensor_is_finite_detects_nan_and_inf() -> None:
    assert tensor_is_finite(torch.tensor([1.0, -2.0]))
    assert not tensor_is_finite(torch.tensor([1.0, float("nan")]))
    assert not tensor_is_finite(torch.tensor([1.0, float("inf")]))
