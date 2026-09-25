"""Pre-treatment hidden-state capture for reflection-utility states.

A state s is a token prefix. Its representation must therefore be read from a
forward pass over exactly that prefix, before either branch action is applied:

    h_l(s) = hidden state at layer l for the final token of the exact
             pre-treatment prefix

Nothing downstream of the state may contribute. The reflection instruction, the
direct continuation and the reflection continuation are all post-treatment, so
none of them is ever part of the input here. The prefix is used as stored token
IDs and is never decoded and re-tokenized, which would silently redefine the
state.

Capture uses the model's own ``output_hidden_states`` API rather than hooks, so
there is nothing to keep in sync with the model implementation.
"""

import hashlib
from collections.abc import Sequence
from typing import Any

import torch


def state_id(problem_id: str, boundary_token_index: int) -> str:
    """Return the stable identifier for one reflection-utility state.

    A state is pinned by its source problem and the trace boundary it was cut
    at, both of which are fixed when the state is created and never depend on
    an outcome. Checkpoint selection can only ever pick one state per boundary,
    so the pair identifies it uniquely within a run.
    """
    return f"{problem_id}#b{boundary_token_index:05d}"


def prefix_fingerprint(prefix_token_ids: Sequence[int]) -> str:
    """Return a content hash of the exact pre-treatment token prefix.

    Identity is over token IDs, not text: two different token sequences can
    decode to the same string, and those are different states.
    """
    payload = ",".join(str(token_id) for token_id in prefix_token_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hidden_state_indexing(
    model: Any,
    hidden_states: tuple[torch.Tensor, ...],
    outputs: Any,
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    """Describe what the returned ``hidden_states`` tuple actually contains.

    The layout is checked against the model rather than assumed: that the tuple
    is one entry longer than the layer count, that entry 0 is the embedding
    output, and what the last entry is. Callers record the result so the
    meaning of a layer index is recoverable from the saved artifact alone.

    A causal-LM forward pass exposes no ``last_hidden_state``, so the last
    entry is identified through the LM head instead: if feeding it to the head
    reproduces the returned logits, it is the final normalized hidden state the
    head reads, not the last block's raw output.
    """
    num_layers = model.config.num_hidden_layers
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    lm_head = getattr(model, "lm_head", None)
    logits = getattr(outputs, "logits", None)

    with torch.inference_mode():
        embedding_output = model.get_input_embeddings()(input_ids)
        first_matches_embedding_output = bool(
            torch.equal(hidden_states[0].to(embedding_output.dtype), embedding_output)
        )
        final_matches_lm_head_input = (
            None
            if lm_head is None or logits is None
            else bool(torch.equal(lm_head(hidden_states[-1]), logits))
        )

    return {
        "num_hidden_states": len(hidden_states),
        "num_hidden_layers": num_layers,
        "hidden_size": model.config.hidden_size,
        "includes_embedding_output": len(hidden_states) == num_layers + 1,
        "first_matches_embedding_output": first_matches_embedding_output,
        "last_matches_last_hidden_state": (
            None
            if last_hidden_state is None
            else bool(torch.equal(hidden_states[-1], last_hidden_state))
        ),
        "final_matches_lm_head_input": final_matches_lm_head_input,
        "layer_labels": ["embedding_output"]
        + [f"layer_{index}_output" for index in range(1, len(hidden_states))],
    }


def capture_prefix_hidden_states(
    model: Any,
    prefix_token_ids: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    """Capture every layer's hidden state at the final token of one exact prefix.

    The returned tensor is ``[num_hidden_states, hidden_size]`` in float32 on
    the CPU, detached from the graph. Casting to a storage dtype is left to the
    caller so that the full-precision values stay available for verification.

    ``use_cache=False`` keeps this a plain forward pass: no KV cache is built
    and no generation state is touched, so capturing a representation cannot
    perturb any rollout.
    """
    prefix = list(prefix_token_ids)
    if not prefix:
        raise ValueError("Cannot capture hidden states for an empty prefix")

    input_ids = torch.tensor([prefix], device=device)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    # The state is the prefix, so only its final token position is read.
    hidden_states = outputs.hidden_states
    final_token = torch.stack(
        [layer[0, -1, :].detach().to(torch.float32).cpu() for layer in hidden_states]
    )

    captured_token_ids = input_ids[0].tolist()
    if captured_token_ids != prefix:
        raise ValueError("Forward pass did not run on the exact stored prefix token IDs")

    return {
        "hidden_states": final_token,
        "indexing": hidden_state_indexing(model, hidden_states, outputs, input_ids),
        "captured_token_ids": captured_token_ids,
    }


def tensor_is_finite(tensor: torch.Tensor) -> bool:
    """Return whether every value is finite, i.e. no NaN and no infinity."""
    return bool(torch.isfinite(tensor).all())
