"""Deterministic checkpoint selection from candidate reasoning states."""

import random
from collections.abc import Sequence
from typing import Any

STATE_FIELDS = (
    "step_index",
    "boundary_token_index",
    "prefix_token_ids",
    "prefix_text",
    "relative_position",
)

MIN_ELIGIBLE_RELATIVE_POSITION = 0.10
MAX_ELIGIBLE_RELATIVE_POSITION = 0.90


def eligible_candidates(
    candidate_states: Sequence[dict[str, Any]],
    min_relative_position: float = MIN_ELIGIBLE_RELATIVE_POSITION,
    max_relative_position: float = MAX_ELIGIBLE_RELATIVE_POSITION,
) -> list[dict[str, Any]]:
    """Restrict candidates to the interior of the trace.

    Near-terminal boundaries are excluded because the model has usually already
    derived its answer by then, which leaves the direct branch degenerate: it
    only has to restate a result it already wrote. Trailing boundaries are
    excluded for the symmetric reason that too little reasoning precedes them.

    The filter reads ``relative_position`` only. It never inspects correctness,
    reference answers, or any branch outcome.
    """
    return [
        state
        for state in candidate_states
        if min_relative_position <= state["relative_position"] <= max_relative_position
    ]


def select_checkpoints(
    candidate_states: Sequence[dict[str, Any]],
    seed: int,
    min_relative_position: float = MIN_ELIGIBLE_RELATIVE_POSITION,
    max_relative_position: float = MAX_ELIGIBLE_RELATIVE_POSITION,
) -> list[dict[str, Any]]:
    """Select distinct early, middle, late, and seeded-random checkpoints.

    Only eligible interior candidates are considered, including for the random
    draw. Early, middle, and late are nearest to 0.25, 0.50, and 0.75
    respectively among states still remaining. Ties use the lower boundary
    token index. The returned list follows trace order.
    """
    remaining = eligible_candidates(candidate_states, min_relative_position, max_relative_position)
    selected: list[dict[str, Any]] = []

    def take(state: dict[str, Any], selection_reason: str) -> None:
        remaining.remove(state)
        checkpoint = {field: state[field] for field in STATE_FIELDS}
        checkpoint["selection_reason"] = selection_reason
        selected.append(checkpoint)

    for selection_reason, target in (("early", 0.25), ("middle", 0.5), ("late", 0.75)):
        if remaining:
            take(
                min(
                    remaining,
                    key=lambda state: (
                        abs(state["relative_position"] - target),
                        state["boundary_token_index"],
                    ),
                ),
                selection_reason,
            )
    if remaining:
        take(random.Random(seed).choice(remaining), "random")

    return sorted(selected, key=lambda state: state["boundary_token_index"])
