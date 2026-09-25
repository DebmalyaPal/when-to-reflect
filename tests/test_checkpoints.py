from when_to_reflect.checkpoints import (
    STATE_FIELDS,
    eligible_candidates,
    select_checkpoints,
)


def candidate_state(step_index: int, relative_position: float) -> dict:
    return {
        "step_index": step_index,
        "boundary_token_index": (step_index + 1) * 10,
        "prefix_token_ids": [1, step_index],
        "prefix_text": f"prefix {step_index}",
        "relative_position": relative_position,
        "reference_solution": "ignored",
        "outcome": "ignored",
    }


def test_checkpoint_selection_is_deterministic_and_distinct() -> None:
    candidates = [candidate_state(index, position) for index, position in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])]

    first = select_checkpoints(candidates, seed=42)
    second = select_checkpoints(candidates, seed=42)

    assert first == second
    assert len(first) == 4
    assert len({checkpoint["boundary_token_index"] for checkpoint in first}) == len(first)


def test_checkpoint_selection_uses_quarter_half_and_three_quarter_targets() -> None:
    candidates = [
        candidate_state(index, position)
        for index, position in enumerate([0.1, 0.24, 0.26, 0.49, 0.51, 0.74, 0.76, 0.9])
    ]

    selected = select_checkpoints(candidates, seed=42)
    selected_by_reason = {checkpoint["selection_reason"]: checkpoint for checkpoint in selected}

    assert selected_by_reason["early"]["relative_position"] == 0.24
    assert selected_by_reason["middle"]["relative_position"] == 0.49
    assert selected_by_reason["late"]["relative_position"] == 0.74


def test_checkpoint_selection_breaks_position_ties_by_lower_boundary() -> None:
    candidates = [
        candidate_state(index, position)
        for index, position in enumerate([0.125, 0.375, 0.5, 0.625, 0.875])
    ]

    selected = select_checkpoints(candidates, seed=1)
    selected_by_reason = {checkpoint["selection_reason"]: checkpoint for checkpoint in selected}

    assert selected_by_reason["early"]["boundary_token_index"] == 10
    assert selected_by_reason["late"]["boundary_token_index"] == 40


def test_selected_checkpoints_are_candidate_members_in_trace_order() -> None:
    candidates = [candidate_state(index, position) for index, position in enumerate([0.2, 0.4, 0.6, 0.8])]

    selected = select_checkpoints(candidates, seed=7)

    candidate_signatures = {tuple(state[field] if field != "prefix_token_ids" else tuple(state[field]) for field in STATE_FIELDS) for state in candidates}
    assert all(
        tuple(
            checkpoint[field] if field != "prefix_token_ids" else tuple(checkpoint[field])
            for field in STATE_FIELDS
        )
        in candidate_signatures
        for checkpoint in selected
    )
    assert [checkpoint["boundary_token_index"] for checkpoint in selected] == sorted(
        checkpoint["boundary_token_index"] for checkpoint in selected
    )


def test_near_terminal_candidates_are_ineligible() -> None:
    # 0.977 is the answer-emission state observed in the one-problem smoke run.
    candidates = [
        candidate_state(index, position)
        for index, position in enumerate([0.067, 0.5, 0.908, 0.977])
    ]

    eligible = eligible_candidates(candidates)
    selected = select_checkpoints(candidates, seed=42)

    assert [state["relative_position"] for state in eligible] == [0.5]
    assert [checkpoint["relative_position"] for checkpoint in selected] == [0.5]
    assert all(checkpoint["relative_position"] <= 0.90 for checkpoint in selected)
    assert all(checkpoint["relative_position"] >= 0.10 for checkpoint in selected)


def test_eligibility_band_is_inclusive_at_both_edges() -> None:
    candidates = [
        candidate_state(index, position)
        for index, position in enumerate([0.099, 0.10, 0.90, 0.901])
    ]

    eligible = eligible_candidates(candidates)

    assert [state["relative_position"] for state in eligible] == [0.10, 0.90]


def test_random_checkpoint_is_drawn_only_from_eligible_candidates() -> None:
    # Only three interior candidates exist, so any fourth pick would be terminal.
    candidates = [
        candidate_state(index, position)
        for index, position in enumerate([0.2, 0.5, 0.8, 0.95, 0.99])
    ]

    for seed in range(20):
        selected = select_checkpoints(candidates, seed=seed)
        assert [checkpoint["relative_position"] for checkpoint in selected] == [0.2, 0.5, 0.8]


def test_random_checkpoint_uses_a_remaining_eligible_candidate() -> None:
    candidates = [
        candidate_state(index, position)
        for index, position in enumerate([0.2, 0.5, 0.8, 0.85, 0.95])
    ]

    selected = select_checkpoints(candidates, seed=3)
    random_checkpoint = next(
        checkpoint for checkpoint in selected if checkpoint["selection_reason"] == "random"
    )

    assert random_checkpoint["relative_position"] == 0.85


def test_checkpoint_selection_ignores_non_candidate_outcome_fields() -> None:
    candidates = [candidate_state(index, position) for index, position in enumerate([0.1, 0.5, 0.9])]
    altered_candidates = [dict(state, reference_solution="different", outcome="different") for state in candidates]

    assert select_checkpoints(candidates, seed=3) == select_checkpoints(altered_candidates, seed=3)
