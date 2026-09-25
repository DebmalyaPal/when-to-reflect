from typing import Any

import pytest

from when_to_reflect.branching import build_branch_inputs
from when_to_reflect.utility import (
    REWARD_BY_OUTCOME,
    ROLLOUT_OUTCOMES,
    classify_rollout_outcome,
    estimate_checkpoint_utilities,
    estimate_state_utility,
    generate_paired_rollouts,
    paired_rollout_seeds,
    rollout_outcome_fields,
    rollout_seed,
    run_summary,
)

SHARED_PREFIX_TOKEN_IDS = [10, 11, 12]
INTERVENTION_TOKEN_IDS = [20, 21]


def branches() -> dict[str, Any]:
    return build_branch_inputs(SHARED_PREFIX_TOKEN_IDS, INTERVENTION_TOKEN_IDS)


def recording_generator(calls: list[dict[str, Any]]):
    """Return a fake generator that logs what it was asked to continue from."""

    def generate(input_token_ids, seed):
        calls.append({"input_token_ids": list(input_token_ids), "seed": seed})
        return [seed, len(input_token_ids)]

    return generate


def rollout(
    rollout_index: int,
    correct: bool | None,
    *,
    generation_complete: bool = True,
    answer_extraction_success: bool = True,
    reference_extraction_success: bool = True,
) -> dict[str, Any]:
    """Build one graded rollout carrying the reward view the estimator reads."""
    graded = {
        "rollout_index": rollout_index,
        "generation_complete": generation_complete,
        "answer_extraction_success": answer_extraction_success,
        "correct": correct,
    }
    return {**graded, **rollout_outcome_fields(graded, reference_extraction_success)}


def graded(correctness: list[bool], **kwargs: Any) -> list[dict[str, Any]]:
    return [rollout(index, correct, **kwargs) for index, correct in enumerate(correctness)]


# --- seed derivation -------------------------------------------------------


def test_rollout_seeds_are_deterministic_per_index() -> None:
    assert rollout_seed(42, 0) == rollout_seed(42, 0) == 42
    assert rollout_seed(42, 3) == rollout_seed(42, 3) == 45


def test_different_rollout_indices_receive_different_seeds() -> None:
    seeds = paired_rollout_seeds(42, 4)

    assert seeds == [42, 43, 44, 45]
    assert len(set(seeds)) == len(seeds)


def test_seed_derivation_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="rollout_index"):
        rollout_seed(42, -1)
    with pytest.raises(ValueError, match="num_rollouts"):
        paired_rollout_seeds(42, 0)


# --- paired rollout generation ---------------------------------------------


def test_generates_exactly_k_rollouts_per_branch() -> None:
    calls: list[dict[str, Any]] = []

    rollouts = generate_paired_rollouts(branches(), 42, 4, recording_generator(calls))

    assert len(rollouts["direct"]) == 4
    assert len(rollouts["reflect"]) == 4
    assert len(calls) == 8
    assert [entry["rollout_index"] for entry in rollouts["direct"]] == [0, 1, 2, 3]
    assert [entry["rollout_index"] for entry in rollouts["reflect"]] == [0, 1, 2, 3]


def test_direct_and_reflection_rollouts_share_matched_seeds() -> None:
    rollouts = generate_paired_rollouts(branches(), 42, 4, recording_generator([]))

    direct_seeds = [entry["seed"] for entry in rollouts["direct"]]
    reflect_seeds = [entry["seed"] for entry in rollouts["reflect"]]

    assert direct_seeds == reflect_seeds
    assert direct_seeds == paired_rollout_seeds(42, 4) == [42, 43, 44, 45]
    assert len(set(direct_seeds)) == 4


def test_every_rollout_continues_from_the_exact_shared_prefix() -> None:
    calls: list[dict[str, Any]] = []
    branch_inputs = branches()

    rollouts = generate_paired_rollouts(branch_inputs, 42, 4, recording_generator(calls))

    direct_calls = calls[0::2]
    reflect_calls = calls[1::2]
    assert all(call["input_token_ids"] == SHARED_PREFIX_TOKEN_IDS for call in direct_calls)
    assert all(
        call["input_token_ids"][: len(SHARED_PREFIX_TOKEN_IDS)] == SHARED_PREFIX_TOKEN_IDS
        for call in reflect_calls
    )
    assert all(
        call["input_token_ids"] == SHARED_PREFIX_TOKEN_IDS + INTERVENTION_TOKEN_IDS
        for call in reflect_calls
    )
    assert branch_inputs["shared_prefix_token_ids"] == SHARED_PREFIX_TOKEN_IDS
    assert all(
        entry["shared_prefix_preserved"]
        for entry in rollouts["direct"] + rollouts["reflect"]
    )


def test_rejects_branch_inputs_that_lost_the_shared_prefix() -> None:
    branch_inputs = branches()
    branch_inputs["reflect"]["input_token_ids"] = [99, *INTERVENTION_TOKEN_IDS]

    with pytest.raises(ValueError, match="shared prefix"):
        generate_paired_rollouts(branch_inputs, 42, 1, recording_generator([]))


def test_rerunning_the_same_configuration_reproduces_seeds_and_outputs() -> None:
    first = generate_paired_rollouts(branches(), 42, 4, recording_generator([]))
    second = generate_paired_rollouts(branches(), 42, 4, recording_generator([]))

    assert first == second


# --- reward availability ---------------------------------------------------


def test_every_outcome_has_a_declared_reward() -> None:
    assert set(REWARD_BY_OUTCOME) == set(ROLLOUT_OUTCOMES)
    assert REWARD_BY_OUTCOME == {
        "correct": 1.0,
        "incorrect": 0.0,
        "no_answer": 0.0,
        "generation_incomplete": None,
        "grader_failure": None,
        "reference_failure": None,
    }


def test_correct_and_incorrect_answers_are_scored() -> None:
    assert rollout(0, True)["outcome"] == "correct"
    assert rollout(0, True)["reward"] == 1.0
    assert rollout(0, False)["outcome"] == "incorrect"
    assert rollout(0, False)["reward"] == 0.0


def test_completed_generation_without_an_answer_scores_zero_as_no_answer() -> None:
    no_answer = rollout(0, False, answer_extraction_success=False)

    assert no_answer["outcome"] == "no_answer"
    assert no_answer["reward_available"] is True
    assert no_answer["reward"] == 0.0
    # The raw grading fields stay visible for auditing.
    assert no_answer["answer_extraction_success"] is False
    assert no_answer["correct"] is False


def test_empty_eos_only_continuation_is_classified_no_answer() -> None:
    # An immediate EOS: complete, one token, nothing extractable.
    empty_rollout = {
        "rollout_index": 0,
        "generated_token_ids": [151645],
        "generated_text": "",
        "generated_token_count": 1,
        "terminated_by_eos": True,
        "generation_complete": True,
        "answer_extraction_success": False,
        "correct": False,
    }

    assert classify_rollout_outcome(empty_rollout, True) == "no_answer"
    assert rollout_outcome_fields(empty_rollout, True)["reward"] == 0.0


def test_incomplete_and_reference_failures_yield_no_reward() -> None:
    incomplete = rollout(0, None, generation_complete=False)
    assert incomplete["outcome"] == "generation_incomplete"
    assert incomplete["reward_available"] is False
    assert incomplete["reward"] is None

    without_reference = rollout(0, True, reference_extraction_success=False)
    assert without_reference["outcome"] == "reference_failure"
    assert without_reference["reward_available"] is False
    assert without_reference["reward"] is None


def test_reference_failure_outranks_every_other_outcome() -> None:
    # Whatever the rollout did, a state with no reference answer is ungradable.
    for generation_complete in (True, False):
        for answer_extraction_success in (True, False):
            graded = {
                "generation_complete": generation_complete,
                "answer_extraction_success": answer_extraction_success,
                "correct": True,
            }
            assert classify_rollout_outcome(graded, False) == "reference_failure"


def test_grader_failure_is_declared_but_unreachable_with_the_current_grader() -> None:
    # score_answer only returns None for an incomplete generation, which is
    # classified earlier, so this label needs a hypothetical grader to appear.
    hypothetical = {
        "generation_complete": True,
        "answer_extraction_success": True,
        "correct": None,
    }

    assert classify_rollout_outcome(hypothetical, True) == "grader_failure"
    assert REWARD_BY_OUTCOME["grader_failure"] is None


# --- utility aggregation ---------------------------------------------------


def test_utility_hat_is_the_paired_mean_reward_difference() -> None:
    utility = estimate_state_utility(
        graded([True, False, False, False]),
        graded([True, True, True, False]),
        num_rollouts=4,
    )

    assert utility["direct_rewards"] == [1.0, 0.0, 0.0, 0.0]
    assert utility["reflect_rewards"] == [1.0, 1.0, 1.0, 0.0]
    assert utility["direct_mean_reward"] == 0.25
    assert utility["reflect_mean_reward"] == 0.75
    assert utility["utility_hat"] == 0.5
    assert utility["all_rollouts_valid"] is True
    assert utility["num_valid_direct_rollouts"] == 4
    assert utility["num_valid_reflect_rollouts"] == 4
    assert utility["invalid_rollouts"] == []


def test_harmful_reflection_yields_negative_utility() -> None:
    utility = estimate_state_utility(graded([True, True]), graded([False, True]), num_rollouts=2)

    assert utility["utility_hat"] == -0.5


def test_incomplete_rollout_is_not_counted_as_reward_zero() -> None:
    direct = [rollout(0, None, generation_complete=False), rollout(1, True)]

    utility = estimate_state_utility(direct, graded([True, True]), num_rollouts=2)

    assert utility["direct_rewards"] == [None, 1.0]
    assert utility["direct_mean_reward"] is None
    assert utility["utility_hat"] is None
    assert utility["all_rollouts_valid"] is False
    assert utility["num_valid_direct_rollouts"] == 1
    assert utility["reflect_mean_reward"] == 1.0
    assert utility["invalid_rollouts"] == [
        {"action": "direct", "rollout_index": 0, "reason": "generation_incomplete"}
    ]


def test_no_answer_rollout_is_scored_rather_than_dropped() -> None:
    reflect = [rollout(0, False, answer_extraction_success=False), rollout(1, True)]

    utility = estimate_state_utility(graded([True, True]), reflect, num_rollouts=2)

    assert utility["reflect_rewards"] == [0.0, 1.0]
    assert utility["reflect_mean_reward"] == 0.5
    assert utility["utility_hat"] == -0.5
    assert utility["all_rollouts_valid"] is True
    assert utility["invalid_rollouts"] == []


def test_incomplete_reflect_rollout_still_invalidates_the_estimate() -> None:
    reflect = [rollout(0, None, generation_complete=False), rollout(1, True)]

    utility = estimate_state_utility(graded([True, True]), reflect, num_rollouts=2)

    assert utility["reflect_rewards"] == [None, 1.0]
    assert utility["reflect_mean_reward"] is None
    assert utility["utility_hat"] is None
    assert utility["num_valid_reflect_rollouts"] == 1
    assert utility["invalid_rollouts"] == [
        {"action": "reflect", "rollout_index": 0, "reason": "generation_incomplete"}
    ]


def test_missing_reference_answer_invalidates_every_reward() -> None:
    utility = estimate_state_utility(
        graded([True, True], reference_extraction_success=False),
        graded([True, True], reference_extraction_success=False),
        num_rollouts=2,
    )

    assert utility["direct_rewards"] == [None, None]
    assert utility["reflect_rewards"] == [None, None]
    assert utility["direct_mean_reward"] is None
    assert utility["reflect_mean_reward"] is None
    assert utility["utility_hat"] is None
    assert utility["all_rollouts_valid"] is False
    assert len(utility["invalid_rollouts"]) == 4
    assert {invalid["reason"] for invalid in utility["invalid_rollouts"]} == {
        "reference_failure"
    }


def test_missing_planned_rollout_invalidates_the_estimate() -> None:
    utility = estimate_state_utility(graded([True]), graded([True, True]), num_rollouts=2)

    assert utility["direct_rewards"] == [1.0, None]
    assert utility["direct_mean_reward"] is None
    assert utility["utility_hat"] is None
    assert utility["all_rollouts_valid"] is False
    assert utility["invalid_rollouts"] == [
        {"action": "direct", "rollout_index": 1, "reason": "rollout_missing"}
    ]


def test_estimate_rejects_invalid_rollout_counts() -> None:
    with pytest.raises(ValueError, match="num_rollouts"):
        estimate_state_utility([], [], num_rollouts=0)
    with pytest.raises(ValueError, match="more rollouts than"):
        estimate_state_utility(graded([True, True]), graded([True]), num_rollouts=1)


# --- per-checkpoint utility over one problem --------------------------------

CHECKPOINTS = [
    {
        "selection_reason": "early",
        "step_index": 0,
        "boundary_token_index": 3,
        "relative_position": 0.25,
        "prefix_token_ids": [1, 2, 3],
        "prefix_text": "early prefix",
    },
    {
        "selection_reason": "middle",
        "step_index": 1,
        "boundary_token_index": 6,
        "relative_position": 0.5,
        "prefix_token_ids": [1, 2, 3, 4, 5, 6],
        "prefix_text": "middle prefix",
    },
    {
        "selection_reason": "late",
        "step_index": 2,
        "boundary_token_index": 8,
        "relative_position": 0.75,
        "prefix_token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "prefix_text": "late prefix",
    },
    {
        "selection_reason": "random",
        "step_index": 3,
        "boundary_token_index": 9,
        "relative_position": 0.9,
        "prefix_token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "prefix_text": "random prefix",
    },
]


def checkpoint_generator(calls: list[dict[str, Any]]):
    """Return a fake generator whose output identifies the state it ran from."""

    def generate(input_token_ids, seed):
        calls.append({"input_token_ids": list(input_token_ids), "seed": seed})
        return [len(input_token_ids), seed]

    return generate


def checkpoint_grader(correct_for):
    """Return a fake grader driven by ``correct_for(prefix_length, seed)``."""

    def grade(rollout):
        prefix_length, seed = rollout["generated_token_ids"]
        correct = correct_for(prefix_length, seed)
        # ``None`` stands for a continuation that never finished, the one case
        # that still leaves a checkpoint without a reward.
        graded_rollout = {
            **rollout,
            "generation_complete": correct is not None,
            "answer_extraction_success": correct is not None,
            "correct": correct,
        }
        return {**graded_rollout, **rollout_outcome_fields(graded_rollout, True)}

    return grade


def run_checkpoints(correct_for, calls: list[dict[str, Any]] | None = None):
    return estimate_checkpoint_utilities(
        CHECKPOINTS,
        [90, 91],
        42,
        4,
        checkpoint_generator(calls if calls is not None else []),
        checkpoint_grader(correct_for),
    )


def test_every_selected_checkpoint_is_processed_exactly_once() -> None:
    calls: list[dict[str, Any]] = []

    records = run_checkpoints(lambda prefix_length, seed: True, calls)

    assert [record["selection_reason"] for record in records] == [
        "early",
        "middle",
        "late",
        "random",
    ]
    assert [record["boundary_token_index"] for record in records] == [3, 6, 8, 9]
    assert len(calls) == len(CHECKPOINTS) * 2 * 4


def test_duplicate_checkpoints_are_rejected() -> None:
    with pytest.raises(ValueError, match="supplied twice"):
        estimate_checkpoint_utilities(
            [CHECKPOINTS[0], CHECKPOINTS[0]],
            [90, 91],
            42,
            4,
            checkpoint_generator([]),
            checkpoint_grader(lambda prefix_length, seed: True),
        )


def test_each_checkpoint_keeps_its_own_exact_prefix() -> None:
    calls: list[dict[str, Any]] = []

    records = run_checkpoints(lambda prefix_length, seed: True, calls)

    for record, checkpoint in zip(records, CHECKPOINTS, strict=True):
        prefix = checkpoint["prefix_token_ids"]
        assert record["shared_prefix_token_ids"] == prefix
        assert record["direct"]["input_token_ids"] == prefix
        assert record["reflect"]["input_token_ids"] == prefix + [90, 91]
        assert record["reflect"]["input_token_ids"][: len(prefix)] == prefix

    # Eight consecutive calls belong to one checkpoint: direct/reflect per rollout.
    for index, checkpoint in enumerate(CHECKPOINTS):
        prefix = checkpoint["prefix_token_ids"]
        checkpoint_calls = calls[index * 8 : (index + 1) * 8]
        assert all(
            call["input_token_ids"][: len(prefix)] == prefix for call in checkpoint_calls
        )


def test_rollout_seeds_are_paired_within_every_checkpoint() -> None:
    records = run_checkpoints(lambda prefix_length, seed: True)

    for record in records:
        direct_seeds = [rollout["seed"] for rollout in record["direct"]["rollouts"]]
        reflect_seeds = [rollout["seed"] for rollout in record["reflect"]["rollouts"]]
        assert direct_seeds == reflect_seeds == [42, 43, 44, 45]


def test_utility_is_calculated_independently_per_checkpoint() -> None:
    # Reflection helps only at the "late" checkpoint, whose prefix is 8 tokens.
    def correct_for(prefix_length: int, seed: int) -> bool:
        is_reflect_branch = prefix_length in {len(c["prefix_token_ids"]) + 2 for c in CHECKPOINTS}
        return is_reflect_branch and prefix_length == 10

    records = {record["selection_reason"]: record for record in run_checkpoints(correct_for)}

    assert records["late"]["utility"]["direct_mean_reward"] == 0.0
    assert records["late"]["utility"]["reflect_mean_reward"] == 1.0
    assert records["late"]["utility"]["utility_hat"] == 1.0
    for selection_reason in ("early", "middle", "random"):
        assert records[selection_reason]["utility"]["utility_hat"] == 0.0


def test_one_invalid_checkpoint_does_not_corrupt_the_others() -> None:
    # The middle checkpoint's prefix is 6 tokens; one of its rollouts is ungradable.
    def correct_for(prefix_length: int, seed: int) -> bool | None:
        if prefix_length == 6 and seed == 44:
            return None
        return True

    records = {record["selection_reason"]: record for record in run_checkpoints(correct_for)}

    middle = records["middle"]["utility"]
    assert middle["utility_hat"] is None
    assert middle["direct_mean_reward"] is None
    assert middle["all_rollouts_valid"] is False
    assert middle["invalid_rollouts"] == [
        {"action": "direct", "rollout_index": 2, "reason": "generation_incomplete"}
    ]

    for selection_reason in ("early", "late", "random"):
        utility = records[selection_reason]["utility"]
        assert utility["utility_hat"] == 0.0
        assert utility["all_rollouts_valid"] is True
        assert utility["invalid_rollouts"] == []


# --- run summary ------------------------------------------------------------


def problem_record(
    direct: list[dict[str, Any]],
    reflect: list[dict[str, Any]],
) -> dict[str, Any]:
    state = {"direct": {"rollouts": direct}, "reflect": {"rollouts": reflect}}
    state["utility"] = estimate_state_utility(direct, reflect, num_rollouts=len(direct))
    return {"checkpoint_utilities": [state], "problem_completed": True}


def test_run_summary_counts_problems_states_and_outcomes() -> None:
    records = [
        problem_record(graded([True, False]), graded([True, True])),
        problem_record(
            [rollout(0, True), rollout(1, None, generation_complete=False)],
            [rollout(0, True), rollout(1, False, answer_extraction_success=False)],
        ),
    ]

    summary = run_summary(records)

    assert summary["problems_attempted"] == 2
    assert summary["problems_completed"] == 2
    assert summary["states_attempted"] == 2
    assert summary["states_with_valid_utility"] == 1
    assert summary["states_unavailable"] == 1
    assert summary["rollout_count"] == 8
    assert summary["correct"] == 5
    assert summary["incorrect"] == 1
    assert summary["generation_incomplete"] == 1
    assert summary["no_answer"] == 1
    assert summary["grader_failure"] == 0
    assert summary["reference_failure"] == 0


def test_run_summary_counts_a_problem_without_states_as_incomplete() -> None:
    summary = run_summary([{"checkpoint_utilities": [], "problem_completed": False}])

    assert summary["problems_attempted"] == 1
    assert summary["problems_completed"] == 0
    assert summary["states_attempted"] == 0
    assert summary["rollout_count"] == 0
    assert all(summary[outcome] == 0 for outcome in ROLLOUT_OUTCOMES)
