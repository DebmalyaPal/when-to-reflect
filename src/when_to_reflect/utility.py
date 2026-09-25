"""Paired repeated rollouts and state-level reflection-utility estimation.

Implements EXPERIMENT_SPEC section 7 (repeated futures) and section 9:

    U_hat(s) = mean(reflect rewards) - mean(continue rewards)

where the reward is binary final-answer correctness.

Direct and reflection rollouts at a given index share a seed, so the two
actions are compared under common random numbers while their branch inputs stay
exactly as built.

Reward semantics are decided here rather than in ``grading``, whose
``score_answer`` behaviour stays frozen. Every rollout is classified into an
explicit outcome (see ``ROLLOUT_OUTCOMES``) and the outcome determines the
reward. A completed continuation counts as an observation of the model's
behaviour even when it produced no answer; an unfinished or ungradable one
yields no reward at all and is recorded as missing rather than as reward 0. If
any planned rollout has no reward, the affected branch mean and ``utility_hat``
are left unavailable rather than averaged over only the successful rollouts.

Known ambiguity: with the current surface-form grader, ``no_answer`` and
``grader_failure`` are not separable. ``grading.extract_generated_answer``
returns ``None`` both when the model genuinely emitted no answer and when it
emitted one this parser cannot recognise, and ``grading.score_answer`` never
raises. Every such rollout is therefore classified ``no_answer``, and
``grader_failure`` is presently unreachable. It is defined here so that a
future symbolic or verifier-based grader, which can fail on an answer that is
demonstrably present, has a distinct label that maps to "no reward" instead of
being silently scored 0.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from when_to_reflect.branching import build_branch_inputs, preserves_shared_prefix

BRANCH_ACTIONS = ("direct", "reflect")

CHECKPOINT_FIELDS = (
    "selection_reason",
    "step_index",
    "boundary_token_index",
    "relative_position",
)


def rollout_seed(base_seed: int, rollout_index: int) -> int:
    """Return the seed shared by both branch actions at ``rollout_index``."""
    if rollout_index < 0:
        raise ValueError("rollout_index must be non-negative")
    return base_seed + rollout_index


def paired_rollout_seeds(base_seed: int, num_rollouts: int) -> list[int]:
    """Return one seed per rollout index, shared by both branch actions."""
    if num_rollouts < 1:
        raise ValueError("num_rollouts must be at least 1")
    return [rollout_seed(base_seed, index) for index in range(num_rollouts)]


def generate_paired_rollouts(
    branches: Mapping[str, Any],
    base_seed: int,
    num_rollouts: int,
    generate: Callable[[Sequence[int], int], list[int]],
) -> dict[str, list[dict[str, Any]]]:
    """Generate matched-seed direct and reflection rollouts from one shared prefix.

    ``generate`` receives the complete branch input token IDs and the rollout
    seed, and returns only the newly generated continuation token IDs. It is
    responsible for resetting the random seed immediately before generating.
    Branch inputs are never rebuilt between rollouts, and each one is rechecked
    against the stored shared prefix before it is used.
    """
    shared_prefix_token_ids = branches["shared_prefix_token_ids"]
    rollouts: dict[str, list[dict[str, Any]]] = {action: [] for action in BRANCH_ACTIONS}

    for rollout_index, seed in enumerate(paired_rollout_seeds(base_seed, num_rollouts)):
        for action in BRANCH_ACTIONS:
            input_token_ids = branches[action]["input_token_ids"]
            if not preserves_shared_prefix(input_token_ids, shared_prefix_token_ids):
                raise ValueError(
                    f"{action} rollout {rollout_index} does not begin with the shared prefix"
                )
            rollouts[action].append(
                {
                    "rollout_index": rollout_index,
                    "seed": seed,
                    "shared_prefix_preserved": True,
                    "generated_token_ids": generate(input_token_ids, seed),
                }
            )

    return rollouts


ROLLOUT_OUTCOMES = (
    "correct",
    "incorrect",
    "no_answer",
    "generation_incomplete",
    "grader_failure",
    "reference_failure",
)

REWARD_BY_OUTCOME: dict[str, float | None] = {
    "correct": 1.0,
    "incorrect": 0.0,
    "no_answer": 0.0,
    "generation_incomplete": None,
    "grader_failure": None,
    "reference_failure": None,
}


def classify_rollout_outcome(
    rollout: Mapping[str, Any],
    reference_extraction_success: bool,
) -> str:
    """Classify one graded rollout into the reward outcome taxonomy.

    Order matters. A state whose reference answer could not be extracted is
    ungradable no matter what the rollout produced, and an unfinished
    continuation is ungradable no matter what text it happens to contain so
    far. ``no_answer`` then covers every completed continuation from which no
    answer could be extracted, an empty EOS-only continuation included: such a
    rollout is a real observation of the model failing to answer, not a missing
    measurement, so it is scored rather than dropped.
    """
    if not reference_extraction_success:
        return "reference_failure"
    if not rollout["generation_complete"]:
        return "generation_incomplete"
    if not rollout["answer_extraction_success"]:
        return "no_answer"
    if rollout["correct"] is None:
        return "grader_failure"
    return "correct" if rollout["correct"] else "incorrect"


def rollout_outcome_fields(
    rollout: Mapping[str, Any],
    reference_extraction_success: bool,
) -> dict[str, Any]:
    """Return the outcome and reward view of one graded rollout.

    The raw ``extracted_answer``, ``answer_extraction_success`` and ``correct``
    fields are left exactly as produced, so this view never hides what the
    grader actually saw.
    """
    outcome = classify_rollout_outcome(rollout, reference_extraction_success)
    reward = REWARD_BY_OUTCOME[outcome]
    return {
        "outcome": outcome,
        "reward_available": reward is not None,
        "reward": reward,
    }


def _branch_rewards(
    action: str,
    rollouts: Sequence[Mapping[str, Any]],
    num_rollouts: int,
) -> tuple[list[float | None], list[dict[str, Any]]]:
    if len(rollouts) > num_rollouts:
        raise ValueError(f"{action} produced more rollouts than the {num_rollouts} planned")

    rewards: list[float | None] = []
    invalid_rollouts: list[dict[str, Any]] = []
    for rollout_index in range(num_rollouts):
        if rollout_index >= len(rollouts):
            rewards.append(None)
            invalid_rollouts.append(
                {"action": action, "rollout_index": rollout_index, "reason": "rollout_missing"}
            )
            continue

        rollout = rollouts[rollout_index]
        if rollout["reward_available"]:
            rewards.append(rollout["reward"])
        else:
            rewards.append(None)
            invalid_rollouts.append(
                {
                    "action": action,
                    "rollout_index": rollout["rollout_index"],
                    "reason": rollout["outcome"],
                }
            )

    return rewards, invalid_rollouts


def estimate_state_utility(
    direct_rollouts: Sequence[Mapping[str, Any]],
    reflect_rollouts: Sequence[Mapping[str, Any]],
    num_rollouts: int,
) -> dict[str, Any]:
    """Estimate U_hat(s) from matched direct and reflection rollouts."""
    if num_rollouts < 1:
        raise ValueError("num_rollouts must be at least 1")

    direct_rewards, invalid_direct = _branch_rewards("direct", direct_rollouts, num_rollouts)
    reflect_rewards, invalid_reflect = _branch_rewards("reflect", reflect_rollouts, num_rollouts)
    invalid_rollouts = invalid_direct + invalid_reflect

    direct_mean_reward = None if invalid_direct else sum(direct_rewards) / num_rollouts
    reflect_mean_reward = None if invalid_reflect else sum(reflect_rewards) / num_rollouts
    all_rollouts_valid = not invalid_rollouts
    utility_hat = reflect_mean_reward - direct_mean_reward if all_rollouts_valid else None

    return {
        "num_rollouts": num_rollouts,
        "direct_rewards": direct_rewards,
        "reflect_rewards": reflect_rewards,
        "direct_mean_reward": direct_mean_reward,
        "reflect_mean_reward": reflect_mean_reward,
        "utility_hat": utility_hat,
        "num_valid_direct_rollouts": sum(1 for reward in direct_rewards if reward is not None),
        "num_valid_reflect_rollouts": sum(1 for reward in reflect_rewards if reward is not None),
        "all_rollouts_valid": all_rollouts_valid,
        "invalid_rollouts": invalid_rollouts,
    }


def checkpoint_utility_record(
    checkpoint: Mapping[str, Any],
    branches: Mapping[str, Any],
    base_seed: int,
    num_rollouts: int,
    generate: Callable[[Sequence[int], int], list[int]],
    grade: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Generate, grade, and score every rollout for one checkpoint.

    ``U_hat`` stays a state-level quantity: this record describes exactly one
    checkpoint and is never combined with another checkpoint's rollouts.
    """
    if list(branches["shared_prefix_token_ids"]) != list(checkpoint["prefix_token_ids"]):
        raise ValueError(
            f"branches for checkpoint {checkpoint['selection_reason']} do not match its prefix"
        )

    rollouts = generate_paired_rollouts(branches, base_seed, num_rollouts, generate)
    graded_rollouts = {
        action: [grade(rollout) for rollout in rollouts[action]] for action in BRANCH_ACTIONS
    }
    utility = estimate_state_utility(
        graded_rollouts["direct"],
        graded_rollouts["reflect"],
        num_rollouts,
    )

    return {
        **{field: checkpoint[field] for field in CHECKPOINT_FIELDS},
        "shared_prefix_token_ids": branches["shared_prefix_token_ids"],
        "shared_prefix_text": checkpoint["prefix_text"],
        "direct": {
            "intervention_token_ids": branches["direct"]["intervention_token_ids"],
            "input_token_ids": branches["direct"]["input_token_ids"],
            "rollouts": graded_rollouts["direct"],
        },
        "reflect": {
            "intervention_token_ids": branches["reflect"]["intervention_token_ids"],
            "input_token_ids": branches["reflect"]["input_token_ids"],
            "rollouts": graded_rollouts["reflect"],
        },
        "utility": utility,
    }


def estimate_checkpoint_utilities(
    checkpoints: Sequence[Mapping[str, Any]],
    intervention_token_ids: Sequence[int],
    base_seed: int,
    num_rollouts: int,
    generate: Callable[[Sequence[int], int], list[int]],
    grade: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run the paired-rollout utility procedure once per selected checkpoint.

    Every checkpoint's branch inputs are built from its own stored prefix token
    IDs before any generation starts, so no rollout outcome can influence how a
    later checkpoint's state is constructed. Each checkpoint is then scored
    independently; an invalid rollout only nulls that checkpoint's utility.
    """
    planned: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    seen_boundaries: set[int] = set()
    for checkpoint in checkpoints:
        boundary_token_index = checkpoint["boundary_token_index"]
        if boundary_token_index in seen_boundaries:
            raise ValueError(f"checkpoint at boundary {boundary_token_index} was supplied twice")
        seen_boundaries.add(boundary_token_index)
        planned.append(
            (checkpoint, build_branch_inputs(checkpoint["prefix_token_ids"], intervention_token_ids))
        )

    return [
        checkpoint_utility_record(checkpoint, branches, base_seed, num_rollouts, generate, grade)
        for checkpoint, branches in planned
    ]


def run_summary(problem_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count problems, states, and rollout outcomes across a dataset run.

    Utility stays a state-level quantity: states are counted, never averaged
    into a problem-level or run-level utility.
    """
    states = [
        state for problem in problem_records for state in problem["checkpoint_utilities"]
    ]
    outcomes = [
        rollout["outcome"]
        for state in states
        for action in BRANCH_ACTIONS
        for rollout in state[action]["rollouts"]
    ]
    outcome_counts = dict.fromkeys(ROLLOUT_OUTCOMES, 0)
    for outcome in outcomes:
        outcome_counts[outcome] += 1

    states_with_valid_utility = sum(
        1 for state in states if state["utility"]["utility_hat"] is not None
    )
    return {
        "problems_attempted": len(problem_records),
        "problems_completed": sum(
            1 for problem in problem_records if problem["problem_completed"]
        ),
        "states_attempted": len(states),
        "states_with_valid_utility": states_with_valid_utility,
        "states_unavailable": len(states) - states_with_valid_utility,
        "rollout_count": len(outcomes),
        **outcome_counts,
    }
