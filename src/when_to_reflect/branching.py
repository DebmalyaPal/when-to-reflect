"""Token-level construction of direct and reflection branch inputs."""

from collections.abc import Sequence


def build_branch_inputs(
    shared_prefix_token_ids: Sequence[int],
    reflection_intervention_token_ids: Sequence[int],
) -> dict[str, dict[str, list[int]] | list[int]]:
    """Build branch inputs without decoding or re-tokenizing the shared prefix."""
    shared_prefix = list(shared_prefix_token_ids)
    intervention = list(reflection_intervention_token_ids)
    direct_input = list(shared_prefix)
    reflect_input = shared_prefix + intervention

    assert direct_input == shared_prefix
    assert reflect_input[: len(shared_prefix)] == shared_prefix

    return {
        "shared_prefix_token_ids": shared_prefix,
        "direct": {
            "intervention_token_ids": [],
            "input_token_ids": direct_input,
        },
        "reflect": {
            "intervention_token_ids": intervention,
            "input_token_ids": reflect_input,
        },
    }


def preserves_shared_prefix(
    input_token_ids: Sequence[int],
    shared_prefix_token_ids: Sequence[int],
) -> bool:
    """Return whether branch input token IDs still begin with the stored state."""
    shared_prefix = list(shared_prefix_token_ids)
    return list(input_token_ids[: len(shared_prefix)]) == shared_prefix


def greedy_generation_config(
    max_new_tokens: int,
    repetition_penalty: float | None = None,
) -> dict[str, int | float | bool]:
    """Return the deterministic generation settings for the base trajectory.

    ``repetition_penalty`` changes greedy decoding as much as it changes
    sampling, so a config that states it has it pinned here rather than
    inherited from the model's own ``generation_config``. The sampling warpers
    (``temperature``, ``top_p``, ``top_k``) are inert without ``do_sample`` and
    are deliberately not passed. Omitting a value leaves the model default in
    place, which is what configurations written before this option did.
    """
    config: dict[str, int | float | bool] = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
    }
    if repetition_penalty is not None:
        config["repetition_penalty"] = repetition_penalty
    return config


def sampling_generation_config(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, int | float | bool]:
    """Return the stochastic decoding settings shared by every branch rollout.

    Repeated-future estimation needs sampling, and EXPERIMENT_SPEC section 6
    requires identical decoding parameters across actions, so one config object
    is built once and reused by both branches.

    ``top_k`` and ``repetition_penalty`` are optional only so that
    configurations written before they were exposed keep behaving exactly as
    they did. Anything left out here is filled in from the model's shipped
    ``generation_config``, which differs between models, so a configuration
    that cares about a setting should state it.
    """
    config: dict[str, int | float | bool] = {
        "do_sample": True,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if top_k is not None:
        config["top_k"] = top_k
    if repetition_penalty is not None:
        config["repetition_penalty"] = repetition_penalty
    return config
