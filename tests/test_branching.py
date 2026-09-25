from when_to_reflect.branching import (
    build_branch_inputs,
    greedy_generation_config,
    preserves_shared_prefix,
    sampling_generation_config,
)


def test_branch_inputs_share_the_exact_checkpoint_prefix_without_mutation() -> None:
    checkpoint = {"prefix_token_ids": [10, 11, 12]}
    original_prefix = list(checkpoint["prefix_token_ids"])

    branches = build_branch_inputs(checkpoint["prefix_token_ids"], [20, 21])

    assert checkpoint["prefix_token_ids"] == original_prefix
    assert branches["shared_prefix_token_ids"] == checkpoint["prefix_token_ids"]
    assert branches["direct"]["input_token_ids"] == checkpoint["prefix_token_ids"]
    assert branches["reflect"]["input_token_ids"][: len(checkpoint["prefix_token_ids"])] == checkpoint[
        "prefix_token_ids"
    ]


def test_direct_has_no_intervention_and_reflection_appends_only_after_shared_prefix() -> None:
    branches = build_branch_inputs([1, 2], [3, 4])

    assert branches["direct"]["intervention_token_ids"] == []
    assert branches["direct"]["input_token_ids"] == [1, 2]
    assert branches["reflect"]["intervention_token_ids"] == [3, 4]
    assert branches["reflect"]["input_token_ids"] == [1, 2, 3, 4]


def test_greedy_generation_configuration_is_deterministic() -> None:
    assert greedy_generation_config(128) == {"do_sample": False, "max_new_tokens": 128}


def test_shared_prefix_preservation_check_accepts_both_branch_inputs() -> None:
    branches = build_branch_inputs([1, 2], [3, 4])
    shared_prefix = branches["shared_prefix_token_ids"]

    assert preserves_shared_prefix(branches["direct"]["input_token_ids"], shared_prefix)
    assert preserves_shared_prefix(branches["reflect"]["input_token_ids"], shared_prefix)
    assert not preserves_shared_prefix([9, 2, 3, 4], shared_prefix)
    assert not preserves_shared_prefix([1], shared_prefix)


def test_sampling_generation_configuration_is_shared_by_both_branches() -> None:
    assert sampling_generation_config(512, 0.7, 0.95) == {
        "do_sample": True,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
    }


def test_generation_configurations_omit_unstated_settings() -> None:
    # Nothing extra is passed, so the model's own defaults still decide these.
    assert "top_k" not in sampling_generation_config(512, 0.7, 0.95)
    assert "repetition_penalty" not in sampling_generation_config(512, 0.7, 0.95)
    assert "repetition_penalty" not in greedy_generation_config(128)


def test_generation_configurations_pin_stated_settings() -> None:
    assert sampling_generation_config(512, 0.7, 0.95, 20, 1.0) == {
        "do_sample": True,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
    }
    # repetition_penalty applies under greedy decoding too; the warpers do not.
    assert greedy_generation_config(128, 1.1) == {
        "do_sample": False,
        "max_new_tokens": 128,
        "repetition_penalty": 1.1,
    }
