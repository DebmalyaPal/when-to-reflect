from copy import deepcopy

from when_to_reflect.branching import build_branch_inputs
from when_to_reflect.grading import (
    answers_match,
    extract_final_boxed_answer,
    extract_generated_answer,
    extract_reference_answer,
    normalize_answer,
    score_answer,
)


def test_extracts_simple_and_final_boxed_answers() -> None:
    assert extract_final_boxed_answer(r"Answer: \boxed{42}") == "42"
    assert extract_final_boxed_answer(r"\boxed{1}, then \boxed{2}") == "2"


def test_extracts_nested_braces_in_boxed_answer() -> None:
    assert extract_reference_answer(r"Result: \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extracts_explicit_final_answer_or_reports_failure() -> None:
    assert extract_generated_answer("Work complete. Final answer: $42$") == "$42$"
    assert extract_generated_answer("The result appears above.") is None


def test_normalized_comparison_handles_harmless_formatting() -> None:
    assert answers_match("42", r"\( 42 \)")
    assert answers_match(r"\left( \frac{1}{2} \right)", r"\frac{1}{2}") is False
    assert normalize_answer(r" $ \dfrac{ 1 }{ 2 } $ ") == r"\frac{1}{2}"
    assert not answers_match("42", "41")


def test_post_generation_grading_does_not_mutate_branch_inputs_or_checkpoint() -> None:
    checkpoint = {"prefix_token_ids": [1, 2, 3]}
    branches = build_branch_inputs(checkpoint["prefix_token_ids"], [4, 5])
    original_checkpoint = deepcopy(checkpoint)
    original_branches = deepcopy(branches)

    reference_answer = extract_reference_answer(r"\boxed{42}")
    generated_answer = extract_generated_answer("Final answer: 42")

    assert answers_match(reference_answer, generated_answer)
    assert checkpoint == original_checkpoint
    assert branches == original_branches


def test_incomplete_generation_is_not_scored_incorrect() -> None:
    assert score_answer("42", "41", generation_complete=False) is None
