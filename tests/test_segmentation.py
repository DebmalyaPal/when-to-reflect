from typing import ClassVar

from when_to_reflect.segmentation import segment_reasoning_trace


class FakeTokenizer:
    token_text: ClassVar = {
        10: "Prompt: ",
        1: "First step.",
        2: "\n",
        3: "\n",
        4: "Second step.",
        5: "\n",
        6: "\n",
        7: "Final answer.",
        8: "<eos>",
    }

    def decode(self, token_ids, skip_special_tokens=True) -> str:
        return "".join(
            self.token_text[token_id]
            for token_id in token_ids
            if not (skip_special_tokens and token_id == 8)
        )


def test_segment_states_are_ordered_true_nonfinal_prefixes() -> None:
    prompt_ids = [10]
    generated_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    tokenizer = FakeTokenizer()

    states = segment_reasoning_trace(tokenizer, prompt_ids, generated_ids)

    assert [state["boundary_token_index"] for state in states] == [3, 6]
    assert [state["step_index"] for state in states] == [0, 1]
    full_token_ids = prompt_ids + generated_ids
    for state in states:
        prefix_token_ids = state["prefix_token_ids"]
        assert prefix_token_ids == full_token_ids[: len(prefix_token_ids)]
        assert prefix_token_ids != full_token_ids
        assert tokenizer.decode(prefix_token_ids) == state["prefix_text"]
        assert state["relative_position"] < 1


def test_segmentation_is_deterministic() -> None:
    tokenizer = FakeTokenizer()
    first = segment_reasoning_trace(tokenizer, [10], [1, 2, 3, 4, 5, 6, 7, 8])
    second = segment_reasoning_trace(tokenizer, [10], [1, 2, 3, 4, 5, 6, 7, 8])

    assert first == second


def test_segmentation_excludes_final_text_before_end_of_sequence_token() -> None:
    tokenizer = FakeTokenizer()

    states = segment_reasoning_trace(tokenizer, [10], [1, 2, 3, 8])

    assert states == []
