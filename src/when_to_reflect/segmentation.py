"""Token-preserving segmentation of generated reasoning traces."""

import re
from collections.abc import Sequence
from typing import Any


def segment_reasoning_trace(
    tokenizer: Any,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Return paragraph-delimited candidate states from an assistant trace.

    ``boundary_token_index`` is the one-based count of generated tokens included
    at the boundary. Each ``prefix_token_ids`` value contains the original chat
    template token IDs plus that exact generated-token prefix.
    """
    prompt_ids = list(prompt_token_ids)
    generated_ids = list(generated_token_ids)
    complete_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    states = []

    for boundary_index in range(1, len(generated_ids)):
        reasoning_prefix_ids = generated_ids[:boundary_index]
        reasoning_prefix_text = tokenizer.decode(reasoning_prefix_ids, skip_special_tokens=True)
        if (
            not reasoning_prefix_text.strip()
            or re.search(r"\n\s*\n$", reasoning_prefix_text) is None
            or reasoning_prefix_text == complete_generated_text
        ):
            continue

        prefix_token_ids = prompt_ids + reasoning_prefix_ids
        states.append(
            {
                "step_index": len(states),
                "boundary_token_index": boundary_index,
                "prefix_token_ids": prefix_token_ids,
                "prefix_text": tokenizer.decode(prefix_token_ids, skip_special_tokens=True),
                "relative_position": boundary_index / len(generated_ids),
            }
        )

    return states
