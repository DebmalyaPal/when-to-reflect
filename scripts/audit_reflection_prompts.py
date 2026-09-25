"""Development-time audit of reflection-intervention wording and format.

This is prompt tooling, not a research experiment. It replays the reasoning
states the main runner already produced, so the intervention is the only thing
that varies: base traces are not regenerated and no direct branch is run.

Two intervention formats are supported:

``assistant_text``
    Plain text appended inside the running assistant turn.
``user_turn``
    The assistant turn is closed, a user instruction is inserted, and a new
    assistant turn is opened. The role-transition tokens are derived from the
    tokenizer's own chat template, never hand-written, and the derivation is
    checked for context-independence before use.

In both cases the pre-intervention shared prefix is untouched; the intervention
tokens are part of the treatment and occur only after it.

Fidelity labels are assigned by human inspection of the saved continuations,
not by this script.
"""

import json
import sys
from collections.abc import Iterator, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from when_to_reflect.branching import build_branch_inputs, sampling_generation_config
from when_to_reflect.config import load_config
from when_to_reflect.device import select_device
from when_to_reflect.generation import generation_status, terminal_token_ids
from when_to_reflect.grading import extract_generated_answer, score_answer
from when_to_reflect.model import load_causal_lm
from when_to_reflect.records import save_record
from when_to_reflect.reproducibility import set_random_seed

PROBE_USER = "X"
PROBE_ASSISTANT = "Y"


def assistant_text_intervention(tokenizer: Any, instruction: str) -> list[int]:
    """Tokenize an intervention that continues the current assistant turn."""
    return tokenizer.encode(instruction, add_special_tokens=False)


def user_turn_intervention(tokenizer: Any, instruction: str) -> list[int]:
    """Derive assistant-end + user-turn + assistant-start tokens from the template.

    The sequence is taken from the tokenizer's own chat template by rendering a
    probe conversation and removing its head, rather than by writing control
    tokens by hand. Two checks guard the result: the extended rendering must
    extend the assistant turn in place, and the derived tail must tokenize
    identically standalone and in context.
    """
    messages = [
        {"role": "user", "content": PROBE_USER},
        {"role": "assistant", "content": PROBE_ASSISTANT},
        {"role": "user", "content": instruction},
    ]
    base_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROBE_USER}],
        add_generation_prompt=True,
        tokenize=False,
    )
    extended_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    head_text = base_text + PROBE_ASSISTANT
    if not extended_text.startswith(head_text):
        raise ValueError("Chat template does not extend the assistant turn in place")

    tail_token_ids = tokenizer.encode(extended_text[len(head_text) :], add_special_tokens=False)
    extended_token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    if extended_token_ids[-len(tail_token_ids) :] != tail_token_ids:
        raise ValueError("Role-transition tokens are not context-independent")
    return tail_token_ids


INTERVENTION_BUILDERS = {
    "assistant_text": assistant_text_intervention,
    "user_turn": user_turn_intervention,
}


def stored_states(
    dataset_record: dict[str, Any],
    selection_reasons: Sequence[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield each stored state with the problem context needed to grade it.

    ``selection_reasons`` restricts the replay to checkpoints chosen for those
    reasons, which is how a run is kept to a couple of states per problem. The
    filter reads only how a checkpoint was selected, never any rollout outcome,
    reward, or correctness.
    """
    for problem in dataset_record["problems"]:
        for state in problem["checkpoint_utilities"]:
            if selection_reasons is not None and state["selection_reason"] not in selection_reasons:
                continue
            yield {
                "problem_id": problem["problem_id"],
                "reference_answer": problem["reference_answer"],
                "reference_extraction_success": problem["reference_extraction_success"],
                "state": state,
            }


def main() -> None:
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT / "configs" / "reflection_prompt_audit.yaml"
    )
    config = load_config(config_path)
    generation = config["generation"]
    if generation["do_sample"] is not True:
        raise ValueError("The audit compares sampled continuations; set do_sample: true")

    dataset_record = json.loads(
        (PROJECT_ROOT / config["input_path"]).read_text(encoding="utf-8")
    )
    if dataset_record["model"]["model_id"] != config["model_id"]:
        raise ValueError("Configured model_id does not match the stored dataset")

    # Snapshot the stored states so mutation can be ruled out at the end.
    stored_snapshot = deepcopy(dataset_record["problems"])

    seeds = config["seeds"]
    # Absent from a config, every stored checkpoint is replayed, as before.
    selection_reasons = config.get("selection_reasons")
    device = select_device()
    tokenizer, model = load_causal_lm(config["model_id"], device)
    stop_token_ids = terminal_token_ids(tokenizer, model)
    generation_config = sampling_generation_config(
        generation["max_new_tokens"],
        generation["temperature"],
        generation["top_p"],
    )
    conditions = []
    for condition in config["conditions"]:
        intervention_format = condition.get("format", "assistant_text")
        if intervention_format not in INTERVENTION_BUILDERS:
            raise ValueError(f"Unknown intervention format: {intervention_format}")
        conditions.append(
            {
                "name": condition["name"],
                "format": intervention_format,
                "instruction": condition["instruction"],
                "intervention_token_ids": INTERVENTION_BUILDERS[intervention_format](
                    tokenizer, condition["instruction"]
                ),
            }
        )

    continuations = []
    for entry in stored_states(dataset_record, selection_reasons):
        state = entry["state"]
        shared_prefix_token_ids = state["shared_prefix_token_ids"]
        prefix_length = len(shared_prefix_token_ids)
        for condition in conditions:
            intervention_token_ids = condition["intervention_token_ids"]
            branches = build_branch_inputs(shared_prefix_token_ids, intervention_token_ids)
            input_token_ids = branches["reflect"]["input_token_ids"]
            invariants = {
                "shared_prefix_preserved": (
                    input_token_ids[:prefix_length] == shared_prefix_token_ids
                ),
                "intervention_after_shared_prefix": (
                    input_token_ids[prefix_length:] == intervention_token_ids
                ),
                "input_length_matches": (
                    len(input_token_ids) == prefix_length + len(intervention_token_ids)
                ),
            }
            for seed in seeds:
                set_random_seed(seed)
                input_ids = torch.tensor([input_token_ids], device=device)
                with torch.inference_mode():
                    output_ids = model.generate(input_ids, **generation_config)
                generated_token_ids = output_ids[0, len(input_token_ids) :].tolist()

                generated_text = tokenizer.decode(
                    generated_token_ids, skip_special_tokens=True
                )
                status = generation_status(
                    generated_token_ids, generation["max_new_tokens"], stop_token_ids
                )
                extracted_answer = extract_generated_answer(generated_text)
                continuations.append(
                    {
                        "problem_id": entry["problem_id"],
                        "selection_reason": state["selection_reason"],
                        "relative_position": state["relative_position"],
                        "boundary_token_index": state["boundary_token_index"],
                        "condition": condition["name"],
                        "format": condition["format"],
                        "instruction": condition["instruction"],
                        "seed": seed,
                        **invariants,
                        "shared_prefix_token_count": prefix_length,
                        "intervention_token_count": len(intervention_token_ids),
                        "generated_token_ids": generated_token_ids,
                        "generated_text": generated_text,
                        **status,
                        "extracted_answer": extracted_answer,
                        "answer_extraction_success": extracted_answer is not None,
                        "correct": score_answer(
                            entry["reference_answer"],
                            extracted_answer,
                            status["generation_complete"],
                        ),
                        # Assigned by human inspection after this script runs.
                        "fidelity_label": None,
                        "instruction_echo": None,
                    }
                )
        print(
            f"{entry['problem_id']} | {state['selection_reason']:<7} | "
            f"pos={state['relative_position']:.3f} | done"
        )

    stored_states_unmutated = dataset_record["problems"] == stored_snapshot
    result = {
        "audit": config.get("audit", "reflection_intervention"),
        "input_path": config["input_path"],
        "model": dataset_record["model"],
        "seeds": seeds,
        "selection_reasons": selection_reasons,
        "generation_config": generation_config,
        "conditions": [
            {
                "name": condition["name"],
                "format": condition["format"],
                "instruction": condition["instruction"],
                "intervention_token_count": len(condition["intervention_token_ids"]),
                "intervention_token_ids": condition["intervention_token_ids"],
            }
            for condition in conditions
        ],
        "stored_states_unmutated": stored_states_unmutated,
        "continuations": continuations,
    }
    output_path = PROJECT_ROOT / config["output_path"]
    save_record(result, output_path)

    total = len(continuations)
    print(f"\nStates: {total // (len(conditions) * len(seeds))}  Generations: {total}")
    for name in ("shared_prefix_preserved", "intervention_after_shared_prefix",
                 "input_length_matches"):
        print(f"  {name}: {sum(c[name] for c in continuations)}/{total}")
    print(f"  stored_states_unmutated: {stored_states_unmutated}")

    print(f"\n{'condition':<28}{'fmt':<16}{'mean_tok':>9}{'empty':>8}{'answered':>10}{'correct':>9}")
    for condition in conditions:
        rows = [c for c in continuations if c["condition"] == condition["name"]]
        empty = sum(1 for c in rows if c["generated_token_count"] <= 1)
        print(
            f"{condition['name']:<28}{condition['format']:<16}"
            f"{sum(c['generated_token_count'] for c in rows) / len(rows):>9.0f}"
            f"{empty:>5}/{len(rows):<2}"
            f"{sum(c['answer_extraction_success'] for c in rows):>7}/{len(rows):<2}"
            f"{sum(1 for c in rows if c['correct']):>6}/{len(rows):<2}"
        )
    print(f"\nSaved result: {output_path}")


if __name__ == "__main__":
    main()
