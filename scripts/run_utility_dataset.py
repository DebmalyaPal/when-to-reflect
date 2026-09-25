"""Build a state-level reflection-utility dataset over MATH problems.

One run covers the whole per-problem procedure in EXPERIMENT_SPEC section 10:
generate a base trajectory, segment it, select checkpoints, fork each state
into matched direct and reflection rollouts, grade them, and estimate U_hat per
state. Utility is never aggregated across states.
"""

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import transformers

from when_to_reflect.activations import prefix_fingerprint, state_id
from when_to_reflect.branching import greedy_generation_config, sampling_generation_config
from when_to_reflect.checkpoints import (
    MAX_ELIGIBLE_RELATIVE_POSITION,
    MIN_ELIGIBLE_RELATIVE_POSITION,
    eligible_candidates,
    select_checkpoints,
)
from when_to_reflect.config import load_config
from when_to_reflect.data import load_math_training_example
from when_to_reflect.device import select_device
from when_to_reflect.generation import (
    RESOLVED_GENERATION_FIELDS,
    effective_generation_config,
    generation_status,
    terminal_token_ids,
    verify_generation_settings,
)
from when_to_reflect.grading import (
    extract_generated_answer,
    extract_reference_answer,
    score_answer,
)
from when_to_reflect.model import load_causal_lm
from when_to_reflect.records import DATASET_RECORD_FIELDS, save_record
from when_to_reflect.reproducibility import set_random_seed
from when_to_reflect.segmentation import segment_reasoning_trace
from when_to_reflect.utility import (
    estimate_checkpoint_utilities,
    rollout_outcome_fields,
    run_summary,
)

PROMPT_VERSION = "math_stepwise_v1"
PROMPT_TEMPLATE = (
    "Solve the following math problem concisely, step by step. "
    "Show only the essential equations and end with 'Final answer:'.\n\n"
    "Problem:\n{problem}"
)


def format_number(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:+.3f}" if value < 0 else f"{value:.3f}"


def process_problem(
    problem_index: int,
    config: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    stop_token_ids: set[int],
    intervention_token_ids: list[int],
    chat_template_kwargs: dict[str, Any],
    base_trace_generation_config: dict[str, Any],
    rollout_generation_config: dict[str, Any],
) -> dict[str, Any]:
    """Run the full state-level procedure for one source problem."""
    rollout_config = config["rollout"]
    base_seed = config["base_seed"]
    example = load_math_training_example(
        problem_index,
        PROJECT_ROOT / config["dataset_path"],
    )

    prompt = PROMPT_TEMPLATE.format(problem=example["problem"])
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **chat_template_kwargs,
    ).to(device)
    set_random_seed(base_seed)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **base_trace_generation_config)

    prompt_token_ids = inputs["input_ids"][0].tolist()
    trace_token_ids = output_ids[0, len(prompt_token_ids) :].tolist()
    trace_status = generation_status(
        trace_token_ids,
        config["base_trace"]["max_new_tokens"],
        stop_token_ids,
    )

    # Segmentation and checkpoint selection read only the trace, never outcomes.
    candidate_states = segment_reasoning_trace(tokenizer, prompt_token_ids, trace_token_ids)
    eligible_states = eligible_candidates(candidate_states)
    checkpoints = select_checkpoints(candidate_states, base_seed)
    reference_answer = extract_reference_answer(example["solution"])

    def generate(input_token_ids: list[int], seed: int) -> list[int]:
        """Sample one continuation from a complete, already-tokenized branch input."""
        set_random_seed(seed)
        input_ids = torch.tensor([input_token_ids], device=device)
        with torch.inference_mode():
            branch_output_ids = model.generate(input_ids, **rollout_generation_config)
        return branch_output_ids[0, len(input_token_ids) :].tolist()

    def grade(rollout: dict[str, Any]) -> dict[str, Any]:
        """Score one already-generated rollout and attach its outcome view."""
        generated_token_ids = rollout["generated_token_ids"]
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        status = generation_status(
            generated_token_ids,
            rollout_config["max_new_tokens"],
            stop_token_ids,
        )
        extracted_answer = extract_generated_answer(generated_text)
        graded = {
            **rollout,
            "generated_text": generated_text,
            **status,
            "extracted_answer": extracted_answer,
            "answer_extraction_success": extracted_answer is not None,
            "correct": score_answer(
                reference_answer,
                extracted_answer,
                status["generation_complete"],
            ),
        }
        return {**graded, **rollout_outcome_fields(graded, reference_answer is not None)}

    checkpoint_utilities = estimate_checkpoint_utilities(
        checkpoints,
        intervention_token_ids,
        base_seed,
        rollout_config["num_rollouts"],
        generate,
        grade,
    )
    # A stable identity per state, so a separately stored artifact (activations)
    # can be linked back to exactly this state without duplicating it. Both
    # fields are functions of the state itself, never of any rollout outcome.
    checkpoint_utilities = [
        {
            "state_id": state_id(example["problem_id"], state["boundary_token_index"]),
            "shared_prefix_sha256": prefix_fingerprint(state["shared_prefix_token_ids"]),
            **state,
        }
        for state in checkpoint_utilities
    ]

    return {
        "problem_id": example["problem_id"],
        "problem_index": problem_index,
        "subject": example.get("subject"),
        "level": example.get("level"),
        "problem": example["problem"],
        "reference_solution": example["solution"],
        "reference_answer": reference_answer,
        "reference_extraction_success": reference_answer is not None,
        "base_trace": {
            "prompt": prompt,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": trace_token_ids,
            "generated_text": tokenizer.decode(trace_token_ids, skip_special_tokens=True),
            **trace_status,
        },
        "num_candidate_states": len(candidate_states),
        "num_eligible_candidates": len(eligible_states),
        "selected_checkpoints": checkpoints,
        "checkpoint_utilities": checkpoint_utilities,
        "problem_completed": bool(checkpoint_utilities),
    }


def git_revision() -> dict[str, Any]:
    """Record the checked-out commit and whether the tree had uncommitted edits."""

    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def run_provenance(
    config: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    chat_template_kwargs: dict[str, Any],
    base_trace_generation_config: dict[str, Any],
    rollout_generation_config: dict[str, Any],
) -> dict[str, Any]:
    """Record what produced this run: weights, tokenization, device, decoding, code.

    States are defined by the exact tokens a specific model produced on a
    specific backend, so a run is only comparable to another run that matches on
    all of this. The generation configs are the *resolved* ones: `generate`
    overlays our arguments onto the model's shipped `generation_config`, so
    settings we never pass still take effect.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "model_id": config["model_id"],
        "model_revision": getattr(model.config, "_commit_hash", None),
        "model_class": type(model).__name__,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "prompt_version": PROMPT_VERSION,
        "reflection_prompt_version": config["reflection_prompt_version"],
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "chat_template_sha256": (
                None
                if chat_template is None
                else hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
            ),
            "chat_template_kwargs": chat_template_kwargs,
        },
        "effective_base_trace_generation_config": effective_generation_config(
            model,
            base_trace_generation_config,
        ),
        "effective_rollout_generation_config": effective_generation_config(
            model,
            rollout_generation_config,
        ),
        "git": git_revision(),
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }


def main() -> None:
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT / "configs" / "math_dev_utility.yaml"
    )
    config = load_config(config_path)
    rollout_config = config["rollout"]
    if rollout_config["do_sample"] is not True:
        raise ValueError("Paired rollout seeds are only meaningful with do_sample: true")
    if config["num_problems"] < 1:
        raise ValueError("num_problems must be at least 1")

    # Model-compatibility knobs for the chat template (e.g. Qwen3 thinking
    # mode). Absent from a config, nothing is passed and behaviour is the
    # tokenizer default, exactly as before.
    chat_template_kwargs = dict(config.get("chat_template") or {})

    device = select_device()
    tokenizer, model = load_causal_lm(config["model_id"], device)
    stop_token_ids = terminal_token_ids(tokenizer, model)
    intervention_token_ids = tokenizer.encode(
        config["reflection_instruction"],
        add_special_tokens=False,
    )
    base_trace_config = config["base_trace"]
    base_trace_generation_config = greedy_generation_config(
        base_trace_config["max_new_tokens"],
        base_trace_config.get("repetition_penalty"),
    )
    rollout_generation_config = sampling_generation_config(
        rollout_config["max_new_tokens"],
        rollout_config["temperature"],
        rollout_config["top_p"],
        rollout_config.get("top_k"),
        rollout_config.get("repetition_penalty"),
    )
    # A decoding setting written in a config must be the one `generate` applies,
    # not a model default that happens to agree today.
    for label, block, generation_config in (
        ("base_trace", base_trace_config, base_trace_generation_config),
        ("rollout", rollout_config, rollout_generation_config),
    ):
        verify_generation_settings(
            effective_generation_config(model, generation_config),
            {field: block[field] for field in RESOLVED_GENERATION_FIELDS if field in block},
            label,
        )

    problem_records = []
    for offset in range(config["num_problems"]):
        problem_index = config["first_problem_index"] + offset
        record = process_problem(
            problem_index,
            config,
            tokenizer,
            model,
            device,
            stop_token_ids,
            intervention_token_ids,
            chat_template_kwargs,
            base_trace_generation_config,
            rollout_generation_config,
        )
        problem_records.append(record)
        print(
            f"problem {problem_index} ({record['problem_id']}) | "
            f"candidates={record['num_candidate_states']} "
            f"eligible={record['num_eligible_candidates']} "
            f"states={len(record['checkpoint_utilities'])}"
        )

    summary = run_summary(problem_records)
    result = {
        "dataset": config["dataset"],
        "dataset_path": config["dataset_path"],
        "prompt_version": PROMPT_VERSION,
        "reflection_prompt_version": config["reflection_prompt_version"],
        "reflection_instruction": config["reflection_instruction"],
        "checkpoint_policy": {
            "targets": {"early": 0.25, "middle": 0.5, "late": 0.75},
            "random_draw": "seeded choice from remaining eligible candidates",
            "min_relative_position": MIN_ELIGIBLE_RELATIVE_POSITION,
            "max_relative_position": MAX_ELIGIBLE_RELATIVE_POSITION,
        },
        "base_trace_generation_config": base_trace_generation_config,
        "rollout_generation_config": rollout_generation_config,
        "model": {
            "model_id": config["model_id"],
            "base_seed": config["base_seed"],
            "num_rollouts": rollout_config["num_rollouts"],
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
        },
        "provenance": run_provenance(
            config,
            tokenizer,
            model,
            device,
            chat_template_kwargs,
            base_trace_generation_config,
            rollout_generation_config,
        ),
        "problems": problem_records,
        "summary": summary,
    }
    output_path = PROJECT_ROOT / config["output_path"]
    save_record(result, output_path, DATASET_RECORD_FIELDS)

    print(f"\n{'problem':<20}{'checkpoint':<10}{'position':>9}{'direct':>9}{'reflect':>9}{'U_hat':>9}")
    for problem in problem_records:
        for state in problem["checkpoint_utilities"]:
            utility = state["utility"]
            print(
                f"{problem['problem_id']:<20}"
                f"{state['selection_reason']:<10}"
                f"{state['relative_position']:>9.3f}"
                f"{format_number(utility['direct_mean_reward']):>9}"
                f"{format_number(utility['reflect_mean_reward']):>9}"
                f"{format_number(utility['utility_hat']):>9}"
            )

    print("\nRun summary")
    for key, value in summary.items():
        print(f"  {key:<26} {value}")
    print(f"\nSaved result: {output_path}")


if __name__ == "__main__":
    main()
