"""Capture pre-treatment hidden states for the states of a utility dataset.

For every selected checkpoint in a stored dataset record, the model is run on
exactly that state's ``shared_prefix_token_ids`` and the hidden state at the
final prefix token is kept for every layer. Nothing after the state is part of
the input: not the reflection instruction, not either continuation.

Every selected state is extracted, including states whose utility came out
unavailable. Restricting extraction to states with a valid ``utility_hat``
would let rollout outcomes decide which states get a representation, which is
exactly the kind of outcome leakage the state construction rules forbid.

No rollouts are generated or regenerated here, and the dataset record is only
read.
"""

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from when_to_reflect.activations import (
    capture_prefix_hidden_states,
    prefix_fingerprint,
    state_id,
    tensor_is_finite,
)
from when_to_reflect.config import load_config
from when_to_reflect.device import select_device
from when_to_reflect.model import load_causal_lm
from when_to_reflect.records import save_record

STORAGE_DTYPES = {"float16": torch.float16, "float32": torch.float32}


def stored_states(dataset_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every selected state with the identity needed to link it back."""
    states = []
    for problem in dataset_record["problems"]:
        for state in problem["checkpoint_utilities"]:
            # Records written before state IDs existed are identified by the
            # same rule, applied to the same two fields.
            identifier = state.get("state_id") or state_id(
                problem["problem_id"], state["boundary_token_index"]
            )
            states.append(
                {
                    "state_id": identifier,
                    "problem_id": problem["problem_id"],
                    "problem_index": problem["problem_index"],
                    "selection_reason": state["selection_reason"],
                    "step_index": state["step_index"],
                    "boundary_token_index": state["boundary_token_index"],
                    "relative_position": state["relative_position"],
                    "shared_prefix_token_ids": state["shared_prefix_token_ids"],
                    "shared_prefix_sha256": state.get("shared_prefix_sha256"),
                }
            )
    return states


def main() -> None:
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT / "configs" / "activations_qwen3_dev.yaml"
    )
    config = load_config(config_path)
    storage_dtype = STORAGE_DTYPES[config.get("storage_dtype", "float16")]

    dataset_path = PROJECT_ROOT / config["input_path"]
    dataset_record = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset_record["model"]["model_id"] != config["model_id"]:
        raise ValueError("Configured model_id does not match the stored dataset")

    device = select_device()
    dataset_device = dataset_record["model"]["device"]
    if dataset_device != str(device):
        # CPU/MPS/CUDA traces can diverge numerically, so a representation read
        # on one backend does not describe a state defined on another.
        raise ValueError(
            f"Dataset states were generated on {dataset_device!r} but this host "
            f"selected {str(device)!r}; regenerate the states on this backend"
        )

    states = stored_states(dataset_record)
    if not states:
        raise ValueError("Dataset record contains no selected states")

    tokenizer, model = load_causal_lm(config["model_id"], device)
    del tokenizer  # Identity here is token IDs; nothing is decoded or re-tokenized.

    captured = []
    indexing: dict[str, Any] | None = None
    for state in states:
        prefix_token_ids = state["shared_prefix_token_ids"]
        result = capture_prefix_hidden_states(model, prefix_token_ids, device)

        # CRITICAL INVARIANT: the prefix used for extraction is the prefix that
        # defines the utility state, integer for integer.
        if result["captured_token_ids"] != prefix_token_ids:
            raise ValueError(f"{state['state_id']}: extraction prefix differs from the state")
        fingerprint = prefix_fingerprint(prefix_token_ids)
        stored_fingerprint = state["shared_prefix_sha256"]
        if stored_fingerprint is not None and stored_fingerprint != fingerprint:
            raise ValueError(f"{state['state_id']}: stored prefix fingerprint does not match")

        hidden_states = result["hidden_states"]
        if not tensor_is_finite(hidden_states):
            raise ValueError(f"{state['state_id']}: hidden states contain NaN or Inf")
        if indexing is None:
            indexing = result["indexing"]
        elif result["indexing"] != indexing:
            raise ValueError(f"{state['state_id']}: hidden-state layout changed between states")

        captured.append({**state, "shared_prefix_sha256": fingerprint, "raw": hidden_states})
        print(
            f"{state['state_id']} | {state['selection_reason']:<7} | "
            f"prefix={len(prefix_token_ids):<4} | {tuple(hidden_states.shape)}"
        )

    assert indexing is not None
    stacked = torch.stack([entry["raw"] for entry in captured])
    stored = stacked.to(storage_dtype)
    max_cast_error = float((stacked - stored.to(torch.float32)).abs().max())

    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_path = output_dir / "activations.npz"
    manifest_path = output_dir / "manifest.json"

    state_ids = [entry["state_id"] for entry in captured]
    np.savez(
        activations_path,
        hidden_states=stored.numpy(),
        state_ids=np.array(state_ids),
        layer_labels=np.array(indexing["layer_labels"]),
    )

    provenance = dataset_record.get("provenance", {})
    manifest = {
        "artifact": "pre_treatment_hidden_states",
        "description": (
            "Hidden state at the final token of each state's exact pre-treatment "
            "prefix, for every layer. Captured before any branch action."
        ),
        "input_path": config["input_path"],
        "activations_path": str(activations_path.relative_to(PROJECT_ROOT)),
        "model_id": config["model_id"],
        "model_revision": provenance.get("model_revision"),
        "device": str(device),
        "compute_dtype": str(next(model.parameters()).dtype),
        "storage_dtype": str(storage_dtype).removeprefix("torch."),
        "max_abs_storage_cast_error": max_cast_error,
        "hidden_state_indexing": indexing,
        "array_layout": {
            "hidden_states": ["state", "layer", "hidden_size"],
            "shape": list(stored.shape),
            "state_order": "row index into hidden_states, matching states[] below",
        },
        "num_states": len(captured),
        "states": [
            {
                "row_index": row_index,
                "state_id": entry["state_id"],
                "problem_id": entry["problem_id"],
                "problem_index": entry["problem_index"],
                "selection_reason": entry["selection_reason"],
                "step_index": entry["step_index"],
                "boundary_token_index": entry["boundary_token_index"],
                "relative_position": entry["relative_position"],
                "shared_prefix_token_count": len(entry["shared_prefix_token_ids"]),
                "shared_prefix_sha256": entry["shared_prefix_sha256"],
            }
            for row_index, entry in enumerate(captured)
        ],
    }
    save_record(manifest, manifest_path)

    # Serialization must not have changed what the model computes. Re-running
    # the first state's forward pass has to reproduce the captured vectors
    # exactly, in full precision.
    recheck = capture_prefix_hidden_states(
        model, captured[0]["shared_prefix_token_ids"], device
    )
    recapture_identical = bool(torch.equal(recheck["hidden_states"], captured[0]["raw"]))

    reloaded = np.load(activations_path)
    reload_identical = bool(
        torch.equal(torch.from_numpy(reloaded["hidden_states"]), stored)
    )

    print(f"\nStates: {len(captured)}")
    print(f"  hidden_states shape        {tuple(stored.shape)}  ({manifest['storage_dtype']})")
    print(f"  layers per state           {indexing['num_hidden_states']} "
          f"(= {indexing['num_hidden_layers']} transformer layers + embedding output)")
    print(f"  hidden size                {indexing['hidden_size']}")
    print(f"  entry 0 is embedding out   {indexing['first_matches_embedding_output']}")
    print(f"  entry -1 feeds the LM head {indexing['final_matches_lm_head_input']}")
    print("  finite (no NaN/Inf)        True  (every state checked at capture)")
    print(f"  max storage cast error     {max_cast_error:.3e}")
    print(f"  recapture identical        {recapture_identical}")
    print(f"  reload identical           {reload_identical}")
    print(f"  activations size           {activations_path.stat().st_size / 1e6:.2f} MB")
    print(f"  manifest size              {manifest_path.stat().st_size / 1e3:.1f} kB")
    print(f"\nSaved: {activations_path}\n       {manifest_path}")

    if not (recapture_identical and reload_identical):
        raise ValueError("Serialization changed the captured representation")


if __name__ == "__main__":
    main()
