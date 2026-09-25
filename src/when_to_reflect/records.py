"""Structured experiment output."""

import json
from pathlib import Path
from typing import Any

DATASET_RECORD_FIELDS = frozenset(
    {
        "dataset",
        "dataset_path",
        "model",
        "prompt_version",
        "reflection_prompt_version",
        "reflection_instruction",
        "checkpoint_policy",
        "base_trace_generation_config",
        "rollout_generation_config",
        "provenance",
        "problems",
        "summary",
    }
)


def save_record(
    record: dict[str, Any],
    path: str | Path,
    required_fields: frozenset[str] = frozenset(),
) -> None:
    """Validate and save one experiment record as JSON."""
    missing_fields = required_fields - record.keys()
    if missing_fields:
        raise ValueError(f"Missing record fields: {sorted(missing_fields)}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
