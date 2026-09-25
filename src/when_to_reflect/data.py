"""Local dataset access helpers."""

from pathlib import Path
from typing import Any

from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATH_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "math_train"


def load_math_training_example(
    problem_index: int,
    dataset_path: str | Path = DEFAULT_MATH_TRAIN_PATH,
) -> dict[str, Any]:
    """Load one MATH training example from a dataset saved to disk."""
    dataset = load_from_disk(str(dataset_path))
    if not 0 <= problem_index < len(dataset):
        raise IndexError(f"problem_index must be in [0, {len(dataset)})")
    return dict(dataset[problem_index])
