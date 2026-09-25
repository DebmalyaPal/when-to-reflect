import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

DATASET_ID = "EleutherAI/hendrycks_math"
DATASET_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "math_train"


def main() -> None:
    if OUTPUT_DIR.exists():
        print(f"MATH training data already exists at {OUTPUT_DIR}")
        print("Skipping download.")
        return

    datasets = []

    for subject in SUBJECTS:
        print(f"Loading {subject} ...")

        dataset = load_dataset(
            DATASET_ID,
            subject,
            split="train",
            revision=DATASET_REVISION,
        )

        # Preserve the original MATH subject/configuration.
        dataset = dataset.add_column(
            "subject",
            [subject] * len(dataset),
        )

        datasets.append(dataset)

    combined = concatenate_datasets(datasets)

    # Stable identifier for downstream experiment bookkeeping.
    combined = combined.add_column(
        "problem_id",
        [f"math_train_{i:05d}" for i in range(len(combined))],
    )

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    combined.save_to_disk(OUTPUT_DIR)
    manifest = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": "train",
        "subjects": SUBJECTS,
        "num_examples": len(combined),
        "problem_id_format": "math_train_{index:05d}",
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("MATH training dataset prepared successfully.")
    print(f"Examples : {len(combined)}")
    print(f"Location : {OUTPUT_DIR}")
    print()
    print("Columns:")
    for column in combined.column_names:
        print(f"  - {column}")


if __name__ == "__main__":
    main()
