import pytest
from datasets import Dataset

from when_to_reflect.data import load_math_training_example


def test_load_math_training_example(tmp_path) -> None:
    dataset_path = tmp_path / "math_train"
    Dataset.from_dict(
        {"problem_id": ["math_train_00000"], "problem": ["1 + 1"], "solution": ["2"]}
    ).save_to_disk(dataset_path)

    assert load_math_training_example(0, dataset_path)["problem_id"] == "math_train_00000"


def test_load_math_training_example_rejects_out_of_range_index(tmp_path) -> None:
    dataset_path = tmp_path / "math_train"
    Dataset.from_dict({"problem_id": ["math_train_00000"]}).save_to_disk(dataset_path)

    with pytest.raises(IndexError, match="problem_index"):
        load_math_training_example(1, dataset_path)
