import json

import pytest

from when_to_reflect.records import DATASET_RECORD_FIELDS, save_record


def test_save_record_writes_json(tmp_path) -> None:
    record = {field: "value" for field in DATASET_RECORD_FIELDS}
    output_path = tmp_path / "nested" / "dataset.json"

    save_record(record, output_path, DATASET_RECORD_FIELDS)

    assert set(json.loads(output_path.read_text())) == DATASET_RECORD_FIELDS


def test_save_record_rejects_missing_required_fields(tmp_path) -> None:
    with pytest.raises(ValueError, match="Missing record fields"):
        save_record({}, tmp_path / "dataset.json", DATASET_RECORD_FIELDS)


def test_save_record_without_required_fields_accepts_any_mapping(tmp_path) -> None:
    output_path = tmp_path / "free.json"

    save_record({"anything": 1}, output_path)

    assert json.loads(output_path.read_text()) == {"anything": 1}
