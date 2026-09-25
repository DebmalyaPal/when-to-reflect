import pytest

from when_to_reflect.config import load_config


def test_load_config_reads_yaml_mapping(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("seed: 7\ndevice: auto\n", encoding="utf-8")

    assert load_config(config_path) == {"seed": 7, "device": "auto"}


def test_load_config_rejects_non_mapping(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(TypeError, match="YAML mapping"):
        load_config(config_path)
