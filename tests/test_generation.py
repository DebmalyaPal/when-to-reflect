from typing import ClassVar

import pytest

from when_to_reflect.generation import (
    effective_generation_config,
    generation_status,
    terminal_token_ids,
    verify_generation_settings,
)


class FakeTokenizer:
    eos_token_id = 9


class FakeGenerationConfig:
    eos_token_id: ClassVar = [9, 10]
    forced_eos_token_id = 11


class FakeModel:
    generation_config = FakeGenerationConfig()


class ShippedGenerationConfig:
    """Stands in for a model's own generation_config.json."""

    def to_dict(self) -> dict[str, object]:
        return {"do_sample": True, "temperature": 0.6, "top_k": 20, "unrelated": 1}


class ShippedDefaultsModel:
    generation_config = ShippedGenerationConfig()


def test_generation_status_detects_eos_termination() -> None:
    status = generation_status([1, 2, 9], max_new_tokens=8, terminal_token_ids={9})

    assert status == {
        "generated_token_count": 3,
        "terminated_by_eos": True,
        "hit_max_new_tokens": False,
        "generation_complete": True,
    }


def test_generation_status_detects_max_token_truncation() -> None:
    status = generation_status([1, 2, 3], max_new_tokens=3, terminal_token_ids={9})

    assert status == {
        "generated_token_count": 3,
        "terminated_by_eos": False,
        "hit_max_new_tokens": True,
        "generation_complete": False,
    }


def test_terminal_token_ids_include_tokenizer_and_model_stops() -> None:
    assert terminal_token_ids(FakeTokenizer(), FakeModel()) == {9, 10, 11}


def test_effective_generation_config_keeps_unpassed_shipped_defaults() -> None:
    resolved = effective_generation_config(
        ShippedDefaultsModel(),
        {"do_sample": True, "max_new_tokens": 1024, "temperature": 0.7, "top_p": 0.95},
    )

    # Passed arguments win, and top_k survives from the model's own defaults
    # even though the caller never mentioned it.
    assert resolved["temperature"] == 0.7
    assert resolved["top_p"] == 0.95
    assert resolved["max_new_tokens"] == 1024
    assert resolved["top_k"] == 20
    # A setting neither passed nor shipped is reported as absent, not omitted.
    assert resolved["repetition_penalty"] is None
    assert "unrelated" not in resolved


def test_verify_generation_settings_accepts_applied_values() -> None:
    verify_generation_settings({"top_k": 20, "temperature": 0.7}, {"top_k": 20}, "rollout")


def test_verify_generation_settings_rejects_a_setting_that_never_reached_generate() -> None:
    # Configured top_k: 40, but generate would apply the model's own 20.
    with pytest.raises(ValueError, match="rollout generation settings were not applied"):
        verify_generation_settings({"top_k": 20}, {"top_k": 40}, "rollout")
