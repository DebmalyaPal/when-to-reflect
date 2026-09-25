"""Generation-completion checks and resolved generation settings."""

from collections.abc import Iterable, Sequence
from typing import Any


def _as_token_ids(value: int | Iterable[int] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return set(value)


def terminal_token_ids(tokenizer: Any, model: Any) -> set[int]:
    """Collect EOS and configured stop-token IDs used by the tokenizer/model."""
    generation_config = model.generation_config
    return (
        _as_token_ids(getattr(tokenizer, "eos_token_id", None))
        | _as_token_ids(getattr(generation_config, "eos_token_id", None))
        | _as_token_ids(getattr(generation_config, "forced_eos_token_id", None))
    )


def generation_status(
    generated_token_ids: Sequence[int],
    max_new_tokens: int,
    terminal_token_ids: set[int],
) -> dict[str, int | bool]:
    """Report whether a continuation ended naturally or reached its token cap."""
    token_count = len(generated_token_ids)
    terminated_by_eos = bool(generated_token_ids) and generated_token_ids[-1] in terminal_token_ids
    hit_max_new_tokens = token_count >= max_new_tokens
    return {
        "generated_token_count": token_count,
        "terminated_by_eos": terminated_by_eos,
        "hit_max_new_tokens": hit_max_new_tokens,
        "generation_complete": terminated_by_eos,
    }


# Settings that change what `generate` produces. Anything we do not pass
# explicitly is filled in from the model's own `generation_config.json`, so the
# resolved values are recorded rather than the arguments we happened to pass.
RESOLVED_GENERATION_FIELDS = (
    "do_sample",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "eos_token_id",
    "pad_token_id",
)


def effective_generation_config(model: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Report the settings ``model.generate(**overrides)`` will actually apply.

    ``generate`` starts from ``model.generation_config`` and overlays the keyword
    arguments it is given, so shipped defaults such as ``top_k`` or
    ``repetition_penalty`` take effect even when the caller never mentions them.
    This makes the resolved values explicit in the run record.

    The logit warpers (``temperature``, ``top_p``, ``top_k``, ``min_p``,
    ``typical_p``) are inert when ``do_sample`` is false; ``repetition_penalty``
    and ``no_repeat_ngram_size`` apply either way.
    """
    resolved = model.generation_config.to_dict()
    resolved.update(overrides)
    return {field: resolved.get(field) for field in RESOLVED_GENERATION_FIELDS}


def verify_generation_settings(
    resolved: dict[str, Any],
    configured: dict[str, Any],
    label: str,
) -> None:
    """Fail if a decoding setting stated in a config is not the one that applies.

    ``configured`` holds the values a configuration file asked for and
    ``resolved`` what ``generate`` will actually use. The two disagree when a
    setting was written down but never plumbed through to ``generate``, in
    which case the model's own default silently decides it instead. That is a
    change in the experiment, so it raises rather than warns.
    """
    mismatched = {
        field: {"configured": value, "resolved": resolved.get(field)}
        for field, value in configured.items()
        if resolved.get(field) != value
    }
    if mismatched:
        raise ValueError(f"{label} generation settings were not applied: {mismatched}")
