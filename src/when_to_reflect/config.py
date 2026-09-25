"""Configuration loading for experiments."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment's YAML mapping."""
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return config
