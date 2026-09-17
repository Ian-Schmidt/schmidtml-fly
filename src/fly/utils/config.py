"""Project configs: every number and name that used to be hard-coded lives in `fly/configs/*.json`.

The config folder is part of the package, not of the repository root: only the `fly` package is
stored next to a model in MLflow, and the model must load without the repository.
"""

import json
from functools import cache
from pathlib import Path
from typing import Any

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@cache
def load(name: str) -> dict[str, Any]:
    """Reads a config by name; repeated calls return the same dict without touching the disk.

    Args:
        name: File name in `fly/configs/` without the extension: `model`, `data`, `training` or `viz`.

    Returns:
        The config contents.

    Raises:
        FileNotFoundError: If there is no such config.

    Example:
        >>> load("model")["steps"]
        32
    """
    return json.loads((CONFIGS / f"{name}.json").read_text())
