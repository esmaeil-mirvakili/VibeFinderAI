"""Load generated retrieval prompt configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RETRIEVAL_PROMPT_CONFIG_PATH = Path("config/retrieval_prompt_config.json")


def load_retrieval_prompt_config(
    path: str | Path = DEFAULT_RETRIEVAL_PROMPT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the generated retrieval prompt config JSON."""

    config_path = Path(path)
    return json.loads(config_path.read_text(encoding="utf-8"))
