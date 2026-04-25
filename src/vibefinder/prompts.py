"""Prompt templates for LLM-backed agent calls."""

from __future__ import annotations

import json
from typing import Any


def build_agent_prompt(agent_name: str, task: str, payload: dict[str, Any]) -> str:
    """Build the shared stage prompt before the output schema is attached."""

    return (
        f"You are {agent_name}.\n\n"
        f"Task:\n{task}\n\n"
        "Return ONLY one valid JSON object.\n"
        "Do not return markdown.\n"
        "Do not use code fences.\n"
        "Do not add commentary before or after the JSON.\n"
        "Do not explain your reasoning.\n"
        "Think through the task internally, but expose only concise public summaries when useful.\n\n"
        "Output rules:\n"
        "1. The output must be parseable by json.loads.\n"
        "2. Return a JSON object only.\n"
        "3. Include only fields that are supported by the context and relevant to the task.\n"
        "4. Omit fields that are unknown, null, empty, unsupported, or not needed.\n"
        "5. Do not invent values.\n"
        "6. Do not include hidden chain-of-thought.\n"
        "7. Strings must use double quotes.\n\n"
        "Context:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True, default=str)}\n\n"
        "Return only the JSON object."
    )


def build_schema_prompt(prompt: str, schema_spec: dict[str, Any]) -> str:
    """Append the requested structured-output schema to a stage prompt."""

    return (
        f"{prompt.strip()}\n\n"
        "Return only a valid JSON object matching this schema and constraints. "
        "Do not wrap the JSON in Markdown.\n\n"
        f"{json.dumps(schema_spec, sort_keys=True)}"
    )
