from __future__ import annotations

import json

from vibefinder.prompts import build_agent_prompt, build_schema_prompt


def test_build_agent_prompt_contains_shared_template_and_payload():
    prompt = build_agent_prompt(
        "Preference Extraction Agent",
        "Extract structured music preferences.",
        {"query": "sad English songs", "requirements": ["Do not invent values."]},
    )

    assert prompt.startswith("You are Preference Extraction Agent.")
    assert "Task:\nExtract structured music preferences." in prompt
    assert "Do not include hidden chain-of-thought" in prompt
    assert '"query": "sad English songs"' in prompt
    assert "Return only the JSON object." in prompt


def test_build_schema_prompt_appends_requested_schema():
    schema_spec = {
        "schema_name": "PreferenceExtractionOutput",
        "json_schema": {"type": "object", "properties": {"raw_query": {"type": "string"}}},
        "constraints": {"no_invented_dataset_columns": True},
    }

    prompt = build_schema_prompt("Base prompt.\n", schema_spec)

    assert prompt.startswith("Base prompt.")
    assert "Return only a valid JSON object matching this schema and constraints." in prompt
    encoded_schema = prompt.split("\n\n")[-1]
    assert json.loads(encoded_schema) == schema_spec
