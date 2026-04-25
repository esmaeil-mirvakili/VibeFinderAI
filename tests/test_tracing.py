from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vibefinder.tracing import (
    _summarize_llm_output,
    context_summary,
    graph_invoke_config,
    langsmith_extra,
    tool_component,
)
from vibefinder.variants import VariantConfig


@dataclass(frozen=True)
class _LLM:
    provider: str = "ollama"
    model: str = "llama3.2"


@dataclass(frozen=True)
class _Context:
    songs: pd.DataFrame
    llm_client: _LLM
    variant_config: VariantConfig
    max_revision_count: int = 1


def test_langsmith_extra_groups_runs_by_backend_model_variant_and_stage():
    context = _test_context()

    extra = langsmith_extra(
        context=context,
        run_id="run-1",
        stage="retrieve_candidates",
        tool_name="lyric_retrieval",
        metadata={"component": tool_component("lyric_retrieval")},
    )

    assert extra["metadata"]["run_id"] == "run-1"
    assert extra["metadata"]["llm_provider"] == "ollama"
    assert extra["metadata"]["llm_model"] == "llama3.2"
    assert extra["metadata"]["variant_name"] == "full"
    assert extra["metadata"]["component"] == "faiss_lyric_retrieval"
    assert "backend:ollama" in extra["tags"]
    assert "model:llama3.2" in extra["tags"]
    assert "variant:full" in extra["tags"]
    assert "tool:lyric_retrieval" in extra["tags"]
    assert "component:faiss_lyric_retrieval" in extra["tags"]


def test_graph_invoke_config_keeps_runtime_objects_out_of_metadata():
    context = _test_context()

    config = graph_invoke_config(context=context, run_id="run-2")

    assert config["metadata"]["run_id"] == "run-2"
    assert config["metadata"]["variant_name"] == "full"
    assert config["metadata"]["llm_provider"] == "ollama"
    assert "songs" not in config["metadata"]
    assert "lyric_index" not in config["metadata"]
    assert "stage:graph_execution" in config["tags"]
    assert config["run_name"] == "vibefinder:full:ollama"


def test_context_summary_is_small_and_serializable():
    context = _test_context()

    summary = context_summary(context)

    assert summary == {
        "row_count": 2,
        "column_count": 1,
        "variant_name": "full",
        "llm_provider": "ollama",
        "llm_model": "llama3.2",
        "max_revision_count": 1,
    }


def test_llm_trace_output_includes_raw_text_generations_and_validated_json():
    summary = _summarize_llm_output(
        {
            "raw_response": '{"raw_query":"English songs","rationale":"Language requested."}',
            "parsed_response": {"raw_query": "English songs", "rationale": "Language requested."},
            "validated_output": {
                "raw_query": "English songs",
                "rationale": "Language requested.",
                "issues": [],
                "warnings": [],
            },
        }
    )

    assert summary["raw_response"] == '{"raw_query":"English songs","rationale":"Language requested."}'
    assert summary["response"]["raw_query"] == "English songs"
    assert summary["validated_output"]["warnings"] == []
    assert summary["generations"][0][0]["text"] == summary["raw_response"]
    assert summary["generations"][0][0]["message"]["role"] == "assistant"


def _test_context() -> _Context:
    return _Context(
        songs=pd.DataFrame({"track_id": ["a", "b"]}),
        llm_client=_LLM(),
        variant_config=VariantConfig(
            name="full",
            use_multi_step_reasoning=True,
            use_critic_revision=True,
            use_lyric_retriever=True,
        ),
    )
