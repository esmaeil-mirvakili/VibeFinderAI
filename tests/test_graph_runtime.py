from __future__ import annotations

from typing import Any

import pandas as pd

from vibefinder.graph_runtime import (
    DEFAULT_VARIANT_CONFIG,
    GraphRuntimeContext,
    complete_agent_step,
    create_graph_runtime_context,
    enabled_tool_names,
    run_tool_step,
)
from vibefinder.tools import ToolRunner
from vibefinder.variants import VariantConfig


class _FakeLLM:
    def __init__(self, output: dict[str, Any] | None = None, error: Exception | None = None):
        self.output = output or {}
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        self.calls.append((prompt, schema_name))
        if self.error is not None:
            raise self.error
        return self.output


def test_create_graph_runtime_context_keeps_runtime_dependencies_out_of_state():
    songs = _sample_rows()
    retrieval_prompt_config = _test_config()
    context = create_graph_runtime_context(
        songs=songs,
        retrieval_prompt_config=retrieval_prompt_config,
        llm_client=_FakeLLM(),
    )

    assert isinstance(context, GraphRuntimeContext)
    assert context.songs is songs
    assert context.retrieval_prompt_config is retrieval_prompt_config
    assert context.tool_context.songs is songs
    assert context.tool_context.retrieval_prompt_config is retrieval_prompt_config
    assert isinstance(context.tool_runner, ToolRunner)
    assert context.tool_runner.context is context.tool_context
    assert context.variant_config == DEFAULT_VARIANT_CONFIG
    assert context.max_revision_count == 1


def test_enabled_tool_names_respects_no_lyric_variant():
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=_FakeLLM(),
        variant_config=VariantConfig(
            name="no_lyric",
            use_multi_step_reasoning=True,
            use_critic_revision=True,
            use_lyric_retriever=False,
        ),
    )

    assert "lyric_retrieval" not in enabled_tool_names(context)
    assert "metadata_retrieval" in enabled_tool_names(context)


def test_complete_agent_step_returns_state_update_with_trace_and_warnings():
    llm = _FakeLLM(
        {
            "raw_query": "English high energy songs",
            "rationale": "The query asks for English songs with high energy.",
            "issues": ("ambiguous genre",),
            "warnings": ("Genre was not specified.",),
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )

    update = complete_agent_step(
        context=context,
        prompt="Extract preferences.",
        schema_name="PreferenceExtractionOutput",
        state_key="preferences",
        stage="extract_preferences",
    )

    assert update["preferences"]["raw_query"] == "English high energy songs"
    assert update["warnings"] == ("Genre was not specified.",)
    assert update["trace"][0]["stage"] == "extract_preferences"
    assert update["trace"][0]["details"]["issue_count"] == 1
    assert llm.calls == [("Extract preferences.", "PreferenceExtractionOutput")]


def test_complete_agent_step_returns_structured_failure_update():
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=_FakeLLM(error=RuntimeError("bad model output")),
    )

    update = complete_agent_step(
        context=context,
        prompt="Extract preferences.",
        schema_name="PreferenceExtractionOutput",
        state_key="preferences",
        stage="extract_preferences",
    )

    assert "preferences" not in update
    assert update["warnings"] == ("extract_preferences failed: bad model output",)
    assert update["trace"][0]["details"]["error_type"] == "RuntimeError"


def test_run_tool_step_records_retrieval_output_candidates_and_mode():
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=_FakeLLM(),
    )

    update = run_tool_step(
        context=context,
        tool_name="metadata_retrieval",
        raw_input={
            "categorical_filters": [{"field": "language", "values": ["en"]}],
            "top_k": 5,
        },
        stage="retrieve_candidates",
    )

    assert update["retrieval_modes_used"] == ("metadata_retrieval",)
    assert update["candidate_ids"] == ("song-a", "song-b")
    assert update["tool_outputs"]["metadata_retrieval"]["output_count"] == 2
    assert update["trace"][0]["details"]["candidate_count"] == 2


def test_run_tool_step_skips_disabled_tool_without_calling_runner():
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=_FakeLLM(),
        variant_config=VariantConfig(
            name="no_lyric",
            use_multi_step_reasoning=True,
            use_critic_revision=True,
            use_lyric_retriever=False,
        ),
    )

    update = run_tool_step(
        context=context,
        tool_name="lyric_retrieval",
        raw_input={"query": "betrayal", "top_k": 5},
        stage="retrieve_candidates",
    )

    assert update["warnings"] == ("Tool is disabled by variant config: lyric_retrieval",)
    assert update["trace"][0]["message"] == "Tool step skipped."
    assert "tool_outputs" not in update


def test_run_tool_step_records_tool_errors():
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=_FakeLLM(),
    )

    update = run_tool_step(
        context=context,
        tool_name="metadata_retrieval",
        raw_input={"text_queries": [{"field": "playlist_genre", "query": "pop"}]},
        stage="retrieve_candidates",
    )

    assert update["tool_errors"][0]["tool_name"] == "metadata_retrieval"
    assert update["tool_errors"][0]["error_type"] == "validation_error"
    assert update["warnings"][0].startswith("metadata_retrieval failed:")
    assert update["trace"][0]["message"] == "Tool step failed."


def test_run_tool_step_maps_scoring_and_reliability_outputs():
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=_FakeLLM(),
    )

    scoring_update = run_tool_step(
        context=context,
        tool_name="candidate_scoring",
        raw_input={
            "candidate_track_ids": ["song-a"],
            "metadata_evidence": [{"track_id": "song-a", "score": 1.0}],
        },
        stage="rank_candidates",
    )
    reliability_update = run_tool_step(
        context=context,
        tool_name="reliability",
        raw_input={
            "final_candidates": [
                {
                    "track_id": "song-a",
                    "rank": 1,
                    "final_score": 0.9,
                    "evidence_sources": ["metadata"],
                }
            ],
            "requested_count": 1,
        },
        stage="build_reliability_report",
    )

    assert scoring_update["scored_candidates"][0]["track_id"] == "song-a"
    assert reliability_update["reliability"]["support_summary"]["candidate_count"] == 1


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "song-a",
                "language": "en",
                "playlist_genre": "pop",
                "playlist_subgenre": "dance pop",
                "playlist_name": "Pop Room",
                "track_artist": "Aster",
                "track_album_name": "Night Signals",
                "track_popularity": 80,
            },
            {
                "track_id": "song-b",
                "language": "en",
                "playlist_genre": "rock",
                "playlist_subgenre": "hard rock",
                "playlist_name": "Rock Room",
                "track_artist": "Boreal",
                "track_album_name": "Daylight",
                "track_popularity": 40,
            },
        ]
    )


def _test_config() -> dict:
    return {
        "llm_prompt_constraints": {
            "categorical_values": {
                "playlist_genre": ["pop", "rock"],
                "playlist_subgenre": ["dance pop", "hard rock"],
                "language": ["en", "es"],
            },
            "full_text_search_columns": {
                "playlist_name": {"search_method": "full_text_search", "unique_count": 2},
                "track_artist": {"search_method": "full_text_search", "unique_count": 2},
                "track_album_name": {"search_method": "full_text_search", "unique_count": 2},
            },
        }
    }
