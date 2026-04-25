from __future__ import annotations

from typing import Annotated, get_args, get_origin, get_type_hints

import pytest

from vibefinder.pipeline_state import (
    PipelineState,
    append_tuples,
    create_initial_pipeline_state,
    make_trace_event,
    merge_dicts,
    merge_unique_strings,
    replace_tuple,
)


def test_create_initial_pipeline_state_is_complete_and_serializable():
    state = create_initial_pipeline_state("  English songs with high energy  ", variant_name="full")

    assert state["query"] == "English songs with high energy"
    assert state["variant_name"] == "full"
    assert state["preferences"] is None
    assert state["retrieval_plan"] is None
    assert state["candidate_ids"] == ()
    assert state["retrieval_modes_used"] == ()
    assert state["tool_outputs"] == {}
    assert state["tool_errors"] == ()
    assert state["revision_count"] == 0
    assert state["warnings"] == ()
    assert state["trace"][0]["stage"] == "received_query"
    assert state["trace"][0]["details"] == {"query_length": 30}


def test_create_initial_pipeline_state_rejects_blank_query():
    with pytest.raises(ValueError, match="query cannot be blank"):
        create_initial_pipeline_state("   ")


def test_pipeline_state_annotations_define_langgraph_reducers():
    hints = get_type_hints(PipelineState, include_extras=True)

    assert get_origin(hints["candidate_ids"]) is Annotated
    assert get_args(hints["candidate_ids"])[1] is replace_tuple
    assert get_args(hints["retrieval_modes_used"])[1] is replace_tuple
    assert get_args(hints["tool_outputs"])[1] is merge_dicts
    assert get_args(hints["tool_errors"])[1] is append_tuples
    assert get_args(hints["recommendations"])[1] is replace_tuple
    assert get_args(hints["trace"])[1] is append_tuples


def test_pipeline_state_excludes_heavy_runtime_objects():
    keys = set(get_type_hints(PipelineState, include_extras=True))

    assert "songs" not in keys
    assert "dataset" not in keys
    assert "lyric_index" not in keys
    assert "lyric_embedder" not in keys
    assert "llm_client" not in keys


def test_merge_unique_strings_preserves_order_and_dedupes_blanks():
    merged = merge_unique_strings(
        ("lyric_retrieval", "metadata_retrieval"),
        ["metadata_retrieval", " ", "feature_filter"],
    )

    assert merged == ("lyric_retrieval", "metadata_retrieval", "feature_filter")


def test_merge_dicts_does_not_mutate_current_value():
    current = {"lyric_retrieval": {"output_count": 5}}
    merged = merge_dicts(current, {"feature_filter": {"output_count": 3}})

    assert current == {"lyric_retrieval": {"output_count": 5}}
    assert merged == {
        "lyric_retrieval": {"output_count": 5},
        "feature_filter": {"output_count": 3},
    }


def test_merge_dicts_deletes_none_update_values():
    current = {
        "lyric_retrieval": {"output_count": 5},
        "candidate_scoring": {"output_count": 2},
    }
    merged = merge_dicts(current, {"lyric_retrieval": None})

    assert merged == {"candidate_scoring": {"output_count": 2}}
    assert current["lyric_retrieval"] == {"output_count": 5}


def test_append_and_replace_tuple_reducers():
    assert append_tuples(("first",), ["second"]) == ("first", "second")
    assert replace_tuple(("old",), ["new"]) == ("new",)
    assert replace_tuple(("old",), None) == ()


def test_make_trace_event_omits_empty_details():
    event = make_trace_event("plan_retrieval", "Planned retrieval.")
    detailed = make_trace_event("retrieve_candidates", "Ran tools.", {"candidate_count": 5})

    assert event == {"stage": "plan_retrieval", "message": "Planned retrieval."}
    assert detailed == {
        "stage": "retrieve_candidates",
        "message": "Ran tools.",
        "details": {"candidate_count": 5},
    }
