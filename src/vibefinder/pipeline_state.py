"""LangGraph state definitions for the recommendation pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, NotRequired, TypedDict


PipelineStage = Literal[
    "received_query",
    "extract_preferences",
    "plan_retrieval",
    "retrieve_candidates",
    "verify_candidates",
    "critique_results",
    "revise_plan",
    "rank_candidates",
    "explain_recommendations",
    "build_reliability_report",
    "finished",
]


RetrievalMode = Literal[
    "lyric_retrieval",
    "metadata_retrieval",
    "feature_filter",
]


class PipelineTraceEvent(TypedDict):
    """One lightweight trace event stored in LangGraph state."""

    stage: PipelineStage
    message: str
    details: NotRequired[dict[str, Any]]


class PipelineState(TypedDict, total=False):
    """Serializable state shared across LangGraph recommendation nodes.

    Large runtime objects such as the songs DataFrame, FAISS index, embedding model,
    and LLM client must stay outside this state and be injected into graph nodes.
    Agent outputs and tool outputs should be stored as Pydantic `model_dump`
    dictionaries so the state remains inspectable and checkpoint-friendly.
    """

    query: str
    variant_name: str | None

    preferences: dict[str, Any] | None
    retrieval_plan: dict[str, Any] | None
    revision_plan: dict[str, Any] | None
    critique: dict[str, Any] | None
    reliability: dict[str, Any] | None
    fatal_error: str | None

    candidate_ids: Annotated[tuple[str, ...], replace_tuple]
    retrieval_modes_used: Annotated[tuple[str, ...], replace_tuple]
    tool_outputs: Annotated[dict[str, Any], merge_dicts]
    tool_errors: Annotated[tuple[dict[str, Any], ...], append_tuples]
    verified_candidates: Annotated[tuple[dict[str, Any], ...], replace_tuple]
    scored_candidates: Annotated[tuple[dict[str, Any], ...], replace_tuple]
    recommendations: Annotated[tuple[dict[str, Any], ...], replace_tuple]
    explanations: Annotated[tuple[dict[str, Any], ...], replace_tuple]

    revision_count: int
    warnings: Annotated[tuple[str, ...], merge_unique_strings]
    trace: Annotated[tuple[PipelineTraceEvent, ...], append_tuples]


def create_initial_pipeline_state(
    query: str,
    *,
    variant_name: str | None = None,
) -> PipelineState:
    """Create a complete initial state for one recommendation run."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query cannot be blank.")

    return PipelineState(
        query=cleaned_query,
        variant_name=variant_name,
        preferences=None,
        retrieval_plan=None,
        revision_plan=None,
        critique=None,
        reliability=None,
        fatal_error=None,
        candidate_ids=(),
        retrieval_modes_used=(),
        tool_outputs={},
        tool_errors=(),
        verified_candidates=(),
        scored_candidates=(),
        recommendations=(),
        explanations=(),
        revision_count=0,
        warnings=(),
        trace=(
            PipelineTraceEvent(
                stage="received_query",
                message="Received user query.",
                details={"query_length": len(cleaned_query)},
            ),
        ),
    )


def make_trace_event(
    stage: PipelineStage,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> PipelineTraceEvent:
    """Build a trace event for graph node updates."""

    event = PipelineTraceEvent(stage=stage, message=message)
    if details:
        event["details"] = dict(details)
    return event


def merge_unique_strings(
    current: Iterable[str] | str | None,
    update: Iterable[str] | str | None,
) -> tuple[str, ...]:
    """Merge string sequences while preserving order and removing blanks."""

    merged: list[str] = []
    seen: set[str] = set()
    for value in (*_as_string_tuple(current), *_as_string_tuple(update)):
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            merged.append(cleaned)
            seen.add(cleaned)
    return tuple(merged)


def merge_dicts(
    current: Mapping[str, Any] | None,
    update: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge dict state updates without mutating the previous state.

    Update values set to None delete the corresponding current key. This lets
    retry nodes clear stale tool outputs while preserving other tool results.
    """

    merged = dict(current or {})
    for key, value in dict(update or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def append_tuples(
    current: Iterable[Any] | Any | None,
    update: Iterable[Any] | Any | None,
) -> tuple[Any, ...]:
    """Append tuple-like state updates."""

    return (*_as_tuple(current), *_as_tuple(update))


def replace_tuple(
    _current: Iterable[Any] | Any | None,
    update: Iterable[Any] | Any | None,
) -> tuple[Any, ...]:
    """Replace list-like state for node outputs where latest value should win."""

    return _as_tuple(update)


def _as_string_tuple(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _as_tuple(value: Iterable[Any] | Any | None) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)
