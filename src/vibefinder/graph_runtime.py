"""Runtime helpers shared by LangGraph recommendation nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from loguru import logger

from vibefinder.agents import AGENT_OUTPUT_SCHEMAS
from vibefinder.llm import LLMClient, create_llm_client
from vibefinder.pipeline_state import PipelineStage, PipelineState, make_trace_event
from vibefinder.tracing import context_summary, langsmith_extra, trace_agent_node, trace_tool_call, tool_component
from vibefinder.tools import (
    ToolContext,
    ToolError,
    ToolResult,
    ToolRunner,
    get_tool_registry,
)
from vibefinder.variants import VariantConfig


AgentStateKey = Literal["preferences", "retrieval_plan", "critique", "revision_plan", "explanations"]

RETRIEVAL_TOOL_NAMES: tuple[str, ...] = (
    "lyric_retrieval",
    "metadata_retrieval",
    "feature_filter",
)

DEFAULT_VARIANT_CONFIG = VariantConfig(
    name="full",
    use_multi_step_reasoning=True,
    use_critic_revision=True,
    use_lyric_retriever=True,
)


@dataclass(frozen=True)
class GraphRuntimeContext:
    """Runtime dependencies kept outside serializable LangGraph state."""

    songs: pd.DataFrame
    retrieval_prompt_config: dict[str, Any]
    llm_client: LLMClient
    tool_context: ToolContext
    tool_runner: ToolRunner
    variant_config: VariantConfig = DEFAULT_VARIANT_CONFIG
    max_revision_count: int = 1


def create_graph_runtime_context(
    *,
    songs: pd.DataFrame,
    retrieval_prompt_config: dict[str, Any],
    llm_client: LLMClient | None = None,
    variant_config: VariantConfig = DEFAULT_VARIANT_CONFIG,
    lyric_index: Any | None = None,
    lyric_embedder: Any | None = None,
    max_revision_count: int = 1,
) -> GraphRuntimeContext:
    """Create graph runtime dependencies for one app or evaluation run."""

    tool_context = ToolContext(
        songs=songs,
        retrieval_prompt_config=retrieval_prompt_config,
        lyric_index=lyric_index,
        lyric_embedder=lyric_embedder,
    )
    return GraphRuntimeContext(
        songs=songs,
        retrieval_prompt_config=retrieval_prompt_config,
        llm_client=llm_client or create_llm_client(),
        tool_context=tool_context,
        tool_runner=ToolRunner(context=tool_context, registry=get_tool_registry()),
        variant_config=variant_config,
        max_revision_count=max_revision_count,
    )


def enabled_tool_names(context: GraphRuntimeContext) -> tuple[str, ...]:
    """Return registry tool names enabled for this runtime variant."""

    names = tuple(context.tool_runner.registry)
    if context.variant_config.use_lyric_retriever:
        return names
    return tuple(name for name in names if name != "lyric_retrieval")


def complete_agent_step(
    *,
    context: GraphRuntimeContext,
    prompt: str,
    schema_name: str,
    state_key: AgentStateKey,
    stage: PipelineStage,
    runtime_preferences: dict[str, Any] | None = None,
) -> PipelineState:
    """Call one LLM agent and return a LangGraph state update."""

    return trace_agent_node(
        stage=stage,
        schema_name=schema_name,
        prompt=prompt,
        context_summary=context_summary(context),
        execute=lambda: _complete_agent_step_untraced(
            context=context,
            prompt=prompt,
            schema_name=schema_name,
            state_key=state_key,
            stage=stage,
            runtime_preferences=runtime_preferences,
        ),
        langsmith_extra=langsmith_extra(
            context=context,
            stage=stage,
            schema_name=schema_name,
            metadata={"component": "agent_node"},
        ),
    )


def _complete_agent_step_untraced(
    *,
    context: GraphRuntimeContext,
    prompt: str,
    schema_name: str,
    state_key: AgentStateKey,
    stage: PipelineStage,
    runtime_preferences: dict[str, Any] | None = None,
) -> PipelineState:
    """Call one LLM agent and return a state update without adding another trace span."""

    try:
        output = context.llm_client.complete_json(prompt=prompt, schema_name=schema_name)
        if schema_name == "RetrievalStrategyOutput" and runtime_preferences:
            output = {**output, "_runtime_preferences": runtime_preferences}
        output = normalize_agent_output_for_runtime(context, schema_name, output)
    except Exception as exc:
        logger.exception("agent_step_failed", stage=stage, schema_name=schema_name)
        update = PipelineState(
            warnings=(f"{stage} failed: {exc}",),
            trace=(
                make_trace_event(
                    stage,
                    "Agent step failed.",
                    {
                        "schema_name": schema_name,
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                ),
            ),
        )
        if stage in {"extract_preferences", "plan_retrieval"}:
            update["fatal_error"] = f"{stage} failed: {exc}"
        return update

    warnings = tuple(output.get("warnings", ()))
    issues = tuple(output.get("issues", ()))
    logger.info(
        "agent_step_finished",
        stage=stage,
        schema_name=schema_name,
        warning_count=len(warnings),
        issue_count=len(issues),
    )
    return PipelineState(
        **{state_key: output},
        warnings=warnings,
        trace=(
            make_trace_event(
                stage,
                "Agent step completed.",
                {
                    "schema_name": schema_name,
                    "warning_count": len(warnings),
                    "issue_count": len(issues),
                },
            ),
        ),
    )


def normalize_agent_output_for_runtime(
    context: GraphRuntimeContext,
    schema_name: str,
    output: dict[str, Any],
) -> dict[str, Any]:
    """Normalize and validate schema-correct LLM output against runtime constraints."""

    normalized = dict(output)
    if schema_name == "PreferenceExtractionOutput":
        normalized = _normalize_preferences_for_runtime(context, normalized)
    elif schema_name == "RetrievalStrategyOutput":
        normalized = _normalize_retrieval_plan_for_runtime(context, normalized)
    elif schema_name == "RevisionOutput" and isinstance(normalized.get("revised_retrieval_plan"), dict):
        normalized["revised_retrieval_plan"] = _normalize_retrieval_plan_for_runtime(
            context,
            normalized["revised_retrieval_plan"],
        )

    if schema_name not in AGENT_OUTPUT_SCHEMAS:
        return normalized
    return AGENT_OUTPUT_SCHEMAS[schema_name].model_validate(normalized).model_dump(mode="json")


def _normalize_preferences_for_runtime(
    context: GraphRuntimeContext,
    output: dict[str, Any],
) -> dict[str, Any]:
    warnings = _runtime_warnings(output)
    normalized = dict(output)
    normalized["categorical_filters"] = _normalize_categorical_filters(
        context,
        normalized.get("categorical_filters", ()),
        warnings,
        owner="preference extraction",
    )
    normalized["feature_range_filters"] = _normalize_feature_constraints(
        context,
        normalized.get("feature_range_filters", ()),
        warnings,
        kind="range_filters",
        owner="preference extraction",
    )
    normalized["feature_targets"] = _normalize_feature_constraints(
        context,
        normalized.get("feature_targets", ()),
        warnings,
        kind="targets",
        owner="preference extraction",
    )
    normalized["warnings"] = warnings
    return normalized


def _normalize_retrieval_plan_for_runtime(
    context: GraphRuntimeContext,
    plan: dict[str, Any],
) -> dict[str, Any]:
    warnings = _runtime_warnings(plan)
    normalized = dict(plan)
    preferences = _preferences_from_plan(normalized)
    normalized.pop("_runtime_preferences", None)
    enabled_modes = set(enabled_tool_names(context))
    modes = [
        mode
        for mode in normalized.get("modes", ())
        if isinstance(mode, str) and mode in {"lyric_retrieval", "metadata_retrieval", "feature_filter"}
    ]

    kept_modes: list[str] = []
    for mode in modes:
        if mode not in enabled_modes:
            warnings.append(f"Removed disabled retrieval mode from plan: {mode}")
            normalized[_request_key_for_mode(mode)] = None
            continue
        request_key = _request_key_for_mode(mode)
        request = normalized.get(request_key)
        if not isinstance(request, dict):
            request = _repair_missing_tool_request(mode, preferences, warnings)
            if request is None:
                warnings.append(f"Removed retrieval mode without a tool-ready request: {mode}")
                continue
            warnings.append(f"Repaired missing tool-ready request for retrieval mode: {mode}")
        request = _normalize_tool_request_for_runtime(context, mode, request, warnings)
        if request is None:
            warnings.append(f"Removed retrieval mode after runtime validation: {mode}")
            normalized[request_key] = None
            continue
        normalized[request_key] = request
        kept_modes.append(mode)

    if not kept_modes and preferences:
        fallback = _fallback_retrieval_plan_from_preferences(
            preferences,
            enabled_modes=enabled_modes,
            warning="Generated deterministic fallback retrieval plan after all LLM-selected modes were invalid.",
        )
        if fallback is not None:
            fallback = _normalize_retrieval_plan_for_runtime(context, fallback)
            fallback_warnings = list(fallback.get("warnings", ()))
            fallback["warnings"] = [*warnings, *fallback_warnings]
            return fallback

    if not kept_modes:
        raise ValueError("Retrieval plan has no enabled modes with valid runtime requests.")

    primary_mode = normalized.get("primary_mode")
    if primary_mode not in kept_modes:
        warnings.append(f"Changed primary_mode to first valid retrieval mode: {kept_modes[0]}")
        normalized["primary_mode"] = kept_modes[0]
    normalized["modes"] = kept_modes
    normalized["warnings"] = warnings
    return normalized


def _preferences_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    preferences = plan.get("_runtime_preferences")
    return preferences if isinstance(preferences, dict) else {}


def _repair_missing_tool_request(
    mode: str,
    preferences: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any] | None:
    if not preferences:
        return None
    if mode == "lyric_retrieval":
        lyric_intent = preferences.get("lyric_intent")
        if isinstance(lyric_intent, str) and lyric_intent.strip():
            request: dict[str, Any] = {"query": lyric_intent.strip(), "top_k": 50}
            language = _preferred_language(preferences)
            if language:
                request["language"] = language
            return request
        return None
    if mode == "metadata_retrieval":
        categorical_filters = list(preferences.get("categorical_filters") or ())
        text_queries = list(preferences.get("text_queries") or ())
        if categorical_filters or text_queries:
            return {
                "categorical_filters": categorical_filters,
                "text_queries": text_queries,
                "top_k": 50,
            }
        return None
    if mode == "feature_filter":
        range_filters = list(preferences.get("feature_range_filters") or ())
        targets = list(preferences.get("feature_targets") or ())
        if range_filters or targets:
            return {"range_filters": range_filters, "targets": targets, "top_k": 50}
        return None
    warnings.append(f"Cannot repair unsupported retrieval mode: {mode}")
    return None


def _fallback_retrieval_plan_from_preferences(
    preferences: dict[str, Any],
    *,
    enabled_modes: set[str],
    warning: str,
) -> dict[str, Any] | None:
    plan: dict[str, Any] = {
        "_runtime_preferences": preferences,
        "scoring_weights": {
            "lyric": 0.4,
            "metadata": 0.25,
            "feature": 0.3,
            "popularity": 0.05,
            "diversity_penalty": 0.05,
        },
        "top_k_final": 10,
        "rationale": "Deterministic fallback plan rebuilt from extracted preferences.",
        "issues": [],
        "warnings": [warning],
    }
    modes: list[str] = []
    for mode in ("lyric_retrieval", "metadata_retrieval", "feature_filter"):
        if mode not in enabled_modes:
            continue
        request = _repair_missing_tool_request(mode, preferences, [])
        if request is None:
            continue
        plan[_request_key_for_mode(mode)] = request
        modes.append(mode)
    if not modes:
        return None
    plan["primary_mode"] = modes[0]
    plan["modes"] = modes
    return plan


def _preferred_language(preferences: dict[str, Any]) -> str | None:
    for item in preferences.get("categorical_filters") or ():
        if not isinstance(item, dict) or item.get("field") != "language":
            continue
        values = item.get("values")
        if isinstance(values, (list, tuple)) and values:
            value = str(values[0]).strip()
            if value:
                return value
    return None


def _normalize_tool_request_for_runtime(
    context: GraphRuntimeContext,
    mode: str,
    request: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any] | None:
    normalized = dict(request)
    if mode == "metadata_retrieval":
        normalized["categorical_filters"] = _normalize_categorical_filters(
            context,
            normalized.get("categorical_filters", ()),
            warnings,
            owner="metadata retrieval request",
        )
        normalized["text_queries"] = _normalize_text_queries(
            context,
            normalized.get("text_queries", ()),
            warnings,
        )
        if not normalized["categorical_filters"] and not normalized["text_queries"]:
            return None
    elif mode == "feature_filter":
        normalized["range_filters"] = _normalize_feature_constraints(
            context,
            normalized.get("range_filters", ()),
            warnings,
            kind="range_filters",
            owner="feature filter request",
        )
        normalized["targets"] = _normalize_feature_constraints(
            context,
            normalized.get("targets", ()),
            warnings,
            kind="targets",
            owner="feature filter request",
        )
        if not normalized["range_filters"] and not normalized["targets"]:
            return None
    elif mode == "lyric_retrieval":
        language = normalized.get("language")
        valid_languages = _categorical_values(context).get("language", ())
        if language is not None and valid_languages and language not in valid_languages:
            warnings.append(f"Removed invalid Lyric RAG language: {language}")
            normalized["language"] = None
    return normalized


def _normalize_categorical_filters(
    context: GraphRuntimeContext,
    filters: Any,
    warnings: list[str],
    *,
    owner: str,
) -> list[dict[str, Any]]:
    valid_values = _categorical_values(context)
    if not isinstance(filters, (list, tuple)):
        return []

    kept_filters: list[dict[str, Any]] = []
    for item in filters:
        if not isinstance(item, dict):
            warnings.append(f"Dropped malformed categorical filter in {owner}.")
            continue
        field = item.get("field")
        values = item.get("values")
        allowed = valid_values.get(field)
        if not isinstance(field, str) or allowed is None:
            warnings.append(f"Dropped unsupported categorical field in {owner}: {field}")
            continue
        if not isinstance(values, (list, tuple)):
            warnings.append(f"Dropped categorical filter without values in {owner}: {field}")
            continue
        cleaned_values = [str(value) for value in values if str(value) in allowed]
        dropped = [str(value) for value in values if str(value) not in allowed]
        if dropped:
            warnings.append(f"Dropped invalid categorical values for {field}: {', '.join(dropped)}")
        if not cleaned_values:
            continue
        kept = dict(item)
        kept["values"] = cleaned_values
        kept_filters.append(kept)
    return kept_filters


def _normalize_text_queries(
    context: GraphRuntimeContext,
    queries: Any,
    warnings: list[str],
) -> list[dict[str, Any]]:
    full_text_columns = set(
        context.retrieval_prompt_config.get("llm_prompt_constraints", {})
        .get("full_text_search_columns", {})
    )
    if not isinstance(queries, (list, tuple)):
        return []
    if not full_text_columns:
        return list(queries)

    kept_queries: list[dict[str, Any]] = []
    for item in queries:
        if not isinstance(item, dict):
            warnings.append("Dropped malformed metadata text query.")
            continue
        field = item.get("field")
        query = item.get("query")
        if field not in full_text_columns:
            warnings.append(f"Dropped unsupported full-text metadata field: {field}")
            continue
        if not isinstance(query, str) or not query.strip():
            warnings.append(f"Dropped blank full-text metadata query for field: {field}")
            continue
        kept_queries.append(item)
    return kept_queries


def _normalize_feature_constraints(
    context: GraphRuntimeContext,
    constraints: Any,
    warnings: list[str],
    *,
    kind: str,
    owner: str,
) -> list[dict[str, Any]]:
    numeric_ranges = (
        context.retrieval_prompt_config.get("llm_prompt_constraints", {}).get("numeric_ranges", {})
    )
    if not isinstance(constraints, (list, tuple)):
        return []
    if not numeric_ranges:
        return list(constraints)

    kept_constraints: list[dict[str, Any]] = []
    for item in constraints:
        if not isinstance(item, dict):
            warnings.append(f"Dropped malformed feature constraint in {owner}.")
            continue
        feature = item.get("feature")
        feature_range = numeric_ranges.get(feature)
        if not isinstance(feature, str) or feature_range is None:
            warnings.append(f"Dropped unsupported numeric feature in {owner}: {feature}")
            continue
        kept = dict(item)
        if kind == "range_filters":
            kept = _clamp_range_filter(kept, feature_range, warnings)
            if kept is None:
                continue
        kept_constraints.append(kept)
    return kept_constraints


def _clamp_range_filter(
    item: dict[str, Any],
    feature_range: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any] | None:
    lower = feature_range.get("min")
    upper = feature_range.get("max")
    min_value = item.get("min_value")
    max_value = item.get("max_value")
    if isinstance(min_value, (int, float)) and isinstance(lower, (int, float)) and min_value < lower:
        warnings.append(f"Clamped min_value for {item.get('feature')} to configured range.")
        item["min_value"] = lower
    if isinstance(max_value, (int, float)) and isinstance(upper, (int, float)) and max_value > upper:
        warnings.append(f"Clamped max_value for {item.get('feature')} to configured range.")
        item["max_value"] = upper
    if (
        item.get("min_value") is not None
        and item.get("max_value") is not None
        and item["min_value"] > item["max_value"]
    ):
        warnings.append(f"Dropped impossible range filter for {item.get('feature')}.")
        return None
    return item


def _categorical_values(context: GraphRuntimeContext) -> dict[str, list[str]]:
    raw_values = (
        context.retrieval_prompt_config.get("llm_prompt_constraints", {}).get("categorical_values", {})
    )
    return {
        str(field): [str(value) for value in values]
        for field, values in raw_values.items()
        if isinstance(values, list)
    }


def _runtime_warnings(output: dict[str, Any]) -> list[str]:
    warnings = output.get("warnings", ())
    if isinstance(warnings, list):
        return list(warnings)
    if isinstance(warnings, tuple):
        return list(warnings)
    return []


def _request_key_for_mode(mode: str) -> str:
    return {
        "lyric_retrieval": "lyric_request",
        "metadata_retrieval": "metadata_request",
        "feature_filter": "feature_request",
    }[mode]


def run_tool_step(
    *,
    context: GraphRuntimeContext,
    tool_name: str,
    raw_input: dict[str, Any],
    stage: PipelineStage,
) -> PipelineState:
    """Run one deterministic tool and return a LangGraph state update."""

    return trace_tool_call(
        stage=stage,
        tool_name=tool_name,
        raw_input=raw_input,
        context_summary=context_summary(context),
        execute=lambda: _run_tool_step_untraced(
            context=context,
            tool_name=tool_name,
            raw_input=raw_input,
            stage=stage,
        ),
        langsmith_extra=langsmith_extra(
            context=context,
            stage=stage,
            tool_name=tool_name,
            metadata={"component": tool_component(tool_name)},
        ),
    )


def _run_tool_step_untraced(
    *,
    context: GraphRuntimeContext,
    tool_name: str,
    raw_input: dict[str, Any],
    stage: PipelineStage,
) -> PipelineState:
    """Run one deterministic tool and return a state update without adding another trace span."""

    if tool_name not in enabled_tool_names(context):
        warning = f"Tool is disabled by variant config: {tool_name}"
        logger.warning(
            "tool_step_disabled",
            tool_name=tool_name,
            variant_name=context.variant_config.name,
        )
        return PipelineState(
            warnings=(warning,),
            trace=(
                make_trace_event(
                    stage,
                    "Tool step skipped.",
                    {"tool_name": tool_name, "reason": "disabled_by_variant"},
                ),
            ),
        )

    result = context.tool_runner.run(tool_name, raw_input)
    if isinstance(result, ToolError):
        return _tool_error_update(stage, result)
    return _tool_success_update(stage, result)


def _tool_success_update(stage: PipelineStage, result: ToolResult) -> PipelineState:
    output = result.output
    candidates = tuple(output.get("candidates", ())) if isinstance(output, dict) else ()
    candidate_ids = _candidate_ids(candidates)
    update: PipelineState = PipelineState(
        tool_outputs={result.tool_name: output},
        warnings=result.warnings,
        trace=(
            make_trace_event(
                stage,
                "Tool step completed.",
                {
                    "tool_name": result.tool_name,
                    "candidate_count": len(candidates),
                    "warning_count": len(result.warnings),
                    **result.trace,
                },
            ),
        ),
    )
    if result.tool_name in RETRIEVAL_TOOL_NAMES:
        update["retrieval_modes_used"] = (result.tool_name,)
    if candidate_ids:
        update["candidate_ids"] = candidate_ids
    if result.tool_name == "candidate_scoring":
        update["scored_candidates"] = candidates
    if result.tool_name == "reliability":
        update["reliability"] = output
    return update


def _tool_error_update(stage: PipelineStage, result: ToolError) -> PipelineState:
    warning = f"{result.tool_name} failed: {result.message}"
    logger.warning(
        "tool_step_failed",
        tool_name=result.tool_name,
        error_type=result.error_type,
        message=result.message,
    )
    return PipelineState(
        tool_errors=(result.model_dump(mode="json"),),
        warnings=(warning,),
        trace=(
            make_trace_event(
                stage,
                "Tool step failed.",
                {
                    "tool_name": result.tool_name,
                    "error_type": result.error_type,
                    "message": result.message,
                    **result.trace,
                },
            ),
        ),
    )


def _candidate_ids(candidates: tuple[Any, ...]) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        track_id = candidate.get("track_id") if isinstance(candidate, dict) else None
        if not isinstance(track_id, str):
            continue
        cleaned = track_id.strip()
        if cleaned and cleaned not in seen:
            ids.append(cleaned)
            seen.add(cleaned)
    return tuple(ids)
