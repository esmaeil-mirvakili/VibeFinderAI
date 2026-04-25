"""LangSmith tracing helpers for VibeFinder workflow instrumentation."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - dependency is expected, fallback keeps app importable.
    traceable = None  # type: ignore[assignment]


R = TypeVar("R")


def trace_metadata(
    *,
    context: Any,
    run_id: str | None = None,
    stage: str | None = None,
    tool_name: str | None = None,
    schema_name: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build LangSmith metadata shared by graph, agent, LLM, and tool spans."""

    llm_client = getattr(context, "llm_client", None)
    variant = getattr(context, "variant_config", None)
    provider = _clean_scalar(getattr(llm_client, "provider", None)) or llm_client.__class__.__name__
    model = _clean_scalar(getattr(llm_client, "model", None)) or "unknown"
    metadata: dict[str, Any] = {
        "application": "vibefinder-ai",
        "run_id": run_id,
        "stage": stage,
        "tool_name": tool_name,
        "schema_name": schema_name,
        "variant_name": _clean_scalar(getattr(variant, "name", None)),
        "use_multi_step_reasoning": getattr(variant, "use_multi_step_reasoning", None),
        "use_critic_revision": getattr(variant, "use_critic_revision", None),
        "use_lyric_retriever": getattr(variant, "use_lyric_retriever", None),
        "max_revision_count": getattr(context, "max_revision_count", None),
        "llm_provider": provider,
        "llm_model": model,
        "ls_provider": provider,
        "ls_model_name": model,
        "ls_model_type": "chat",
    }
    if extra:
        metadata.update(dict(extra))
    return {key: value for key, value in metadata.items() if value is not None}


def trace_tags(
    *,
    context: Any,
    stage: str | None = None,
    tool_name: str | None = None,
    schema_name: str | None = None,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """Build filterable LangSmith tags for run, model backend, variant, and stage."""

    llm_client = getattr(context, "llm_client", None)
    variant = getattr(context, "variant_config", None)
    tags = [
        "vibefinder",
        f"variant:{getattr(variant, 'name', 'unknown')}",
        f"backend:{getattr(llm_client, 'provider', llm_client.__class__.__name__)}",
        f"model:{getattr(llm_client, 'model', 'unknown')}",
    ]
    if stage:
        tags.append(f"stage:{stage}")
    if tool_name:
        tags.append(f"tool:{tool_name}")
        tags.append(f"component:{tool_component(tool_name)}")
    if schema_name:
        tags.append(f"schema:{schema_name}")
    tags.extend(extra)
    return [tag for tag in tags if tag and not tag.endswith(":None")]


def langsmith_extra(
    *,
    context: Any,
    run_id: str | None = None,
    stage: str | None = None,
    tool_name: str | None = None,
    schema_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return invocation-level LangSmith metadata/tags/project config."""

    extra: dict[str, Any] = {
        "metadata": trace_metadata(
            context=context,
            run_id=run_id,
            stage=stage,
            tool_name=tool_name,
            schema_name=schema_name,
            extra=metadata,
        ),
        "tags": trace_tags(
            context=context,
            stage=stage,
            tool_name=tool_name,
            schema_name=schema_name,
            extra=tags,
        ),
    }
    project_name = os.getenv("VIBEFINDER_LANGSMITH_PROJECT") or os.getenv("LANGSMITH_PROJECT")
    if project_name:
        extra["project_name"] = project_name
    return extra


def graph_invoke_config(*, context: Any, run_id: str) -> dict[str, Any]:
    """Build LangGraph invoke config so graph-level traces are tagged in LangSmith."""

    metadata = trace_metadata(context=context, run_id=run_id, stage="graph_execution")
    tags = trace_tags(context=context, stage="graph_execution")
    return {
        "metadata": metadata,
        "tags": tags,
        "run_name": f"vibefinder:{metadata.get('variant_name', 'unknown')}:{metadata.get('llm_provider', 'unknown')}",
    }


def tool_component(tool_name: str) -> str:
    """Map tool names to inspectable tracing component labels."""

    return {
        "lyric_retrieval": "faiss_lyric_retrieval",
        "metadata_retrieval": "pandas_metadata_retrieval",
        "feature_filter": "pandas_feature_filter",
        "candidate_scoring": "deterministic_ranking",
        "reliability": "deterministic_reliability",
    }.get(tool_name, "deterministic_tool")


def trace_recommendation_run(
    *,
    query: str,
    context_summary: dict[str, Any],
    execute: Callable[[], R],
    langsmith_extra: dict[str, Any],
) -> R:
    """Trace one top-level recommendation run."""

    return _trace_recommendation_run(
        query=query,
        context_summary=context_summary,
        execute=execute,
        langsmith_extra=langsmith_extra,
    )


def trace_agent_node(
    *,
    stage: str,
    schema_name: str,
    prompt: str,
    context_summary: dict[str, Any],
    execute: Callable[[], R],
    langsmith_extra: dict[str, Any],
) -> R:
    """Trace one LLM-backed agent node."""

    return _trace_agent_node(
        stage=stage,
        schema_name=schema_name,
        prompt=prompt,
        context_summary=context_summary,
        execute=execute,
        langsmith_extra=langsmith_extra,
    )


def trace_llm_json_call(
    *,
    provider: str,
    model: str,
    schema_name: str,
    prompt: str,
    execute: Callable[[], dict[str, Any]],
    langsmith_extra: dict[str, Any],
) -> dict[str, Any]:
    """Trace one provider-agnostic structured LLM call."""

    return _trace_llm_json_call(
        provider=provider,
        model=model,
        schema_name=schema_name,
        prompt=prompt,
        execute=execute,
        langsmith_extra=langsmith_extra,
    )


def trace_tool_call(
    *,
    stage: str,
    tool_name: str,
    raw_input: dict[str, Any],
    context_summary: dict[str, Any],
    execute: Callable[[], R],
    langsmith_extra: dict[str, Any],
) -> R:
    """Trace one deterministic tool call."""

    return _trace_tool_call(
        stage=stage,
        tool_name=tool_name,
        raw_input=raw_input,
        context_summary=context_summary,
        execute=execute,
        langsmith_extra=langsmith_extra,
    )


def trace_route_decision(
    *,
    route_name: str,
    route: str,
    details: dict[str, Any],
    langsmith_extra: dict[str, Any],
) -> str:
    """Trace a graph route or retry decision."""

    return _trace_route_decision(
        route_name=route_name,
        route=route,
        details=details,
        langsmith_extra=langsmith_extra,
    )


def context_summary(context: Any) -> dict[str, Any]:
    """Return a small serializable summary of runtime context for traces."""

    songs = getattr(context, "songs", None)
    variant = getattr(context, "variant_config", None)
    llm_client = getattr(context, "llm_client", None)
    return {
        "row_count": len(songs) if songs is not None else None,
        "column_count": len(getattr(songs, "columns", ())) if songs is not None else None,
        "variant_name": getattr(variant, "name", None),
        "llm_provider": getattr(llm_client, "provider", llm_client.__class__.__name__),
        "llm_model": getattr(llm_client, "model", None),
        "max_revision_count": getattr(context, "max_revision_count", None),
    }


def _traceable_or_passthrough(*, run_type: str, name: str, process_inputs: Callable, process_outputs: Callable):
    if traceable is None:
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            return func

        return decorator
    return traceable(
        run_type=run_type,
        name=name,
        process_inputs=process_inputs,
        process_outputs=process_outputs,
    )


@_traceable_or_passthrough(
    run_type="chain",
    name="VibeFinder Recommendation Run",
    process_inputs=lambda inputs: {
        "query": inputs.get("query"),
        "context_summary": inputs.get("context_summary"),
    },
    process_outputs=lambda output: _summarize_final_state(output),
)
def _trace_recommendation_run(
    *,
    query: str,
    context_summary: dict[str, Any],
    execute: Callable[[], R],
) -> R:
    return execute()


@_traceable_or_passthrough(
    run_type="chain",
    name="VibeFinder Agent Node",
    process_inputs=lambda inputs: {
        "stage": inputs.get("stage"),
        "schema_name": inputs.get("schema_name"),
        "prompt": inputs.get("prompt"),
        "context_summary": inputs.get("context_summary"),
    },
    process_outputs=lambda output: _summarize_agent_output(output),
)
def _trace_agent_node(
    *,
    stage: str,
    schema_name: str,
    prompt: str,
    context_summary: dict[str, Any],
    execute: Callable[[], R],
) -> R:
    return execute()


@_traceable_or_passthrough(
    run_type="llm",
    name="VibeFinder LLM JSON Call",
    process_inputs=lambda inputs: {
        "provider": inputs.get("provider"),
        "model": inputs.get("model"),
        "schema_name": inputs.get("schema_name"),
        "prompt": inputs.get("prompt"),
        "messages": [
            {
                "role": "user",
                "content": inputs.get("prompt"),
            }
        ],
    },
    process_outputs=lambda output: _summarize_llm_output(output),
)
def _trace_llm_json_call(
    *,
    provider: str,
    model: str,
    schema_name: str,
    prompt: str,
    execute: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    return execute()


@_traceable_or_passthrough(
    run_type="tool",
    name="VibeFinder Tool Call",
    process_inputs=lambda inputs: {
        "stage": inputs.get("stage"),
        "tool_name": inputs.get("tool_name"),
        "component": tool_component(str(inputs.get("tool_name", ""))),
        "raw_input": inputs.get("raw_input"),
        "context_summary": inputs.get("context_summary"),
    },
    process_outputs=lambda output: _summarize_tool_output(output),
)
def _trace_tool_call(
    *,
    stage: str,
    tool_name: str,
    raw_input: dict[str, Any],
    context_summary: dict[str, Any],
    execute: Callable[[], R],
) -> R:
    return execute()


@_traceable_or_passthrough(
    run_type="chain",
    name="VibeFinder Route Decision",
    process_inputs=lambda inputs: {
        "route_name": inputs.get("route_name"),
        "details": inputs.get("details"),
    },
    process_outputs=lambda output: {"route": output},
)
def _trace_route_decision(
    *,
    route_name: str,
    route: str,
    details: dict[str, Any],
) -> str:
    return route


def _summarize_final_state(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {"output": _jsonable(output)}
    reliability = output.get("reliability") if isinstance(output.get("reliability"), dict) else {}
    return {
        "variant_name": output.get("variant_name"),
        "candidate_count": len(output.get("candidate_ids", ())),
        "scored_count": len(output.get("scored_candidates", ())),
        "verified_count": len(output.get("verified_candidates", ())),
        "explanation_count": len(output.get("explanations", ())),
        "revision_count": output.get("revision_count"),
        "confidence_label": reliability.get("confidence_label"),
        "confidence_score": reliability.get("confidence_score"),
        "scored_candidates": output.get("scored_candidates", ()),
        "explanations": output.get("explanations", ()),
        "warnings": output.get("warnings", ()),
        "trace": output.get("trace", ()),
    }


def _summarize_agent_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {"output": _jsonable(output)}
    trace_events = output.get("trace", ())
    return {
        "state_keys": sorted(output.keys()),
        "warnings": output.get("warnings", ()),
        "trace": trace_events,
        "preferences": output.get("preferences"),
        "retrieval_plan": output.get("retrieval_plan"),
        "verified_candidates": output.get("verified_candidates"),
        "critique": output.get("critique"),
        "revision_plan": output.get("revision_plan"),
        "explanations": output.get("explanations"),
    }


def _summarize_llm_output(output: Any) -> dict[str, Any]:
    """Return LangSmith-friendly LLM output with raw text and structured JSON."""

    if not isinstance(output, dict):
        text = str(output)
        return {
            "raw_response": text,
            "response": text,
            "generations": [[{"text": text, "message": {"role": "assistant", "content": text}}]],
        }

    raw_response = output.get("raw_response")
    parsed_response = output.get("parsed_response")
    validated_output = output.get("validated_output")
    raw_text = "" if raw_response is None else str(raw_response)
    return {
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "validated_output": validated_output,
        "response": validated_output,
        "generations": [
            [
                {
                    "text": raw_text,
                    "message": {
                        "role": "assistant",
                        "content": raw_text,
                    },
                }
            ]
        ],
    }


def _summarize_tool_output(output: Any) -> dict[str, Any]:
    if isinstance(output, BaseModel):
        output = output.model_dump(mode="json")
    if not isinstance(output, dict):
        return {"output": _jsonable(output)}
    if "tool_outputs" in output or "tool_errors" in output:
        tool_outputs = output.get("tool_outputs", {})
        tool_name = next(iter(tool_outputs), None) if isinstance(tool_outputs, dict) else None
        tool_output = tool_outputs.get(tool_name, {}) if tool_name else {}
        candidates = tool_output.get("candidates", ()) if isinstance(tool_output, dict) else ()
        errors = output.get("tool_errors", ())
        return {
            "ok": not bool(errors),
            "tool_name": tool_name,
            "error_count": len(errors),
            "candidate_count": len(candidates),
            "input_count": tool_output.get("input_count") if isinstance(tool_output, dict) else None,
            "output_count": tool_output.get("output_count") if isinstance(tool_output, dict) else None,
            "candidate_ids": output.get("candidate_ids", ()),
            "retrieval_modes_used": output.get("retrieval_modes_used", ()),
            "warnings": output.get("warnings", ()),
            "trace": output.get("trace", ()),
            "tool_output": tool_output,
            "tool_errors": errors,
            "scored_candidates": output.get("scored_candidates", ()),
            "reliability": output.get("reliability"),
        }
    tool_output = output.get("output") if isinstance(output.get("output"), dict) else {}
    candidates = tool_output.get("candidates", ()) if isinstance(tool_output, dict) else ()
    return {
        "ok": output.get("ok"),
        "tool_name": output.get("tool_name"),
        "error_type": output.get("error_type"),
        "message": output.get("message"),
        "candidate_count": len(candidates),
        "input_count": tool_output.get("input_count") if isinstance(tool_output, dict) else None,
        "output_count": tool_output.get("output_count") if isinstance(tool_output, dict) else None,
        "warnings": output.get("warnings", ()),
        "trace": output.get("trace", {}),
        "output": tool_output,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
