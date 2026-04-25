"""LangGraph assembly for the recommendation pipeline."""

from __future__ import annotations

from uuid import uuid4
from typing import Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from vibefinder.agent_nodes import (
    critique_results_node,
    explain_recommendations_node,
    extract_preferences_node,
    plan_retrieval_node,
    revise_plan_node,
    verify_candidates_node,
)
from vibefinder.graph_runtime import GraphRuntimeContext, RETRIEVAL_TOOL_NAMES, run_tool_step
from vibefinder.pipeline_state import (
    PipelineState,
    append_tuples,
    create_initial_pipeline_state,
    make_trace_event,
    merge_dicts,
    merge_unique_strings,
    replace_tuple,
)
from vibefinder.tracing import (
    context_summary,
    graph_invoke_config,
    langsmith_extra,
    trace_recommendation_run,
    trace_route_decision,
)
from vibefinder.tools.schemas import DEFAULT_FINAL_TOP_K


def build_recommendation_graph(context: GraphRuntimeContext) -> Any:
    """Build the executable LangGraph recommendation pipeline."""

    graph = StateGraph(PipelineState)
    graph.add_node("extract_preferences", lambda state: extract_preferences_node(state, context))
    graph.add_node("plan_retrieval", lambda state: plan_retrieval_node(state, context))
    graph.add_node("retrieve_candidates", lambda state: retrieve_candidates_node(state, context))
    graph.add_node("rank_candidates", lambda state: rank_candidates_node(state, context))
    graph.add_node("verify_candidates", lambda state: verify_candidates_node(state, context))
    graph.add_node("critique_results", lambda state: critique_results_node(state, context))
    graph.add_node("revise_plan", lambda state: revise_plan_node(state, context))
    graph.add_node("build_reliability_report", lambda state: build_reliability_node(state, context))
    graph.add_node("explain_recommendations", lambda state: explain_recommendations_node(state, context))
    graph.add_node("finish", finish_node)

    graph.add_edge(START, "extract_preferences")
    graph.add_conditional_edges(
        "extract_preferences",
        lambda state: _route_after_extract_preferences(state, context),
        {
            "plan_retrieval": "plan_retrieval",
            "finish": "finish",
        },
    )
    graph.add_conditional_edges(
        "plan_retrieval",
        lambda state: _route_after_plan_retrieval(state, context),
        {
            "retrieve_candidates": "retrieve_candidates",
            "finish": "finish",
        },
    )
    graph.add_edge("retrieve_candidates", "rank_candidates")
    graph.add_conditional_edges(
        "rank_candidates",
        lambda state: _route_after_ranking(context),
        {
            "verify_candidates": "verify_candidates",
            "build_reliability_report": "build_reliability_report",
        },
    )
    graph.add_edge("verify_candidates", "critique_results")
    graph.add_conditional_edges(
        "critique_results",
        lambda state: _route_after_critique(state, context),
        {
            "revise_plan": "revise_plan",
            "build_reliability_report": "build_reliability_report",
        },
    )
    graph.add_conditional_edges(
        "revise_plan",
        lambda state: _route_after_revision(state, context),
        {
            "retrieve_candidates": "retrieve_candidates",
            "build_reliability_report": "build_reliability_report",
        },
    )
    graph.add_conditional_edges(
        "build_reliability_report",
        lambda state: _route_after_reliability(context),
        {
            "explain_recommendations": "explain_recommendations",
            "finish": "finish",
        },
    )
    graph.add_edge("explain_recommendations", "finish")
    graph.add_edge("finish", END)

    return graph.compile()


def run_recommendation(query: str, context: GraphRuntimeContext) -> PipelineState:
    """Run the compiled recommendation graph for one query and return final state."""

    run_id = uuid4().hex
    initial_state = create_initial_pipeline_state(query, variant_name=context.variant_config.name)
    graph = build_recommendation_graph(context)
    return trace_recommendation_run(
        query=query,
        context_summary=context_summary(context),
        execute=lambda: graph.invoke(
            initial_state,
            config=graph_invoke_config(context=context, run_id=run_id),
        ),
        langsmith_extra=langsmith_extra(
            context=context,
            run_id=run_id,
            stage="recommendation_run",
            metadata={
                "component": "langgraph_workflow",
                "query_length": len(query.strip()),
            },
        ),
    )


def retrieve_candidates_node(state: PipelineState, context: GraphRuntimeContext) -> PipelineState:
    """Run retrieval tools requested by the retrieval strategy plan."""

    retrieval_plan = state.get("retrieval_plan")
    if not isinstance(retrieval_plan, dict):
        return _missing_state_update("retrieve_candidates", "Missing required state object: retrieval_plan")

    modes = tuple(retrieval_plan.get("modes", ()))
    if not modes:
        return _missing_state_update("retrieve_candidates", "Retrieval plan did not select any modes.")

    aggregate = PipelineState(
        candidate_ids=(),
        retrieval_modes_used=(),
        tool_outputs={tool_name: None for tool_name in RETRIEVAL_TOOL_NAMES},
    )
    for mode in modes:
        request_key = _request_key_for_mode(mode)
        if request_key is None:
            aggregate = _merge_state_updates(
                aggregate,
                _missing_state_update("retrieve_candidates", f"Unsupported retrieval mode: {mode}"),
            )
            continue

        raw_input = retrieval_plan.get(request_key)
        if not isinstance(raw_input, dict):
            aggregate = _merge_state_updates(
                aggregate,
                _missing_state_update(
                    "retrieve_candidates",
                    f"Missing tool request for retrieval mode: {mode}",
                ),
            )
            continue

        aggregate = _merge_state_updates(
            aggregate,
            run_tool_step(
                context=context,
                tool_name=mode,
                raw_input=raw_input,
                stage="retrieve_candidates",
            ),
        )

    logger.info(
        "retrieve_candidates_node_finished",
        modes=modes,
        candidate_count=len(aggregate.get("candidate_ids", ())),
        warning_count=len(aggregate.get("warnings", ())),
    )
    return aggregate


def rank_candidates_node(state: PipelineState, context: GraphRuntimeContext) -> PipelineState:
    """Run deterministic candidate scoring from retrieval tool evidence."""

    raw_input = _candidate_scoring_input(state)
    if raw_input is None:
        update = _missing_state_update("rank_candidates", "No retrieval candidates were available for scoring.")
        update["scored_candidates"] = ()
        return update
    return run_tool_step(
        context=context,
        tool_name="candidate_scoring",
        raw_input=raw_input,
        stage="rank_candidates",
    )


def build_reliability_node(state: PipelineState, context: GraphRuntimeContext) -> PipelineState:
    """Run deterministic reliability assessment from final scored candidates."""

    scored_candidates = tuple(state.get("scored_candidates", ()))
    if not scored_candidates:
        raw_candidates: list[dict[str, Any]] = []
    else:
        verification_lookup = _verification_lookup(state)
        raw_candidates = [
            _reliability_candidate(candidate, verification_lookup)
            for candidate in scored_candidates
            if isinstance(candidate, dict) and candidate.get("track_id")
        ]

    retrieval_plan = state.get("retrieval_plan") or {}
    raw_input = {
        "final_candidates": raw_candidates,
        "requested_count": int(retrieval_plan.get("top_k_final", DEFAULT_FINAL_TOP_K))
        if isinstance(retrieval_plan, dict)
        else DEFAULT_FINAL_TOP_K,
        "retrieval_modes_used": list(state.get("retrieval_modes_used", ())),
        "prior_warnings": list(state.get("warnings", ())),
        "verifier_warnings": _verifier_warnings(state),
        "critic_issues": _critic_issues(state),
        "hard_constraint_violations": _hard_constraint_violations(state),
        "revision_used": int(state.get("revision_count", 0)) > 0,
        "revision_succeeded": _revision_succeeded(state),
    }
    return run_tool_step(
        context=context,
        tool_name="reliability",
        raw_input=raw_input,
        stage="build_reliability_report",
    )


def finish_node(state: PipelineState) -> PipelineState:
    """Append a final trace event."""

    return PipelineState(
        trace=(
            make_trace_event(
                "finished",
                "Recommendation graph finished.",
                {
                    "candidate_count": len(state.get("candidate_ids", ())),
                    "scored_count": len(state.get("scored_candidates", ())),
                    "verified_count": len(state.get("verified_candidates", ())),
                    "explanation_count": len(state.get("explanations", ())),
                    "warning_count": len(state.get("warnings", ())),
                    "confidence": (state.get("reliability") or {}).get("confidence_label")
                    if isinstance(state.get("reliability"), dict)
                    else None,
                },
            ),
        )
    )


def _route_after_critique(state: PipelineState, context: GraphRuntimeContext) -> str:
    critique = state.get("critique")
    if not context.variant_config.use_critic_revision:
        logger.info(
            "graph_revision_route",
            route="build_reliability_report",
            reason="critic_revision_disabled",
        )
        return _trace_route(
            context=context,
            route_name="after_critique",
            route="build_reliability_report",
            details={"reason": "critic_revision_disabled"},
        )
    if int(state.get("revision_count", 0)) >= context.max_revision_count:
        logger.info(
            "graph_revision_route",
            route="build_reliability_report",
            reason="max_revision_count_reached",
            revision_count=int(state.get("revision_count", 0)),
            max_revision_count=context.max_revision_count,
        )
        return _trace_route(
            context=context,
            route_name="after_critique",
            route="build_reliability_report",
            details={
                "reason": "max_revision_count_reached",
                "revision_count": int(state.get("revision_count", 0)),
                "max_revision_count": context.max_revision_count,
            },
        )
    if isinstance(critique, dict) and critique.get("should_revise") is True:
        logger.info(
            "graph_revision_route",
            route="revise_plan",
            revision_count=int(state.get("revision_count", 0)),
        )
    deterministic_trigger = _deterministic_revision_trigger(state)
    if deterministic_trigger is not None:
        logger.info(
            "graph_revision_route",
            route="revise_plan",
            reason=deterministic_trigger,
            revision_count=int(state.get("revision_count", 0)),
        )
        return _trace_route(
            context=context,
            route_name="after_critique",
            route="revise_plan",
            details={
                "reason": deterministic_trigger,
                "revision_count": int(state.get("revision_count", 0)),
            },
        )
        return _trace_route(
            context=context,
            route_name="after_critique",
            route="revise_plan",
            details={
                "reason": "critique_requested_revision",
                "revision_count": int(state.get("revision_count", 0)),
                "critic_issue_count": len(critique.get("issues", ())),
            },
        )
    logger.info(
        "graph_revision_route",
        route="build_reliability_report",
        reason="critique_did_not_request_revision",
    )
    return _trace_route(
        context=context,
        route_name="after_critique",
        route="build_reliability_report",
        details={"reason": "critique_did_not_request_revision"},
    )


def _deterministic_revision_trigger(state: PipelineState) -> str | None:
    if not state.get("scored_candidates"):
        return "no_candidates"
    preferences = state.get("preferences")
    if isinstance(preferences, dict) and preferences.get("lyric_intent"):
        scored = [item for item in state.get("scored_candidates", ()) if isinstance(item, dict)]
        if scored:
            lyric_count = sum(
                1
                for item in scored[:10]
                if "lyric_retrieval" in tuple(item.get("evidence_sources", ()))
            )
            if lyric_count / min(10, len(scored)) < 0.3:
                return "low_lyric_evidence"
    if _hard_constraint_violations(state):
        return "hard_constraint_violations"
    return None


def _route_after_extract_preferences(state: PipelineState, context: GraphRuntimeContext) -> str:
    if isinstance(state.get("preferences"), dict) and not state.get("fatal_error"):
        return _trace_route(
            context=context,
            route_name="after_extract_preferences",
            route="plan_retrieval",
            details={"reason": "preferences_available"},
        )
    return _trace_route(
        context=context,
        route_name="after_extract_preferences",
        route="finish",
        details={"reason": "missing_required_preferences"},
    )


def _route_after_plan_retrieval(state: PipelineState, context: GraphRuntimeContext) -> str:
    if isinstance(state.get("retrieval_plan"), dict) and not state.get("fatal_error"):
        return _trace_route(
            context=context,
            route_name="after_plan_retrieval",
            route="retrieve_candidates",
            details={"reason": "retrieval_plan_available"},
        )
    return _trace_route(
        context=context,
        route_name="after_plan_retrieval",
        route="finish",
        details={"reason": "missing_required_retrieval_plan"},
    )


def _route_after_ranking(context: GraphRuntimeContext) -> str:
    if context.variant_config.use_multi_step_reasoning:
        return _trace_route(
            context=context,
            route_name="after_ranking",
            route="verify_candidates",
            details={"reason": "multi_step_reasoning_enabled"},
        )
    logger.info(
        "graph_multi_step_route",
        route="build_reliability_report",
        reason="multi_step_reasoning_disabled",
    )
    return _trace_route(
        context=context,
        route_name="after_ranking",
        route="build_reliability_report",
        details={"reason": "multi_step_reasoning_disabled"},
    )


def _route_after_reliability(context: GraphRuntimeContext) -> str:
    if context.variant_config.use_multi_step_reasoning:
        return _trace_route(
            context=context,
            route_name="after_reliability",
            route="explain_recommendations",
            details={"reason": "multi_step_reasoning_enabled"},
        )
    logger.info(
        "graph_multi_step_route",
        route="finish",
        reason="multi_step_reasoning_disabled",
    )
    return _trace_route(
        context=context,
        route_name="after_reliability",
        route="finish",
        details={"reason": "multi_step_reasoning_disabled"},
    )


def _route_after_revision(state: PipelineState, context: GraphRuntimeContext) -> str:
    revision_plan = state.get("revision_plan")
    if isinstance(revision_plan, dict) and revision_plan.get("should_retry") is True:
        return _trace_route(
            context=context,
            route_name="after_revision",
            route="retrieve_candidates",
            details={
                "reason": "revision_requested_retry",
                "revision_count": int(state.get("revision_count", 0)),
            },
        )
    return _trace_route(
        context=context,
        route_name="after_revision",
        route="build_reliability_report",
        details={
            "reason": "revision_did_not_retry",
            "revision_count": int(state.get("revision_count", 0)),
        },
    )


def _trace_route(
    *,
    context: GraphRuntimeContext,
    route_name: str,
    route: str,
    details: dict[str, Any],
) -> str:
    return trace_route_decision(
        route_name=route_name,
        route=route,
        details=details,
        langsmith_extra=langsmith_extra(
            context=context,
            stage=route_name,
            metadata={"component": "graph_routing", **details},
            tags=("route",),
        ),
    )


def _candidate_scoring_input(state: PipelineState) -> dict[str, Any] | None:
    evidence = _retrieval_evidence(state)
    candidate_ids = _candidate_ids_from_evidence(evidence) or tuple(state.get("candidate_ids", ()))
    if not candidate_ids and not any(evidence.values()):
        return None

    retrieval_plan = state.get("retrieval_plan") or {}
    raw_input: dict[str, Any] = {
        "candidate_track_ids": list(candidate_ids) if candidate_ids else None,
        "lyric_evidence": evidence["lyric_evidence"],
        "metadata_evidence": evidence["metadata_evidence"],
        "feature_evidence": evidence["feature_evidence"],
        "top_k": int(retrieval_plan.get("top_k_final", DEFAULT_FINAL_TOP_K))
        if isinstance(retrieval_plan, dict)
        else DEFAULT_FINAL_TOP_K,
    }
    preferences = state.get("preferences")
    if (
        isinstance(preferences, dict)
        and isinstance(preferences.get("lyric_intent"), str)
        and "lyric_retrieval" in _active_retrieval_modes(state)
    ):
        raw_input["lyric_intent"] = preferences["lyric_intent"]
    if isinstance(retrieval_plan, dict) and isinstance(retrieval_plan.get("scoring_weights"), dict):
        raw_input["weights"] = retrieval_plan["scoring_weights"]
    return raw_input


def _retrieval_evidence(state: PipelineState) -> dict[str, list[dict[str, Any]]]:
    tool_outputs = state.get("tool_outputs", {})
    active_modes = _active_retrieval_modes(state)
    return {
        "lyric_evidence": _evidence_from_tool(tool_outputs.get("lyric_retrieval", {}))
        if "lyric_retrieval" in active_modes
        else [],
        "metadata_evidence": _evidence_from_tool(tool_outputs.get("metadata_retrieval", {}))
        if "metadata_retrieval" in active_modes
        else [],
        "feature_evidence": _evidence_from_tool(tool_outputs.get("feature_filter", {}))
        if "feature_filter" in active_modes
        else [],
    }


def _active_retrieval_modes(state: PipelineState) -> tuple[str, ...]:
    retrieval_plan = state.get("retrieval_plan")
    if isinstance(retrieval_plan, dict):
        modes = tuple(str(mode) for mode in retrieval_plan.get("modes", ()) if mode)
        if modes:
            return modes
    return ("lyric_retrieval", "metadata_retrieval", "feature_filter")


def _candidate_ids_from_evidence(evidence: dict[str, list[dict[str, Any]]]) -> tuple[str, ...]:
    stats: dict[str, dict[str, Any]] = {}
    source_order = {
        "metadata_evidence": 0,
        "feature_evidence": 1,
        "lyric_evidence": 2,
    }
    for source_name, items in evidence.items():
        for item in items:
            track_id = item.get("track_id")
            if not isinstance(track_id, str):
                continue
            cleaned = track_id.strip()
            if not cleaned:
                continue
            score = item.get("score")
            source_stat = stats.setdefault(
                cleaned,
                {"sources": set(), "score": 0.0, "first_source_order": source_order.get(source_name, 99)},
            )
            source_stat["sources"].add(source_name)
            if isinstance(score, (int, float)):
                source_stat["score"] = max(float(source_stat["score"]), float(score))
            source_stat["first_source_order"] = min(
                int(source_stat["first_source_order"]),
                source_order.get(source_name, 99),
            )
    ordered = sorted(
        stats,
        key=lambda track_id: (
            -len(stats[track_id]["sources"]),
            -float(stats[track_id]["score"]),
            int(stats[track_id]["first_source_order"]),
            track_id,
        ),
    )
    return tuple(ordered[:150])


def _evidence_from_tool(tool_output: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_output, dict):
        return []
    evidence: list[dict[str, Any]] = []
    for candidate in tool_output.get("candidates", ()):
        if not isinstance(candidate, dict):
            continue
        track_id = candidate.get("track_id")
        score = candidate.get("score", candidate.get("final_score"))
        if not track_id or score is None:
            continue
        evidence.append(
            {
                "track_id": track_id,
                "score": score,
                "rank": candidate.get("rank"),
                "details": _compact_details(candidate),
            }
        )
    return evidence


def _compact_details(candidate: dict[str, Any]) -> dict[str, Any]:
    excluded = {"track_id", "score", "rank"}
    return {
        key: value
        for key, value in candidate.items()
        if key not in excluded and (isinstance(value, (str, int, float, bool)) or value is None)
    }


def _reliability_candidate(
    candidate: dict[str, Any],
    verification_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    track_id = str(candidate["track_id"])
    verification = verification_lookup.get(track_id, {})
    return {
        "track_id": track_id,
        "rank": candidate.get("rank", 1),
        "final_score": candidate.get("final_score", 0.0),
        "evidence_sources": list(candidate.get("evidence_sources", ())),
        "score_components": candidate.get("score_components", {}),
        "verified": verification.get("verified"),
        "verifier_score": verification.get("verifier_score"),
        "constraint_violations": list(verification.get("violations", ())),
        "warnings": list(verification.get("warnings", ())),
        "track_artist": candidate.get("track_artist"),
        "track_album_name": candidate.get("track_album_name"),
    }


def _verification_lookup(state: PipelineState) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["track_id"]): candidate
        for candidate in state.get("verified_candidates", ())
        if isinstance(candidate, dict) and candidate.get("track_id")
    }


def _verifier_warnings(state: PipelineState) -> list[str]:
    warnings: list[str] = []
    for candidate in state.get("verified_candidates", ()):
        if isinstance(candidate, dict):
            warnings.extend(str(warning) for warning in candidate.get("warnings", ()))
    return warnings


def _critic_issues(state: PipelineState) -> list[str]:
    critique = state.get("critique")
    if isinstance(critique, dict):
        return [str(issue) for issue in critique.get("issues", ())]
    return []


def _hard_constraint_violations(state: PipelineState) -> list[str]:
    violations: list[str] = []
    for candidate in state.get("verified_candidates", ()):
        if isinstance(candidate, dict):
            violations.extend(str(violation) for violation in candidate.get("violations", ()))
    return violations


def _revision_succeeded(state: PipelineState) -> bool | None:
    revision_plan = state.get("revision_plan")
    if not isinstance(revision_plan, dict):
        return None
    if not revision_plan.get("should_retry"):
        return None
    return bool(state.get("scored_candidates"))


def _request_key_for_mode(mode: str) -> str | None:
    if mode == "lyric_retrieval":
        return "lyric_request"
    if mode == "metadata_retrieval":
        return "metadata_request"
    if mode == "feature_filter":
        return "feature_request"
    return None


def _merge_state_updates(current: PipelineState, update: PipelineState) -> PipelineState:
    merged = PipelineState(**current)
    if "candidate_ids" in update:
        merged["candidate_ids"] = merge_unique_strings(current.get("candidate_ids", ()), update["candidate_ids"])
    if "retrieval_modes_used" in update:
        merged["retrieval_modes_used"] = merge_unique_strings(
            current.get("retrieval_modes_used", ()),
            update["retrieval_modes_used"],
        )
    if "tool_outputs" in update:
        merged["tool_outputs"] = merge_dicts(current.get("tool_outputs", {}), update["tool_outputs"])
    if "tool_errors" in update:
        merged["tool_errors"] = append_tuples(current.get("tool_errors", ()), update["tool_errors"])
    if "warnings" in update:
        merged["warnings"] = merge_unique_strings(current.get("warnings", ()), update["warnings"])
    if "trace" in update:
        merged["trace"] = append_tuples(current.get("trace", ()), update["trace"])
    if "scored_candidates" in update:
        merged["scored_candidates"] = replace_tuple(current.get("scored_candidates", ()), update["scored_candidates"])
    if "reliability" in update:
        merged["reliability"] = update["reliability"]
    return merged


def _missing_state_update(stage: str, message: str) -> PipelineState:
    return PipelineState(
        warnings=(message,),
        trace=(make_trace_event(stage, "Graph node skipped.", {"reason": message}),),
    )
