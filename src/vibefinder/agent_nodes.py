"""LangGraph-ready wrappers for LLM agent nodes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from vibefinder.graph_runtime import (
    GraphRuntimeContext,
    complete_agent_step,
    enabled_tool_names,
    normalize_agent_output_for_runtime,
)
from vibefinder.pipeline_state import PipelineStage, PipelineState, make_trace_event
from vibefinder.prompts import build_agent_prompt
from vibefinder.tracing import context_summary, langsmith_extra, trace_agent_node


AGENT_CANDIDATE_CONTEXT_LIMIT = 10


def extract_preferences_node(
    state: PipelineState, context: GraphRuntimeContext
) -> PipelineState:
    """Run the preference extraction agent."""

    query = state.get("query")
    if not isinstance(query, str) or not query.strip():
        return _missing_state_update(
            "extract_preferences", "Missing required state value: query"
        )
    query = query.strip()

    prompt = build_agent_prompt(
        "Preference Extraction Agent",
        "Extract structured music preferences from the user query.",
        {
            "query": query,
            "retrieval_prompt_config": context.retrieval_prompt_config,
            "requirements": [
                "Use only approved dataset and retrieval fields.",
                "Artist, album, and playlist names are full-text search terms, not categorical enums.",
                "Do not invent genre taxonomies, semantic tags, or dataset columns.",
                "Set lyric_intent only to a lyric/theme/story intent from the user's query; use null when no such intent is stated.",
                "Use feature_range_filters only for hard numeric bounds with min_value or max_value; use feature_targets for low, medium, or high preferences.",
                "Return concise public rationale, issues, and warnings.",
            ],
        },
    )
    return complete_agent_step(
        context=context,
        prompt=prompt,
        schema_name="PreferenceExtractionOutput",
        state_key="preferences",
        stage="extract_preferences",
    )


def plan_retrieval_node(
    state: PipelineState, context: GraphRuntimeContext
) -> PipelineState:
    """Run the retrieval strategy agent."""

    preferences = state.get("preferences")
    if not isinstance(preferences, dict):
        return _missing_state_update(
            "plan_retrieval", "Missing required state object: preferences"
        )

    prompt = build_agent_prompt(
        "Retrieval Strategy Agent",
        "Choose tool-ready retrieval requests for the extracted preferences.",
        {
            "query": state.get("query"),
            "preferences": preferences,
            "enabled_tool_specs": _enabled_tool_prompt_specs(context),
            "variant": {
                "name": context.variant_config.name,
                "use_multi_step_reasoning": context.variant_config.use_multi_step_reasoning,
                "use_critic_revision": context.variant_config.use_critic_revision,
                "use_lyric_retriever": context.variant_config.use_lyric_retriever,
            },
            "requirements": [
                "Only select enabled retrieval tools.",
                "Every selected retrieval mode must include its corresponding tool-ready request.",
                "Use popularity only as a small ranking adjustment.",
                "Return concise public rationale, issues, and warnings.",
            ],
        },
    )
    return complete_agent_step(
        context=context,
        prompt=prompt,
        schema_name="RetrievalStrategyOutput",
        state_key="retrieval_plan",
        stage="plan_retrieval",
        runtime_preferences=preferences,
    )


def verify_candidates_node(
    state: PipelineState, context: GraphRuntimeContext
) -> PipelineState:
    """Run the verifier agent and map candidate results into graph state."""

    missing = _require_keys(
        state, ("preferences", "retrieval_plan"), "verify_candidates"
    )
    if missing is not None:
        return missing

    prompt = build_agent_prompt(
        "Verifier Agent",
        "Check whether retrieved candidates satisfy the extracted preferences.",
        {
            "query": state.get("query"),
            "preferences": state.get("preferences"),
            "retrieval_plan": state.get("retrieval_plan"),
            "candidate_context_records": _candidate_context_records(
                context,
                state,
                include_lyrics=True,
                limit=AGENT_CANDIDATE_CONTEXT_LIMIT,
            ),
            "requirements": [
                "Ground verification in tool outputs and approved dataset fields only.",
                "Use the provided lyrics only for verification; do not quote full lyrics in the response.",
                "Do not infer unsupported lyric meaning with certainty.",
                "Use violations only for explicit hard-constraint mismatches grounded in dataset fields.",
                "Treat weak lyric/theme evidence, missing evidence, uncertainty, or partial matches as warnings, not violations.",
                "When unsure, keep the candidate unverified with warnings instead of rejecting it.",
                "Use exact evidence_sources names when possible: lyric_retrieval, metadata_retrieval, feature_filter.",
                "Return concise public rationale, issues, and warnings.",
            ],
        },
    )
    output, error_update = _complete_agent_json(
        context=context,
        prompt=prompt,
        schema_name="VerificationOutput",
        stage="verify_candidates",
    )
    if error_update is not None:
        return error_update
    candidates = tuple(output.get("candidates", ()))
    update = PipelineState(
        verified_candidates=candidates,
        warnings=tuple(output.get("warnings", ())),
        trace=(
            make_trace_event(
                "verify_candidates",
                "Verifier agent completed.",
                {
                    "candidate_count": len(candidates),
                    "issue_count": len(output.get("issues", ())),
                    "warning_count": len(output.get("warnings", ())),
                    "summary": output.get("summary"),
                },
            ),
        ),
    )
    if state.get("scored_candidates"):
        filtered_candidates, filter_warnings = _filter_scored_candidates_by_verification(
            context,
            state,
            state.get("scored_candidates", ()),
            candidates,
        )
        update["scored_candidates"] = filtered_candidates
        if filter_warnings:
            update["warnings"] = tuple(output.get("warnings", ())) + tuple(filter_warnings)
    return update


def critique_results_node(
    state: PipelineState, context: GraphRuntimeContext
) -> PipelineState:
    """Run the critic agent, or skip it when the variant disables critic/revision."""

    if not context.variant_config.use_critic_revision:
        critique = {
            "issues": [],
            "should_revise": False,
            "revision_focus": [],
            "summary": "Critique and revision are disabled for this variant.",
            "rationale": "The active evaluation variant bypasses critic and revision steps.",
            "confidence": "medium",
            "warnings": [],
        }
        return PipelineState(
            critique=critique,
            trace=(
                make_trace_event(
                    "critique_results",
                    "Critique skipped by variant.",
                    {"variant_name": context.variant_config.name},
                ),
            ),
        )

    prompt = build_agent_prompt(
        "Critic Agent",
        "Review verified and scored candidates as a group and decide whether revision is needed.",
        {
            "query": state.get("query"),
            "preferences": state.get("preferences"),
            "retrieval_plan": state.get("retrieval_plan"),
            "verified_candidates": _limit_candidate_items(state.get("verified_candidates", ())),
            "scored_candidates": _limit_candidate_items(state.get("scored_candidates", ())),
            "candidate_context_records": _candidate_context_records(
                context,
                state,
                include_lyrics=True,
                limit=AGENT_CANDIDATE_CONTEXT_LIMIT,
            ),
            "warnings": state.get("warnings", ()),
            "requirements": [
                "Look for weak evidence, narrow candidate pools, hard constraint violations, and repetition.",
                "Use the provided lyrics only for critique; do not quote full lyrics in the response.",
                "Set should_revise only when a bounded retry can plausibly improve results.",
                "Return concise public rationale, issues, and warnings.",
            ],
        },
    )
    return complete_agent_step(
        context=context,
        prompt=prompt,
        schema_name="CritiqueOutput",
        state_key="critique",
        stage="critique_results",
    )


def revise_plan_node(
    state: PipelineState, context: GraphRuntimeContext
) -> PipelineState:
    """Run the revision agent for the bounded retry loop."""

    if not context.variant_config.use_critic_revision:
        return _skip_update(
            "revise_plan",
            "Revision skipped by variant.",
            {"variant_name": context.variant_config.name},
        )
    if int(state.get("revision_count", 0)) >= context.max_revision_count:
        return _skip_update(
            "revise_plan",
            "Revision skipped because max revision count was reached.",
            {
                "revision_count": int(state.get("revision_count", 0)),
                "max_revision_count": context.max_revision_count,
            },
        )

    critique = state.get("critique") or {}
    if not critique.get("should_revise", False):
        fallback = _deterministic_revision_for_trigger(state, context)
        if fallback is not None:
            return fallback
        return _skip_update(
            "revise_plan",
            "Revision skipped because critique did not request retry.",
            {"should_revise": False},
        )

    prompt = build_agent_prompt(
        "Revision Agent",
        "Revise the retrieval plan for one bounded retry.",
        {
            "query": state.get("query"),
            "preferences": state.get("preferences"),
            "current_retrieval_plan": state.get("retrieval_plan"),
            "critique": critique,
            "scored_candidates": _limit_candidate_items(state.get("scored_candidates", ())),
            "verified_candidates": _limit_candidate_items(state.get("verified_candidates", ())),
            "candidate_context_records": _candidate_context_records(
                context,
                state,
                include_lyrics=False,
                limit=AGENT_CANDIDATE_CONTEXT_LIMIT,
            ),
            "warnings": state.get("warnings", ()),
            "enabled_tool_specs": _enabled_tool_prompt_specs(context),
            "revision_count": int(state.get("revision_count", 0)),
            "max_revision_count": context.max_revision_count,
            "requirements": [
                "Only revise the plan when should_retry is true.",
                "Only select enabled retrieval tools.",
                "Preserve hard constraints from preferences.",
                "Return concise public rationale, issues, and warnings.",
            ],
        },
    )
    output, error_update = _complete_agent_json(
        context=context,
        prompt=prompt,
        schema_name="RevisionOutput",
        stage="revise_plan",
    )
    if error_update is not None:
        fallback = _deterministic_revision_fallback(state, context)
        if fallback is not None:
            return _merge_agent_updates(error_update, fallback)
        error_update["revision_plan"] = {
            "should_retry": False,
            "revised_retrieval_plan": None,
            "rationale": "Revision failed, so the previous retrieval plan and candidates were preserved.",
            "issues": ["Revision agent failed validation."],
            "warnings": list(error_update.get("warnings", ())),
        }
        return error_update
    update = PipelineState(
        revision_plan=output,
        revision_count=int(state.get("revision_count", 0))
        + (1 if output.get("should_retry") else 0),
        warnings=tuple(output.get("warnings", ())),
        trace=(
            make_trace_event(
                "revise_plan",
                "Revision agent completed.",
                {
                    "should_retry": output.get("should_retry"),
                    "issue_count": len(output.get("issues", ())),
                    "warning_count": len(output.get("warnings", ())),
                },
            ),
        ),
    )
    revised_plan = output.get("revised_retrieval_plan")
    if output.get("should_retry") and isinstance(revised_plan, dict):
        update["retrieval_plan"] = revised_plan
    return update


def explain_recommendations_node(
    state: PipelineState, context: GraphRuntimeContext
) -> PipelineState:
    """Run the explanation agent for final recommendations."""

    target_candidates = _limit_candidate_items(
        state.get("scored_candidates", ()), limit=AGENT_CANDIDATE_CONTEXT_LIMIT
    )
    required_track_ids = tuple(
        str(candidate["track_id"])
        for candidate in target_candidates
        if isinstance(candidate, dict) and candidate.get("track_id")
    )
    prompt = build_agent_prompt(
        "Explanation Agent",
        "Write grounded recommendation explanations from final candidates and dataset evidence.",
        {
            "query": state.get("query"),
            "preferences": state.get("preferences"),
            "scored_candidates": target_candidates,
            "verified_candidates": _limit_candidate_items(
                state.get("verified_candidates", ()), limit=AGENT_CANDIDATE_CONTEXT_LIMIT
            ),
            "reliability": state.get("reliability"),
            "candidate_dataset_records": _candidate_dataset_records(context, state),
            "required_explanation_track_ids": required_track_ids,
            "requirements": [
                "Return exactly one explanation recommendation for every required_explanation_track_id.",
                "Do not omit any required track_id.",
                "Preserve the ranked order from scored_candidates.",
                "Use only supporting_fields from approved dataset columns.",
                "Use supporting_tool_outputs only from available tool outputs: lyric_retrieval, metadata_retrieval, feature_filter, candidate_scoring, reliability.",
                "Do not quote or include full lyrics.",
                "Return concise public rationale, issues, and warnings.",
            ],
        },
    )
    output, error_update = _complete_agent_json(
        context=context,
        prompt=prompt,
        schema_name="ExplanationOutput",
        stage="explain_recommendations",
    )
    if error_update is not None:
        return error_update
    explanations, fallback_warnings = _complete_missing_explanations(
        context,
        state,
        tuple(output.get("recommendations", ())),
    )
    return PipelineState(
        explanations=explanations,
        recommendations=explanations,
        warnings=tuple(output.get("warnings", ())) + fallback_warnings,
        trace=(
            make_trace_event(
                "explain_recommendations",
                "Explanation agent completed.",
                {
                    "explanation_count": len(explanations),
                    "issue_count": len(output.get("issues", ())),
                    "warning_count": len(output.get("warnings", ())) + len(fallback_warnings),
                    "overall_summary": output.get("overall_summary"),
                },
            ),
        ),
    )


def _complete_missing_explanations(
    context: GraphRuntimeContext,
    state: PipelineState,
    explanations: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    completed, fallback_warnings = _fill_missing_explanations(context, state, explanations)
    if not any("deterministic evidence summary" in warning for warning in fallback_warnings):
        return completed, fallback_warnings

    existing_track_ids = {
        str(item["track_id"])
        for item in explanations
        if isinstance(item, dict) and item.get("track_id")
    }
    missing_candidates = tuple(
        candidate
        for candidate in _limit_candidate_items(
            state.get("scored_candidates", ()), limit=AGENT_CANDIDATE_CONTEXT_LIMIT
        )
        if isinstance(candidate, dict)
        and candidate.get("track_id")
        and str(candidate["track_id"]) not in existing_track_ids
    )
    if not missing_candidates:
        return completed, fallback_warnings

    repair_prompt = build_agent_prompt(
        "Explanation Agent",
        "Write grounded recommendation explanations only for the missing final candidates.",
        {
            "query": state.get("query"),
            "preferences": state.get("preferences"),
            "missing_scored_candidates": missing_candidates,
            "reliability": state.get("reliability"),
            "candidate_dataset_records": _candidate_context_records(
                context,
                state,
                include_lyrics=False,
                limit=len(missing_candidates),
                candidate_ids=tuple(str(candidate["track_id"]) for candidate in missing_candidates),
            ),
            "required_explanation_track_ids": tuple(
                str(candidate["track_id"]) for candidate in missing_candidates
            ),
            "requirements": [
                "Return exactly one explanation recommendation for every required_explanation_track_id.",
                "Return explanations only for the missing track_ids listed here.",
                "Preserve the ranked order from missing_scored_candidates.",
                "Use only supporting_fields from approved dataset columns.",
                "Use supporting_tool_outputs only from available tool outputs: lyric_retrieval, metadata_retrieval, feature_filter, candidate_scoring, reliability.",
                "Do not quote or include full lyrics.",
                "Return concise public rationale, issues, and warnings.",
            ],
        },
    )
    repair_output, repair_error = _complete_agent_json(
        context=context,
        prompt=repair_prompt,
        schema_name="ExplanationOutput",
        stage="explain_recommendations",
    )
    if repair_error is not None:
        return completed, fallback_warnings

    merged_lookup = {
        str(item["track_id"]): dict(item)
        for item in explanations
        if isinstance(item, dict) and item.get("track_id")
    }
    for item in repair_output.get("recommendations", ()):
        if isinstance(item, dict) and item.get("track_id"):
            merged_lookup[str(item["track_id"])] = dict(item)

    repaired, repaired_fallback_warnings = _fill_missing_explanations(
        context,
        state,
        tuple(merged_lookup.values()),
    )
    return repaired, repaired_fallback_warnings


def _fill_missing_explanations(
    context: GraphRuntimeContext,
    state: PipelineState,
    explanations: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    scored = tuple(
        candidate
        for candidate in state.get("scored_candidates", ())
        if isinstance(candidate, dict) and candidate.get("track_id")
    )
    if not scored:
        return explanations, ()
    by_track_id = {
        str(item["track_id"]): dict(item)
        for item in explanations
        if isinstance(item, dict) and item.get("track_id")
    }
    missing = [candidate for candidate in scored if str(candidate["track_id"]) not in by_track_id]
    if not missing:
        return explanations, ()
    row_lookup = _dataset_row_lookup(context, tuple(str(candidate["track_id"]) for candidate in missing))
    for candidate in missing:
        track_id = str(candidate["track_id"])
        by_track_id[track_id] = _fallback_explanation(candidate, row_lookup.get(track_id))
    ordered = tuple(
        by_track_id[str(candidate["track_id"])]
        for candidate in scored
        if str(candidate["track_id"]) in by_track_id
    )
    return ordered, ("LLM explanation unavailable; deterministic evidence summary used.",)


def _dataset_row_lookup(
    context: GraphRuntimeContext,
    track_ids: tuple[str, ...],
) -> dict[str, pd.Series]:
    if not track_ids or "track_id" not in context.songs.columns:
        return {}
    rows = context.songs[context.songs["track_id"].astype(str).isin(track_ids)]
    return {str(row["track_id"]): row for _, row in rows.iterrows()}


def _fallback_explanation(candidate: dict[str, Any], row: pd.Series | None) -> dict[str, Any]:
    track_id = str(candidate["track_id"])
    rank = int(candidate.get("rank") or 1)
    fields: list[str] = []
    fragments: list[str] = []
    if row is not None:
        title = _clean_row_text(row.get("track_name"))
        artist = _clean_row_text(row.get("track_artist"))
        language = _clean_row_text(row.get("language"))
        genre = _clean_row_text(row.get("playlist_genre"))
        subgenre = _clean_row_text(row.get("playlist_subgenre"))
        if title and artist:
            fragments.append(f"{title} by {artist}")
            fields.extend(["track_name", "track_artist"])
        if language:
            fragments.append(f"language={language}")
            fields.append("language")
        if genre:
            fragments.append(f"genre={genre}")
            fields.append("playlist_genre")
        if subgenre:
            fragments.append(f"subgenre={subgenre}")
            fields.append("playlist_subgenre")
    score = candidate.get("final_score")
    if isinstance(score, (int, float)):
        fragments.append(f"score={round(float(score), 3)}")
    evidence_sources = [
        str(source)
        for source in candidate.get("evidence_sources", ())
        if str(source) in {"lyric_retrieval", "metadata_retrieval", "feature_filter", "candidate_scoring", "reliability"}
    ]
    explanation = "Recommended from available dataset fields and retrieval evidence"
    if fragments:
        explanation += ": " + "; ".join(fragments)
    return {
        "track_id": track_id,
        "rank": rank,
        "explanation": explanation + ".",
        "supporting_fields": sorted(set(fields)),
        "supporting_tool_outputs": [*evidence_sources, "candidate_scoring"],
        "warnings": ["LLM explanation unavailable; deterministic evidence summary used."],
    }


def _clean_row_text(value: Any) -> str | None:
    if _is_null(value):
        return None
    text = str(value).strip()
    return text or None


def _complete_agent_json(
    *,
    context: GraphRuntimeContext,
    prompt: str,
    schema_name: str,
    stage: PipelineStage,
) -> tuple[dict[str, Any], None] | tuple[None, PipelineState]:
    return trace_agent_node(
        stage=stage,
        schema_name=schema_name,
        prompt=prompt,
        context_summary=context_summary(context),
        execute=lambda: _complete_agent_json_untraced(
            context=context,
            prompt=prompt,
            schema_name=schema_name,
            stage=stage,
        ),
        langsmith_extra=langsmith_extra(
            context=context,
            stage=stage,
            schema_name=schema_name,
            metadata={"component": "agent_node"},
        ),
    )


def _complete_agent_json_untraced(
    *,
    context: GraphRuntimeContext,
    prompt: str,
    schema_name: str,
    stage: PipelineStage,
) -> tuple[dict[str, Any], None] | tuple[None, PipelineState]:
    try:
        return (
            normalize_agent_output_for_runtime(
                context,
                schema_name,
                context.llm_client.complete_json(prompt=prompt, schema_name=schema_name),
            ),
            None,
        )
    except Exception as exc:
        logger.exception("agent_node_failed", stage=stage, schema_name=schema_name)
        return None, PipelineState(
            warnings=(f"{stage} failed: {exc}",),
            trace=(
                make_trace_event(
                    stage,
                    "Agent node failed.",
                    {
                        "schema_name": schema_name,
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                ),
            ),
        )


def _enabled_tool_prompt_specs(context: GraphRuntimeContext) -> list[dict[str, Any]]:
    enabled = set(enabled_tool_names(context))
    return [
        tool.to_prompt_spec()
        for name, tool in context.tool_runner.registry.items()
        if name in enabled
    ]


def _candidate_dataset_records(
    context: GraphRuntimeContext,
    state: PipelineState,
    limit: int = AGENT_CANDIDATE_CONTEXT_LIMIT,
) -> list[dict[str, Any]]:
    return _candidate_context_records(context, state, include_lyrics=False, limit=limit)


def _candidate_context_records(
    context: GraphRuntimeContext,
    state: PipelineState,
    *,
    include_lyrics: bool,
    limit: int,
    candidate_ids: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    selected_candidate_ids = candidate_ids or _state_candidate_ids(state)
    selected_candidate_ids = tuple(selected_candidate_ids[:limit])
    if not selected_candidate_ids or "track_id" not in context.songs.columns:
        return []

    allowed_columns = list(context.songs.columns) if include_lyrics else [
        column for column in context.songs.columns if column != "lyrics"
    ]
    rows = context.songs[context.songs["track_id"].astype(str).isin(selected_candidate_ids)]
    row_lookup = {str(row["track_id"]): row for _, row in rows.iterrows()}
    scored_lookup = _candidate_lookup(state.get("scored_candidates", ()))
    verified_lookup = _candidate_lookup(state.get("verified_candidates", ()))
    evidence_lookup = _tool_evidence_lookup(state.get("tool_outputs", {}))

    records: list[dict[str, Any]] = []
    for track_id in selected_candidate_ids:
        row = row_lookup.get(track_id)
        if row is None:
            continue
        record = {
            key: None if _is_null(row.get(key)) else row.get(key)
            for key in allowed_columns
        }
        scored = scored_lookup.get(track_id)
        if scored:
            record["scoring"] = {
                key: scored.get(key)
                for key in ("rank", "final_score", "score_components", "evidence_sources", "verifier_score", "verified")
                if key in scored
            }
        verified = verified_lookup.get(track_id)
        if verified:
            record["verification"] = {
                key: verified.get(key)
                for key in ("verified", "verifier_score", "matched_constraints", "violations", "evidence_sources", "warnings")
                if key in verified
            }
        evidence = evidence_lookup.get(track_id)
        if evidence:
            record["retrieval_evidence"] = evidence
        records.append(record)
    return records


def _candidate_lookup(values: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return {}
    return {
        str(item["track_id"]): item
        for item in values
        if isinstance(item, dict) and item.get("track_id")
    }


def _tool_evidence_lookup(tool_outputs: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(tool_outputs, dict):
        return {}
    evidence_by_track_id: dict[str, list[dict[str, Any]]] = {}
    for tool_name, output in tool_outputs.items():
        if not isinstance(output, dict):
            continue
        for candidate in output.get("candidates", ()):
            if not isinstance(candidate, dict) or not candidate.get("track_id"):
                continue
            track_id = str(candidate["track_id"])
            evidence = {
                "tool_name": tool_name,
                "rank": candidate.get("rank"),
                "score": candidate.get("score", candidate.get("final_score")),
                "lyric_preview": candidate.get("lyric_preview"),
            }
            evidence_by_track_id.setdefault(track_id, []).append(
                {key: value for key, value in evidence.items() if value is not None}
            )
    return evidence_by_track_id


def _limit_candidate_items(values: Any, limit: int = AGENT_CANDIDATE_CONTEXT_LIMIT) -> tuple[Any, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(values[:limit])


def _state_candidate_ids(state: PipelineState) -> tuple[str, ...]:
    if state.get("scored_candidates"):
        return tuple(
            candidate["track_id"]
            for candidate in state["scored_candidates"]
            if isinstance(candidate, dict) and candidate.get("track_id")
        )
    return tuple(state.get("candidate_ids", ()))


def _filter_scored_candidates_by_verification(
    context: GraphRuntimeContext,
    state: PipelineState,
    scored_candidates: tuple[dict[str, Any], ...],
    verified_candidates: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    verification_lookup = {
        str(candidate["track_id"]): candidate
        for candidate in verified_candidates
        if isinstance(candidate, dict) and candidate.get("track_id")
    }
    if not verification_lookup:
        return scored_candidates, ("Verifier returned no per-candidate judgments; kept ranked candidates.",)

    kept: list[dict[str, Any]] = []
    rejected_count = 0
    soft_count = 0
    deterministic_rejected_count = 0
    for candidate in scored_candidates:
        if not isinstance(candidate, dict):
            continue
        track_id = str(candidate.get("track_id", ""))
        verification = verification_lookup.get(track_id)
        deterministic_violations = _deterministic_hard_violations(context, state, track_id)
        if deterministic_violations:
            deterministic_rejected_count += 1
            continue
        if not verification:
            kept.append(candidate)
            continue
        violations = tuple(str(item) for item in verification.get("violations", ()) if str(item).strip())
        hard_violations = tuple(violation for violation in violations if _is_hard_verifier_violation(violation))
        soft_violations = tuple(violation for violation in violations if violation not in hard_violations)
        if hard_violations:
            rejected_count += 1
            continue
        updated = _verifier_adjusted_candidate(candidate, verification, soft_violations)
        updated["verified"] = verification.get("verified")
        updated["verifier_score"] = verification.get("verifier_score")
        if soft_violations:
            updated["verification_status"] = "soft_violation"
            updated["verification_warnings"] = list(soft_violations)
            soft_count += 1
        elif verification.get("verified") is not True:
            soft_count += 1
            updated["verification_status"] = "soft_unverified"
        kept.append(updated)

    kept.sort(key=lambda item: (-float(item.get("final_score", 0.0)), str(item.get("track_id", ""))))
    kept = [
        {**candidate, "rank": rank}
        for rank, candidate in enumerate(kept, start=1)
    ]

    warnings: list[str] = []
    if rejected_count:
        warnings.append(f"Verifier rejected {rejected_count} candidates with explicit violations.")
    if deterministic_rejected_count:
        warnings.append(
            f"Rejected {deterministic_rejected_count} candidates with deterministic hard constraint violations."
        )
    if soft_count:
        warnings.append(
            f"Kept {soft_count} candidates with soft verifier concerns; only hard constraint mismatches are rejected."
        )
    if not kept and scored_candidates:
        fallback_count = min(5, len(scored_candidates))
        warnings.append(
            f"Verifier rejected all candidates; kept top {fallback_count} scored candidates with low-confidence warnings."
        )
        return tuple(scored_candidates[:fallback_count]), tuple(warnings)
    return tuple(kept), tuple(warnings)


def _verifier_adjusted_candidate(
    candidate: dict[str, Any],
    verification: dict[str, Any],
    soft_violations: tuple[str, ...],
) -> dict[str, Any]:
    updated = dict(candidate)
    score = float(updated.get("final_score", 0.0))
    adjustment = 0.0
    reasons: list[str] = []
    if verification.get("verified") is True:
        adjustment += 0.03
        reasons.append("verified boost")
    if verification.get("verified") is not True:
        adjustment -= 0.04
        reasons.append("unverified demotion")
    warning_text = " ".join(
        [
            *[str(item) for item in soft_violations],
            *[str(item) for item in verification.get("warnings", ())],
            str(verification.get("rationale", "")),
        ]
    ).casefold()
    if any(marker in warning_text for marker in ("weak evidence", "insufficient evidence", "missing evidence")):
        adjustment -= 0.06
        reasons.append("weak evidence demotion")
    if any(marker in warning_text for marker in ("opposite perspective", "does not match", "not match", "mismatch")):
        adjustment -= 0.12
        reasons.append("explicit mismatch demotion")
    adjusted_score = max(0.0, min(1.0, score + adjustment))
    updated["pre_verification_score"] = score
    updated["final_score"] = round(adjusted_score, 6)
    if reasons:
        updated["verification_adjustment"] = round(adjustment, 6)
        updated["verification_adjustment_reasons"] = reasons
    return updated


def _deterministic_hard_violations(
    context: GraphRuntimeContext,
    state: PipelineState,
    track_id: str,
) -> tuple[str, ...]:
    preferences = state.get("preferences")
    if not isinstance(preferences, dict) or not track_id:
        return ()
    if "track_id" not in context.songs.columns:
        return ()
    rows = context.songs[context.songs["track_id"].astype(str) == track_id]
    if rows.empty:
        return ("not present in dataset",)
    row = rows.iloc[0]
    violations: list[str] = []
    for item in preferences.get("categorical_filters") or ():
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        values = item.get("values")
        if field not in {"language", "playlist_genre", "playlist_subgenre"}:
            continue
        accepted = {str(value).casefold() for value in values or () if str(value).strip()}
        actual = row.get(str(field))
        actual_text = "" if _is_null(actual) else str(actual).casefold()
        if accepted and actual_text not in accepted:
            violations.append(f"{field} constraint mismatch")
    for query in preferences.get("text_queries") or ():
        if not isinstance(query, dict):
            continue
        field = query.get("field")
        if field not in {"track_artist", "track_album_name"}:
            continue
        expected = str(query.get("query", "")).strip().casefold()
        actual = row.get(str(field))
        actual_text = "" if _is_null(actual) else str(actual).casefold()
        if expected and expected not in actual_text:
            violations.append(f"{field} constraint mismatch")
    return tuple(violations)


def _is_hard_verifier_violation(violation: str) -> bool:
    normalized = " ".join(violation.casefold().split())
    soft_markers = (
        "weak",
        "uncertain",
        "not enough evidence",
        "insufficient evidence",
        "missing evidence",
        "missing lyric",
        "no lyric",
        "theme",
        "mood",
        "perspective",
        "may not",
        "might not",
        "unclear",
        "partial",
        "low verifier",
    )
    if any(marker in normalized for marker in soft_markers):
        return False
    hard_markers = (
        "language mismatch",
        "wrong language",
        "genre mismatch",
        "subgenre mismatch",
        "artist mismatch",
        "album mismatch",
        "playlist mismatch",
        "explicit exclusion",
        "excluded",
        "violates exclusion",
        "outside requested range",
        "outside hard range",
        "below minimum",
        "above maximum",
        "not in dataset",
        "not present in dataset",
        "constraint mismatch",
    )
    return any(marker in normalized for marker in hard_markers)


def _deterministic_revision_fallback(
    state: PipelineState,
    context: GraphRuntimeContext,
) -> PipelineState | None:
    plan = state.get("retrieval_plan")
    if not isinstance(plan, dict):
        return None
    if state.get("candidate_ids"):
        return None
    modes = [mode for mode in plan.get("modes", ()) if mode in enabled_tool_names(context)]
    if modes:
        return None

    preferences = state.get("preferences")
    if not isinstance(preferences, dict):
        return None
    fallback_plan = _fallback_retrieval_plan_from_preferences(preferences)
    if fallback_plan is None:
        return None
    return PipelineState(
        revision_plan={
            "should_retry": True,
            "revised_retrieval_plan": fallback_plan,
            "rationale": "The revision agent failed, so a deterministic fallback plan was built from extracted preferences.",
            "issues": ["The previous plan produced no usable retrieval modes."],
            "warnings": ["Used deterministic revision fallback after revision agent failure."],
        },
        retrieval_plan=fallback_plan,
        revision_count=int(state.get("revision_count", 0)) + 1,
        warnings=("Used deterministic revision fallback after revision agent failure.",),
        trace=(
            make_trace_event(
                "revise_plan",
                "Deterministic revision fallback completed.",
                {"modes": fallback_plan.get("modes", ())},
            ),
        ),
    )


def _deterministic_revision_for_trigger(
    state: PipelineState,
    context: GraphRuntimeContext,
) -> PipelineState | None:
    trigger = _revision_trigger(state)
    if trigger is None:
        return None
    plan = state.get("retrieval_plan")
    if not isinstance(plan, dict):
        return None
    revised_plan = _tool_specific_revised_plan(state, context, trigger)
    if revised_plan is None:
        return None
    before_ids = tuple(
        str(candidate.get("track_id"))
        for candidate in state.get("scored_candidates", ())
        if isinstance(candidate, dict) and candidate.get("track_id")
    )
    warning = f"Used deterministic revision trigger: {trigger}."
    return PipelineState(
        revision_plan={
            "should_retry": True,
            "revised_retrieval_plan": revised_plan,
            "rationale": "A deterministic quality trigger requested one bounded retry.",
            "issues": [trigger],
            "warnings": [warning],
        },
        retrieval_plan=revised_plan,
        revision_count=int(state.get("revision_count", 0)) + 1,
        warnings=(warning,),
        trace=(
            make_trace_event(
                "revise_plan",
                "Deterministic revision trigger completed.",
                {
                    "revision_trigger": trigger,
                    "revision_before_candidate_count": len(before_ids),
                    "revision_after_candidate_count": None,
                    "revision_changed_top_k": None,
                },
            ),
        ),
    )


def _revision_trigger(state: PipelineState) -> str | None:
    if not state.get("scored_candidates"):
        return "no_candidates"
    preferences = state.get("preferences")
    scored = [item for item in state.get("scored_candidates", ()) if isinstance(item, dict)]
    if isinstance(preferences, dict) and preferences.get("lyric_intent") and scored:
        top = scored[:10]
        lyric_count = sum(1 for item in top if "lyric_retrieval" in tuple(item.get("evidence_sources", ())))
        if lyric_count / max(1, len(top)) < 0.3:
            return "low_lyric_evidence"
    hard_violations = [
        violation
        for candidate in state.get("verified_candidates", ())
        if isinstance(candidate, dict)
        for violation in candidate.get("violations", ())
        if _is_hard_verifier_violation(str(violation))
    ]
    if hard_violations:
        return "hard_constraint_violations"
    return None


def _tool_specific_revised_plan(
    state: PipelineState,
    context: GraphRuntimeContext,
    trigger: str,
) -> dict[str, Any] | None:
    plan = dict(state.get("retrieval_plan") or {})
    preferences = state.get("preferences")
    if not isinstance(preferences, dict):
        preferences = {}
    modes = [str(mode) for mode in plan.get("modes", ()) if str(mode) in enabled_tool_names(context)]
    if trigger == "low_lyric_evidence" and context.variant_config.use_lyric_retriever:
        if "lyric_retrieval" not in modes:
            modes.insert(0, "lyric_retrieval")
        lyric_intent = preferences.get("lyric_intent")
        if isinstance(lyric_intent, str) and lyric_intent.strip():
            lyric_request = dict(plan.get("lyric_request") or {})
            lyric_request["query"] = lyric_intent.strip()
            lyric_request["top_k"] = max(50, int(lyric_request.get("top_k") or 50))
            plan["lyric_request"] = lyric_request
        weights = dict(plan.get("scoring_weights") or {})
        weights.update({"lyric": max(float(weights.get("lyric", 0.4)), 0.65), "metadata": min(float(weights.get("metadata", 0.25)), 0.2)})
        plan["scoring_weights"] = weights
    elif trigger == "hard_constraint_violations":
        metadata_request = dict(plan.get("metadata_request") or {})
        metadata_request["categorical_filters"] = list(preferences.get("categorical_filters") or metadata_request.get("categorical_filters") or [])
        metadata_request["text_queries"] = list(preferences.get("text_queries") or metadata_request.get("text_queries") or [])
        if metadata_request.get("categorical_filters") or metadata_request.get("text_queries"):
            if "metadata_retrieval" not in modes:
                modes.insert(0, "metadata_retrieval")
            plan["metadata_request"] = metadata_request
    elif trigger == "no_candidates":
        fallback = _fallback_retrieval_plan_from_preferences(preferences)
        if fallback is not None:
            return fallback
        return None
    if not modes:
        return None
    plan["modes"] = modes
    plan["primary_mode"] = modes[0]
    plan["warnings"] = [*list(plan.get("warnings") or []), f"Revised deterministically for trigger: {trigger}"]
    plan["rationale"] = plan.get("rationale") or "Revised deterministically from workflow quality triggers."
    plan["issues"] = list(plan.get("issues") or [])
    plan["top_k_final"] = int(plan.get("top_k_final") or 10)
    return plan


def _fallback_retrieval_plan_from_preferences(preferences: dict[str, Any]) -> dict[str, Any] | None:
    modes: list[str] = []
    plan: dict[str, Any] = {
        "scoring_weights": {
            "lyric": 0.4,
            "metadata": 0.25,
            "feature": 0.3,
            "popularity": 0.05,
            "diversity_penalty": 0.05,
        },
        "top_k_final": 10,
        "rationale": "Fallback plan rebuilt from extracted preferences.",
        "issues": [],
        "warnings": ["Generated by deterministic revision fallback."],
    }
    lyric_intent = preferences.get("lyric_intent")
    if isinstance(lyric_intent, str) and lyric_intent.strip():
        modes.append("lyric_retrieval")
        plan["lyric_request"] = {"query": lyric_intent.strip(), "top_k": 50}
    categorical_filters = preferences.get("categorical_filters") or ()
    text_queries = preferences.get("text_queries") or ()
    if categorical_filters or text_queries:
        modes.append("metadata_retrieval")
        plan["metadata_request"] = {
            "categorical_filters": list(categorical_filters),
            "text_queries": list(text_queries),
            "top_k": 50,
        }
    range_filters = preferences.get("feature_range_filters") or ()
    targets = preferences.get("feature_targets") or ()
    if range_filters or targets:
        modes.append("feature_filter")
        plan["feature_request"] = {
            "range_filters": list(range_filters),
            "targets": list(targets),
            "top_k": 50,
        }
    if not modes:
        return None
    plan["primary_mode"] = modes[0]
    plan["modes"] = modes
    return plan


def _merge_agent_updates(first: PipelineState, second: PipelineState) -> PipelineState:
    merged = PipelineState(**first)
    for key, value in second.items():
        if key in {"warnings", "trace"}:
            merged[key] = tuple(first.get(key, ())) + tuple(value)
        else:
            merged[key] = value
    return merged


def _require_keys(
    state: PipelineState, keys: tuple[str, ...], stage: PipelineStage
) -> PipelineState | None:
    missing = [key for key in keys if state.get(key) in (None, {}, ())]
    if missing:
        return _missing_state_update(
            stage, f"Missing required state values: {', '.join(missing)}"
        )
    return None


def _missing_state_update(stage: PipelineStage, message: str) -> PipelineState:
    return PipelineState(
        warnings=(message,),
        trace=(make_trace_event(stage, "Agent node skipped.", {"reason": message}),),
    )


def _skip_update(
    stage: PipelineStage, message: str, details: dict[str, Any]
) -> PipelineState:
    return PipelineState(
        trace=(make_trace_event(stage, message, details),),
    )


def _is_null(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False
