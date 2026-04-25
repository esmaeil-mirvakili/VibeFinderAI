"""Evaluation helpers for reproducible ablation runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from vibefinder.pipeline_state import PipelineState
from vibefinder.variants import VariantConfig


FeatureDirection = Literal["low", "medium", "high"]
TextMatchMode = Literal["all", "any"]


@dataclass(frozen=True)
class ExpectedFeatureTarget:
    """Expected numeric feature behavior for one benchmark query."""

    feature: str
    direction: FeatureDirection
    weight: float = 1.0


@dataclass(frozen=True)
class ExpectedTextMatch:
    """Expected full-text metadata match for one benchmark query."""

    field: str
    query: str
    match_mode: TextMatchMode = "all"


@dataclass(frozen=True)
class EvaluationQuery:
    """One benchmark query and optional measurable expectations."""

    id: str
    query: str
    group: str
    expected_language: str | None = None
    expected_genres: tuple[str, ...] = ()
    expected_subgenres: tuple[str, ...] = ()
    expected_feature_targets: tuple[ExpectedFeatureTarget, ...] = ()
    expected_feature_exclusions: tuple[ExpectedFeatureTarget, ...] = ()
    expected_text_matches: tuple[ExpectedTextMatch, ...] = ()
    expected_retrieval_modes: tuple[str, ...] = ()
    lyric_intent: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class EvaluationRunResult:
    """Compact result for one query/variant graph execution."""

    query_id: str
    query: str
    group: str
    variant: str
    elapsed_seconds: float
    success: bool
    summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    state: dict[str, Any] | None = None


DEFAULT_EVALUATION_VARIANTS: dict[str, VariantConfig] = {
    "full": VariantConfig(
        name="full",
        use_multi_step_reasoning=True,
        use_critic_revision=True,
        use_lyric_retriever=True,
    ),
    "no_critic_revision": VariantConfig(
        name="no_critic_revision",
        use_multi_step_reasoning=True,
        use_critic_revision=False,
        use_lyric_retriever=True,
    ),
    "no_lyric_retriever": VariantConfig(
        name="no_lyric_retriever",
        use_multi_step_reasoning=True,
        use_critic_revision=True,
        use_lyric_retriever=False,
    ),
}


def load_evaluation_queries(path: str | Path) -> tuple[EvaluationQuery, ...]:
    """Load benchmark queries from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation query file must contain a JSON list.")
    return tuple(_evaluation_query_from_mapping(item) for item in data)


def evaluation_variant_configs(names: list[str] | tuple[str, ...] | None = None) -> tuple[VariantConfig, ...]:
    """Return evaluation variant configs in a stable order."""

    selected_names = tuple(names or DEFAULT_EVALUATION_VARIANTS)
    unknown = [name for name in selected_names if name not in DEFAULT_EVALUATION_VARIANTS]
    if unknown:
        allowed = ", ".join(DEFAULT_EVALUATION_VARIANTS)
        raise ValueError(f"Unknown evaluation variant(s): {', '.join(unknown)}. Allowed: {allowed}")
    return tuple(DEFAULT_EVALUATION_VARIANTS[name] for name in selected_names)


def summarize_pipeline_state(
    *,
    state: PipelineState,
    songs: pd.DataFrame,
    evaluation_query: EvaluationQuery,
    variant_name: str,
    elapsed_seconds: float,
    top_k: int = 10,
    include_state: bool = False,
) -> EvaluationRunResult:
    """Build a compact, JSON-safe result from a completed pipeline state."""

    final_track_ids = _final_track_ids(state, top_k=top_k)
    raw_candidate_ids = [str(track_id) for track_id in state.get("candidate_ids", ()) if str(track_id).strip()]
    scored_track_ids = [
        str(candidate["track_id"])
        for candidate in state.get("scored_candidates", ())
        if isinstance(candidate, dict) and candidate.get("track_id")
    ][:top_k]
    summary = {
        "candidate_count": len(tuple(state.get("candidate_ids", ()))),
        "scored_count": len(tuple(state.get("scored_candidates", ()))),
        "recommendation_count": len(tuple(state.get("explanations", ()))),
        "warning_count": len(tuple(state.get("warnings", ()))),
        "tool_error_count": len(tuple(state.get("tool_errors", ()))),
        "retrieval_modes_used": list(state.get("retrieval_modes_used", ())),
        "revision_count": int(state.get("revision_count", 0)),
        "revision_used": int(state.get("revision_count", 0)) > 0,
        "confidence_label": _reliability_value(state, "confidence_label"),
        "confidence_score": _reliability_value(state, "confidence_score"),
        "raw_candidate_track_ids": raw_candidate_ids[:top_k],
        "scored_track_ids": scored_track_ids,
        "final_track_ids": final_track_ids,
        "final_candidates": _final_candidate_records(state=state, songs=songs, track_ids=final_track_ids),
        "final_explanations": _final_explanations(state=state, track_ids=final_track_ids),
        "failure_category": _failure_category(state),
        "trace_stages": [event.get("stage") for event in state.get("trace", ()) if isinstance(event, dict)],
        "warnings": list(state.get("warnings", ())),
        "tool_errors": list(state.get("tool_errors", ())),
    }
    metrics = compute_evaluation_metrics(
        songs=songs,
        track_ids=tuple(final_track_ids),
        evaluation_query=evaluation_query,
        state=state,
    )
    return EvaluationRunResult(
        query_id=evaluation_query.id,
        query=evaluation_query.query,
        group=evaluation_query.group,
        variant=variant_name,
        elapsed_seconds=round(elapsed_seconds, 4),
        success=True,
        summary=_json_safe(summary),
        metrics=_json_safe(metrics),
        state=_json_safe(state) if include_state else None,
    )


def failed_run_result(
    *,
    evaluation_query: EvaluationQuery,
    variant_name: str,
    elapsed_seconds: float,
    error: Exception,
) -> EvaluationRunResult:
    """Build a compact result for a failed graph execution."""

    return EvaluationRunResult(
        query_id=evaluation_query.id,
        query=evaluation_query.query,
        group=evaluation_query.group,
        variant=variant_name,
        elapsed_seconds=round(elapsed_seconds, 4),
        success=False,
        error={
            "type": error.__class__.__name__,
            "message": str(error),
        },
    )


def compute_evaluation_metrics(
    *,
    songs: pd.DataFrame,
    track_ids: tuple[str, ...],
    evaluation_query: EvaluationQuery,
    state: PipelineState | None = None,
) -> dict[str, Any]:
    """Compute lightweight automatic metrics for one ranked result list."""

    records = _records_for_track_ids(songs, track_ids)
    candidate_count = len(records)
    expected = evaluation_query
    metrics: dict[str, Any] = {
        "evaluated_candidate_count": candidate_count,
        "missing_dataset_count": max(0, len(track_ids) - candidate_count),
        "unique_artist_count": _unique_count(records, "track_artist"),
        "unique_album_count": _unique_count(records, "track_album_name"),
        "unique_subgenre_count": _unique_count(records, "playlist_subgenre"),
    }
    metrics["artist_diversity"] = _ratio(metrics["unique_artist_count"], candidate_count)
    metrics["album_diversity"] = _ratio(metrics["unique_album_count"], candidate_count)
    metrics["subgenre_diversity"] = _ratio(metrics["unique_subgenre_count"], candidate_count)

    if expected.expected_language:
        metrics["language_match_rate"] = _categorical_match_rate(
            records,
            "language",
            (expected.expected_language,),
        )
    else:
        metrics["language_match_rate"] = None

    metrics["genre_match_rate"] = (
        _categorical_match_rate(records, "playlist_genre", expected.expected_genres)
        if expected.expected_genres
        else None
    )
    metrics["subgenre_match_rate"] = (
        _categorical_match_rate(records, "playlist_subgenre", expected.expected_subgenres)
        if expected.expected_subgenres
        else None
    )
    metrics["feature_fit_score"] = (
        _feature_fit_score(songs, records, expected.expected_feature_targets)
        if expected.expected_feature_targets
        else None
    )
    metrics["feature_exclusion_pass_rate"] = (
        _feature_exclusion_pass_rate(songs, records, expected.expected_feature_exclusions)
        if expected.expected_feature_exclusions
        else None
    )
    metrics["metadata_text_match_rate"] = (
        _text_match_rate(records, expected.expected_text_matches)
        if expected.expected_text_matches
        else None
    )
    metrics["expected_retrieval_mode_recall"] = (
        _expected_retrieval_mode_recall(state, expected.expected_retrieval_modes)
        if expected.expected_retrieval_modes
        else None
    )
    metrics["unexpected_retrieval_mode_count"] = (
        _unexpected_retrieval_mode_count(state, expected.expected_retrieval_modes)
        if expected.expected_retrieval_modes
        else None
    )
    metrics["lyric_retrieval_used"] = _retrieval_mode_used(state, "lyric_retrieval")
    metrics["lyric_evidence_rate"] = _evidence_source_rate(state, "lyric", tuple(track_ids))
    metrics["lyric_intent_expected"] = expected.lyric_intent is not None
    metrics["lyric_intent_retrieval_match"] = (
        metrics["lyric_retrieval_used"] if expected.lyric_intent else None
    )
    metrics["automatic_constraint_score"] = _average_optional_scores(
        (
            metrics["language_match_rate"],
            metrics["genre_match_rate"],
            metrics["subgenre_match_rate"],
            metrics["metadata_text_match_rate"],
            metrics["feature_fit_score"],
            metrics["feature_exclusion_pass_rate"],
            metrics["expected_retrieval_mode_recall"],
            metrics["lyric_intent_retrieval_match"],
        )
    )
    return metrics


def result_to_dict(result: EvaluationRunResult) -> dict[str, Any]:
    """Convert an evaluation result dataclass to a JSON-safe dictionary."""

    return _json_safe(
        {
            "query_id": result.query_id,
            "query": result.query,
            "group": result.group,
            "variant": result.variant,
            "elapsed_seconds": result.elapsed_seconds,
            "success": result.success,
            "summary": result.summary,
            "metrics": result.metrics,
            "error": result.error,
            "state": result.state,
        }
    )


def result_from_dict(data: dict[str, Any]) -> EvaluationRunResult:
    """Build an evaluation result dataclass from a JSON-safe dictionary."""

    return EvaluationRunResult(
        query_id=str(data["query_id"]),
        query=str(data["query"]),
        group=str(data["group"]),
        variant=str(data["variant"]),
        elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
        success=bool(data.get("success")),
        summary=dict(data.get("summary") or {}),
        metrics=dict(data.get("metrics") or {}),
        error=dict(data["error"]) if isinstance(data.get("error"), dict) else None,
        state=dict(data["state"]) if isinstance(data.get("state"), dict) else None,
    )


def build_evaluation_report(
    *,
    results: tuple[EvaluationRunResult, ...],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the top-level JSON report payload."""

    result_dicts = [result_to_dict(result) for result in results]
    return _json_safe(
        {
            "metadata": metadata,
            "aggregate": aggregate_results(result_dicts),
            "results": result_dicts,
        }
    )


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate evaluation results by variant."""

    aggregate: dict[str, dict[str, Any]] = {}
    for result in results:
        variant = str(result["variant"])
        bucket = aggregate.setdefault(
            variant,
            {
                "run_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "avg_elapsed_seconds": None,
                "avg_confidence_score": None,
                "avg_automatic_constraint_score": None,
                "avg_feature_fit_score": None,
                "avg_metadata_text_match_rate": None,
                "avg_feature_exclusion_pass_rate": None,
                "avg_expected_retrieval_mode_recall": None,
                "avg_lyric_evidence_rate": None,
                "avg_warning_count": None,
                "avg_final_candidate_count": None,
                "failure_categories": {},
                "revision_used_count": 0,
            },
        )
        bucket["run_count"] += 1
        if result.get("success"):
            bucket["success_count"] += 1
        else:
            bucket["failure_count"] += 1
        if result.get("summary", {}).get("revision_used"):
            bucket["revision_used_count"] += 1
        failure_category = result.get("summary", {}).get("failure_category")
        if failure_category:
            categories = bucket.setdefault("failure_categories", {})
            categories[failure_category] = categories.get(failure_category, 0) + 1

    for variant, bucket in aggregate.items():
        variant_results = [result for result in results if result["variant"] == variant]
        bucket["avg_elapsed_seconds"] = _average(
            result.get("elapsed_seconds") for result in variant_results
        )
        bucket["avg_confidence_score"] = _average(
            result.get("summary", {}).get("confidence_score") for result in variant_results
        )
        bucket["avg_automatic_constraint_score"] = _average(
            result.get("metrics", {}).get("automatic_constraint_score") for result in variant_results
        )
        bucket["avg_feature_fit_score"] = _average(
            result.get("metrics", {}).get("feature_fit_score") for result in variant_results
        )
        bucket["avg_metadata_text_match_rate"] = _average(
            result.get("metrics", {}).get("metadata_text_match_rate") for result in variant_results
        )
        bucket["avg_feature_exclusion_pass_rate"] = _average(
            result.get("metrics", {}).get("feature_exclusion_pass_rate") for result in variant_results
        )
        bucket["avg_expected_retrieval_mode_recall"] = _average(
            result.get("metrics", {}).get("expected_retrieval_mode_recall") for result in variant_results
        )
        bucket["avg_lyric_evidence_rate"] = _average(
            result.get("metrics", {}).get("lyric_evidence_rate") for result in variant_results
        )
        bucket["avg_warning_count"] = _average(
            result.get("summary", {}).get("warning_count") for result in variant_results
        )
        bucket["avg_final_candidate_count"] = _average(
            len(result.get("summary", {}).get("final_track_ids", ())) for result in variant_results
        )
    return aggregate


def markdown_summary(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for the JSON evaluation report."""

    metadata = report.get("metadata", {})
    lines = [
        "# VibeFinder Evaluation Summary",
        "",
        f"- Generated at: {metadata.get('generated_at', 'unknown')}",
        f"- Query count: {metadata.get('query_count', 'unknown')}",
        f"- Variants: {', '.join(metadata.get('variants', ())) if metadata.get('variants') else 'unknown'}",
        f"- Model backend: {metadata.get('llm_provider', 'unknown')}",
        f"- Model name: {metadata.get('llm_model', 'unknown')}",
        "",
        "| Variant | Runs | Success | Avg Confidence | Avg Constraint | Avg Feature Fit | Text Match | Mode Recall | Lyric Evidence | Avg Warnings | Revisions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, aggregate in sorted(report.get("aggregate", {}).items()):
        lines.append(
            "| {variant} | {run_count} | {success_count} | {confidence} | {constraint} | "
            "{feature} | {text} | {mode} | {lyric} | {warnings} | {revision} |".format(
                variant=variant,
                run_count=aggregate["run_count"],
                success_count=aggregate["success_count"],
                confidence=_format_metric(aggregate.get("avg_confidence_score")),
                constraint=_format_metric(aggregate.get("avg_automatic_constraint_score")),
                feature=_format_metric(aggregate.get("avg_feature_fit_score")),
                text=_format_metric(aggregate.get("avg_metadata_text_match_rate")),
                mode=_format_metric(aggregate.get("avg_expected_retrieval_mode_recall")),
                lyric=_format_metric(aggregate.get("avg_lyric_evidence_rate")),
                warnings=_format_metric(aggregate.get("avg_warning_count")),
                revision=aggregate["revision_used_count"],
            )
        )
    return "\n".join(lines) + "\n"


def _evaluation_query_from_mapping(item: Any) -> EvaluationQuery:
    if not isinstance(item, dict):
        raise ValueError("Each evaluation query must be a JSON object.")
    try:
        query_id = str(item["id"]).strip()
        query = str(item["query"]).strip()
    except KeyError as exc:
        raise ValueError(f"Evaluation query is missing required field: {exc.args[0]}") from exc
    if not query_id or not query:
        raise ValueError("Evaluation query id and query cannot be blank.")

    expected = item.get("expected", {})
    if expected is None:
        expected = {}
    if not isinstance(expected, dict):
        raise ValueError(f"Evaluation query {query_id} expected field must be an object.")

    return EvaluationQuery(
        id=query_id,
        query=query,
        group=str(item.get("group", "general")).strip() or "general",
        expected_language=_optional_string(expected.get("language")),
        expected_genres=_string_tuple(expected.get("genres")),
        expected_subgenres=_string_tuple(expected.get("subgenres")),
        expected_feature_targets=_feature_targets(expected.get("feature_targets"), query_id),
        expected_feature_exclusions=_feature_targets(expected.get("feature_exclusions"), query_id),
        expected_text_matches=_text_matches(expected.get("text_matches"), query_id),
        expected_retrieval_modes=_string_tuple(expected.get("retrieval_modes")),
        lyric_intent=_optional_string(expected.get("lyric_intent")),
        notes=_optional_string(item.get("notes")),
    )


def _feature_targets(value: Any, query_id: str) -> tuple[ExpectedFeatureTarget, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Evaluation query {query_id} feature_targets must be a list.")
    targets = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation query {query_id} feature target must be an object.")
        direction = item.get("direction")
        if direction not in {"low", "medium", "high"}:
            raise ValueError(
                f"Evaluation query {query_id} feature target direction must be low, medium, or high."
            )
        weight = float(item.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"Evaluation query {query_id} feature target weight must be positive.")
        feature = str(item.get("feature", "")).strip()
        if not feature:
            raise ValueError(f"Evaluation query {query_id} feature target feature cannot be blank.")
        targets.append(
            ExpectedFeatureTarget(
                feature=feature,
                direction=direction,
                weight=weight,
            )
        )
    return tuple(targets)


def _text_matches(value: Any, query_id: str) -> tuple[ExpectedTextMatch, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Evaluation query {query_id} text_matches must be a list.")
    matches = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation query {query_id} text match must be an object.")
        field = str(item.get("field", "")).strip()
        query = str(item.get("query", "")).strip()
        match_mode = item.get("match_mode", "all")
        if not field or not query:
            raise ValueError(f"Evaluation query {query_id} text match field and query cannot be blank.")
        if field not in {"playlist_name", "track_artist", "track_album_name"}:
            raise ValueError(f"Evaluation query {query_id} text match field is unsupported: {field}")
        if match_mode not in {"all", "any"}:
            raise ValueError(f"Evaluation query {query_id} text match mode must be all or any.")
        matches.append(ExpectedTextMatch(field=field, query=query, match_mode=match_mode))
    return tuple(matches)


def _final_track_ids(state: PipelineState, top_k: int) -> list[str]:
    scored = tuple(state.get("scored_candidates", ()))
    return [
        str(candidate["track_id"])
        for candidate in scored
        if isinstance(candidate, dict) and candidate.get("track_id")
    ][:top_k]


def _final_candidate_records(
    *,
    state: PipelineState,
    songs: pd.DataFrame,
    track_ids: list[str],
) -> list[dict[str, Any]]:
    lookup = _record_lookup(songs)
    score_lookup = {
        str(candidate["track_id"]): candidate
        for candidate in state.get("scored_candidates", ())
        if isinstance(candidate, dict) and candidate.get("track_id")
    }
    records = []
    for rank, track_id in enumerate(track_ids, start=1):
        row = lookup.get(track_id, {})
        score = score_lookup.get(track_id, {})
        records.append(
            {
                "rank": rank,
                "track_id": track_id,
                "track_name": row.get("track_name"),
                "track_artist": row.get("track_artist"),
                "track_album_name": row.get("track_album_name"),
                "playlist_genre": row.get("playlist_genre"),
                "playlist_subgenre": row.get("playlist_subgenre"),
                "language": row.get("language"),
                "final_score": score.get("final_score"),
                "evidence_sources": list(score.get("evidence_sources", ())),
            }
        )
    return _json_safe(records)


def _final_explanations(*, state: PipelineState, track_ids: list[str]) -> list[dict[str, Any]]:
    explanation_lookup = {
        str(item["track_id"]): item
        for item in state.get("explanations", ())
        if isinstance(item, dict) and item.get("track_id")
    }
    explanations = []
    for rank, track_id in enumerate(track_ids, start=1):
        item = explanation_lookup.get(track_id)
        if not item:
            continue
        explanations.append(
            {
                "rank": rank,
                "track_id": track_id,
                "explanation": item.get("explanation"),
                "warnings": list(item.get("warnings", ())) if isinstance(item.get("warnings"), (list, tuple)) else [],
            }
        )
    return _json_safe(explanations)


def _records_for_track_ids(songs: pd.DataFrame, track_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    lookup = _record_lookup(songs)
    return [lookup[track_id] for track_id in track_ids if track_id in lookup]


def _record_lookup(songs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "track_id" not in songs.columns:
        return {}
    records = songs.to_dict(orient="records")
    return {str(record["track_id"]): record for record in records if record.get("track_id") is not None}


def _categorical_match_rate(
    records: list[dict[str, Any]],
    field_name: str,
    accepted_values: tuple[str, ...],
) -> float | None:
    if not records:
        return None
    accepted = {value.casefold() for value in accepted_values}
    if not accepted:
        return None
    matched = 0
    for record in records:
        value = record.get(field_name)
        if isinstance(value, str) and value.casefold() in accepted:
            matched += 1
    return round(matched / len(records), 4)


def _feature_fit_score(
    songs: pd.DataFrame,
    records: list[dict[str, Any]],
    targets: tuple[ExpectedFeatureTarget, ...],
) -> float | None:
    if not records or not targets:
        return None
    total = 0.0
    weight_total = 0.0
    for target in targets:
        if target.feature not in songs.columns:
            continue
        series = pd.to_numeric(songs[target.feature], errors="coerce").dropna()
        if series.empty:
            continue
        values = [pd.to_numeric(record.get(target.feature), errors="coerce") for record in records]
        numeric_values = [float(value) for value in values if pd.notna(value)]
        if not numeric_values:
            continue
        feature_scores = [
            _direction_score(value=value, series=series, direction=target.direction)
            for value in numeric_values
        ]
        total += sum(feature_scores) / len(feature_scores) * target.weight
        weight_total += target.weight
    if weight_total == 0:
        return None
    return round(total / weight_total, 4)


def _feature_exclusion_pass_rate(
    songs: pd.DataFrame,
    records: list[dict[str, Any]],
    exclusions: tuple[ExpectedFeatureTarget, ...],
) -> float | None:
    if not records or not exclusions:
        return None
    evaluated = 0
    passed = 0
    for exclusion in exclusions:
        if exclusion.feature not in songs.columns:
            continue
        series = pd.to_numeric(songs[exclusion.feature], errors="coerce").dropna()
        if series.empty:
            continue
        for record in records:
            value = pd.to_numeric(record.get(exclusion.feature), errors="coerce")
            if pd.isna(value):
                continue
            evaluated += 1
            excluded_direction_score = _direction_score(
                value=float(value),
                series=series,
                direction=exclusion.direction,
            )
            if excluded_direction_score < 0.67:
                passed += 1
    if evaluated == 0:
        return None
    return round(passed / evaluated, 4)


def _direction_score(*, value: float, series: pd.Series, direction: FeatureDirection) -> float:
    minimum = float(series.min())
    maximum = float(series.max())
    if math.isclose(minimum, maximum):
        percentile = 0.5
    else:
        percentile = (value - minimum) / (maximum - minimum)
    percentile = max(0.0, min(1.0, percentile))
    if direction == "high":
        return percentile
    if direction == "low":
        return 1.0 - percentile
    return max(0.0, 1.0 - abs(percentile - 0.5) * 2.0)


def _text_match_rate(records: list[dict[str, Any]], matches: tuple[ExpectedTextMatch, ...]) -> float | None:
    if not records or not matches:
        return None
    evaluated = 0
    matched = 0
    for record in records:
        for expected in matches:
            evaluated += 1
            if _record_matches_text(record, expected):
                matched += 1
    if evaluated == 0:
        return None
    return round(matched / evaluated, 4)


def _record_matches_text(record: dict[str, Any], expected: ExpectedTextMatch) -> bool:
    value_terms = set(_text_terms(str(record.get(expected.field, ""))))
    terms = _text_terms(expected.query)
    if not terms:
        return False
    if expected.match_mode == "any":
        return any(term in value_terms for term in terms)
    return all(term in value_terms for term in terms)


def _text_terms(value: str) -> tuple[str, ...]:
    terms = []
    for raw_term in value.casefold().replace("/", " ").split():
        term = "".join(character for character in raw_term if character.isalnum() or character in {"&"})
        if term:
            terms.append(term)
    return tuple(terms)


def _expected_retrieval_mode_recall(
    state: PipelineState | None,
    expected_modes: tuple[str, ...],
) -> float | None:
    if not expected_modes:
        return None
    used = set(_retrieval_modes_used(state))
    expected = set(expected_modes)
    if not expected:
        return None
    return round(len(used & expected) / len(expected), 4)


def _unexpected_retrieval_mode_count(
    state: PipelineState | None,
    expected_modes: tuple[str, ...],
) -> int | None:
    if not expected_modes:
        return None
    used = set(_retrieval_modes_used(state))
    return len(used - set(expected_modes))


def _retrieval_mode_used(state: PipelineState | None, mode: str) -> bool | None:
    if state is None:
        return None
    return mode in _retrieval_modes_used(state)


def _retrieval_modes_used(state: PipelineState | None) -> tuple[str, ...]:
    if state is None:
        return ()
    return tuple(str(mode) for mode in state.get("retrieval_modes_used", ()) if str(mode).strip())


def _evidence_source_rate(
    state: PipelineState | None,
    source: str,
    track_ids: tuple[str, ...],
) -> float | None:
    if state is None or not track_ids:
        return None
    source_by_track_id = {
        str(candidate["track_id"]): tuple(str(item) for item in candidate.get("evidence_sources", ()))
        for candidate in state.get("scored_candidates", ())
        if isinstance(candidate, dict) and candidate.get("track_id")
    }
    evaluated = 0
    matched = 0
    for track_id in track_ids:
        sources = source_by_track_id.get(track_id)
        if sources is None:
            continue
        evaluated += 1
        if _evidence_source_matches(sources, source):
            matched += 1
    if evaluated == 0:
        return None
    return round(matched / evaluated, 4)


def _evidence_source_matches(sources: tuple[str, ...], source: str) -> bool:
    aliases = {
        "lyric": {"lyric", "lyric_retrieval", "lyric_evidence"},
        "metadata": {"metadata", "metadata_retrieval", "metadata_evidence"},
        "feature": {"feature", "feature_filter", "feature_evidence"},
    }
    accepted = aliases.get(source, {source})
    return any(item in accepted for item in sources)


def _failure_category(state: PipelineState) -> str | None:
    warnings = tuple(str(warning) for warning in state.get("warnings", ()))
    if any("not valid JSON" in warning for warning in warnings):
        return "llm_parse_failure"
    if any("validation error" in warning.casefold() or "validation failed" in warning.casefold() for warning in warnings):
        return "validation_failure"
    if state.get("tool_errors"):
        return "tool_failure"
    if not state.get("candidate_ids"):
        return "empty_retrieval"
    if state.get("candidate_ids") and not state.get("scored_candidates"):
        return "empty_after_scoring_or_verification"
    if state.get("scored_candidates") and not state.get("explanations"):
        return "explanation_missing"
    return None


def _unique_count(records: list[dict[str, Any]], field_name: str) -> int:
    return len(
        {
            str(record.get(field_name)).strip().casefold()
            for record in records
            if str(record.get(field_name, "")).strip()
        }
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _average_optional_scores(values: tuple[Any, ...]) -> float | None:
    return _average(value for value in values if value is not None)


def _average(values: Any) -> float | None:
    numeric = []
    for value in values:
        if value is None:
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 4)


def _reliability_value(state: PipelineState, key: str) -> Any:
    reliability = state.get("reliability")
    if isinstance(reliability, dict):
        return reliability.get(key)
    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError("Expected a string or list of strings.")
    return tuple(str(item).strip() for item in values if str(item).strip())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
