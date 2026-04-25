"""Reliability tool implementation."""

from __future__ import annotations

from collections import Counter

import pandas as pd
from loguru import logger

from vibefinder.tools.schemas import (
    ConfidenceLabel,
    ReliabilityCandidate,
    ReliabilityInput,
    ReliabilityOutput,
    ReliabilitySupportSummary,
)


def assess_reliability(
    songs: pd.DataFrame,
    request: ReliabilityInput,
) -> ReliabilityOutput:
    """Compute confidence and warnings from measurable pipeline evidence."""

    _validate_required_columns(songs)

    candidates = _dedupe_candidates(request.final_candidates)
    dataset_track_ids = {str(track_id).strip() for track_id in songs["track_id"].dropna()}
    missing_dataset_count = sum(
        1 for candidate in candidates if candidate.track_id.strip() not in dataset_track_ids
    )
    support_summary = _support_summary(candidates, request, missing_dataset_count)
    warnings = _warnings(candidates, request, support_summary)
    confidence_score = _confidence_score(candidates, request, support_summary)
    confidence_label = _confidence_label(confidence_score)

    logger.info(
        "reliability_assessment_finished",
        confidence_label=confidence_label,
        confidence_score=confidence_score,
        candidate_count=support_summary.candidate_count,
        warning_count=len(warnings),
        revision_used=request.revision_used,
        retrieval_modes_used=request.retrieval_modes_used,
    )
    return ReliabilityOutput(
        confidence_label=confidence_label,
        confidence_score=confidence_score,
        warnings=warnings,
        support_summary=support_summary,
    )


def reliability_prompt_constraints() -> dict[str, object]:
    """Return prompt-safe reliability constraints."""

    return {
        "required_candidate_fields": ["track_id", "rank", "final_score"],
        "optional_candidate_fields": [
            "evidence_sources",
            "score_components",
            "verified",
            "verifier_score",
            "constraint_violations",
            "track_artist",
            "track_album_name",
        ],
        "confidence_labels": ["low", "medium", "high"],
        "confidence_basis": [
            "candidate_count",
            "score_distribution",
            "retrieval_modes_used",
            "evidence_source_coverage",
            "verifier_results",
            "critic_issues",
            "hard_constraint_violations",
            "prior_tool_warnings",
            "revision_outcome",
            "artist_album_repetition",
        ],
        "note": (
            "Confidence is computed from measurable pipeline evidence only. "
            "Do not supply large text bodies or invented semantic labels."
        ),
    }


def _validate_required_columns(songs: pd.DataFrame) -> None:
    if "track_id" not in songs.columns:
        raise ValueError("Songs DataFrame is missing reliability columns: track_id")


def _dedupe_candidates(
    candidates: tuple[ReliabilityCandidate, ...],
) -> tuple[ReliabilityCandidate, ...]:
    seen: set[str] = set()
    deduped: list[ReliabilityCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.rank, item.track_id)):
        track_id = candidate.track_id.strip()
        if track_id and track_id not in seen:
            deduped.append(candidate)
            seen.add(track_id)
    return tuple(deduped)


def _support_summary(
    candidates: tuple[ReliabilityCandidate, ...],
    request: ReliabilityInput,
    missing_dataset_count: int,
) -> ReliabilitySupportSummary:
    scores = [candidate.final_score for candidate in candidates]
    verified_values = [candidate.verified for candidate in candidates if candidate.verified is not None]
    evidence_counts = Counter(
        source.strip()
        for candidate in candidates
        for source in candidate.evidence_sources
        if source.strip()
    )
    constraint_violation_count = len(request.hard_constraint_violations) + sum(
        len(candidate.constraint_violations) for candidate in candidates
    )

    return ReliabilitySupportSummary(
        candidate_count=len(candidates),
        requested_count=request.requested_count,
        average_final_score=round(sum(scores) / len(scores), 6) if scores else 0.0,
        top_score=round(max(scores), 6) if scores else 0.0,
        score_spread=round(max(scores) - min(scores), 6) if scores else 0.0,
        verified_count=sum(1 for value in verified_values if value is True),
        unverified_count=sum(1 for value in verified_values if value is False),
        constraint_violation_count=constraint_violation_count,
        low_score_count=sum(score < request.minimum_acceptable_score for score in scores),
        missing_dataset_count=missing_dataset_count,
        evidence_source_counts=dict(sorted(evidence_counts.items())),
        retrieval_modes_used=request.retrieval_modes_used,
        revision_used=request.revision_used,
        revision_succeeded=request.revision_succeeded,
        prior_warning_count=(
            len(request.prior_warnings)
            + len(request.verifier_warnings)
            + sum(len(candidate.warnings) for candidate in candidates)
        ),
        critic_issue_count=len(request.critic_issues),
        repeated_artist_count=_repeat_count(candidate.track_artist for candidate in candidates),
        repeated_album_count=_repeat_count(candidate.track_album_name for candidate in candidates),
    )


def _warnings(
    candidates: tuple[ReliabilityCandidate, ...],
    request: ReliabilityInput,
    support_summary: ReliabilitySupportSummary,
) -> tuple[str, ...]:
    warnings: list[str] = []

    if support_summary.candidate_count == 0:
        warnings.append("No final candidates were available for reliability assessment.")
        return tuple(warnings)

    if support_summary.candidate_count < request.requested_count:
        warnings.append(
            f"Only {support_summary.candidate_count} final candidates were returned for requested "
            f"{request.requested_count}."
        )
    if support_summary.top_score < request.minimum_acceptable_score:
        warnings.append(
            f"Top final score {support_summary.top_score:.2f} is below the minimum acceptable score "
            f"{request.minimum_acceptable_score:.2f}."
        )
    if support_summary.average_final_score < request.minimum_acceptable_score:
        warnings.append(
            f"Average final score {support_summary.average_final_score:.2f} is below the minimum "
            f"acceptable score {request.minimum_acceptable_score:.2f}."
        )
    if support_summary.low_score_count:
        warnings.append(f"{support_summary.low_score_count} final candidates have weak final scores.")
    if not request.retrieval_modes_used:
        warnings.append("No retrieval modes were recorded in the trace.")
    if not support_summary.evidence_source_counts:
        warnings.append("Final candidates do not include retrieval evidence sources.")
    if all(candidate.verified is None for candidate in candidates):
        warnings.append("No verifier outcomes were supplied for the final candidates.")
    elif support_summary.unverified_count:
        warnings.append(f"{support_summary.unverified_count} final candidates failed verifier checks.")
    if support_summary.constraint_violation_count:
        warnings.append(
            f"{support_summary.constraint_violation_count} hard constraint violations were reported."
        )
    if support_summary.missing_dataset_count:
        warnings.append(f"{support_summary.missing_dataset_count} final candidates were missing from the dataset.")
    if support_summary.repeated_artist_count:
        warnings.append(f"{support_summary.repeated_artist_count} repeated artist placements were detected.")
    if support_summary.repeated_album_count:
        warnings.append(f"{support_summary.repeated_album_count} repeated album placements were detected.")
    if request.revision_used and request.revision_succeeded is False:
        warnings.append("Revision was used but did not resolve the reported issues.")

    warnings.extend(_limited_prefixed("Prior tool warning", request.prior_warnings))
    warnings.extend(_limited_prefixed("Verifier warning", request.verifier_warnings))
    warnings.extend(_limited_prefixed("Critic issue", request.critic_issues))

    return _dedupe_strings(tuple(warnings))


def _confidence_score(
    candidates: tuple[ReliabilityCandidate, ...],
    request: ReliabilityInput,
    support_summary: ReliabilitySupportSummary,
) -> float:
    if support_summary.candidate_count == 0:
        return 0.0

    coverage_score = min(support_summary.candidate_count / request.requested_count, 1.0)
    quality_score = support_summary.average_final_score
    top_score = support_summary.top_score
    evidence_score = min(len(support_summary.evidence_source_counts) / 3, 1.0)
    verifier_score = _verifier_score(candidates)
    constraint_score = 1.0 - min(
        support_summary.constraint_violation_count / max(support_summary.candidate_count, 1),
        1.0,
    )

    score = (
        coverage_score * 0.2
        + quality_score * 0.3
        + top_score * 0.15
        + evidence_score * 0.15
        + verifier_score * 0.1
        + constraint_score * 0.1
    )
    score -= min(support_summary.prior_warning_count * 0.03, 0.15)
    score -= min(support_summary.critic_issue_count * 0.05, 0.2)
    score -= min(support_summary.missing_dataset_count * 0.1, 0.3)
    score -= min((support_summary.repeated_artist_count + support_summary.repeated_album_count) * 0.03, 0.15)
    if request.revision_used and request.revision_succeeded is False:
        score -= 0.1
    return round(_clamp(score), 6)


def _verifier_score(candidates: tuple[ReliabilityCandidate, ...]) -> float:
    verifier_scores = [candidate.verifier_score for candidate in candidates if candidate.verifier_score is not None]
    if verifier_scores:
        return _clamp(sum(verifier_scores) / len(verifier_scores))

    verified_values = [candidate.verified for candidate in candidates if candidate.verified is not None]
    if verified_values:
        return sum(1 for value in verified_values if value is True) / len(verified_values)

    return 0.6


def _confidence_label(score: float) -> ConfidenceLabel:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _repeat_count(values: object) -> int:
    counts = Counter(_normalize(value) for value in values)
    return sum(count - 1 for value, count in counts.items() if value and count > 1)


def _normalize(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).casefold().strip().split())


def _limited_prefixed(prefix: str, values: tuple[str, ...], limit: int = 3) -> list[str]:
    cleaned = [value.strip() for value in values if value.strip()]
    return [f"{prefix}: {value}" for value in cleaned[:limit]]


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return tuple(deduped)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
