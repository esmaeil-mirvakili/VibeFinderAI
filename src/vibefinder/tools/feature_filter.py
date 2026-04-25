"""Feature filter tool implementation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from vibefinder.config import load_retrieval_prompt_config
from vibefinder.tools.schemas import (
    FEATURE_COLUMNS,
    FeatureCandidateMatch,
    FeatureFilterInput,
    FeatureFilterOutput,
    FeatureRangeFilter,
    FeatureTarget,
)


def filter_by_features(
    songs: pd.DataFrame,
    request: FeatureFilterInput,
    retrieval_prompt_config: dict[str, Any] | None = None,
) -> FeatureFilterOutput:
    """Filter and score songs by numeric audio feature constraints."""

    config = retrieval_prompt_config or load_retrieval_prompt_config()
    numeric_ranges = config["llm_prompt_constraints"]["numeric_ranges"]
    _validate_request_against_config(request, numeric_ranges)

    working = songs.copy()
    if request.candidate_track_ids is not None:
        candidate_ids = set(request.candidate_track_ids)
        working = working[working["track_id"].astype(str).isin(candidate_ids)].copy()

    input_count = len(working)
    warnings: list[str] = []
    passed_range_filters = tuple(_range_label(range_filter) for range_filter in request.range_filters)

    for range_filter in request.range_filters:
        series = pd.to_numeric(working[range_filter.feature], errors="coerce")
        mask = pd.Series(True, index=working.index)
        if range_filter.min_value is not None:
            mask &= series >= range_filter.min_value
        if range_filter.max_value is not None:
            mask &= series <= range_filter.max_value
        working = working[mask].copy()

    if working.empty:
        warnings.append("No candidates matched the feature range filters.")
        logger.info(
            "feature_filter_finished",
            input_count=input_count,
            output_count=0,
            range_filter_count=len(request.range_filters),
            target_count=len(request.targets),
            warnings=warnings,
        )
        return FeatureFilterOutput(
            candidates=(),
            input_count=input_count,
            output_count=0,
            warnings=tuple(warnings),
        )

    matches: list[FeatureCandidateMatch] = []
    for _, row in working.iterrows():
        feature_scores = _score_targets(row, request.targets, numeric_ranges)
        score = _weighted_average(feature_scores, request.targets) if request.targets else 1.0
        feature_values = {
            feature: _clean_number(row.get(feature))
            for feature in FEATURE_COLUMNS
            if feature in row.index
        }
        matches.append(
            FeatureCandidateMatch(
                track_id=str(row["track_id"]),
                score=round(score, 6),
                passed_range_filters=passed_range_filters,
                feature_scores={feature: round(value, 6) for feature, value in feature_scores.items()},
                feature_values=feature_values,
            )
        )

    matches.sort(key=lambda item: (-item.score, item.track_id))
    limited = tuple(matches[: request.top_k])

    if len(matches) > request.top_k:
        warnings.append(f"Returned top {request.top_k} of {len(matches)} feature-matched candidates.")

    logger.info(
        "feature_filter_finished",
        input_count=input_count,
        output_count=len(limited),
        range_filter_count=len(request.range_filters),
        target_count=len(request.targets),
        warnings=warnings,
    )
    return FeatureFilterOutput(
        candidates=limited,
        input_count=input_count,
        output_count=len(limited),
        warnings=tuple(warnings),
    )


def _validate_request_against_config(
    request: FeatureFilterInput,
    numeric_ranges: dict[str, dict[str, float | int | None]],
) -> None:
    for range_filter in request.range_filters:
        _validate_feature_bounds(range_filter, numeric_ranges)
    for target in request.targets:
        if target.feature not in numeric_ranges:
            raise ValueError(f"Feature is not present in retrieval prompt config: {target.feature}")
        if target.target_value is not None:
            bounds = numeric_ranges[target.feature]
            minimum = bounds["min"]
            maximum = bounds["max"]
            if minimum is not None and target.target_value < minimum:
                raise ValueError(f"target_value for {target.feature} is below dataset minimum {minimum}.")
            if maximum is not None and target.target_value > maximum:
                raise ValueError(f"target_value for {target.feature} is above dataset maximum {maximum}.")


def _validate_feature_bounds(
    range_filter: FeatureRangeFilter,
    numeric_ranges: dict[str, dict[str, float | int | None]],
) -> None:
    if range_filter.feature not in numeric_ranges:
        raise ValueError(f"Feature is not present in retrieval prompt config: {range_filter.feature}")
    bounds = numeric_ranges[range_filter.feature]
    minimum = bounds["min"]
    maximum = bounds["max"]
    if minimum is not None and range_filter.min_value is not None and range_filter.min_value < minimum:
        raise ValueError(f"min_value for {range_filter.feature} is below dataset minimum {minimum}.")
    if maximum is not None and range_filter.max_value is not None and range_filter.max_value > maximum:
        raise ValueError(f"max_value for {range_filter.feature} is above dataset maximum {maximum}.")


def _score_targets(
    row: pd.Series,
    targets: tuple[FeatureTarget, ...],
    numeric_ranges: dict[str, dict[str, float | int | None]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for target in targets:
        value = _clean_number(row.get(target.feature))
        if value is None:
            scores[target.feature] = 0.0
            continue

        bounds = numeric_ranges[target.feature]
        minimum = float(bounds["min"])
        maximum = float(bounds["max"])
        if target.target_value is not None:
            scores[target.feature] = _closeness_score(float(value), float(target.target_value), minimum, maximum)
        else:
            low, high = _direction_range(target.direction, minimum, maximum)
            scores[target.feature] = _range_closeness_score(float(value), low, high, minimum, maximum)
    return scores


def _direction_range(direction: str | None, minimum: float, maximum: float) -> tuple[float, float]:
    span = maximum - minimum
    if direction == "low":
        return minimum, minimum + span / 3
    if direction == "medium":
        return minimum + span / 3, minimum + 2 * span / 3
    if direction == "high":
        return minimum + 2 * span / 3, maximum
    raise ValueError("direction is required when target_value is not provided.")


def _closeness_score(value: float, target: float, minimum: float, maximum: float) -> float:
    span = max(maximum - minimum, 1e-9)
    return max(0.0, 1.0 - abs(value - target) / span)


def _range_closeness_score(
    value: float,
    low: float,
    high: float,
    minimum: float,
    maximum: float,
) -> float:
    if low <= value <= high:
        return 1.0
    target = low if value < low else high
    return _closeness_score(value, target, minimum, maximum)


def _weighted_average(
    feature_scores: dict[str, float],
    targets: tuple[FeatureTarget, ...],
) -> float:
    weights = {target.feature: target.weight for target in targets}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    return sum(feature_scores[feature] * weights[feature] for feature in feature_scores) / total_weight


def _range_label(range_filter: FeatureRangeFilter) -> str:
    lower = "*" if range_filter.min_value is None else range_filter.min_value
    upper = "*" if range_filter.max_value is None else range_filter.max_value
    return f"{range_filter.feature}:{lower}..{upper}"


def _clean_number(value: Any) -> float | int | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric
