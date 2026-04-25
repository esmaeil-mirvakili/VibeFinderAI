"""Metadata retrieval tool implementation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from vibefinder.config import load_retrieval_prompt_config
from vibefinder.tools.schemas import (
    CATEGORICAL_METADATA_COLUMNS,
    FULL_TEXT_METADATA_COLUMNS,
    MetadataCandidateMatch,
    MetadataRetrievalInput,
    MetadataRetrievalOutput,
)


def retrieve_by_metadata(
    songs: pd.DataFrame,
    request: MetadataRetrievalInput,
    retrieval_prompt_config: dict[str, Any] | None = None,
) -> MetadataRetrievalOutput:
    """Retrieve songs using categorical and full-text metadata constraints."""

    config = retrieval_prompt_config or load_retrieval_prompt_config()
    _validate_request_against_config(request, config)

    working = songs.copy()
    if request.candidate_track_ids is not None:
        candidate_ids = set(request.candidate_track_ids)
        working = working[working["track_id"].astype(str).isin(candidate_ids)].copy()

    input_count = len(working)
    warnings: list[str] = []

    for categorical_filter in request.categorical_filters:
        accepted = {_normalize(value) for value in categorical_filter.values}
        field_values = working[categorical_filter.field].fillna("").astype(str).map(_normalize)
        working = working[field_values.isin(accepted)].copy()

    if working.empty:
        warnings.append("No candidates matched the metadata categorical filters.")
        logger.info(
            "metadata_retrieval_finished",
            input_count=input_count,
            output_count=0,
            categorical_filter_count=len(request.categorical_filters),
            text_query_count=len(request.text_queries),
            warnings=warnings,
        )
        return MetadataRetrievalOutput(
            candidates=(),
            input_count=input_count,
            output_count=0,
            warnings=tuple(warnings),
        )

    matches: list[MetadataCandidateMatch] = []
    for _, row in working.iterrows():
        categorical_matches = _categorical_matches(row, request)
        text_matches, text_score, matched_text_query_count = _text_matches(row, request)
        if not _passes_text_match_mode(request, matched_text_query_count):
            continue

        categorical_score = float(len(categorical_matches))
        score = categorical_score + text_score
        reasons = tuple(
            [f"{field}={value}" for field, value in categorical_matches.items()]
            + [f"{field} contains '{query}'" for field, query in text_matches.items()]
        )
        matches.append(
            MetadataCandidateMatch(
                track_id=str(row["track_id"]),
                score=round(score, 6),
                categorical_matches=categorical_matches,
                text_matches=text_matches,
                match_reasons=reasons,
            )
        )

    matches.sort(key=lambda item: (-item.score, item.track_id))
    limited = tuple(matches[: request.top_k])

    if not limited:
        warnings.append("No candidates matched the metadata text queries.")
    elif len(matches) > request.top_k:
        warnings.append(f"Returned top {request.top_k} of {len(matches)} metadata-matched candidates.")

    logger.info(
        "metadata_retrieval_finished",
        input_count=input_count,
        output_count=len(limited),
        categorical_filter_count=len(request.categorical_filters),
        text_query_count=len(request.text_queries),
        warnings=warnings,
    )
    return MetadataRetrievalOutput(
        candidates=limited,
        input_count=input_count,
        output_count=len(limited),
        warnings=tuple(warnings),
    )


def _validate_request_against_config(
    request: MetadataRetrievalInput,
    config: dict[str, Any],
) -> None:
    categorical_values = config["llm_prompt_constraints"]["categorical_values"]
    full_text_columns = config["llm_prompt_constraints"]["full_text_search_columns"]

    for categorical_filter in request.categorical_filters:
        if categorical_filter.field not in categorical_values:
            raise ValueError(f"Field is not configured as categorical metadata: {categorical_filter.field}")
        valid_values = {_normalize(value) for value in categorical_values[categorical_filter.field]}
        invalid = [value for value in categorical_filter.values if _normalize(value) not in valid_values]
        if invalid:
            raise ValueError(
                f"Invalid values for {categorical_filter.field}: {', '.join(invalid)}"
            )

    for text_query in request.text_queries:
        if text_query.field not in full_text_columns:
            raise ValueError(f"Field is not configured for full-text metadata search: {text_query.field}")


def _categorical_matches(
    row: pd.Series,
    request: MetadataRetrievalInput,
) -> dict[str, str]:
    matches: dict[str, str] = {}
    for categorical_filter in request.categorical_filters:
        value = str(row.get(categorical_filter.field, ""))
        accepted = {_normalize(item) for item in categorical_filter.values}
        if _normalize(value) in accepted:
            matches[categorical_filter.field] = value
    return matches


def _text_matches(
    row: pd.Series,
    request: MetadataRetrievalInput,
) -> tuple[dict[str, str], float, int]:
    matches: dict[str, str] = {}
    score = 0.0
    matched_count = 0
    for text_query in request.text_queries:
        haystack = str(row.get(text_query.field, ""))
        if _normalize(text_query.query) in _normalize(haystack):
            matches[text_query.field] = text_query.query
            score += text_query.weight
            matched_count += 1
    return matches, score, matched_count


def _passes_text_match_mode(request: MetadataRetrievalInput, matched_text_query_count: int) -> bool:
    if not request.text_queries:
        return True
    if request.text_match_mode == "all":
        return matched_text_query_count == len(request.text_queries)
    if request.text_match_mode == "any":
        return matched_text_query_count > 0
    raise ValueError(f"Unsupported text_match_mode: {request.text_match_mode}")


def _normalize(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def metadata_prompt_constraints(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return prompt-safe metadata constraints."""

    active_config = config or load_retrieval_prompt_config()
    return {
        "categorical_fields": list(CATEGORICAL_METADATA_COLUMNS),
        "categorical_values": active_config["llm_prompt_constraints"]["categorical_values"],
        "full_text_fields": list(FULL_TEXT_METADATA_COLUMNS),
        "full_text_search_columns": active_config["llm_prompt_constraints"]["full_text_search_columns"],
        "text_match_modes": ["all", "any"],
        "default_text_match_mode": "all",
        "note": "Do not treat full_text_fields as enum values.",
    }
