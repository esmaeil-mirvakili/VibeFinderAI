"""Candidate scoring tool implementation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from vibefinder.tools.schemas import (
    CandidateEvidence,
    CandidateScoreMatch,
    CandidateScoringInput,
    CandidateScoringOutput,
)


@dataclass(frozen=True)
class _CandidateScore:
    track_id: str
    base_score: float
    final_score: float
    score_components: dict[str, float]
    evidence_sources: tuple[str, ...]
    popularity: float | None
    diversity_penalty: float
    track_artist: str | None
    track_album_name: str | None


def score_candidates(
    songs: pd.DataFrame,
    request: CandidateScoringInput,
) -> CandidateScoringOutput:
    """Combine retrieval evidence into deterministic ranked candidates."""

    _validate_required_columns(songs)
    song_lookup = _song_lookup(songs)
    candidate_ids = _candidate_ids(request)
    input_count = len(candidate_ids)
    warnings: list[str] = []

    if not candidate_ids:
        warnings.append("No candidate ids were available for scoring.")
        return CandidateScoringOutput(candidates=(), input_count=0, output_count=0, warnings=tuple(warnings))

    missing_ids = tuple(track_id for track_id in candidate_ids if track_id not in song_lookup)
    if missing_ids:
        warnings.append(f"Skipped {len(missing_ids)} candidates missing from the dataset.")

    lyric_scores = _normalized_evidence_scores(request.lyric_evidence)
    metadata_scores = _normalized_evidence_scores(request.metadata_evidence)
    feature_scores = _normalized_evidence_scores(request.feature_evidence)
    has_lyric_intent = bool(request.lyric_intent and request.lyric_intent.strip())

    scored: list[_CandidateScore] = []
    for track_id in candidate_ids:
        row = song_lookup.get(track_id)
        if row is None:
            continue

        components = {
            "lyric": lyric_scores.get(track_id, 0.0),
            "metadata": metadata_scores.get(track_id, 0.0),
            "feature": feature_scores.get(track_id, 0.0),
            "popularity": _popularity_score(row),
        }
        evidence_sources = tuple(
            source_name
            for source, source_name in (
                ("lyric", "lyric_retrieval"),
                ("metadata", "metadata_retrieval"),
                ("feature", "feature_filter"),
            )
            if components[source] > 0
        )
        base_score = _base_score(
            components,
            request,
            source_count=len(evidence_sources),
            has_lyric_intent=has_lyric_intent,
        )
        scored.append(
            _CandidateScore(
                track_id=track_id,
                base_score=base_score,
                final_score=base_score,
                score_components=components,
                evidence_sources=evidence_sources,
                popularity=_clean_optional_float(row.get("track_popularity")),
                diversity_penalty=0.0,
                track_artist=_clean_text(row.get("track_artist")),
                track_album_name=_clean_text(row.get("track_album_name")),
            )
        )

    if not scored:
        warnings.append("No candidates could be scored after dataset lookup.")
        return CandidateScoringOutput(
            candidates=(),
            input_count=input_count,
            output_count=0,
            warnings=tuple(warnings),
        )

    scored.sort(key=lambda item: (-item.base_score, item.track_id))
    if request.apply_diversity_penalty and request.weights.diversity_penalty > 0:
        scored = _apply_diversity_penalty(scored, request.weights.diversity_penalty)

    scored.sort(key=lambda item: (-item.final_score, item.track_id))
    before_lyric_rate = _lyric_evidence_rate(scored[: request.top_k])
    if has_lyric_intent and request.lyric_evidence:
        scored = _promote_lyric_supported_candidates(scored, request.top_k)
        scored.sort(key=lambda item: (-item.final_score, item.track_id))
    after_lyric_rate = _lyric_evidence_rate(scored[: request.top_k])
    limited = scored[: request.top_k]

    if len(scored) > request.top_k:
        warnings.append(f"Returned top {request.top_k} of {len(scored)} scored candidates.")
    if any(not item.evidence_sources for item in scored):
        warnings.append("Some scored candidates had no retrieval evidence and were ranked by popularity only.")
    if has_lyric_intent and request.lyric_evidence:
        lyric_available_count = sum(1 for item in scored if "lyric_retrieval" in item.evidence_sources)
        lyric_top_count = sum(1 for item in limited if "lyric_retrieval" in item.evidence_sources)
        if lyric_available_count and lyric_top_count == 0:
            warnings.append("Lyric retrieval produced candidates, but none survived the final top-k ranking.")

    matches = tuple(
        CandidateScoreMatch(
            track_id=item.track_id,
            rank=rank,
            final_score=round(item.final_score, 6),
            score_components={key: round(value, 6) for key, value in item.score_components.items()},
            evidence_sources=item.evidence_sources,
            popularity=item.popularity,
            diversity_penalty=round(item.diversity_penalty, 6),
            track_artist=item.track_artist,
            track_album_name=item.track_album_name,
        )
        for rank, item in enumerate(limited, start=1)
    )

    logger.info(
        "candidate_scoring_finished",
        input_count=input_count,
        output_count=len(matches),
        lyric_evidence_count=len(request.lyric_evidence),
        lyric_intent_present=has_lyric_intent,
        lyric_evidence_rate_before=round(before_lyric_rate, 4),
        lyric_evidence_rate_after=round(after_lyric_rate, 4),
        metadata_evidence_count=len(request.metadata_evidence),
        feature_evidence_count=len(request.feature_evidence),
        apply_diversity_penalty=request.apply_diversity_penalty,
        warnings=warnings,
    )
    return CandidateScoringOutput(
        candidates=matches,
        input_count=input_count,
        output_count=len(matches),
        warnings=tuple(warnings),
    )


def candidate_scoring_prompt_constraints() -> dict[str, object]:
    """Return prompt-safe candidate scoring constraints."""

    return {
        "required_fields": ["track_id", "score"],
        "accepted_evidence_sources": ["lyric_evidence", "metadata_evidence", "feature_evidence"],
        "default_weights": {
            "lyric": 0.4,
            "metadata": 0.25,
            "feature": 0.3,
            "popularity": 0.05,
            "diversity_penalty": 0.05,
        },
        "popularity_weight_max": 0.2,
        "note": (
            "Use retrieval tool outputs as evidence. Popularity is a small adjustment only. "
            "The tool uses dataset artist and album fields only for diversity penalties."
        ),
    }


def _validate_required_columns(songs: pd.DataFrame) -> None:
    required = ("track_id", "track_popularity", "track_artist", "track_album_name")
    missing = [column for column in required if column not in songs.columns]
    if missing:
        raise ValueError(f"Songs DataFrame is missing candidate scoring columns: {', '.join(missing)}")


def _song_lookup(songs: pd.DataFrame) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    for _, row in songs.iterrows():
        track_id = str(row["track_id"])
        if track_id not in lookup:
            lookup[track_id] = row
    return lookup


def _candidate_ids(request: CandidateScoringInput) -> tuple[str, ...]:
    if request.candidate_track_ids is not None:
        return _dedupe_preserving_order(request.candidate_track_ids)
    evidence_ids = tuple(
        item.track_id
        for evidence in (request.lyric_evidence, request.metadata_evidence, request.feature_evidence)
        for item in evidence
    )
    return _dedupe_preserving_order(evidence_ids)


def _dedupe_preserving_order(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            deduped.append(cleaned)
            seen.add(cleaned)
    return tuple(deduped)


def _normalized_evidence_scores(evidence: tuple[CandidateEvidence, ...]) -> dict[str, float]:
    raw_scores: dict[str, float] = {}
    for item in evidence:
        current = raw_scores.get(item.track_id)
        score = float(item.score)
        if current is None or score > current:
            raw_scores[item.track_id] = score

    if not raw_scores:
        return {}

    max_score = max(raw_scores.values())
    if max_score > 1:
        return {track_id: _clamp(score / max_score) for track_id, score in raw_scores.items()}
    return {track_id: _clamp(score) for track_id, score in raw_scores.items()}


def _base_score(
    components: dict[str, float],
    request: CandidateScoringInput,
    *,
    source_count: int,
    has_lyric_intent: bool,
) -> float:
    weights = request.weights
    lyric_weight = weights.lyric
    metadata_weight = weights.metadata
    feature_weight = weights.feature
    if has_lyric_intent and request.lyric_evidence:
        lyric_weight = max(lyric_weight, 0.55)
        metadata_weight *= 0.75
        feature_weight *= 0.75
    weighted_sum = (
        components["lyric"] * lyric_weight
        + components["metadata"] * metadata_weight
        + components["feature"] * feature_weight
        + components["popularity"] * weights.popularity
    )
    total_weight = lyric_weight + metadata_weight + feature_weight + weights.popularity
    source_agreement_bonus = 0.03 * max(0, source_count - 1)
    lyric_adjustment = 0.0
    if has_lyric_intent and request.lyric_evidence:
        lyric_adjustment = 0.08 if components["lyric"] > 0 else -0.12
    return _clamp(weighted_sum / total_weight + source_agreement_bonus + lyric_adjustment)


def _lyric_evidence_rate(scored: list[_CandidateScore]) -> float:
    if not scored:
        return 0.0
    lyric_count = sum(1 for item in scored if "lyric_retrieval" in item.evidence_sources)
    return lyric_count / len(scored)


def _promote_lyric_supported_candidates(
    scored: list[_CandidateScore],
    top_k: int,
) -> list[_CandidateScore]:
    if not scored or top_k <= 0:
        return scored
    top = scored[:top_k]
    if any("lyric_retrieval" in item.evidence_sources for item in top):
        return scored
    lyric_candidates = [item for item in scored[top_k:] if "lyric_retrieval" in item.evidence_sources]
    if not lyric_candidates:
        return scored
    promoted = lyric_candidates[0]
    adjusted = [
        _replace_final_score(item, _clamp(top[-1].final_score + 0.001))
        if item.track_id == promoted.track_id
        else item
        for item in scored
    ]
    return adjusted


def _replace_final_score(item: _CandidateScore, final_score: float) -> _CandidateScore:
    return _CandidateScore(
        track_id=item.track_id,
        base_score=item.base_score,
        final_score=final_score,
        score_components=item.score_components,
        evidence_sources=item.evidence_sources,
        popularity=item.popularity,
        diversity_penalty=item.diversity_penalty,
        track_artist=item.track_artist,
        track_album_name=item.track_album_name,
    )


def _apply_diversity_penalty(
    scored: list[_CandidateScore],
    penalty_weight: float,
) -> list[_CandidateScore]:
    artist_counts: dict[str, int] = {}
    album_counts: dict[str, int] = {}
    adjusted: list[_CandidateScore] = []

    for item in scored:
        artist_key = _normalize(item.track_artist)
        album_key = _normalize(item.track_album_name)
        repeat_count = 0
        if artist_key:
            repeat_count += artist_counts.get(artist_key, 0)
            artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
        if album_key:
            repeat_count += album_counts.get(album_key, 0)
            album_counts[album_key] = album_counts.get(album_key, 0) + 1

        penalty = min(0.3, penalty_weight * repeat_count)
        adjusted.append(
            _CandidateScore(
                track_id=item.track_id,
                base_score=item.base_score,
                final_score=_clamp(item.base_score - penalty),
                score_components=item.score_components,
                evidence_sources=item.evidence_sources,
                popularity=item.popularity,
                diversity_penalty=penalty,
                track_artist=item.track_artist,
                track_album_name=item.track_album_name,
            )
        )
    return adjusted


def _popularity_score(row: pd.Series) -> float:
    popularity = _clean_optional_float(row.get("track_popularity"))
    if popularity is None:
        return 0.0
    return _clamp(popularity / 100.0)


def _clean_optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.casefold().strip().split())


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
