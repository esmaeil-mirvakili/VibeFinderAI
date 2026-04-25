from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from vibefinder.tools import (
    ReliabilityCandidate,
    ReliabilityInput,
    ToolContext,
    ToolError,
    ToolResult,
    ToolRunner,
    assess_reliability,
    get_tool_registry,
)


def test_reliability_returns_high_confidence_for_strong_supported_results():
    request = ReliabilityInput(
        final_candidates=(
            _candidate("song-a", 1, 0.95, artist="Aster", album="Night Signals"),
            _candidate("song-b", 2, 0.9, artist="Boreal", album="Daylight"),
            _candidate("song-c", 3, 0.84, artist="Cinder", album="Road"),
            _candidate("song-d", 4, 0.8, artist="Dune", album="Long Drive"),
            _candidate("song-e", 5, 0.76, artist="Elm", album="City Room"),
        ),
        requested_count=5,
        retrieval_modes_used=("lyric_retrieval", "metadata_retrieval", "feature_filter"),
    )

    output = assess_reliability(_sample_rows(), request)

    assert output.confidence_label == "high"
    assert output.confidence_score >= 0.75
    assert output.warnings == ()
    assert output.support_summary.evidence_source_counts == {
        "feature": 5,
        "lyric": 5,
        "metadata": 5,
    }


def test_reliability_warns_for_empty_results():
    output = assess_reliability(
        _sample_rows(),
        ReliabilityInput(final_candidates=(), requested_count=5),
    )

    assert output.confidence_label == "low"
    assert output.confidence_score == 0.0
    assert output.support_summary.candidate_count == 0
    assert output.warnings == ("No final candidates were available for reliability assessment.",)


def test_reliability_warns_for_weak_scores_constraints_and_repetition():
    request = ReliabilityInput(
        final_candidates=(
            ReliabilityCandidate(
                track_id="song-a",
                rank=1,
                final_score=0.4,
                evidence_sources=("metadata",),
                verified=False,
                constraint_violations=("language mismatch",),
                track_artist="Aster",
                track_album_name="Night Signals",
            ),
            ReliabilityCandidate(
                track_id="song-x",
                rank=2,
                final_score=0.3,
                evidence_sources=(),
                verified=True,
                track_artist="Aster",
                track_album_name="Night Signals",
            ),
        ),
        requested_count=5,
        retrieval_modes_used=("metadata_retrieval",),
        prior_warnings=("Metadata retrieval returned few candidates.",),
        verifier_warnings=("One candidate violated language.",),
        critic_issues=("Results are too narrow.",),
        hard_constraint_violations=("global language mismatch",),
        revision_used=True,
        revision_succeeded=False,
    )

    output = assess_reliability(_sample_rows(), request)

    assert output.confidence_label == "low"
    assert output.support_summary.constraint_violation_count == 2
    assert output.support_summary.missing_dataset_count == 1
    assert output.support_summary.repeated_artist_count == 1
    assert output.support_summary.repeated_album_count == 1
    assert "Only 2 final candidates were returned for requested 5." in output.warnings
    assert "2 hard constraint violations were reported." in output.warnings
    assert "1 final candidates were missing from the dataset." in output.warnings
    assert "Revision was used but did not resolve the reported issues." in output.warnings
    assert "Prior tool warning: Metadata retrieval returned few candidates." in output.warnings


def test_reliability_rejects_invalid_candidate_score():
    with pytest.raises(ValidationError):
        ReliabilityInput(
            final_candidates=(
                ReliabilityCandidate(track_id="song-a", rank=1, final_score=1.2),
            )
        )


def test_reliability_runs_through_tool_runner():
    context = ToolContext(songs=_sample_rows(), retrieval_prompt_config={})
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run(
        "reliability",
        {
            "final_candidates": [
                {
                    "track_id": "song-a",
                    "rank": 1,
                    "final_score": 0.9,
                    "evidence_sources": ["lyric", "feature"],
                    "verified": True,
                }
            ],
            "requested_count": 1,
            "retrieval_modes_used": ["lyric_retrieval", "feature_filter"],
        },
    )

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.output["confidence_label"] in {"medium", "high"}
    assert result.output["support_summary"]["candidate_count"] == 1


def test_reliability_tool_runner_returns_validation_error():
    context = ToolContext(songs=_sample_rows(), retrieval_prompt_config={})
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run(
        "reliability",
        {"final_candidates": [{"track_id": "song-a", "rank": 1, "final_score": -0.1}]},
    )

    assert isinstance(result, ToolError)
    assert result.error_type == "validation_error"


def test_reliability_tool_runner_returns_runtime_error_for_missing_columns():
    context = ToolContext(songs=pd.DataFrame([{"track_name": "No ID"}]), retrieval_prompt_config={})
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run(
        "reliability",
        {"final_candidates": [{"track_id": "song-a", "rank": 1, "final_score": 0.9}]},
    )

    assert isinstance(result, ToolError)
    assert result.error_type == "ValueError"
    assert "missing reliability columns" in result.message


def test_reliability_prompt_spec_is_schema_backed_and_prompt_safe():
    spec = get_tool_registry()["reliability"].to_prompt_spec()

    assert spec["name"] == "reliability"
    assert spec["input_schema"]["title"] == "ReliabilityInput"
    assert spec["output_schema"]["title"] == "ReliabilityOutput"
    assert spec["constraints"]["confidence_labels"] == ["low", "medium", "high"]
    assert "lyrics" not in str(spec)
    assert "invented semantic labels" in str(spec)


def test_tool_registry_contains_reliability_tool():
    assert "reliability" in get_tool_registry()


def _candidate(
    track_id: str,
    rank: int,
    score: float,
    artist: str,
    album: str,
) -> ReliabilityCandidate:
    return ReliabilityCandidate(
        track_id=track_id,
        rank=rank,
        final_score=score,
        evidence_sources=("lyric", "metadata", "feature"),
        verified=True,
        verifier_score=score,
        track_artist=artist,
        track_album_name=album,
    )


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"track_id": "song-a"},
            {"track_id": "song-b"},
            {"track_id": "song-c"},
            {"track_id": "song-d"},
            {"track_id": "song-e"},
        ]
    )
