from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from vibefinder.tools import (
    CandidateEvidence,
    CandidateScoringInput,
    CandidateScoringWeights,
    MAX_FINAL_TOP_K,
    ToolContext,
    ToolError,
    ToolResult,
    ToolRunner,
    get_tool_registry,
    score_candidates,
)


def test_candidate_scoring_combines_retrieval_evidence_and_popularity():
    request = CandidateScoringInput(
        lyric_evidence=(
            CandidateEvidence(track_id="song-a", score=0.9),
            CandidateEvidence(track_id="song-b", score=0.5),
        ),
        metadata_evidence=(CandidateEvidence(track_id="song-a", score=2),),
        feature_evidence=(
            CandidateEvidence(track_id="song-b", score=0.95),
            CandidateEvidence(track_id="song-c", score=0.4),
        ),
        top_k=3,
    )

    output = score_candidates(_sample_scoring_rows(), request)

    assert output.output_count == 3
    assert [candidate.track_id for candidate in output.candidates] == ["song-a", "song-b", "song-c"]
    assert output.candidates[0].evidence_sources == ("lyric_retrieval", "metadata_retrieval")
    assert output.candidates[0].score_components["metadata"] == 1.0
    assert output.candidates[0].score_components["popularity"] == 0.1


def test_candidate_scoring_keeps_popularity_as_small_adjustment():
    request = CandidateScoringInput(
        lyric_evidence=(
            CandidateEvidence(track_id="song-a", score=0.9),
            CandidateEvidence(track_id="song-c", score=0.4),
        ),
        candidate_track_ids=("song-a", "song-c"),
        top_k=2,
    )

    output = score_candidates(_sample_scoring_rows(), request)

    assert [candidate.track_id for candidate in output.candidates] == ["song-a", "song-c"]
    assert output.candidates[1].popularity == 100


def test_candidate_scoring_promotes_lyric_evidence_when_lyric_intent_exists():
    request = CandidateScoringInput(
        lyric_intent="betrayal and regret",
        lyric_evidence=(CandidateEvidence(track_id="song-c", score=0.7),),
        feature_evidence=(CandidateEvidence(track_id="song-b", score=1.0),),
        candidate_track_ids=("song-b", "song-c"),
        top_k=2,
    )

    output = score_candidates(_sample_scoring_rows(), request)

    assert output.candidates[0].track_id == "song-c"
    assert "lyric_retrieval" in output.candidates[0].evidence_sources


def test_candidate_scoring_applies_diversity_penalty_after_base_scoring():
    request = CandidateScoringInput(
        lyric_evidence=(
            CandidateEvidence(track_id="song-a", score=0.9),
            CandidateEvidence(track_id="song-d", score=0.89),
        ),
        candidate_track_ids=("song-a", "song-d"),
        weights=CandidateScoringWeights(lyric=1, metadata=0, feature=0, popularity=0, diversity_penalty=0.05),
        top_k=2,
    )

    output = score_candidates(_sample_scoring_rows(), request)

    assert output.candidates[0].track_id == "song-a"
    assert output.candidates[1].track_id == "song-d"
    assert output.candidates[1].diversity_penalty == 0.1
    assert output.candidates[1].final_score < output.candidates[1].score_components["lyric"]


def test_candidate_scoring_rejects_empty_input():
    with pytest.raises(ValidationError):
        CandidateScoringInput()


def test_candidate_scoring_rejects_popularity_as_large_weight():
    with pytest.raises(ValidationError):
        CandidateScoringInput(
            lyric_evidence=(CandidateEvidence(track_id="song-a", score=0.9),),
            weights=CandidateScoringWeights(popularity=0.5),
        )


def test_candidate_scoring_clamps_oversized_top_k():
    request = CandidateScoringInput(
        lyric_evidence=(CandidateEvidence(track_id="song-a", score=0.9),),
        top_k=999,
    )

    assert request.top_k == MAX_FINAL_TOP_K


def test_candidate_scoring_runs_through_tool_runner():
    context = ToolContext(songs=_sample_scoring_rows(), retrieval_prompt_config={})
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run(
        "candidate_scoring",
        {
            "lyric_evidence": [{"track_id": "song-a", "score": 0.9}],
            "feature_evidence": [{"track_id": "song-b", "score": 0.8}],
            "top_k": 1,
        },
    )

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.output["output_count"] == 1
    assert result.output["candidates"][0]["track_id"] == "song-a"


def test_candidate_scoring_tool_runner_returns_validation_error():
    context = ToolContext(songs=_sample_scoring_rows(), retrieval_prompt_config={})
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run("candidate_scoring", {"lyric_evidence": [{"track_id": "", "score": 0.9}]})

    assert isinstance(result, ToolError)
    assert result.error_type == "validation_error"


def test_candidate_scoring_tool_runner_returns_runtime_error_for_missing_columns():
    context = ToolContext(songs=pd.DataFrame([{"track_id": "song-a"}]), retrieval_prompt_config={})
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run(
        "candidate_scoring",
        {"lyric_evidence": [{"track_id": "song-a", "score": 0.9}]},
    )

    assert isinstance(result, ToolError)
    assert result.error_type == "ValueError"
    assert "missing candidate scoring columns" in result.message


def test_candidate_scoring_prompt_spec_is_schema_backed_and_prompt_safe():
    spec = get_tool_registry()["candidate_scoring"].to_prompt_spec()

    assert spec["name"] == "candidate_scoring"
    assert spec["input_schema"]["title"] == "CandidateScoringInput"
    assert spec["output_schema"]["title"] == "CandidateScoringOutput"
    assert spec["constraints"]["popularity_weight_max"] == 0.2
    assert "lyrics" not in str(spec)
    assert "full lyric text" not in str(spec)


def _sample_scoring_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "song-a",
                "track_popularity": 10,
                "track_artist": "Aster",
                "track_album_name": "Night Signals",
            },
            {
                "track_id": "song-b",
                "track_popularity": 90,
                "track_artist": "Boreal",
                "track_album_name": "Daylight",
            },
            {
                "track_id": "song-c",
                "track_popularity": 100,
                "track_artist": "Cinder",
                "track_album_name": "Road",
            },
            {
                "track_id": "song-d",
                "track_popularity": 20,
                "track_artist": "Aster",
                "track_album_name": "Night Signals",
            },
        ]
    )
