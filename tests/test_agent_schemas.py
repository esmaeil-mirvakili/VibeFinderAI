from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibefinder.agents import (
    AGENT_OUTPUT_SCHEMAS,
    CritiqueOutput,
    ExplanationCandidate,
    ExplanationOutput,
    PreferenceExtractionOutput,
    RevisionOutput,
    RetrievalStrategyOutput,
    VerificationCandidateResult,
    VerificationOutput,
    agent_output_schema_prompt_specs,
)
from vibefinder.tools.schemas import MAX_FINAL_TOP_K, MAX_RETRIEVAL_TOP_K


def test_preference_extraction_accepts_tool_ready_preferences():
    output = PreferenceExtractionOutput(
        raw_query="English pop songs about betrayal with high energy",
        lyric_intent="betrayal and regret",
        categorical_filters=(
            {"field": "language", "values": ("en",)},
            {"field": "playlist_genre", "values": ("pop",)},
        ),
        text_queries=({"field": "track_artist", "query": "Aster"},),
        feature_targets=({"feature": "energy", "direction": "high"},),
        hard_constraints=("language must be en",),
        rationale="The query states language, genre, theme, and energy preferences.",
    )

    assert output.raw_query == "English pop songs about betrayal with high energy"
    assert output.categorical_filters[0].field == "language"
    assert output.text_queries[0].field == "track_artist"
    assert output.feature_targets[0].feature == "energy"
    assert output.issues == ()
    assert output.warnings == ()


def test_preference_extraction_rejects_unsupported_fields_and_extra_keys():
    with pytest.raises(ValidationError):
        PreferenceExtractionOutput(
            raw_query="songs by Aster",
            rationale="Artist names are full-text, not categorical metadata.",
            categorical_filters=({"field": "track_artist", "values": ("Aster",)},),
        )

    with pytest.raises(ValidationError):
        PreferenceExtractionOutput(
            raw_query="songs",
            rationale="Reject invented output fields.",
            invented_concepts=("dramatic",),
        )


def test_preference_extraction_drops_empty_feature_range_filters():
    output = PreferenceExtractionOutput.model_validate(
        {
            "raw_query": "find me a sad song to listen to",
            "feature_range_filters": [
                {"feature": "energy", "min_value": None, "max_value": None},
                {"feature": "tempo", "min_value": 80, "max_value": None},
            ],
            "feature_targets": [{"feature": "energy", "direction": "low"}],
            "rationale": "The query asks for a sad song.",
            "issues": [],
            "warnings": [],
        }
    )

    assert len(output.feature_range_filters) == 1
    assert output.feature_range_filters[0].feature == "tempo"
    assert output.warnings == (
        "Dropped 1 feature range filters because they had no min_value or max_value.",
    )


def test_preference_extraction_drops_prompt_artifact_lyric_intent():
    output = PreferenceExtractionOutput.model_validate(
        {
            "raw_query": "find me a sad song to listen to",
            "lyric_intent": "lyrics are retrieved by the Lyric RAG tool and are intentionally excluded from this prompt config",
            "rationale": "The query asks for a sad song.",
            "issues": [],
            "warnings": [],
        }
    )

    assert output.lyric_intent is None
    assert output.warnings == (
        "Dropped lyric_intent because it appeared to copy prompt/config instructions.",
    )


def test_retrieval_strategy_allows_runtime_repair_of_missing_tool_requests():
    output = RetrievalStrategyOutput(
        primary_mode="metadata_retrieval",
        modes=("metadata_retrieval",),
        rationale="Need metadata filters.",
    )

    assert output.metadata_request is None

    with pytest.raises(ValidationError, match="primary_mode must be included"):
        RetrievalStrategyOutput(
            primary_mode="lyric_retrieval",
            modes=("feature_filter",),
            feature_request={"targets": [{"feature": "energy", "direction": "high"}]},
            rationale="Need audio features.",
        )


def test_retrieval_strategy_accepts_hybrid_tool_ready_plan():
    aliased_strategy = RetrievalStrategyOutput(
        primary_mode="metadata",
        modes=("metadata", "feature"),
        metadata_request={
            "categorical_filters": [{"field": "language", "values": ["en"]}],
            "top_k": 25,
        },
        feature_request={
            "targets": [{"feature": "energy", "direction": "high"}],
            "top_k": 25,
        },
        rationale="Common LLM retrieval aliases should normalize.",
    )
    assert aliased_strategy.primary_mode == "metadata_retrieval"
    assert aliased_strategy.modes == ("metadata_retrieval", "feature_filter")

    oversized_strategy = RetrievalStrategyOutput(
        primary_mode="metadata_retrieval",
        modes=("metadata_retrieval",),
        metadata_request={
            "categorical_filters": [{"field": "language", "values": ["en"]}],
            "top_k": 999,
        },
        top_k_final=999,
        rationale="Oversized top-k values should be bounded.",
    )
    assert oversized_strategy.metadata_request is not None
    assert oversized_strategy.metadata_request.top_k == MAX_RETRIEVAL_TOP_K
    assert oversized_strategy.top_k_final == MAX_FINAL_TOP_K

    output = RetrievalStrategyOutput(
        primary_mode="lyric_retrieval",
        modes=("lyric_retrieval", "metadata_retrieval", "feature_filter"),
        lyric_request={"query": "betrayal regret", "language": "en", "top_k": 25},
        metadata_request={
            "categorical_filters": [{"field": "language", "values": ["en"]}],
            "top_k": 25,
        },
        feature_request={
            "targets": [{"feature": "energy", "direction": "high"}],
            "top_k": 25,
        },
        rationale="The query mixes lyrical theme, language, and energy.",
    )

    assert output.lyric_request is not None
    assert output.metadata_request is not None
    assert output.feature_request is not None
    assert output.scoring_weights.popularity == 0.05


def test_verification_output_validates_candidate_results():
    output = VerificationOutput(
        candidates=(
            VerificationCandidateResult(
                track_id="song-a",
                verified=True,
                verifier_score=0.9,
                matched_constraints=("language matched",),
                evidence_sources=("lyric_retrieval", "feature_filter"),
                rationale="The candidate matches the stated language and energy.",
            ),
        ),
        summary="One candidate verified.",
        rationale="The candidate evidence satisfies the stated constraints.",
    )

    assert output.candidates[0].track_id == "song-a"

    aliased_output = VerificationOutput(
        candidates=(
            {
                "track_id": "song-b",
                "verified": True,
                "verifier_score": 0.8,
                "evidence_sources": ["metadata", "lyric", "feature"],
                "rationale": "Common LLM evidence aliases should be normalized.",
            },
        ),
        summary="One candidate verified.",
        rationale="The candidate has normalized evidence sources.",
    )
    assert aliased_output.candidates[0].evidence_sources == (
        "metadata_retrieval",
        "lyric_retrieval",
        "feature_filter",
    )

    sentence_salvaged_output = VerificationOutput(
        candidates=(
            {
                "track_id": "song-b",
                "verified": False,
                "verifier_score": 0.4,
                "evidence_sources": [
                    "metadata_filter",
                    "Lyrical content is about attraction and chemistry, not sadness.",
                ],
                "rationale": "The candidate has weak theme fit.",
            },
        ),
        summary="One candidate checked.",
        rationale="Unsupported sentence-like evidence should become a warning.",
    )
    assert sentence_salvaged_output.candidates[0].evidence_sources == ("metadata_retrieval",)
    assert "Moved unsupported evidence source to warnings" in sentence_salvaged_output.candidates[0].warnings[0]

    salvaged_output = VerificationOutput(
        candidates=(
            {
                "track_id": "song-c",
                "verified": True,
                "verifier_score": 0.8,
                "rationale": "Valid candidate.",
            },
            {
                "track_id": "song-d",
                "verified": True,
                "verifier_score": 0.9,
                "violations": ["contradiction"],
                "rationale": "Invalid candidate.",
            },
            {
                "track_id": "song-c",
                "verified": True,
                "verifier_score": 0.7,
                "rationale": "Duplicate candidate.",
            },
        ),
        summary="Candidates were partially valid.",
        rationale="Invalid candidates should not fail the whole verifier output.",
    )
    assert [candidate.track_id for candidate in salvaged_output.candidates] == ["song-c"]
    assert len(salvaged_output.warnings) == 2

    with pytest.raises(ValidationError, match="verified candidates cannot include violations"):
        VerificationCandidateResult(
            track_id="song-a",
            verified=True,
            verifier_score=0.9,
            violations=("language mismatch",),
            rationale="Contradictory verifier output.",
        )

    with pytest.raises(ValidationError, match="Unsupported tool output names"):
        VerificationCandidateResult(
            track_id="song-a",
            verified=False,
            verifier_score=0.2,
            evidence_sources=("artist_biography_api",),
            rationale="Uses unsupported evidence.",
        )


def test_critique_and_revision_outputs_enforce_retry_contract():
    with pytest.raises(ValidationError, match="should_revise requires"):
        CritiqueOutput(
            should_revise=True,
            summary="Needs revision.",
            rationale="Revision was requested but no issue was supplied.",
        )

    critique = CritiqueOutput(
        issues=("candidate pool too narrow",),
        should_revise=True,
        revision_focus=("broaden metadata filtering",),
        summary="The result set is too small.",
        rationale="The candidate pool does not provide enough viable options.",
    )
    assert critique.should_revise is True

    with pytest.raises(ValidationError, match="revised_retrieval_plan is required"):
        RevisionOutput(should_retry=True, rationale="Retry with broader search.")

    revision = RevisionOutput(
        should_retry=True,
        revised_retrieval_plan={
            "primary_mode": "metadata_retrieval",
            "modes": ("metadata_retrieval",),
            "metadata_request": {
                "text_queries": [{"field": "playlist_name", "query": "pop"}],
                "top_k": 50,
            },
            "broad_search": True,
            "rationale": "Broaden to playlist-name metadata search.",
        },
        rationale="The first pass was too narrow.",
        issues=("candidate pool too narrow",),
    )
    assert revision.revised_retrieval_plan is not None


def test_explanation_output_requires_dataset_fields_and_supported_tool_outputs():
    output = ExplanationOutput(
        recommendations=(
            ExplanationCandidate(
                track_id="song-a",
                rank=1,
                explanation="Recommended because the track is in English and has high energy.",
                supporting_fields=("language", "energy", "track_name"),
                supporting_tool_outputs=("metadata_retrieval", "feature_filter"),
            ),
        ),
        overall_summary="One grounded recommendation.",
        rationale="The explanation cites only dataset fields and tool outputs.",
    )

    assert output.recommendations[0].supporting_fields == ("language", "energy", "track_name")

    aliased_output = ExplanationOutput(
        recommendations=(
            {
                "track_id": "song-b",
                "rank": 1,
                "explanation": "Recommended from metadata and feature evidence.",
                "supporting_tool_outputs": ["metadata", "feature"],
            },
        ),
        rationale="Common LLM tool-output aliases should be normalized.",
    )
    assert aliased_output.recommendations[0].supporting_tool_outputs == (
        "metadata_retrieval",
        "feature_filter",
    )

    salvaged_output = ExplanationOutput(
        recommendations=(
            {
                "track_id": "song-c",
                "rank": 1,
                "explanation": "Valid recommendation.",
            },
            {
                "track_id": "song-d",
                "rank": 2,
                "explanation": "Invalid recommendation.",
                "supporting_fields": ["artist_biography"],
            },
        ),
        rationale="Invalid recommendations should not fail the whole explanation output.",
    )
    assert [candidate.track_id for candidate in salvaged_output.recommendations] == ["song-c"]
    assert len(salvaged_output.warnings) == 1

    with pytest.raises(ValidationError, match="Unsupported dataset evidence fields"):
        ExplanationCandidate(
            track_id="song-a",
            rank=1,
            explanation="Uses unsupported evidence.",
            supporting_fields=("artist_biography",),
        )

    duplicate_output = ExplanationOutput(
        recommendations=(
            {
                "track_id": "song-a",
                "rank": 1,
                "explanation": "First explanation.",
            },
            {
                "track_id": "song-a",
                "rank": 2,
                "explanation": "Duplicate explanation.",
            },
        ),
        rationale="Duplicate track ids should be salvaged.",
    )
    assert [candidate.track_id for candidate in duplicate_output.recommendations] == ["song-a"]
    assert "Dropped duplicate explanation recommendation" in duplicate_output.warnings[0]


def test_agent_schema_prompt_specs_are_available_and_prompt_safe():
    specs = agent_output_schema_prompt_specs()

    assert set(specs) == set(AGENT_OUTPUT_SCHEMAS)
    assert specs["PreferenceExtractionOutput"]["json_schema"]["title"] == "PreferenceExtractionOutput"
    assert specs["RetrievalStrategyOutput"]["constraints"]["requires_tool_ready_requests"] is True
    for spec in specs.values():
        properties = spec["json_schema"]["properties"]
        assert "rationale" in properties
        assert "issues" in properties
        assert "warnings" in properties
        assert spec["constraints"]["required_intermediate_fields"] == ["rationale", "issues", "warnings"]
    assert "artist_biography" not in str(specs)
    assert "listener_history" not in str(specs)
    assert "track_artist" in str(specs)
