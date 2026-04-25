from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vibefinder.evaluation import (
    EvaluationQuery,
    ExpectedFeatureTarget,
    ExpectedTextMatch,
    aggregate_results,
    evaluation_variant_configs,
    load_evaluation_queries,
    markdown_summary,
    summarize_pipeline_state,
)


def test_load_evaluation_queries_parses_expected_constraints(tmp_path):
    query_path = tmp_path / "queries.json"
    query_path.write_text(
        json.dumps(
            [
                {
                    "id": "sad-pop",
                    "group": "mixed",
                    "query": "sad English pop songs",
                    "expected": {
                        "language": "en",
                        "genres": ["pop"],
                        "subgenres": ["dance pop"],
                        "lyric_intent": "sad",
                        "retrieval_modes": ["lyric_retrieval", "metadata_retrieval", "feature_filter"],
                        "text_matches": [
                            {"field": "playlist_name", "query": "sad pop", "match_mode": "any"}
                        ],
                        "feature_targets": [{"feature": "energy", "direction": "low"}],
                        "feature_exclusions": [{"feature": "acousticness", "direction": "high"}],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    queries = load_evaluation_queries(query_path)

    assert len(queries) == 1
    assert queries[0].id == "sad-pop"
    assert queries[0].expected_language == "en"
    assert queries[0].expected_genres == ("pop",)
    assert queries[0].expected_subgenres == ("dance pop",)
    assert queries[0].lyric_intent == "sad"
    assert queries[0].expected_retrieval_modes == (
        "lyric_retrieval",
        "metadata_retrieval",
        "feature_filter",
    )
    assert queries[0].expected_text_matches[0].field == "playlist_name"
    assert queries[0].expected_text_matches[0].match_mode == "any"
    assert queries[0].expected_feature_targets[0].feature == "energy"
    assert queries[0].expected_feature_targets[0].direction == "low"
    assert queries[0].expected_feature_exclusions[0].feature == "acousticness"


def test_load_evaluation_queries_rejects_invalid_feature_direction(tmp_path):
    query_path = tmp_path / "queries.json"
    query_path.write_text(
        json.dumps(
            [
                {
                    "id": "bad",
                    "query": "bad query",
                    "expected": {
                        "feature_targets": [{"feature": "energy", "direction": "extreme"}],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="direction"):
        load_evaluation_queries(query_path)


def test_project_benchmark_queries_cover_required_groups():
    queries = load_evaluation_queries(Path("evaluation/benchmark_queries.json"))
    groups = {query.group for query in queries}

    assert len(queries) == 10
    assert len({query.id for query in queries}) == 10
    assert {"lyric_theme", "mixed", "hard_constraints"} == groups
    assert sum(bool(query.expected_feature_targets) for query in queries) == 10
    assert sum(bool(query.expected_genres) for query in queries) >= 1
    assert sum(query.expected_language is not None for query in queries) >= 3
    assert sum(bool(query.expected_text_matches) for query in queries) == 0
    assert sum(bool(query.expected_retrieval_modes) for query in queries) == 10
    assert sum(query.lyric_intent is not None for query in queries) == 10


def test_evaluation_variant_configs_returns_stable_ablation_variants():
    variants = evaluation_variant_configs(("full", "no_lyric_retriever"))

    assert [variant.name for variant in variants] == ["full", "no_lyric_retriever"]
    assert variants[0].use_lyric_retriever is True
    assert variants[1].use_lyric_retriever is False


def test_evaluation_variant_configs_default_order_excludes_multi_step_ablation():
    variants = evaluation_variant_configs()

    assert [variant.name for variant in variants] == [
        "full",
        "no_critic_revision",
        "no_lyric_retriever",
    ]


def test_evaluation_variant_configs_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown evaluation variant"):
        evaluation_variant_configs(("full", "missing"))


def test_summarize_pipeline_state_computes_metrics_and_compact_candidates():
    state = {
        "candidate_ids": ("song-a", "song-b"),
        "retrieval_modes_used": ("metadata_retrieval", "feature_filter"),
        "scored_candidates": (
            {
                "track_id": "song-a",
                "rank": 1,
                "final_score": 0.9,
                "evidence_sources": ("lyric", "metadata", "feature"),
            },
            {
                "track_id": "song-b",
                "rank": 2,
                "final_score": 0.5,
                "evidence_sources": ("metadata",),
            },
        ),
        "explanations": ({"track_id": "song-a", "rank": 1},),
        "warnings": ("narrow result set",),
        "tool_errors": (),
        "revision_count": 1,
        "reliability": {"confidence_label": "high", "confidence_score": 0.82},
        "trace": ({"stage": "received_query"}, {"stage": "finished"}),
    }
    result = summarize_pipeline_state(
        state=state,
        songs=_songs(),
        evaluation_query=EvaluationQuery(
            id="english-pop-high-energy",
            query="English pop songs with high energy",
            group="mixed",
            expected_language="en",
            expected_genres=("pop",),
            expected_retrieval_modes=("lyric_retrieval", "metadata_retrieval", "feature_filter"),
            lyric_intent="heartbreak",
        ),
        variant_name="full",
        elapsed_seconds=1.23456,
        top_k=2,
    )

    assert result.success is True
    assert result.elapsed_seconds == 1.2346
    assert result.summary["confidence_label"] == "high"
    assert result.summary["revision_used"] is True
    assert result.summary["final_track_ids"] == ["song-a", "song-b"]
    assert result.summary["final_candidates"][0]["track_name"] == "A"
    assert result.metrics["language_match_rate"] == 0.5
    assert result.metrics["genre_match_rate"] == 0.5
    assert result.metrics["artist_diversity"] == 1.0
    assert result.metrics["expected_retrieval_mode_recall"] == 0.6667
    assert result.metrics["unexpected_retrieval_mode_count"] == 0
    assert result.metrics["lyric_retrieval_used"] is False
    assert result.metrics["lyric_evidence_rate"] == 0.5
    assert result.metrics["lyric_intent_retrieval_match"] is False
    assert result.state is None


def test_summarize_pipeline_state_scores_text_matches_and_exclusions():
    state = {
        "candidate_ids": ("song-a", "song-b"),
        "retrieval_modes_used": ("metadata_retrieval", "feature_filter"),
        "scored_candidates": (
            {"track_id": "song-a", "rank": 1, "final_score": 0.9, "evidence_sources": ("metadata", "feature")},
            {"track_id": "song-b", "rank": 2, "final_score": 0.5, "evidence_sources": ("feature",)},
        ),
        "warnings": (),
        "tool_errors": (),
        "revision_count": 0,
        "reliability": {"confidence_label": "medium", "confidence_score": 0.6},
        "trace": (),
    }
    result = summarize_pipeline_state(
        state=state,
        songs=_songs(),
        evaluation_query=EvaluationQuery(
            id="artist-feature",
            query="songs by Artist A that are not high energy",
            group="metadata_text",
            expected_text_matches=(
                ExpectedTextMatch(field="track_artist", query="Artist A", match_mode="all"),
            ),
            expected_feature_exclusions=(
                ExpectedFeatureTarget(feature="energy", direction="high", weight=1.0),
            ),
        ),
        variant_name="full",
        elapsed_seconds=1.0,
        top_k=2,
    )

    assert result.metrics["metadata_text_match_rate"] == 0.5
    assert result.metrics["feature_exclusion_pass_rate"] == 0.5


def test_aggregate_results_and_markdown_summary():
    results = [
        {
            "variant": "full",
            "elapsed_seconds": 1.0,
            "success": True,
            "summary": {
                "confidence_score": 0.8,
                "warning_count": 1,
                "final_track_ids": ["song-a"],
                "revision_used": True,
            },
            "metrics": {"automatic_constraint_score": 0.7, "feature_fit_score": 0.6},
        },
        {
            "variant": "full",
            "elapsed_seconds": 2.0,
            "success": False,
            "summary": {"warning_count": 0, "final_track_ids": [], "revision_used": False},
            "metrics": {},
        },
    ]

    aggregate = aggregate_results(results)
    report = {
        "metadata": {
            "generated_at": "2026-04-18T00:00:00Z",
            "query_count": 2,
            "variants": ["full"],
            "llm_provider": "ollama",
            "llm_model": "qwen2.5:14b",
        },
        "aggregate": aggregate,
    }
    rendered = markdown_summary(report)

    assert aggregate["full"]["run_count"] == 2
    assert aggregate["full"]["success_count"] == 1
    assert aggregate["full"]["failure_count"] == 1
    assert aggregate["full"]["avg_elapsed_seconds"] == 1.5
    assert aggregate["full"]["revision_used_count"] == 1
    assert "| full | 2 | 1 |" in rendered


def _songs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "song-a",
                "track_name": "A",
                "track_artist": "Artist A",
                "track_album_name": "Album A",
                "playlist_genre": "pop",
                "playlist_subgenre": "dance pop",
                "language": "en",
                "energy": 0.9,
            },
            {
                "track_id": "song-b",
                "track_name": "B",
                "track_artist": "Artist B",
                "track_album_name": "Album B",
                "playlist_genre": "rock",
                "playlist_subgenre": "hard rock",
                "language": "es",
                "energy": 0.4,
            },
        ]
    )
