from __future__ import annotations

from typing import Any

import pandas as pd

from vibefinder.graph import build_recommendation_graph, run_recommendation
from vibefinder.graph_runtime import create_graph_runtime_context
from vibefinder.pipeline_state import create_initial_pipeline_state
from vibefinder.variants import VariantConfig


class _FakeLLM:
    def __init__(self, outputs: dict[str, dict[str, Any] | list[dict[str, Any]]]):
        self.outputs = outputs
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        self.calls.append((prompt, schema_name))
        output = self.outputs[schema_name]
        if isinstance(output, list):
            if len(output) == 1:
                return output[0]
            return output.pop(0)
        return output


def test_build_recommendation_graph_runs_minimal_langgraph_pipeline():
    llm = _FakeLLM(
        {
            "PreferenceExtractionOutput": {
                "raw_query": "English pop songs",
                "categorical_filters": [
                    {"field": "language", "values": ["en"]},
                    {"field": "playlist_genre", "values": ["pop"]},
                ],
                "rationale": "The query asks for English pop songs.",
                "issues": [],
                "warnings": [],
            },
            "RetrievalStrategyOutput": {
                "primary_mode": "metadata_retrieval",
                "modes": ["metadata_retrieval"],
                "metadata_request": {
                    "categorical_filters": [
                        {"field": "language", "values": ["en"]},
                        {"field": "playlist_genre", "values": ["pop"]},
                    ],
                    "top_k": 10,
                },
                "scoring_weights": {
                    "lyric": 0.0,
                    "metadata": 0.8,
                    "feature": 0.0,
                    "popularity": 0.05,
                    "diversity_penalty": 0.0,
                },
                "top_k_final": 2,
                "rationale": "Metadata retrieval directly matches the request.",
                "issues": [],
                "warnings": [],
            },
            "VerificationOutput": _verification_output(("song-a", "song-b")),
            "CritiqueOutput": _critique_output(should_revise=False),
            "ExplanationOutput": _explanation_output(("song-a", "song-b")),
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    graph = build_recommendation_graph(context)

    result = graph.invoke(create_initial_pipeline_state("English pop songs", variant_name="full"))

    assert result["preferences"]["raw_query"] == "English pop songs"
    assert result["retrieval_plan"]["primary_mode"] == "metadata_retrieval"
    assert result["retrieval_modes_used"] == ("metadata_retrieval",)
    assert result["candidate_ids"] == ("song-a", "song-b")
    assert [candidate["track_id"] for candidate in result["scored_candidates"]] == ["song-a", "song-b"]
    assert result["reliability"]["support_summary"]["candidate_count"] == 2
    assert result["trace"][-1]["stage"] == "finished"
    assert [schema_name for _, schema_name in llm.calls] == [
        "PreferenceExtractionOutput",
        "RetrievalStrategyOutput",
        "VerificationOutput",
        "CritiqueOutput",
        "ExplanationOutput",
    ]


def test_run_recommendation_uses_context_variant_name():
    llm = _FakeLLM(
        {
            "PreferenceExtractionOutput": {
                "raw_query": "English songs",
                "categorical_filters": [{"field": "language", "values": ["en"]}],
                "rationale": "The query asks for English songs.",
                "issues": [],
                "warnings": [],
            },
            "RetrievalStrategyOutput": {
                "primary_mode": "metadata_retrieval",
                "modes": ["metadata_retrieval"],
                "metadata_request": {
                    "categorical_filters": [{"field": "language", "values": ["en"]}],
                    "top_k": 10,
                },
                "top_k_final": 2,
                "rationale": "Language metadata can satisfy this query.",
                "issues": [],
                "warnings": [],
            },
            "VerificationOutput": _verification_output(("song-a", "song-b")),
            "ExplanationOutput": _explanation_output(("song-a", "song-b")),
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
        variant_config=VariantConfig(
            name="no_critic_revision",
            use_multi_step_reasoning=True,
            use_critic_revision=False,
            use_lyric_retriever=True,
        ),
    )

    result = run_recommendation("English songs", context)

    assert result["variant_name"] == "no_critic_revision"
    assert result["reliability"]["confidence_label"] in {"medium", "high"}
    assert [schema_name for _, schema_name in llm.calls] == [
        "PreferenceExtractionOutput",
        "RetrievalStrategyOutput",
        "VerificationOutput",
        "ExplanationOutput",
    ]


def test_no_multi_step_variant_skips_verifier_critic_revision_and_explanation():
    llm = _FakeLLM(
        {
            "PreferenceExtractionOutput": {
                "raw_query": "English songs",
                "categorical_filters": [{"field": "language", "values": ["en"]}],
                "rationale": "The query asks for English songs.",
                "issues": [],
                "warnings": [],
            },
            "RetrievalStrategyOutput": {
                "primary_mode": "metadata_retrieval",
                "modes": ["metadata_retrieval"],
                "metadata_request": {
                    "categorical_filters": [{"field": "language", "values": ["en"]}],
                    "top_k": 10,
                },
                "top_k_final": 2,
                "rationale": "Language metadata can satisfy this query.",
                "issues": [],
                "warnings": [],
            },
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
        variant_config=VariantConfig(
            name="no_multi_step",
            use_multi_step_reasoning=False,
            use_critic_revision=True,
            use_lyric_retriever=True,
        ),
    )

    result = run_recommendation("English songs", context)

    assert result["variant_name"] == "no_multi_step"
    assert [candidate["track_id"] for candidate in result["scored_candidates"]] == ["song-a", "song-b"]
    assert result["verified_candidates"] == ()
    assert result["explanations"] == ()
    assert result["reliability"]["support_summary"]["candidate_count"] == 2
    assert [schema_name for _, schema_name in llm.calls] == [
        "PreferenceExtractionOutput",
        "RetrievalStrategyOutput",
    ]


def test_graph_records_disabled_lyric_retriever_warning():
    llm = _FakeLLM(
        {
            "PreferenceExtractionOutput": {
                "raw_query": "songs about regret",
                "lyric_intent": "regret",
                "rationale": "The query asks for a lyric theme.",
                "issues": [],
                "warnings": [],
            },
            "RetrievalStrategyOutput": {
                "primary_mode": "lyric_retrieval",
                "modes": ["lyric_retrieval"],
                "lyric_request": {"query": "regret", "top_k": 5},
                "top_k_final": 2,
                "rationale": "A lyric query would normally use Lyric RAG.",
                "issues": [],
                "warnings": [],
            },
            "VerificationOutput": _verification_output(()),
            "CritiqueOutput": _critique_output(should_revise=False),
            "ExplanationOutput": _explanation_output(()),
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
        variant_config=VariantConfig(
            name="no_lyric",
            use_multi_step_reasoning=True,
            use_critic_revision=True,
            use_lyric_retriever=False,
        ),
    )

    result = run_recommendation("songs about regret", context)

    assert result["fatal_error"].startswith("plan_retrieval failed:")
    assert "Retrieval plan has no enabled modes" in result["warnings"][0]
    assert "retrieval_plan" not in result or result["retrieval_plan"] is None
    assert result["candidate_ids"] == ()
    assert result["trace"][-1]["stage"] == "finished"


def test_graph_routes_one_bounded_revision_and_scores_revised_evidence():
    first_plan = {
        "primary_mode": "metadata_retrieval",
        "modes": ["metadata_retrieval"],
        "metadata_request": {
            "categorical_filters": [
                {"field": "language", "values": ["en"]},
                {"field": "playlist_genre", "values": ["pop"]},
            ],
            "top_k": 10,
        },
        "scoring_weights": {
            "lyric": 0.0,
            "metadata": 0.8,
            "feature": 0.0,
            "popularity": 0.05,
            "diversity_penalty": 0.0,
        },
        "top_k_final": 2,
        "rationale": "The first pass uses metadata.",
        "issues": [],
        "warnings": [],
    }
    revised_plan = {
        "primary_mode": "metadata_retrieval",
        "modes": ["metadata_retrieval"],
        "metadata_request": {
            "categorical_filters": [
                {"field": "language", "values": ["es"]},
                {"field": "playlist_genre", "values": ["latin"]},
            ],
            "top_k": 10,
        },
        "scoring_weights": {
            "lyric": 0.0,
            "metadata": 0.8,
            "feature": 0.0,
            "popularity": 0.05,
            "diversity_penalty": 0.0,
        },
        "top_k_final": 2,
        "rationale": "The revised pass preserves the Spanish latin constraint.",
        "issues": [],
        "warnings": [],
    }
    llm = _FakeLLM(
        {
            "PreferenceExtractionOutput": {
                "raw_query": "Spanish latin songs",
                "categorical_filters": [
                    {"field": "language", "values": ["es"]},
                    {"field": "playlist_genre", "values": ["latin"]},
                ],
                "hard_constraints": ["Spanish language", "latin genre"],
                "rationale": "The query asks for Spanish latin songs.",
                "issues": [],
                "warnings": [],
            },
            "RetrievalStrategyOutput": first_plan,
            "VerificationOutput": [
                _verification_output(("song-a", "song-b"), verified=False),
                _verification_output(("song-c",), verified=True),
            ],
            "CritiqueOutput": [
                _critique_output(
                    should_revise=True,
                    issues=("The first pass ignored the Spanish latin constraints.",),
                ),
                _critique_output(should_revise=False),
            ],
            "RevisionOutput": {
                "should_retry": True,
                "revised_retrieval_plan": revised_plan,
                "rationale": "Retry with the correct language and genre filters.",
                "issues": [],
                "warnings": [],
            },
            "ExplanationOutput": _explanation_output(("song-c",)),
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
        max_revision_count=1,
    )

    result = run_recommendation("Spanish latin songs", context)

    assert result["revision_count"] == 1
    assert result["retrieval_plan"]["metadata_request"]["categorical_filters"][0]["values"] == ["es"]
    assert [candidate["track_id"] for candidate in result["scored_candidates"]] == ["song-c"]
    assert [candidate["track_id"] for candidate in result["verified_candidates"]] == ["song-c"]
    assert result["reliability"]["support_summary"]["candidate_count"] == 1
    assert [schema_name for _, schema_name in llm.calls] == [
        "PreferenceExtractionOutput",
        "RetrievalStrategyOutput",
        "VerificationOutput",
        "CritiqueOutput",
        "RevisionOutput",
        "VerificationOutput",
        "CritiqueOutput",
        "ExplanationOutput",
    ]


def test_revision_retry_with_no_results_clears_first_pass_candidates():
    first_plan = {
        "primary_mode": "metadata_retrieval",
        "modes": ["metadata_retrieval"],
        "metadata_request": {
            "categorical_filters": [{"field": "language", "values": ["en"]}],
            "top_k": 10,
        },
        "top_k_final": 2,
        "rationale": "The first pass uses English metadata.",
        "issues": [],
        "warnings": [],
    }
    revised_plan = {
        "primary_mode": "metadata_retrieval",
        "modes": ["metadata_retrieval"],
        "metadata_request": {
            "text_queries": [{"field": "playlist_name", "query": "no such playlist"}],
            "top_k": 10,
        },
        "top_k_final": 2,
        "rationale": "Retry with a valid text query that has no matches.",
        "issues": [],
        "warnings": [],
    }
    llm = _FakeLLM(
        {
            "PreferenceExtractionOutput": {
                "raw_query": "impossible playlist songs",
                "text_queries": [{"field": "playlist_name", "query": "no such playlist"}],
                "rationale": "The query asks for a specific playlist-like phrase.",
                "issues": [],
                "warnings": [],
            },
            "RetrievalStrategyOutput": first_plan,
            "VerificationOutput": [
                _verification_output(("song-a", "song-b"), verified=False),
                _verification_output(()),
            ],
            "CritiqueOutput": [
                _critique_output(should_revise=True, issues=("The first pass is wrong.",)),
                _critique_output(should_revise=False),
            ],
            "RevisionOutput": {
                "should_retry": True,
                "revised_retrieval_plan": revised_plan,
                "rationale": "Retry with the requested playlist text.",
                "issues": [],
                "warnings": [],
            },
            "ExplanationOutput": _explanation_output(()),
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
        max_revision_count=1,
    )

    result = run_recommendation("impossible playlist songs", context)

    assert result["revision_count"] == 1
    assert result["candidate_ids"] == ()
    assert result["scored_candidates"] == ()
    assert result["verified_candidates"] == ()
    assert result["reliability"]["support_summary"]["candidate_count"] == 0


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "song-a",
                "track_popularity": 80,
                "track_artist": "Aster",
                "track_album_name": "Night Signals",
                "language": "en",
                "playlist_genre": "pop",
                "playlist_subgenre": "dance pop",
                "playlist_name": "Pop Room",
            },
            {
                "track_id": "song-b",
                "track_popularity": 50,
                "track_artist": "Boreal",
                "track_album_name": "Daylight",
                "language": "en",
                "playlist_genre": "pop",
                "playlist_subgenre": "indie poptimism",
                "playlist_name": "Bright Pop",
            },
            {
                "track_id": "song-c",
                "track_popularity": 70,
                "track_artist": "Cinder",
                "track_album_name": "Road",
                "language": "es",
                "playlist_genre": "latin",
                "playlist_subgenre": "latin pop",
                "playlist_name": "Latin Pop",
            },
        ]
    )


def _test_config() -> dict:
    return {
        "llm_prompt_constraints": {
            "categorical_values": {
                "playlist_genre": ["pop", "latin"],
                "playlist_subgenre": ["dance pop", "indie poptimism", "latin pop"],
                "language": ["en", "es"],
            },
            "full_text_search_columns": {
                "playlist_name": {"search_method": "full_text_search", "unique_count": 3},
                "track_artist": {"search_method": "full_text_search", "unique_count": 3},
                "track_album_name": {"search_method": "full_text_search", "unique_count": 3},
            },
        }
    }


def _verification_output(track_ids: tuple[str, ...], verified: bool = True) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "track_id": track_id,
                "verified": verified,
                "verifier_score": 0.9 if verified else 0.2,
                "matched_constraints": ["metadata match"] if verified else [],
                "violations": [] if verified else ["constraint mismatch"],
                "evidence_sources": ["metadata_retrieval"],
                "rationale": "The candidate was checked against metadata evidence.",
                "warnings": [],
            }
            for track_id in track_ids
        ],
        "summary": "Verifier checked the candidate set.",
        "rationale": "Candidates were judged against available retrieval evidence.",
        "issues": [] if verified else ["Some candidates failed constraints."],
        "warnings": [],
    }


def _critique_output(
    should_revise: bool,
    issues: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "issues": list(issues),
        "should_revise": should_revise,
        "revision_focus": ["retry metadata filters"] if should_revise else [],
        "summary": "Revision is needed." if should_revise else "The candidate set is acceptable.",
        "rationale": "The critic reviewed candidate quality and constraint fit.",
        "confidence": "medium",
        "warnings": [],
    }


def _explanation_output(track_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "recommendations": [
            {
                "track_id": track_id,
                "rank": rank,
                "explanation": "Recommended because the retrieved metadata fits the request.",
                "supporting_fields": ["language", "playlist_genre"],
                "supporting_tool_outputs": ["metadata_retrieval", "candidate_scoring"],
                "warnings": [],
            }
            for rank, track_id in enumerate(track_ids, start=1)
        ],
        "overall_summary": "Recommendations are grounded in retrieval evidence.",
        "rationale": "Explanations cite only dataset fields and tool outputs.",
        "issues": [],
        "warnings": [],
    }
