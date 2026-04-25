from __future__ import annotations

from typing import Any

import pandas as pd

from vibefinder.agent_nodes import (
    critique_results_node,
    explain_recommendations_node,
    extract_preferences_node,
    plan_retrieval_node,
    revise_plan_node,
    verify_candidates_node,
)
from vibefinder.graph_runtime import create_graph_runtime_context
from vibefinder.pipeline_state import create_initial_pipeline_state
from vibefinder.variants import VariantConfig


class _FakeLLM:
    def __init__(self, outputs: dict[str, dict[str, Any]]):
        self.outputs = outputs
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        self.calls.append((prompt, schema_name))
        output = self.outputs[schema_name]
        if isinstance(output, list):
            if not output:
                raise AssertionError(f"No fake output left for schema {schema_name}")
            return output.pop(0)
        return output


def test_extract_preferences_node_calls_expected_schema_and_updates_state():
    llm = _FakeLLM(
        {
            "PreferenceExtractionOutput": {
                "raw_query": "English high energy songs",
                "feature_targets": [{"feature": "energy", "direction": "high"}],
                "rationale": "The query asks for energetic English music.",
                "issues": [],
                "warnings": ["Genre was not specified."],
            }
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    state = create_initial_pipeline_state("English high energy songs")

    update = extract_preferences_node(state, context)

    assert update["preferences"]["raw_query"] == "English high energy songs"
    assert update["warnings"] == ("Genre was not specified.",)
    assert update["trace"][0]["stage"] == "extract_preferences"
    assert llm.calls[0][1] == "PreferenceExtractionOutput"
    assert "retrieval_prompt_config" in llm.calls[0][0]
    assert "Do not include hidden chain-of-thought" in llm.calls[0][0]


def test_plan_retrieval_node_uses_enabled_tool_specs_and_respects_no_lyric_variant():
    llm = _FakeLLM(
        {
            "RetrievalStrategyOutput": {
                "primary_mode": "metadata_retrieval",
                "modes": ["metadata_retrieval"],
                "metadata_request": {
                    "categorical_filters": [{"field": "language", "values": ["en"]}],
                    "top_k": 25,
                },
                "rationale": "Language metadata is enough for this narrow test.",
                "issues": [],
                "warnings": [],
            }
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
    state = {
        **create_initial_pipeline_state("English songs"),
        "preferences": {
            "raw_query": "English songs",
            "categorical_filters": [{"field": "language", "values": ["en"]}],
            "rationale": "The query asks for English songs.",
            "issues": [],
            "warnings": [],
        },
    }

    update = plan_retrieval_node(state, context)

    assert update["retrieval_plan"]["primary_mode"] == "metadata_retrieval"
    assert llm.calls[0][1] == "RetrievalStrategyOutput"
    assert '"name": "metadata_retrieval"' in llm.calls[0][0]
    assert '"name": "lyric_retrieval"' not in llm.calls[0][0]


def test_plan_retrieval_node_repairs_missing_requests_from_preferences():
    llm = _FakeLLM(
        {
            "RetrievalStrategyOutput": {
                "primary_mode": "metadata_retrieval",
                "modes": ["metadata_retrieval", "feature_filter"],
                "rationale": "The model selected tools but omitted request payloads.",
                "issues": [],
                "warnings": [],
            }
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    state = {
        **create_initial_pipeline_state("English energetic songs"),
        "preferences": {
            "raw_query": "English energetic songs",
            "categorical_filters": [{"field": "language", "values": ["en"]}],
            "feature_targets": [{"feature": "energy", "direction": "high"}],
            "rationale": "The query asks for English energetic music.",
            "issues": [],
            "warnings": [],
        },
    }

    update = plan_retrieval_node(state, context)

    assert update["retrieval_plan"]["metadata_request"]["categorical_filters"][0]["field"] == "language"
    assert update["retrieval_plan"]["feature_request"]["targets"][0]["feature"] == "energy"
    assert any("Repaired missing tool-ready request" in warning for warning in update["warnings"])


def test_verify_candidates_node_maps_agent_candidates_to_verified_candidates():
    llm = _FakeLLM(
        {
            "VerificationOutput": {
                "candidates": [
                    {
                        "track_id": "song-a",
                        "verified": True,
                        "verifier_score": 0.9,
                        "matched_constraints": ["language matched"],
                        "evidence_sources": ["metadata_retrieval"],
                        "rationale": "The metadata evidence satisfies the language request.",
                        "warnings": [],
                    }
                ],
                "summary": "One candidate verified.",
                "rationale": "The candidate matched the available constraints.",
                "issues": [],
                "warnings": [],
            }
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    state = {
        **create_initial_pipeline_state("English songs"),
        "preferences": {"raw_query": "English songs"},
        "retrieval_plan": {"primary_mode": "metadata_retrieval"},
        "candidate_ids": ("song-a",),
        "tool_outputs": {"metadata_retrieval": {"candidates": [{"track_id": "song-a", "score": 1.0}]}},
    }

    update = verify_candidates_node(state, context)

    assert update["verified_candidates"][0]["track_id"] == "song-a"
    assert update["trace"][0]["details"]["candidate_count"] == 1
    assert llm.calls[0][1] == "VerificationOutput"
    assert "secret lyric text should not be in prompts" in llm.calls[0][0]


def test_verify_candidates_node_filters_unverified_scored_candidates():
    llm = _FakeLLM(
        {
            "VerificationOutput": {
                "candidates": [
                    {
                        "track_id": "song-a",
                        "verified": True,
                        "verifier_score": 0.9,
                        "matched_constraints": ["language matched"],
                        "evidence_sources": ["metadata"],
                        "rationale": "The candidate matches the request.",
                    },
                    {
                        "track_id": "song-b",
                        "verified": False,
                        "verifier_score": 0.2,
                        "violations": ["language mismatch"],
                        "evidence_sources": ["metadata"],
                        "rationale": "The candidate violates the request.",
                    },
                ],
                "summary": "Only one candidate verified.",
                "rationale": "Candidates were checked against metadata evidence.",
                "issues": ["One candidate failed verification."],
                "warnings": [],
            }
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    state = {
        **create_initial_pipeline_state("English songs"),
        "preferences": {"raw_query": "English songs"},
        "retrieval_plan": {"primary_mode": "metadata_retrieval"},
        "candidate_ids": ("song-a", "song-b"),
        "scored_candidates": (
            {"track_id": "song-a", "rank": 1, "final_score": 0.9},
            {"track_id": "song-b", "rank": 2, "final_score": 0.8},
        ),
        "tool_outputs": {"metadata_retrieval": {"candidates": [{"track_id": "song-a", "score": 1.0}]}},
    }

    update = verify_candidates_node(state, context)

    assert [candidate["track_id"] for candidate in update["scored_candidates"]] == ["song-a"]
    assert update["verified_candidates"][0]["evidence_sources"] == ["metadata_retrieval"]


def test_verify_candidates_node_keeps_soft_violations_as_warnings():
    llm = _FakeLLM(
        {
            "VerificationOutput": {
                "candidates": [
                    {
                        "track_id": "song-a",
                        "verified": False,
                        "verifier_score": 0.35,
                        "violations": ["weak lyric theme evidence"],
                        "evidence_sources": ["lyric_retrieval"],
                        "rationale": "The lyric evidence is uncertain, but there is no hard metadata mismatch.",
                    },
                    {
                        "track_id": "song-b",
                        "verified": False,
                        "verifier_score": 0.2,
                        "violations": ["wrong language"],
                        "evidence_sources": ["metadata_retrieval"],
                        "rationale": "The candidate violates the requested language.",
                    },
                ],
                "summary": "One soft concern and one hard mismatch.",
                "rationale": "The verifier separated uncertainty from hard constraints.",
                "issues": [],
                "warnings": [],
            }
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    state = {
        **create_initial_pipeline_state("English songs about regret"),
        "preferences": {"raw_query": "English songs about regret"},
        "retrieval_plan": {"primary_mode": "lyric_retrieval"},
        "candidate_ids": ("song-a", "song-b"),
        "scored_candidates": (
            {"track_id": "song-a", "rank": 1, "final_score": 0.9},
            {"track_id": "song-b", "rank": 2, "final_score": 0.8},
        ),
        "tool_outputs": {"lyric_retrieval": {"candidates": [{"track_id": "song-a", "score": 1.0}]}},
    }

    update = verify_candidates_node(state, context)

    assert [candidate["track_id"] for candidate in update["scored_candidates"]] == ["song-a"]
    assert update["scored_candidates"][0]["verification_status"] == "soft_violation"
    assert any("only hard constraint mismatches are rejected" in warning for warning in update["warnings"])


def test_critique_results_node_skips_when_variant_disables_critic_revision():
    llm = _FakeLLM({})
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
        variant_config=VariantConfig(
            name="no_critic",
            use_multi_step_reasoning=True,
            use_critic_revision=False,
            use_lyric_retriever=True,
        ),
    )

    update = critique_results_node(create_initial_pipeline_state("English songs"), context)

    assert update["critique"]["should_revise"] is False
    assert update["trace"][0]["message"] == "Critique skipped by variant."
    assert llm.calls == []


def test_critic_receives_top_ten_candidates_with_lyrics():
    llm = _FakeLLM(
        {
            "CritiqueOutput": {
                "issues": [],
                "should_revise": False,
                "revision_focus": [],
                "summary": "The candidate set is acceptable.",
                "rationale": "The critic reviewed the top candidates.",
                "confidence": "medium",
                "warnings": [],
            }
        }
    )
    track_ids = tuple(f"track-{index:02d}" for index in range(12))
    context = create_graph_runtime_context(
        songs=_many_sample_rows(12),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    state = {
        **create_initial_pipeline_state("lyrics query"),
        "preferences": {"raw_query": "lyrics query"},
        "retrieval_plan": {"primary_mode": "lyric_retrieval"},
        "scored_candidates": tuple(
            {"track_id": track_id, "rank": index + 1, "final_score": 0.8}
            for index, track_id in enumerate(track_ids)
        ),
        "verified_candidates": (),
    }

    critique_results_node(state, context)

    prompt = llm.calls[0][0]
    assert "lyric-09" in prompt
    assert "lyric-10" not in prompt
    assert "track-09" in prompt
    assert "track-10" not in prompt


def test_revise_plan_node_updates_revision_plan_and_retrieval_plan_when_retrying():
    llm = _FakeLLM(
        {
            "RevisionOutput": {
                "should_retry": True,
                "revised_retrieval_plan": {
                    "primary_mode": "metadata_retrieval",
                    "modes": ["metadata_retrieval"],
                    "metadata_request": {
                        "text_queries": [{"field": "playlist_name", "query": "pop"}],
                        "top_k": 50,
                    },
                    "rationale": "Broaden metadata search through playlist text.",
                    "issues": [],
                    "warnings": [],
                },
                "rationale": "The first pass was too narrow.",
                "issues": ["candidate pool too narrow"],
                "warnings": [],
            }
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    state = {
        **create_initial_pipeline_state("English songs"),
        "preferences": {"raw_query": "English songs"},
        "retrieval_plan": {"primary_mode": "metadata_retrieval"},
        "critique": {"should_revise": True, "issues": ["candidate pool too narrow"]},
        "revision_count": 0,
    }

    update = revise_plan_node(state, context)

    assert update["revision_count"] == 1
    assert update["revision_plan"]["should_retry"] is True
    assert update["retrieval_plan"]["metadata_request"]["text_queries"][0]["field"] == "playlist_name"
    assert llm.calls[0][1] == "RevisionOutput"
    assert "secret lyric text should not be in prompts" not in llm.calls[0][0]


def test_reviser_receives_top_ten_candidate_summaries_without_lyrics():
    llm = _FakeLLM(
        {
            "RevisionOutput": {
                "should_retry": False,
                "revised_retrieval_plan": None,
                "rationale": "No retry is needed.",
                "issues": [],
                "warnings": [],
            }
        }
    )
    track_ids = tuple(f"track-{index:02d}" for index in range(12))
    context = create_graph_runtime_context(
        songs=_many_sample_rows(12),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    state = {
        **create_initial_pipeline_state("lyrics query"),
        "preferences": {"raw_query": "lyrics query"},
        "retrieval_plan": {"primary_mode": "lyric_retrieval"},
        "critique": {"should_revise": True, "issues": ["candidate pool is weak"]},
        "revision_count": 0,
        "scored_candidates": tuple(
            {"track_id": track_id, "rank": index + 1, "final_score": 0.8}
            for index, track_id in enumerate(track_ids)
        ),
    }

    revise_plan_node(state, context)

    prompt = llm.calls[0][0]
    assert "track-09" in prompt
    assert "track-10" not in prompt
    assert "lyric-00" not in prompt
    assert '"lyrics":' not in prompt


def test_explain_recommendations_node_maps_explanations_and_excludes_lyrics_from_prompt():
    llm = _FakeLLM(
        {
            "ExplanationOutput": {
                "recommendations": [
                    {
                        "track_id": "song-a",
                        "rank": 1,
                        "explanation": "Recommended because it is English and high energy.",
                        "supporting_fields": ["language", "energy"],
                        "supporting_tool_outputs": ["metadata_retrieval"],
                        "warnings": [],
                    }
                ],
                "overall_summary": "One grounded recommendation.",
                "rationale": "The explanation uses only dataset and tool evidence.",
                "issues": [],
                "warnings": [],
            }
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    state = {
        **create_initial_pipeline_state("English songs"),
        "scored_candidates": [{"track_id": "song-a", "rank": 1, "final_score": 0.9}],
        "verified_candidates": [{"track_id": "song-a", "verified": True}],
        "reliability": {"confidence_label": "high"},
    }

    update = explain_recommendations_node(state, context)

    assert update["explanations"][0]["track_id"] == "song-a"
    assert "secret lyric text should not be in prompts" not in llm.calls[0][0]
    assert '"lyrics"' not in llm.calls[0][0]
    assert llm.calls[0][1] == "ExplanationOutput"


def test_explain_recommendations_node_fills_missing_explanations_without_lyrics():
    llm = _FakeLLM(
        {
            "ExplanationOutput": [
                {
                    "recommendations": [
                        {
                            "track_id": "song-a",
                            "rank": 1,
                            "explanation": "Recommended because it is English.",
                            "supporting_fields": ["language"],
                            "supporting_tool_outputs": ["metadata_retrieval"],
                            "warnings": [],
                        }
                    ],
                    "overall_summary": "One explanation returned.",
                    "rationale": "The model returned fewer explanations than candidates.",
                    "issues": [],
                    "warnings": [],
                },
                {
                    "recommendations": [
                        {
                            "track_id": "song-b",
                            "rank": 2,
                            "explanation": "Recommended because it matches the remaining evidence.",
                            "supporting_fields": ["language", "track_name"],
                            "supporting_tool_outputs": ["feature_filter", "candidate_scoring"],
                            "warnings": [],
                        }
                    ],
                    "overall_summary": "The missing explanation was filled.",
                    "rationale": "The repair pass explains only the missing candidate.",
                    "issues": [],
                    "warnings": [],
                },
            ]
        }
    )
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=llm,
    )
    state = {
        **create_initial_pipeline_state("English songs"),
        "scored_candidates": (
            {"track_id": "song-a", "rank": 1, "final_score": 0.9, "evidence_sources": ("metadata_retrieval",)},
            {"track_id": "song-b", "rank": 2, "final_score": 0.8, "evidence_sources": ("feature_filter",)},
        ),
        "verified_candidates": (),
        "reliability": {"confidence_label": "medium"},
    }

    update = explain_recommendations_node(state, context)

    assert [item["track_id"] for item in update["explanations"]] == ["song-a", "song-b"]
    assert "lyrics" not in update["explanations"][1]["explanation"].casefold()
    assert not any("deterministic evidence summary" in warning for warning in update["warnings"])
    assert len(llm.calls) == 2
    assert '"required_explanation_track_ids"' in llm.calls[0][0]
    assert '"song-b"' in llm.calls[1][0]


def test_agent_nodes_return_missing_state_update_for_missing_requirements():
    context = create_graph_runtime_context(
        songs=_sample_rows(),
        retrieval_prompt_config=_test_config(),
        llm_client=_FakeLLM({}),
    )

    update = plan_retrieval_node(create_initial_pipeline_state("English songs"), context)

    assert update["warnings"] == ("Missing required state object: preferences",)
    assert update["trace"][0]["message"] == "Agent node skipped."


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "song-a",
                "track_name": "Signal",
                "track_artist": "Aster",
                "track_album_name": "Night Signals",
                "playlist_name": "Pop Room",
                "playlist_genre": "pop",
                "playlist_subgenre": "dance pop",
                "language": "en",
                "energy": 0.9,
                "lyrics": "secret lyric text should not be in prompts",
            },
            {
                "track_id": "song-b",
                "track_name": "Daylight",
                "track_artist": "Boreal",
                "track_album_name": "Daylight",
                "playlist_name": "Rock Room",
                "playlist_genre": "rock",
                "playlist_subgenre": "hard rock",
                "language": "en",
                "energy": 0.4,
                "lyrics": "another lyric text should not be in prompts",
            },
        ]
    )


def _many_sample_rows(count: int) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "track_id": f"track-{index:02d}",
                "track_name": f"Track {index:02d}",
                "track_artist": "Aster",
                "track_album_name": "Night Signals",
                "playlist_name": "Pop Room",
                "playlist_genre": "pop",
                "playlist_subgenre": "dance pop",
                "language": "en",
                "energy": 0.5,
                "lyrics": f"lyric-{index:02d}",
            }
        )
    return pd.DataFrame(rows)


def _test_config() -> dict:
    return {
        "llm_prompt_constraints": {
            "categorical_values": {
                "playlist_genre": ["pop", "rock"],
                "playlist_subgenre": ["dance pop", "hard rock"],
                "language": ["en", "es"],
            },
            "full_text_search_columns": {
                "playlist_name": {"search_method": "full_text_search", "unique_count": 2},
                "track_artist": {"search_method": "full_text_search", "unique_count": 2},
                "track_album_name": {"search_method": "full_text_search", "unique_count": 2},
            },
            "numeric_ranges": {
                "energy": {"min": 0, "max": 1},
            },
        }
    }
