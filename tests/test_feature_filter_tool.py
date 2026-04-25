from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from vibefinder.tools import (
    FeatureFilterInput,
    FeatureRangeFilter,
    FeatureTarget,
    MAX_RETRIEVAL_TOP_K,
    ToolContext,
    ToolError,
    ToolResult,
    ToolRunner,
    filter_by_features,
    get_tool_registry,
)


def test_feature_filter_scores_high_energy_direction():
    songs = _sample_feature_rows()
    request = FeatureFilterInput(
        targets=(FeatureTarget(feature="energy", direction="high"),),
        top_k=2,
    )

    output = filter_by_features(songs, request, _test_config())

    assert output.output_count == 2
    assert [candidate.track_id for candidate in output.candidates] == ["high", "mid"]
    assert output.candidates[0].feature_scores["energy"] == 1.0


def test_feature_filter_applies_hard_range_and_candidate_subset():
    songs = _sample_feature_rows()
    request = FeatureFilterInput(
        range_filters=(FeatureRangeFilter(feature="tempo", min_value=100, max_value=140),),
        targets=(FeatureTarget(feature="danceability", target_value=0.7),),
        candidate_track_ids=("high", "low"),
        top_k=10,
    )

    output = filter_by_features(songs, request, _test_config())

    assert output.input_count == 2
    assert output.output_count == 1
    assert output.candidates[0].track_id == "high"
    assert output.candidates[0].passed_range_filters == ("tempo:100.0..140.0",)


def test_feature_filter_rejects_unsupported_feature():
    with pytest.raises(ValidationError):
        FeatureFilterInput(
            targets=(FeatureTarget(feature="playlist_genre", direction="high"),),
        )


def test_feature_filter_clamps_oversized_top_k():
    request = FeatureFilterInput(
        targets=(FeatureTarget(feature="energy", direction="high"),),
        top_k=999,
    )

    assert request.top_k == MAX_RETRIEVAL_TOP_K


def test_feature_filter_rejects_values_outside_config_range():
    songs = _sample_feature_rows()
    request = FeatureFilterInput(
        range_filters=(FeatureRangeFilter(feature="energy", min_value=-0.1, max_value=0.5),),
    )

    with pytest.raises(ValueError, match="below dataset minimum"):
        filter_by_features(songs, request, _test_config())


def test_feature_filter_registry_exposes_schema_and_callable():
    registry = get_tool_registry()

    assert "feature_filter" in registry
    assert registry["feature_filter"].input_schema is FeatureFilterInput
    assert registry["feature_filter"].callable is not filter_by_features


def test_feature_filter_runs_through_tool_runner():
    context = ToolContext(songs=_sample_feature_rows(), retrieval_prompt_config=_test_config())
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run(
        "feature_filter",
        {
            "targets": [{"feature": "energy", "direction": "high"}],
            "top_k": 1,
        },
    )

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.output["output_count"] == 1
    assert result.output["candidates"][0]["track_id"] == "high"
    assert "duration_ms" in result.trace


def test_feature_filter_tool_runner_returns_validation_error():
    context = ToolContext(songs=_sample_feature_rows(), retrieval_prompt_config=_test_config())
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run("feature_filter", {"targets": [{"feature": "bad_feature", "direction": "high"}]})

    assert isinstance(result, ToolError)
    assert result.ok is False
    assert result.error_type == "validation_error"
    assert result.details["errors"]


def test_feature_filter_tool_runner_returns_runtime_error():
    context = ToolContext(songs=_sample_feature_rows(), retrieval_prompt_config=_test_config())
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run(
        "feature_filter",
        {
            "range_filters": [{"feature": "energy", "min_value": -0.1}],
        },
    )

    assert isinstance(result, ToolError)
    assert result.ok is False
    assert result.error_type == "ValueError"
    assert "below dataset minimum" in result.message


def test_feature_filter_prompt_spec_is_schema_backed_and_prompt_safe():
    spec = get_tool_registry()["feature_filter"].to_prompt_spec()

    assert spec["name"] == "feature_filter"
    assert spec["input_schema"]["title"] == "FeatureFilterInput"
    assert spec["output_schema"]["title"] == "FeatureFilterOutput"
    assert "lyrics" not in str(spec)
    assert "feature_columns" in spec["constraints"]


def _sample_feature_rows() -> pd.DataFrame:
    rows = [
        _row("high", energy=0.9, danceability=0.72, tempo=128),
        _row("mid", energy=0.5, danceability=0.55, tempo=110),
        _row("low", energy=0.15, danceability=0.25, tempo=82),
    ]
    return pd.DataFrame(rows)


def _row(track_id: str, energy: float, danceability: float, tempo: float) -> dict[str, float | str]:
    return {
        "track_id": track_id,
        "energy": energy,
        "danceability": danceability,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "valence": 0.4,
        "tempo": tempo,
        "speechiness": 0.05,
        "liveness": 0.1,
        "duration_ms": 200000,
        "loudness": -6.0,
        "track_popularity": 50,
    }


def _test_config() -> dict:
    return {
        "llm_prompt_constraints": {
            "numeric_ranges": {
                "energy": {"min": 0, "max": 1},
                "danceability": {"min": 0, "max": 1},
                "acousticness": {"min": 0, "max": 1},
                "instrumentalness": {"min": 0, "max": 1},
                "valence": {"min": 0, "max": 1},
                "tempo": {"min": 50, "max": 200},
                "speechiness": {"min": 0, "max": 1},
                "liveness": {"min": 0, "max": 1},
                "duration_ms": {"min": 30000, "max": 600000},
                "loudness": {"min": -40, "max": 5},
                "track_popularity": {"min": 0, "max": 100},
            }
        }
    }
