from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from vibefinder.tools import (
    MetadataCategoricalFilter,
    MetadataRetrievalInput,
    MetadataTextQuery,
    ToolContext,
    ToolError,
    ToolResult,
    ToolRunner,
    get_tool_registry,
    retrieve_by_metadata,
)


def test_metadata_retrieval_filters_categorical_values():
    request = MetadataRetrievalInput(
        categorical_filters=(
            MetadataCategoricalFilter(field="playlist_genre", values=("pop",)),
            MetadataCategoricalFilter(field="language", values=("en",)),
        ),
        top_k=10,
    )

    output = retrieve_by_metadata(_sample_metadata_rows(), request, _test_config())

    assert output.output_count == 2
    assert [candidate.track_id for candidate in output.candidates] == ["pop-1", "pop-2"]
    assert output.candidates[0].categorical_matches["playlist_genre"] == "pop"


def test_metadata_retrieval_searches_full_text_case_insensitively():
    request = MetadataRetrievalInput(
        text_queries=(MetadataTextQuery(field="track_artist", query="aStEr"),),
        top_k=10,
    )

    output = retrieve_by_metadata(_sample_metadata_rows(), request, _test_config())

    assert output.output_count == 2
    assert [candidate.track_id for candidate in output.candidates] == ["pop-1", "rock-1"]
    assert output.candidates[0].text_matches == {"track_artist": "aStEr"}


def test_metadata_retrieval_combines_categorical_and_full_text():
    request = MetadataRetrievalInput(
        categorical_filters=(MetadataCategoricalFilter(field="playlist_genre", values=("pop",)),),
        text_queries=(MetadataTextQuery(field="track_album_name", query="night"),),
        top_k=10,
    )

    output = retrieve_by_metadata(_sample_metadata_rows(), request, _test_config())

    assert output.output_count == 1
    assert output.candidates[0].track_id == "pop-1"
    assert output.candidates[0].match_reasons == (
        "playlist_genre=pop",
        "track_album_name contains 'night'",
    )


def test_metadata_retrieval_requires_all_text_queries_by_default():
    request = MetadataRetrievalInput(
        text_queries=(
            MetadataTextQuery(field="track_artist", query="Aster"),
            MetadataTextQuery(field="track_album_name", query="Night"),
        ),
        top_k=10,
    )

    output = retrieve_by_metadata(_sample_metadata_rows(), request, _test_config())

    assert output.output_count == 1
    assert output.candidates[0].track_id == "pop-1"


def test_metadata_retrieval_supports_any_text_query_mode():
    request = MetadataRetrievalInput(
        text_queries=(
            MetadataTextQuery(field="track_artist", query="Aster"),
            MetadataTextQuery(field="track_album_name", query="Night"),
        ),
        text_match_mode="any",
        top_k=10,
    )

    output = retrieve_by_metadata(_sample_metadata_rows(), request, _test_config())

    assert output.output_count == 2
    assert [candidate.track_id for candidate in output.candidates] == ["pop-1", "rock-1"]


def test_metadata_retrieval_rejects_full_text_field_as_categorical():
    with pytest.raises(ValidationError):
        MetadataRetrievalInput(
            categorical_filters=(MetadataCategoricalFilter(field="track_artist", values=("Aster",)),),
        )


def test_metadata_retrieval_rejects_invalid_categorical_value_against_config():
    request = MetadataRetrievalInput(
        categorical_filters=(MetadataCategoricalFilter(field="playlist_genre", values=("classical",)),),
    )

    with pytest.raises(ValueError, match="Invalid values"):
        retrieve_by_metadata(_sample_metadata_rows(), request, _test_config())


def test_metadata_retrieval_runs_through_tool_runner():
    context = ToolContext(songs=_sample_metadata_rows(), retrieval_prompt_config=_test_config())
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run(
        "metadata_retrieval",
        {
            "text_queries": [{"field": "playlist_name", "query": "drama"}],
            "top_k": 1,
        },
    )

    assert isinstance(result, ToolResult)
    assert result.output["output_count"] == 1
    assert result.output["candidates"][0]["track_id"] == "pop-1"


def test_metadata_retrieval_tool_runner_returns_validation_error():
    context = ToolContext(songs=_sample_metadata_rows(), retrieval_prompt_config=_test_config())
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run(
        "metadata_retrieval",
        {"text_queries": [{"field": "playlist_genre", "query": "pop"}]},
    )

    assert isinstance(result, ToolError)
    assert result.error_type == "validation_error"


def test_metadata_retrieval_tool_runner_returns_runtime_error():
    context = ToolContext(songs=_sample_metadata_rows(), retrieval_prompt_config=_test_config())
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run(
        "metadata_retrieval",
        {"categorical_filters": [{"field": "playlist_genre", "values": ["classical"]}]},
    )

    assert isinstance(result, ToolError)
    assert result.error_type == "ValueError"
    assert "Invalid values" in result.message


def test_metadata_retrieval_prompt_spec_marks_full_text_fields_not_enums():
    spec = get_tool_registry()["metadata_retrieval"].to_prompt_spec()

    assert spec["name"] == "metadata_retrieval"
    assert spec["input_schema"]["title"] == "MetadataRetrievalInput"
    assert spec["output_schema"]["title"] == "MetadataRetrievalOutput"
    assert spec["constraints"]["categorical_fields"] == [
        "playlist_genre",
        "playlist_subgenre",
        "language",
    ]
    assert spec["constraints"]["full_text_fields"] == [
        "playlist_name",
        "track_artist",
        "track_album_name",
    ]
    assert spec["constraints"]["text_match_modes"] == ["all", "any"]
    assert spec["constraints"]["default_text_match_mode"] == "all"
    assert "lyrics" not in str(spec)
    assert "Aster" not in str(spec)


def _sample_metadata_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "pop-1",
                "playlist_genre": "pop",
                "playlist_subgenre": "dance pop",
                "language": "en",
                "playlist_name": "Pop Drama",
                "track_artist": "Aster",
                "track_album_name": "Night Signals",
            },
            {
                "track_id": "pop-2",
                "playlist_genre": "pop",
                "playlist_subgenre": "indie poptimism",
                "language": "en",
                "playlist_name": "Bright Pop",
                "track_artist": "Boreal",
                "track_album_name": "Daylight",
            },
            {
                "track_id": "rock-1",
                "playlist_genre": "rock",
                "playlist_subgenre": "hard rock",
                "language": "en",
                "playlist_name": "Rock Night",
                "track_artist": "Aster",
                "track_album_name": "Amp Room",
            },
        ]
    )


def _test_config() -> dict:
    return {
        "llm_prompt_constraints": {
            "categorical_values": {
                "playlist_genre": ["pop", "rock"],
                "playlist_subgenre": ["dance pop", "hard rock", "indie poptimism"],
                "language": ["en", "es"],
            },
            "full_text_search_columns": {
                "playlist_name": {"search_method": "full_text_search", "unique_count": 3},
                "track_artist": {"search_method": "full_text_search", "unique_count": 2},
                "track_album_name": {"search_method": "full_text_search", "unique_count": 3},
            },
        }
    }
