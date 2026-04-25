from __future__ import annotations

import pandas as pd
import pytest
import numpy as np
from pydantic import ValidationError

from vibefinder.embeddings import normalize_embedding_matrix
from vibefinder.tools import (
    LyricRetrievalInput,
    ToolContext,
    ToolError,
    ToolResult,
    ToolRunner,
    build_lyric_index,
    ensure_lyric_index,
    get_tool_registry,
    load_lyric_index,
    lyric_index_files,
    retrieve_by_lyrics,
    save_lyric_index,
)


def test_lyric_retrieval_returns_relevant_track_ids_without_lyrics():
    songs = _sample_lyric_rows()
    embedder = _FakeEmbeddingProvider()
    request = LyricRetrievalInput(query="betrayal regret heartbreak", top_k=2)

    output = retrieve_by_lyrics(
        songs,
        request,
        embedder=embedder,
        retrieval_prompt_config=_test_config(),
    )

    assert output.output_count >= 1
    assert output.candidates[0].track_id == "heartbreak-1"
    assert output.candidates[0].score > 0
    assert "lyrics" not in output.model_dump(mode="json")["candidates"][0]
    assert len(embedder.calls) == 2
    assert len(embedder.calls[0]) == 3
    assert embedder.calls[1] == ("betrayal regret heartbreak",)


def test_lyric_retrieval_supports_language_filter():
    songs = _sample_lyric_rows()
    request = LyricRetrievalInput(query="betrayal regret", language="es", top_k=5)

    output = retrieve_by_lyrics(
        songs,
        request,
        embedder=_FakeEmbeddingProvider(),
        retrieval_prompt_config=_test_config(),
    )

    assert output.output_count == 1
    assert output.input_count == 1
    assert output.candidates[0].track_id == "spanish-1"
    assert output.candidates[0].language == "es"


def test_lyric_retrieval_supports_candidate_subset():
    songs = _sample_lyric_rows()
    request = LyricRetrievalInput(
        query="dance lights",
        candidate_track_ids=("dance-1",),
        top_k=5,
    )

    output = retrieve_by_lyrics(
        songs,
        request,
        embedder=_FakeEmbeddingProvider(),
        retrieval_prompt_config=_test_config(),
    )

    assert output.input_count == 1
    assert output.output_count == 1
    assert output.candidates[0].track_id == "dance-1"


def test_lyric_retrieval_uses_supplied_index():
    songs = _sample_lyric_rows()
    embedder = _FakeEmbeddingProvider()
    index = build_lyric_index(songs, embedder=embedder)
    request = LyricRetrievalInput(query="betrayal regret", top_k=1)

    output = retrieve_by_lyrics(
        songs,
        request,
        lyric_index=index,
        embedder=embedder,
        retrieval_prompt_config=_test_config(),
    )

    assert output.output_count == 1
    assert output.candidates[0].track_id in {"heartbreak-1", "spanish-1"}


def test_lyric_index_save_and_load_round_trip(tmp_path):
    songs = _sample_lyric_rows()
    index = build_lyric_index(songs, embedder=_FakeEmbeddingProvider())

    files = save_lyric_index(index, root=tmp_path)
    loaded = load_lyric_index(root=tmp_path, expected_embedding_model="fake-embedding-model")

    assert files.index_path == lyric_index_files(tmp_path).index_path
    assert files.index_path.exists()
    assert files.metadata_path.exists()
    assert loaded.index.ntotal == index.index.ntotal
    assert loaded.track_ids == index.track_ids
    assert loaded.languages == index.languages
    assert loaded.dimension == index.dimension
    assert loaded.embedding_model == index.embedding_model


def test_ensure_lyric_index_builds_saves_then_reuses_persisted_files(tmp_path):
    songs = _sample_lyric_rows()
    first_embedder = _FakeEmbeddingProvider()

    built = ensure_lyric_index(songs, root=tmp_path, embedder=first_embedder)

    assert built.index.ntotal == 3
    assert len(first_embedder.calls) == 1
    assert lyric_index_files(tmp_path).index_path.exists()
    assert lyric_index_files(tmp_path).metadata_path.exists()

    second_embedder = _FakeEmbeddingProvider()
    loaded = ensure_lyric_index(songs, root=tmp_path, embedder=second_embedder)

    assert loaded.track_ids == built.track_ids
    assert second_embedder.calls == []


def test_lyric_retrieval_rejects_blank_query():
    with pytest.raises(ValidationError):
        LyricRetrievalInput(query="   ")


def test_lyric_retrieval_rejects_invalid_language_against_config():
    request = LyricRetrievalInput(query="betrayal", language="zz")

    with pytest.raises(ValueError, match="Invalid language"):
        retrieve_by_lyrics(
            _sample_lyric_rows(),
            request,
            embedder=_FakeEmbeddingProvider(),
            retrieval_prompt_config=_test_config(),
        )


def test_lyric_retrieval_rejects_mismatched_index_and_query_embedder():
    songs = _sample_lyric_rows()
    index = build_lyric_index(songs, embedder=_FakeEmbeddingProvider(name="index-embedder"))
    request = LyricRetrievalInput(query="betrayal", top_k=1)

    with pytest.raises(ValueError, match="embedding model does not match"):
        retrieve_by_lyrics(
            songs,
            request,
            lyric_index=index,
            embedder=_FakeEmbeddingProvider(name="query-embedder"),
            retrieval_prompt_config=_test_config(),
        )


def test_lyric_retrieval_runs_through_tool_runner():
    songs = _sample_lyric_rows()
    context = ToolContext(
        songs=songs,
        retrieval_prompt_config=_test_config(),
        lyric_index=build_lyric_index(songs, embedder=_FakeEmbeddingProvider()),
        lyric_embedder=_FakeEmbeddingProvider(),
    )
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run("lyric_retrieval", {"query": "betrayal regret", "top_k": 1})

    assert isinstance(result, ToolResult)
    assert result.output["output_count"] == 1
    assert result.output["candidates"][0]["track_id"] in {"heartbreak-1", "spanish-1"}
    assert result.output["retrieval_mode"] == "lyric_faiss"


def test_lyric_retrieval_tool_runner_returns_validation_error():
    context = ToolContext(songs=_sample_lyric_rows(), retrieval_prompt_config=_test_config())
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run("lyric_retrieval", {"query": "betrayal", "top_k": 0})

    assert isinstance(result, ToolError)
    assert result.error_type == "validation_error"


def test_lyric_retrieval_tool_runner_returns_runtime_error():
    context = ToolContext(
        songs=_sample_lyric_rows(),
        retrieval_prompt_config=_test_config(),
        lyric_embedder=_FakeEmbeddingProvider(),
    )
    runner = ToolRunner(context=context, registry=get_tool_registry())

    result = runner.run("lyric_retrieval", {"query": "betrayal", "language": "zz"})

    assert isinstance(result, ToolError)
    assert result.error_type == "ValueError"
    assert "Invalid language" in result.message


def test_lyric_retrieval_prompt_spec_is_schema_backed_and_prompt_safe():
    spec = get_tool_registry()["lyric_retrieval"].to_prompt_spec()

    assert spec["name"] == "lyric_retrieval"
    assert spec["input_schema"]["title"] == "LyricRetrievalInput"
    assert spec["output_schema"]["title"] == "LyricRetrievalOutput"
    assert spec["constraints"]["retrieval_mode"] == "lyric_faiss"
    assert spec["constraints"]["embedding_required"] is True
    assert "secret betrayal lyric" not in str(spec)
    assert "full lyric text" in str(spec)


def test_tool_registry_contains_all_implemented_retrieval_tools():
    registry = get_tool_registry()

    assert {"feature_filter", "metadata_retrieval", "lyric_retrieval", "candidate_scoring"} <= set(registry)


def _sample_lyric_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "heartbreak-1",
                "lyrics": "secret betrayal lyric with heartbreak regret and lonely apologies",
                "language": "en",
            },
            {
                "track_id": "dance-1",
                "lyrics": "dance lights rhythm floor moving together all night",
                "language": "en",
            },
            {
                "track_id": "spanish-1",
                "lyrics": "betrayal regret corazon noche perdida",
                "language": "es",
            },
        ]
    )


def _test_config() -> dict:
    return {
        "llm_prompt_constraints": {
            "categorical_values": {
                "language": ["en", "es"],
            }
        }
    }


class _FakeEmbeddingProvider:
    def __init__(self, name: str = "fake-embedding-model") -> None:
        self._name = name
        self.calls: list[tuple[str, ...]] = []

    @property
    def name(self) -> str:
        return self._name

    def embed_texts(self, texts):
        self.calls.append(tuple(texts))
        return normalize_embedding_matrix(
            np.asarray([self._vector_for_text(text) for text in texts], dtype="float32")
        )

    def _vector_for_text(self, text: str) -> list[float]:
        normalized = text.casefold()
        if "dance" in normalized or "lights" in normalized or "rhythm" in normalized:
            return [0.0, 1.0, 0.0]
        if "heartbreak" in normalized or "lonely" in normalized:
            return [1.0, 0.0, 0.0]
        if "betrayal" in normalized or "regret" in normalized or "corazon" in normalized:
            return [0.85, 0.0, 0.15]
        return [0.0, 0.0, 1.0]
