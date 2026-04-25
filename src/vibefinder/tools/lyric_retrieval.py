"""Lyric retrieval tool implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from loguru import logger

from vibefinder.config import load_retrieval_prompt_config
from vibefinder.embeddings import (
    TextEmbeddingProvider,
    get_default_embedding_provider,
    normalize_embedding_matrix,
)
from vibefinder.tools.schemas import (
    LyricCandidateMatch,
    LyricRetrievalInput,
    LyricRetrievalOutput,
    LYRIC_RETRIEVAL_COLUMNS,
)

DEFAULT_LYRIC_INDEX_FILENAME = "lyric_faiss.index"
DEFAULT_LYRIC_INDEX_METADATA_FILENAME = "lyric_faiss_metadata.json"


@dataclass(frozen=True)
class LyricIndex:
    """In-memory FAISS lyric index and row metadata."""

    index: faiss.Index
    track_ids: tuple[str, ...]
    languages: tuple[str | None, ...]
    dimension: int
    embedding_model: str


@dataclass(frozen=True)
class LyricIndexFiles:
    """Filesystem locations for a persisted lyric FAISS index."""

    index_path: Path
    metadata_path: Path


def build_lyric_index(
    songs: pd.DataFrame,
    embedder: TextEmbeddingProvider | None = None,
) -> LyricIndex:
    """Build an in-memory FAISS index over embedding-model lyric vectors."""

    _validate_required_columns(songs)
    active_embedder = embedder or get_default_embedding_provider()
    track_ids: list[str] = []
    languages: list[str | None] = []
    lyric_texts: list[str] = []

    for _, row in songs.iterrows():
        lyric_text = row.get("lyrics")
        if pd.isna(lyric_text) or not str(lyric_text).strip():
            continue
        track_ids.append(str(row["track_id"]))
        lyric_texts.append(str(lyric_text))
        language = row.get("language")
        languages.append(None if pd.isna(language) else str(language))

    if lyric_texts:
        matrix = normalize_embedding_matrix(active_embedder.embed_texts(lyric_texts))
        dimension = matrix.shape[1]
    else:
        matrix = np.empty((0, 1), dtype="float32")
        dimension = 1

    index = faiss.IndexFlatIP(dimension)
    if len(matrix):
        index.add(matrix)

    logger.info(
        "lyric_index_built",
        indexed_count=len(track_ids),
        source_row_count=len(songs),
        dimension=dimension,
        embedding_model=active_embedder.name,
    )
    return LyricIndex(
        index=index,
        track_ids=tuple(track_ids),
        languages=tuple(languages),
        dimension=dimension,
        embedding_model=active_embedder.name,
    )


def lyric_index_files(root: str | Path = ".") -> LyricIndexFiles:
    """Return project-root file paths for the persisted lyric index."""

    root_path = Path(root).expanduser()
    return LyricIndexFiles(
        index_path=root_path / DEFAULT_LYRIC_INDEX_FILENAME,
        metadata_path=root_path / DEFAULT_LYRIC_INDEX_METADATA_FILENAME,
    )


def save_lyric_index(
    lyric_index: LyricIndex,
    root: str | Path = ".",
) -> LyricIndexFiles:
    """Persist a lyric FAISS index and its row metadata."""

    files = lyric_index_files(root)
    files.index_path.parent.mkdir(parents=True, exist_ok=True)
    files.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(lyric_index.index, str(files.index_path))
    metadata = {
        "embedding_model": lyric_index.embedding_model,
        "dimension": lyric_index.dimension,
        "track_ids": list(lyric_index.track_ids),
        "languages": list(lyric_index.languages),
        "indexed_count": len(lyric_index.track_ids),
    }
    files.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "lyric_index_saved",
        index_path=str(files.index_path),
        metadata_path=str(files.metadata_path),
        indexed_count=len(lyric_index.track_ids),
        dimension=lyric_index.dimension,
        embedding_model=lyric_index.embedding_model,
    )
    return files


def load_lyric_index(
    root: str | Path = ".",
    expected_embedding_model: str | None = None,
) -> LyricIndex:
    """Load a persisted lyric FAISS index from the project root."""

    files = lyric_index_files(root)
    if not files.index_path.exists() or not files.metadata_path.exists():
        raise FileNotFoundError(
            "Persisted lyric index files are missing: "
            f"{files.index_path} and {files.metadata_path}"
        )

    metadata = json.loads(files.metadata_path.read_text(encoding="utf-8"))
    embedding_model = str(metadata["embedding_model"])
    if expected_embedding_model is not None and embedding_model != expected_embedding_model:
        raise ValueError(
            "Persisted lyric index embedding model does not match configured embedder: "
            f"{embedding_model} != {expected_embedding_model}"
        )

    index = faiss.read_index(str(files.index_path))
    track_ids = tuple(str(track_id) for track_id in metadata["track_ids"])
    languages = tuple(
        None if language is None else str(language)
        for language in metadata["languages"]
    )
    dimension = int(metadata["dimension"])
    if index.d != dimension:
        raise ValueError(f"Persisted lyric index dimension mismatch: {index.d} != {dimension}")
    if index.ntotal != len(track_ids):
        raise ValueError(f"Persisted lyric index row count mismatch: {index.ntotal} != {len(track_ids)}")
    if len(languages) != len(track_ids):
        raise ValueError(
            f"Persisted lyric index language count mismatch: {len(languages)} != {len(track_ids)}"
        )

    logger.info(
        "lyric_index_loaded",
        index_path=str(files.index_path),
        metadata_path=str(files.metadata_path),
        indexed_count=len(track_ids),
        dimension=dimension,
        embedding_model=embedding_model,
    )
    return LyricIndex(
        index=index,
        track_ids=track_ids,
        languages=languages,
        dimension=dimension,
        embedding_model=embedding_model,
    )


def ensure_lyric_index(
    songs: pd.DataFrame,
    root: str | Path = ".",
    embedder: TextEmbeddingProvider | None = None,
) -> LyricIndex:
    """Load the persisted lyric index or build and save it if missing or invalid."""

    active_embedder = embedder or get_default_embedding_provider()
    files = lyric_index_files(root)
    try:
        lyric_index = load_lyric_index(
            root=root,
            expected_embedding_model=active_embedder.name,
        )
        logger.info(
            "lyric_index_startup_resolved",
            source="persisted",
            index_path=str(files.index_path),
            metadata_path=str(files.metadata_path),
        )
        return lyric_index
    except FileNotFoundError:
        logger.info(
            "lyric_index_startup_missing",
            index_path=str(files.index_path),
            metadata_path=str(files.metadata_path),
        )
    except Exception as exc:
        logger.warning(
            "lyric_index_startup_invalid",
            index_path=str(files.index_path),
            metadata_path=str(files.metadata_path),
            error_type=exc.__class__.__name__,
            error=str(exc),
        )

    lyric_index = build_lyric_index(songs=songs, embedder=active_embedder)
    save_lyric_index(lyric_index=lyric_index, root=root)
    logger.info(
        "lyric_index_startup_resolved",
        source="rebuilt",
        index_path=str(files.index_path),
        metadata_path=str(files.metadata_path),
    )
    return lyric_index


def retrieve_by_lyrics(
    songs: pd.DataFrame,
    request: LyricRetrievalInput,
    lyric_index: LyricIndex | None = None,
    embedder: TextEmbeddingProvider | None = None,
    retrieval_prompt_config: dict[str, Any] | None = None,
) -> LyricRetrievalOutput:
    """Retrieve songs using the lyrics FAISS index."""

    _validate_required_columns(songs)
    config = retrieval_prompt_config or load_retrieval_prompt_config()
    _validate_language_against_config(request, config)

    active_embedder = embedder or get_default_embedding_provider()
    active_index = lyric_index or build_lyric_index(songs, embedder=active_embedder)
    _validate_embedder_matches_index(active_embedder, active_index)
    input_count = _filtered_input_count(active_index, request)
    warnings: list[str] = []

    if active_index.index.ntotal == 0:
        warnings.append("No lyric rows are available for retrieval.")
        return _empty_output(input_count, warnings)

    query_matrix = normalize_embedding_matrix(active_embedder.embed_texts([request.query]))
    if query_matrix.shape[1] != active_index.dimension:
        raise ValueError(
            "Query embedding dimension does not match lyric index dimension: "
            f"{query_matrix.shape[1]} != {active_index.dimension}"
        )

    search_k = _search_limit(active_index.index.ntotal, request)
    scores, indices = active_index.index.search(query_matrix, search_k)
    candidate_ids = set(request.candidate_track_ids) if request.candidate_track_ids is not None else None
    language = _normalize(request.language) if request.language is not None else None

    lyric_preview_lookup = _lyric_preview_lookup(songs)
    scored_matches: list[tuple[str, str | None, float]] = []
    seen_track_ids: set[str] = set()
    for raw_score, raw_index in zip(scores[0], indices[0], strict=True):
        if raw_index < 0:
            continue
        track_id = active_index.track_ids[int(raw_index)]
        if track_id in seen_track_ids:
            continue
        if candidate_ids is not None and track_id not in candidate_ids:
            continue

        candidate_language = active_index.languages[int(raw_index)]
        if language is not None and _normalize(candidate_language) != language:
            continue

        score = _normalized_similarity_score(float(raw_score))
        if score <= request.min_score:
            continue

        seen_track_ids.add(track_id)
        scored_matches.append((track_id, candidate_language, score))

    scored_matches.sort(key=lambda item: (-item[2], item[0]))
    limited_matches = scored_matches[: request.top_k]
    matches = [
        LyricCandidateMatch(
            track_id=track_id,
            score=round(score, 6),
            rank=rank,
            language=candidate_language,
            lyric_preview=lyric_preview_lookup.get(track_id),
        )
        for rank, (track_id, candidate_language, score) in enumerate(limited_matches, start=1)
    ]

    if not matches:
        warnings.append("No candidates matched the Lyric RAG constraints.")
    elif len(scored_matches) > request.top_k:
        warnings.append(f"Returned top {request.top_k} of {len(scored_matches)} lyric-retrieved candidates.")

    logger.info(
        "lyric_retrieval_finished",
        input_count=input_count,
        output_count=len(matches),
        top_k=request.top_k,
        language=request.language,
        embedding_model=active_index.embedding_model,
        candidate_subset_size=None if request.candidate_track_ids is None else len(request.candidate_track_ids),
        warnings=warnings,
    )
    return LyricRetrievalOutput(
        candidates=tuple(matches),
        input_count=input_count,
        output_count=len(matches),
        warnings=tuple(warnings),
    )


def lyric_prompt_constraints(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return prompt-safe Lyric RAG constraints."""

    active_config = config or load_retrieval_prompt_config()
    return {
        "source_columns": list(LYRIC_RETRIEVAL_COLUMNS),
        "query_type": "natural_language_lyric_intent",
        "embedding_required": True,
        "embedding_provider": "Configured TextEmbeddingProvider. Default: sentence-transformers/all-MiniLM-L6-v2.",
        "optional_filters": {
            "language": active_config["llm_prompt_constraints"]["categorical_values"].get("language", []),
            "candidate_track_ids": "optional subset from prior retrieval tools",
        },
        "retrieval_mode": "lyric_faiss",
        "does_not_return": ["lyrics", "full lyric text"],
        "note": (
            "Use for theme, story, mood, and phrase-level lyric requests. The tool embeds lyrics and queries "
            "with the same configured embedding model, searches FAISS, and does not include full lyrics in "
            "prompts or logs."
        ),
    }


def _validate_required_columns(songs: pd.DataFrame) -> None:
    missing = [column for column in LYRIC_RETRIEVAL_COLUMNS if column not in songs.columns]
    if missing:
        raise ValueError(f"Songs DataFrame is missing Lyric RAG columns: {', '.join(missing)}")


def _validate_language_against_config(
    request: LyricRetrievalInput,
    config: dict[str, Any],
) -> None:
    if request.language is None:
        return
    valid_languages = {
        _normalize(value)
        for value in config["llm_prompt_constraints"]["categorical_values"].get("language", [])
    }
    if valid_languages and _normalize(request.language) not in valid_languages:
        raise ValueError(f"Invalid language for Lyric RAG: {request.language}")


def _validate_embedder_matches_index(
    embedder: TextEmbeddingProvider,
    lyric_index: LyricIndex,
) -> None:
    if embedder.name != lyric_index.embedding_model:
        raise ValueError(
            "Lyric index embedding model does not match query embedding model: "
            f"{lyric_index.embedding_model} != {embedder.name}"
        )


def _filtered_input_count(active_index: LyricIndex, request: LyricRetrievalInput) -> int:
    candidate_ids = set(request.candidate_track_ids) if request.candidate_track_ids is not None else None
    language = _normalize(request.language) if request.language is not None else None
    count = 0
    for track_id, candidate_language in zip(active_index.track_ids, active_index.languages, strict=True):
        if candidate_ids is not None and track_id not in candidate_ids:
            continue
        if language is not None and _normalize(candidate_language) != language:
            continue
        count += 1
    return count


def _search_limit(index_size: int, request: LyricRetrievalInput) -> int:
    if request.candidate_track_ids is not None or request.language is not None:
        return index_size
    return min(index_size, max(request.top_k * 5, request.top_k))


def _empty_output(input_count: int, warnings: list[str]) -> LyricRetrievalOutput:
    logger.info(
        "lyric_retrieval_finished",
        input_count=input_count,
        output_count=0,
        warnings=warnings,
    )
    return LyricRetrievalOutput(
        candidates=(),
        input_count=input_count,
        output_count=0,
        warnings=tuple(warnings),
    )


def _lyric_preview_lookup(songs: pd.DataFrame, max_chars: int = 180) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for _, row in songs.iterrows():
        track_id = str(row.get("track_id", "")).strip()
        if not track_id or track_id in lookup:
            continue
        lyric_text = row.get("lyrics")
        if pd.isna(lyric_text):
            continue
        preview = " ".join(str(lyric_text).split())
        if not preview:
            continue
        lookup[track_id] = preview[:max_chars].rstrip()
    return lookup


def _normalized_similarity_score(score: float) -> float:
    if np.isnan(score):
        return 0.0
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.casefold().strip().split())
