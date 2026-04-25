"""Tool registry for deterministic agent tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from vibefinder.tools.candidate_scoring import candidate_scoring_prompt_constraints, score_candidates
from vibefinder.tools.feature_filter import filter_by_features
from vibefinder.tools.lyric_retrieval import lyric_prompt_constraints, retrieve_by_lyrics
from vibefinder.tools.metadata_retrieval import metadata_prompt_constraints, retrieve_by_metadata
from vibefinder.tools.reliability import assess_reliability, reliability_prompt_constraints
from vibefinder.tools.runtime import ToolContext
from vibefinder.tools.schemas import (
    CandidateScoringInput,
    CandidateScoringOutput,
    FEATURE_COLUMNS,
    FeatureFilterInput,
    FeatureFilterOutput,
    LyricRetrievalInput,
    LyricRetrievalOutput,
    MetadataRetrievalInput,
    MetadataRetrievalOutput,
    ReliabilityInput,
    ReliabilityOutput,
)


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata and callable for one deterministic tool."""

    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    callable: Callable[[ToolContext, BaseModel], Any]
    constraints: dict[str, Any] | None = None

    def to_prompt_spec(self) -> dict[str, Any]:
        """Return prompt-safe tool schema metadata for LLM agents."""

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
            "output_schema": self.output_schema.model_json_schema(),
            "constraints": self.constraints or {},
        }


def run_feature_filter(context: ToolContext, request: FeatureFilterInput) -> FeatureFilterOutput:
    """Context-aware wrapper for LangGraph/tool-runner execution."""

    return filter_by_features(
        songs=context.songs,
        request=request,
        retrieval_prompt_config=context.retrieval_prompt_config,
    )


def run_metadata_retrieval(
    context: ToolContext,
    request: MetadataRetrievalInput,
) -> MetadataRetrievalOutput:
    """Context-aware wrapper for LangGraph/tool-runner execution."""

    return retrieve_by_metadata(
        songs=context.songs,
        request=request,
        retrieval_prompt_config=context.retrieval_prompt_config,
    )


def run_lyric_retrieval(
    context: ToolContext,
    request: LyricRetrievalInput,
) -> LyricRetrievalOutput:
    """Context-aware wrapper for LangGraph/tool-runner execution."""

    return retrieve_by_lyrics(
        songs=context.songs,
        request=request,
        lyric_index=context.lyric_index,
        embedder=context.lyric_embedder,
        retrieval_prompt_config=context.retrieval_prompt_config,
    )


def run_candidate_scoring(
    context: ToolContext,
    request: CandidateScoringInput,
) -> CandidateScoringOutput:
    """Context-aware wrapper for LangGraph/tool-runner execution."""

    return score_candidates(
        songs=context.songs,
        request=request,
    )


def run_reliability(
    context: ToolContext,
    request: ReliabilityInput,
) -> ReliabilityOutput:
    """Context-aware wrapper for LangGraph/tool-runner execution."""

    return assess_reliability(
        songs=context.songs,
        request=request,
    )


def get_tool_registry() -> dict[str, ToolDefinition]:
    """Return registered deterministic tools."""

    feature_filter = ToolDefinition(
        name="feature_filter",
        description=(
            "Filter and score candidate songs using numeric audio features. "
            "Inputs are validated against retrieval_prompt_config numeric ranges."
        ),
        input_schema=FeatureFilterInput,
        output_schema=FeatureFilterOutput,
        callable=run_feature_filter,
        constraints={
            "feature_columns": list(FEATURE_COLUMNS),
            "note": "Feature names are validated by schema and retrieval_prompt_config numeric ranges.",
        },
    )
    metadata_retrieval = ToolDefinition(
        name="metadata_retrieval",
        description=(
            "Retrieve candidate songs with categorical metadata filters and full-text metadata search. "
            "Genres, subgenres, and language are categorical; playlist, artist, and album names are full-text."
        ),
        input_schema=MetadataRetrievalInput,
        output_schema=MetadataRetrievalOutput,
        callable=run_metadata_retrieval,
        constraints=metadata_prompt_constraints(),
    )
    lyric_retrieval = ToolDefinition(
        name="lyric_retrieval",
        description=(
            "Retrieve candidate songs for lyric theme, story, mood, or phrase requests using an in-memory "
            "FAISS lyric index. The output returns track ids and scores, not full lyrics."
        ),
        input_schema=LyricRetrievalInput,
        output_schema=LyricRetrievalOutput,
        callable=run_lyric_retrieval,
        constraints=lyric_prompt_constraints(),
    )
    candidate_scoring = ToolDefinition(
        name="candidate_scoring",
        description=(
            "Combine lyric, metadata, and feature retrieval evidence into deterministic ranked candidates. "
            "Popularity is a small adjustment and artist/album repetition can be penalized."
        ),
        input_schema=CandidateScoringInput,
        output_schema=CandidateScoringOutput,
        callable=run_candidate_scoring,
        constraints=candidate_scoring_prompt_constraints(),
    )
    reliability = ToolDefinition(
        name="reliability",
        description=(
            "Assess final recommendation confidence and warnings from measurable pipeline evidence, "
            "including candidate counts, score distribution, verifier results, critic issues, "
            "retrieval modes, and revision outcome."
        ),
        input_schema=ReliabilityInput,
        output_schema=ReliabilityOutput,
        callable=run_reliability,
        constraints=reliability_prompt_constraints(),
    )
    return {
        feature_filter.name: feature_filter,
        metadata_retrieval.name: metadata_retrieval,
        lyric_retrieval.name: lyric_retrieval,
        candidate_scoring.name: candidate_scoring,
        reliability.name: reliability,
    }
