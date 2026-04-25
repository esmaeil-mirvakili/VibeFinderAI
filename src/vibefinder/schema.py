"""Shared typed objects for the recommendation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FeatureDirection = Literal["low", "medium", "high"]
ConfidenceLabel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Preferences:
    """Structured interpretation of a user's English music request."""

    raw_query: str
    lyric_concepts: tuple[str, ...] = ()
    genre_terms: tuple[str, ...] = ()
    language: str | None = None
    energy: FeatureDirection | None = None
    valence: FeatureDirection | None = None
    tempo: FeatureDirection | None = None
    danceability: FeatureDirection | None = None
    acousticness: FeatureDirection | None = None
    exclusions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalPlan:
    """The retrieval modes and knobs chosen for a request."""

    modes: tuple[str, ...]
    primary_mode: str
    broad_search: bool | None = None
    top_k_per_mode: int | None = None


@dataclass(frozen=True)
class Critique:
    """Group-level review of first-pass candidates."""

    issues: tuple[str, ...] = ()
    should_revise: bool = False


@dataclass(frozen=True)
class ReliabilityReport:
    """Confidence and warnings exposed with final recommendations."""

    confidence: ConfidenceLabel
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    """A final grounded recommendation."""

    track_id: str
    track_name: str
    track_artist: str
    score: float
    explanation: str


@dataclass
class PipelineTrace:
    """Inspectable intermediate state for app display and evaluation."""

    preferences: Preferences
    retrieval_plan: RetrievalPlan
    retrieval_modes_used: list[str] = field(default_factory=list)
    first_pass_count: int = 0
    final_candidate_count: int = 0
    critique: Critique = field(default_factory=Critique)
    revised: bool = False


@dataclass(frozen=True)
class PipelineResult:
    """End-to-end output from a pipeline run."""

    query: str
    recommendations: tuple[Recommendation, ...]
    reliability: ReliabilityReport
    trace: PipelineTrace
