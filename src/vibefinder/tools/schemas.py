"""Pydantic schemas for deterministic agent tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


FEATURE_COLUMNS: tuple[str, ...] = (
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
    "valence",
    "tempo",
    "speechiness",
    "liveness",
    "duration_ms",
    "loudness",
    "track_popularity",
)
MIN_RETRIEVAL_TOP_K = 1
MAX_RETRIEVAL_TOP_K = 100
DEFAULT_RETRIEVAL_TOP_K = 50
MIN_FINAL_TOP_K = 1
MAX_FINAL_TOP_K = 20
DEFAULT_FINAL_TOP_K = 10

FeatureDirection = Literal["low", "medium", "high"]
TextMatchMode = Literal["all", "any"]
ConfidenceLabel = Literal["low", "medium", "high"]

CATEGORICAL_METADATA_COLUMNS: tuple[str, ...] = (
    "playlist_genre",
    "playlist_subgenre",
    "language",
)
FULL_TEXT_METADATA_COLUMNS: tuple[str, ...] = (
    "playlist_name",
    "track_artist",
    "track_album_name",
)
METADATA_COLUMNS: tuple[str, ...] = CATEGORICAL_METADATA_COLUMNS + FULL_TEXT_METADATA_COLUMNS
LYRIC_RETRIEVAL_COLUMNS: tuple[str, ...] = (
    "track_id",
    "lyrics",
    "language",
)


class FeatureRangeFilter(BaseModel):
    """Inclusive hard range constraint for one numeric feature."""

    model_config = ConfigDict(frozen=True)

    feature: str = Field(description="Numeric feature column to filter.")
    min_value: float | None = Field(default=None, description="Inclusive lower bound.")
    max_value: float | None = Field(default=None, description="Inclusive upper bound.")

    @model_validator(mode="after")
    def validate_bounds(self) -> "FeatureRangeFilter":
        if self.feature not in FEATURE_COLUMNS:
            raise ValueError(f"Unsupported feature column: {self.feature}")
        if self.min_value is None and self.max_value is None:
            raise ValueError("At least one of min_value or max_value is required.")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value cannot be greater than max_value.")
        return self


class FeatureTarget(BaseModel):
    """Soft target for one numeric feature."""

    model_config = ConfigDict(frozen=True)

    feature: str = Field(description="Numeric feature column to score.")
    direction: FeatureDirection | None = Field(
        default=None,
        description="Qualitative target direction mapped to dataset numeric range.",
    )
    target_value: float | None = Field(default=None, description="Explicit numeric target.")
    weight: float = Field(default=1.0, gt=0, description="Relative scoring weight.")

    @model_validator(mode="after")
    def validate_target(self) -> "FeatureTarget":
        if self.feature not in FEATURE_COLUMNS:
            raise ValueError(f"Unsupported feature column: {self.feature}")
        if self.direction is None and self.target_value is None:
            raise ValueError("Either direction or target_value is required.")
        return self


class FeatureFilterInput(BaseModel):
    """Input for the feature filter tool."""

    model_config = ConfigDict(frozen=True)

    range_filters: tuple[FeatureRangeFilter, ...] = Field(default_factory=tuple)
    targets: tuple[FeatureTarget, ...] = Field(default_factory=tuple)
    candidate_track_ids: tuple[str, ...] | None = Field(
        default=None,
        description="Optional candidate ids to filter before scoring.",
    )
    top_k: int = Field(
        default=DEFAULT_RETRIEVAL_TOP_K,
        ge=MIN_RETRIEVAL_TOP_K,
        le=MAX_RETRIEVAL_TOP_K,
    )

    @model_validator(mode="before")
    @classmethod
    def clamp_high_top_k(cls, data: object) -> object:
        return _clamp_high_top_k(data, max_value=MAX_RETRIEVAL_TOP_K)

    @model_validator(mode="after")
    def require_constraints(self) -> "FeatureFilterInput":
        if not self.range_filters and not self.targets:
            raise ValueError("At least one range filter or target is required.")
        return self


class FeatureCandidateMatch(BaseModel):
    """One feature-filtered candidate."""

    track_id: str
    score: float = Field(ge=0, le=1)
    passed_range_filters: tuple[str, ...] = Field(default_factory=tuple)
    feature_scores: dict[str, float] = Field(default_factory=dict)
    feature_values: dict[str, float | int | None] = Field(default_factory=dict)


class FeatureFilterOutput(BaseModel):
    """Output from the feature filter tool."""

    candidates: tuple[FeatureCandidateMatch, ...]
    input_count: int
    output_count: int
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class MetadataCategoricalFilter(BaseModel):
    """Exact categorical metadata filter."""

    model_config = ConfigDict(frozen=True)

    field: str = Field(description="Categorical metadata field.")
    values: tuple[str, ...] = Field(min_length=1, description="Accepted values.")

    @model_validator(mode="after")
    def validate_filter(self) -> "MetadataCategoricalFilter":
        if self.field not in CATEGORICAL_METADATA_COLUMNS:
            raise ValueError(f"Unsupported categorical metadata field: {self.field}")
        return self


class MetadataTextQuery(BaseModel):
    """Full-text metadata search query."""

    model_config = ConfigDict(frozen=True)

    field: str = Field(description="Full-text metadata field.")
    query: str = Field(min_length=1, description="Case-insensitive search text.")
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_text_query(self) -> "MetadataTextQuery":
        if self.field not in FULL_TEXT_METADATA_COLUMNS:
            raise ValueError(f"Unsupported full-text metadata field: {self.field}")
        if not self.query.strip():
            raise ValueError("query cannot be blank.")
        return self


class MetadataRetrievalInput(BaseModel):
    """Input for the metadata retrieval tool."""

    model_config = ConfigDict(frozen=True)

    categorical_filters: tuple[MetadataCategoricalFilter, ...] = Field(default_factory=tuple)
    text_queries: tuple[MetadataTextQuery, ...] = Field(default_factory=tuple)
    text_match_mode: TextMatchMode = Field(
        default="all",
        description="Whether all text queries must match or any text query may match.",
    )
    candidate_track_ids: tuple[str, ...] | None = Field(default=None)
    top_k: int = Field(
        default=DEFAULT_RETRIEVAL_TOP_K,
        ge=MIN_RETRIEVAL_TOP_K,
        le=MAX_RETRIEVAL_TOP_K,
    )

    @model_validator(mode="before")
    @classmethod
    def clamp_high_top_k(cls, data: object) -> object:
        return _clamp_high_top_k(data, max_value=MAX_RETRIEVAL_TOP_K)

    @model_validator(mode="after")
    def require_constraints(self) -> "MetadataRetrievalInput":
        if not self.categorical_filters and not self.text_queries:
            raise ValueError("At least one categorical filter or text query is required.")
        return self


class MetadataCandidateMatch(BaseModel):
    """One metadata-retrieved candidate."""

    track_id: str
    score: float = Field(ge=0)
    categorical_matches: dict[str, str] = Field(default_factory=dict)
    text_matches: dict[str, str] = Field(default_factory=dict)
    match_reasons: tuple[str, ...] = Field(default_factory=tuple)


class MetadataRetrievalOutput(BaseModel):
    """Output from the metadata retrieval tool."""

    candidates: tuple[MetadataCandidateMatch, ...]
    input_count: int
    output_count: int
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class LyricRetrievalInput(BaseModel):
    """Input for the Lyric RAG tool."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        min_length=1,
        description="Natural-language lyric intent, theme, story, mood, or phrase to retrieve.",
    )
    language: str | None = Field(
        default=None,
        description="Optional exact language code filter, such as 'en'.",
    )
    candidate_track_ids: tuple[str, ...] | None = Field(
        default=None,
        description="Optional candidate ids to search within after Lyric RAG retrieval.",
    )
    top_k: int = Field(
        default=DEFAULT_RETRIEVAL_TOP_K,
        ge=MIN_RETRIEVAL_TOP_K,
        le=MAX_RETRIEVAL_TOP_K,
    )
    min_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Minimum normalized Lyric RAG score to return.",
    )

    @model_validator(mode="before")
    @classmethod
    def clamp_high_top_k(cls, data: object) -> object:
        return _clamp_high_top_k(data, max_value=MAX_RETRIEVAL_TOP_K)

    @model_validator(mode="after")
    def validate_query(self) -> "LyricRetrievalInput":
        if not self.query.strip():
            raise ValueError("query cannot be blank.")
        if self.language is not None and not self.language.strip():
            raise ValueError("language cannot be blank.")
        return self


class LyricCandidateMatch(BaseModel):
    """One lyric-retrieved candidate."""

    track_id: str
    score: float = Field(ge=0, le=1)
    rank: int = Field(ge=1)
    language: str | None = None
    lyric_preview: str | None = Field(
        default=None,
        description="Short lyric evidence preview. Never contains full lyrics.",
    )
    retrieval_mode: Literal["lyric_faiss"] = "lyric_faiss"


class LyricRetrievalOutput(BaseModel):
    """Output from the Lyric RAG tool."""

    candidates: tuple[LyricCandidateMatch, ...]
    input_count: int
    output_count: int
    retrieval_mode: Literal["lyric_faiss"] = "lyric_faiss"
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class CandidateEvidence(BaseModel):
    """Scoring evidence for one candidate from one retrieval tool."""

    model_config = ConfigDict(frozen=True)

    track_id: str = Field(min_length=1)
    score: float = Field(ge=0, description="Tool-specific candidate score.")
    rank: int | None = Field(default=None, ge=1)
    details: dict[str, float | int | str | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_track_id(self) -> "CandidateEvidence":
        if not self.track_id.strip():
            raise ValueError("track_id cannot be blank.")
        return self


class CandidateScoringWeights(BaseModel):
    """Weights for deterministic candidate scoring."""

    model_config = ConfigDict(frozen=True)

    lyric: float = Field(default=0.4, ge=0, le=1)
    metadata: float = Field(default=0.25, ge=0, le=1)
    feature: float = Field(default=0.3, ge=0, le=1)
    popularity: float = Field(default=0.05, ge=0, le=0.2)
    diversity_penalty: float = Field(default=0.05, ge=0, le=0.2)

    @model_validator(mode="after")
    def validate_weights(self) -> "CandidateScoringWeights":
        if self.lyric + self.metadata + self.feature + self.popularity <= 0:
            raise ValueError("At least one scoring weight must be greater than zero.")
        return self


class CandidateScoringInput(BaseModel):
    """Input for the candidate scoring tool."""

    model_config = ConfigDict(frozen=True)

    candidate_track_ids: tuple[str, ...] | None = Field(
        default=None,
        description="Optional explicit candidate pool. If omitted, uses the union of evidence track ids.",
    )
    lyric_evidence: tuple[CandidateEvidence, ...] = Field(default_factory=tuple)
    metadata_evidence: tuple[CandidateEvidence, ...] = Field(default_factory=tuple)
    feature_evidence: tuple[CandidateEvidence, ...] = Field(default_factory=tuple)
    lyric_intent: str | None = Field(
        default=None,
        description="Optional lyric/theme intent used to make lyric evidence survive ranking.",
    )
    weights: CandidateScoringWeights = Field(default_factory=CandidateScoringWeights)
    apply_diversity_penalty: bool = True
    top_k: int = Field(
        default=DEFAULT_FINAL_TOP_K,
        ge=MIN_FINAL_TOP_K,
        le=MAX_FINAL_TOP_K,
    )

    @model_validator(mode="before")
    @classmethod
    def clamp_high_top_k(cls, data: object) -> object:
        return _clamp_high_top_k(data, max_value=MAX_FINAL_TOP_K)

    @model_validator(mode="after")
    def require_candidates_or_evidence(self) -> "CandidateScoringInput":
        if not self.candidate_track_ids and not (
            self.lyric_evidence or self.metadata_evidence or self.feature_evidence
        ):
            raise ValueError("At least one candidate_track_id or evidence item is required.")
        if self.candidate_track_ids is not None:
            blank_ids = [track_id for track_id in self.candidate_track_ids if not track_id.strip()]
            if blank_ids:
                raise ValueError("candidate_track_ids cannot contain blank values.")
        return self


class CandidateScoreMatch(BaseModel):
    """One ranked candidate with deterministic score components."""

    track_id: str
    rank: int = Field(ge=1)
    final_score: float = Field(ge=0, le=1)
    score_components: dict[str, float] = Field(default_factory=dict)
    evidence_sources: tuple[str, ...] = Field(default_factory=tuple)
    popularity: float | None = None
    diversity_penalty: float = Field(default=0.0, ge=0, le=1)
    track_artist: str | None = None
    track_album_name: str | None = None


class CandidateScoringOutput(BaseModel):
    """Output from the candidate scoring tool."""

    candidates: tuple[CandidateScoreMatch, ...]
    input_count: int
    output_count: int
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class ReliabilityCandidate(BaseModel):
    """Final candidate evidence inspected by the reliability tool."""

    model_config = ConfigDict(frozen=True)

    track_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    final_score: float = Field(ge=0, le=1)
    evidence_sources: tuple[str, ...] = Field(default_factory=tuple)
    score_components: dict[str, float] = Field(default_factory=dict)
    verified: bool | None = Field(
        default=None,
        description="Verifier outcome for this candidate when available.",
    )
    verifier_score: float | None = Field(default=None, ge=0, le=1)
    constraint_violations: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    track_artist: str | None = None
    track_album_name: str | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "ReliabilityCandidate":
        if not self.track_id.strip():
            raise ValueError("track_id cannot be blank.")
        blank_sources = [source for source in self.evidence_sources if not source.strip()]
        if blank_sources:
            raise ValueError("evidence_sources cannot contain blank values.")
        return self


class ReliabilityInput(BaseModel):
    """Input for the reliability tool."""

    model_config = ConfigDict(frozen=True)

    final_candidates: tuple[ReliabilityCandidate, ...] = Field(
        default_factory=tuple,
        description="Final ranked candidates after retrieval, verification, revision, and scoring.",
    )
    requested_count: int = Field(
        default=DEFAULT_FINAL_TOP_K,
        ge=MIN_FINAL_TOP_K,
        le=MAX_FINAL_TOP_K,
    )
    retrieval_modes_used: tuple[str, ...] = Field(default_factory=tuple)
    prior_warnings: tuple[str, ...] = Field(default_factory=tuple)
    verifier_warnings: tuple[str, ...] = Field(default_factory=tuple)
    critic_issues: tuple[str, ...] = Field(default_factory=tuple)
    hard_constraint_violations: tuple[str, ...] = Field(default_factory=tuple)
    revision_used: bool = False
    revision_succeeded: bool | None = None
    minimum_acceptable_score: float = Field(default=0.45, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def clamp_high_requested_count(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        requested_count = normalized.get("requested_count")
        if isinstance(requested_count, int) and requested_count > MAX_FINAL_TOP_K:
            normalized["requested_count"] = MAX_FINAL_TOP_K
        return normalized

    @model_validator(mode="after")
    def validate_trace(self) -> "ReliabilityInput":
        blank_modes = [mode for mode in self.retrieval_modes_used if not mode.strip()]
        if blank_modes:
            raise ValueError("retrieval_modes_used cannot contain blank values.")
        return self


class ReliabilitySupportSummary(BaseModel):
    """Measurable support used to compute reliability."""

    candidate_count: int = Field(ge=0)
    requested_count: int = Field(ge=1)
    average_final_score: float = Field(ge=0, le=1)
    top_score: float = Field(ge=0, le=1)
    score_spread: float = Field(ge=0, le=1)
    verified_count: int = Field(ge=0)
    unverified_count: int = Field(ge=0)
    constraint_violation_count: int = Field(ge=0)
    low_score_count: int = Field(ge=0)
    missing_dataset_count: int = Field(ge=0)
    evidence_source_counts: dict[str, int] = Field(default_factory=dict)
    retrieval_modes_used: tuple[str, ...] = Field(default_factory=tuple)
    revision_used: bool = False
    revision_succeeded: bool | None = None
    prior_warning_count: int = Field(ge=0)
    critic_issue_count: int = Field(ge=0)
    repeated_artist_count: int = Field(ge=0)
    repeated_album_count: int = Field(ge=0)


class ReliabilityOutput(BaseModel):
    """Output from the reliability tool."""

    confidence_label: ConfidenceLabel
    confidence_score: float = Field(ge=0, le=1)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    support_summary: ReliabilitySupportSummary


def _clamp_high_top_k(data: object, *, max_value: int) -> object:
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    top_k = normalized.get("top_k")
    if isinstance(top_k, int) and top_k > max_value:
        normalized["top_k"] = max_value
    return normalized
