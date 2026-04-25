"""Pydantic schemas for LLM agent structured outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vibefinder.data import REQUIRED_COLUMNS
from vibefinder.tools.schemas import (
    CATEGORICAL_METADATA_COLUMNS,
    DEFAULT_FINAL_TOP_K,
    FEATURE_COLUMNS,
    FULL_TEXT_METADATA_COLUMNS,
    MAX_FINAL_TOP_K,
    MIN_FINAL_TOP_K,
    CandidateScoringWeights,
    FeatureFilterInput,
    FeatureRangeFilter,
    FeatureTarget,
    LyricRetrievalInput,
    MetadataCategoricalFilter,
    MetadataRetrievalInput,
    MetadataTextQuery,
)


RetrievalModeName = Literal["lyric_retrieval", "metadata_retrieval", "feature_filter"]
AgentConfidence = Literal["low", "medium", "high"]
AgentOutputSchemaName = Literal[
    "PreferenceExtractionOutput",
    "RetrievalStrategyOutput",
    "VerificationOutput",
    "CritiqueOutput",
    "RevisionOutput",
    "ExplanationOutput",
]

TOOL_OUTPUT_NAMES: tuple[str, ...] = (
    "lyric_retrieval",
    "metadata_retrieval",
    "feature_filter",
    "candidate_scoring",
    "reliability",
)
TOOL_OUTPUT_NAME_ALIASES: dict[str, str] = {
    "lyric": "lyric_retrieval",
    "lyrics": "lyric_retrieval",
    "lyric_evidence": "lyric_retrieval",
    "lyric_search": "lyric_retrieval",
    "lyrics_search": "lyric_retrieval",
    "metadata": "metadata_retrieval",
    "metadata_evidence": "metadata_retrieval",
    "metadata_filter": "metadata_retrieval",
    "metadata_search": "metadata_retrieval",
    "feature": "feature_filter",
    "features": "feature_filter",
    "feature_evidence": "feature_filter",
    "feature_search": "feature_filter",
    "audio_feature_filter": "feature_filter",
    "scoring": "candidate_scoring",
    "ranking": "candidate_scoring",
    "reliability_report": "reliability",
}
RETRIEVAL_MODE_ALIASES: dict[str, str] = {
    "lyric": "lyric_retrieval",
    "lyrics": "lyric_retrieval",
    "lyric_search": "lyric_retrieval",
    "lyrics_search": "lyric_retrieval",
    "lyric_retriever": "lyric_retrieval",
    "metadata": "metadata_retrieval",
    "metadata_search": "metadata_retrieval",
    "metadata_retriever": "metadata_retrieval",
    "text": "metadata_retrieval",
    "full_text": "metadata_retrieval",
    "feature": "feature_filter",
    "features": "feature_filter",
    "feature_search": "feature_filter",
    "feature_retrieval": "feature_filter",
    "audio_features": "feature_filter",
}


class PreferenceExtractionOutput(BaseModel):
    """Structured preferences extracted from a user's natural-language query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_query: str = Field(min_length=1, description="Original user query.")
    lyric_intent: str | None = Field(
        default=None,
        description="Free-form lyric/theme/story intent to pass to the Lyric RAG tool.",
    )
    categorical_filters: tuple[MetadataCategoricalFilter, ...] = Field(
        default_factory=tuple,
        description="Exact categorical metadata constraints using approved categorical fields only.",
    )
    text_queries: tuple[MetadataTextQuery, ...] = Field(
        default_factory=tuple,
        description="Free-form metadata search terms using approved full-text fields only.",
    )
    feature_range_filters: tuple[FeatureRangeFilter, ...] = Field(default_factory=tuple)
    feature_targets: tuple[FeatureTarget, ...] = Field(default_factory=tuple)
    exclusions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="User-stated exclusions in free text; not dataset columns.",
    )
    hard_constraints: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Human-readable hard constraints that later agents must preserve.",
    )
    ambiguity_notes: tuple[str, ...] = Field(default_factory=tuple)
    confidence: AgentConfidence = "medium"
    rationale: str = Field(
        min_length=1,
        description="Concise public summary of why these preferences were extracted.",
    )
    issues: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_artifacts(cls, data: object) -> object:
        """Drop common local-LLM artifacts that are not meaningful preferences."""

        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        warnings = list(normalized.get("warnings") or [])

        lyric_intent = normalized.get("lyric_intent")
        if isinstance(lyric_intent, str) and _looks_like_prompt_artifact(lyric_intent):
            normalized["lyric_intent"] = None
            warnings.append("Dropped lyric_intent because it appeared to copy prompt/config instructions.")

        range_filters = data.get("feature_range_filters")
        if not isinstance(range_filters, list):
            if warnings:
                normalized["warnings"] = warnings
                return normalized
            return data

        kept_filters = []
        dropped_count = 0
        for item in range_filters:
            if (
                isinstance(item, dict)
                and item.get("min_value") is None
                and item.get("max_value") is None
            ):
                dropped_count += 1
                continue
            kept_filters.append(item)

        if not dropped_count:
            if warnings:
                normalized["warnings"] = warnings
                return normalized
            return data

        normalized["feature_range_filters"] = kept_filters
        warnings.append(
            f"Dropped {dropped_count} feature range filters because they had no min_value or max_value."
        )
        normalized["warnings"] = warnings
        return normalized

    @model_validator(mode="after")
    def validate_preferences(self) -> "PreferenceExtractionOutput":
        _reject_blank_string("raw_query", self.raw_query)
        _reject_optional_blank_string("lyric_intent", self.lyric_intent)
        _reject_blank_sequence("exclusions", self.exclusions)
        _reject_blank_sequence("hard_constraints", self.hard_constraints)
        _reject_blank_sequence("ambiguity_notes", self.ambiguity_notes)
        _reject_blank_string("rationale", self.rationale)
        _reject_blank_sequence("issues", self.issues)
        _reject_blank_sequence("warnings", self.warnings)
        return self


class RetrievalStrategyOutput(BaseModel):
    """Tool-ready retrieval strategy selected by the retrieval strategy agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary_mode: RetrievalModeName
    modes: tuple[RetrievalModeName, ...] = Field(min_length=1)
    lyric_request: LyricRetrievalInput | None = None
    metadata_request: MetadataRetrievalInput | None = None
    feature_request: FeatureFilterInput | None = None
    scoring_weights: CandidateScoringWeights = Field(default_factory=CandidateScoringWeights)
    broad_search: bool = False
    top_k_final: int = Field(default=DEFAULT_FINAL_TOP_K, ge=MIN_FINAL_TOP_K, le=MAX_FINAL_TOP_K)
    rationale: str = Field(
        min_length=1,
        description="Concise public summary of why this retrieval strategy was selected.",
    )
    issues: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def normalize_retrieval_modes(cls, data: object) -> object:
        return _normalize_retrieval_strategy_payload(data)

    @model_validator(mode="after")
    def validate_strategy(self) -> "RetrievalStrategyOutput":
        if self.primary_mode not in self.modes:
            raise ValueError("primary_mode must be included in modes.")
        if self.lyric_request is not None and "lyric_retrieval" not in self.modes:
            raise ValueError("lyric_request cannot be supplied unless lyric_retrieval is selected.")
        if self.metadata_request is not None and "metadata_retrieval" not in self.modes:
            raise ValueError("metadata_request cannot be supplied unless metadata_retrieval is selected.")
        if self.feature_request is not None and "feature_filter" not in self.modes:
            raise ValueError("feature_request cannot be supplied unless feature_filter is selected.")
        _reject_blank_string("rationale", self.rationale)
        _reject_blank_sequence("issues", self.issues)
        _reject_blank_sequence("warnings", self.warnings)
        return self


class VerificationCandidateResult(BaseModel):
    """Verifier result for one candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: str = Field(min_length=1)
    verified: bool
    verifier_score: float = Field(ge=0, le=1)
    matched_constraints: tuple[str, ...] = Field(default_factory=tuple)
    violations: tuple[str, ...] = Field(default_factory=tuple)
    evidence_sources: tuple[str, ...] = Field(default_factory=tuple)
    rationale: str = Field(min_length=1)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def normalize_evidence_sources(cls, data: object) -> object:
        if isinstance(data, dict):
            normalized = dict(data)
            if "evidence_sources" in normalized:
                normalized["evidence_sources"], source_warnings = _normalize_tool_output_names_with_warnings(
                    normalized.get("evidence_sources"),
                    warning_prefix="Moved unsupported evidence source to warnings",
                )
                if source_warnings:
                    normalized["warnings"] = [*list(normalized.get("warnings") or []), *source_warnings]
            return normalized
        return data

    @model_validator(mode="after")
    def validate_candidate(self) -> "VerificationCandidateResult":
        _reject_blank_string("track_id", self.track_id)
        _reject_blank_string("rationale", self.rationale)
        _reject_blank_sequence("matched_constraints", self.matched_constraints)
        _reject_blank_sequence("violations", self.violations)
        _reject_blank_sequence("warnings", self.warnings)
        _validate_tool_output_names(self.evidence_sources, "evidence_sources")
        if self.verified and self.violations:
            raise ValueError("verified candidates cannot include violations.")
        return self


class VerificationOutput(BaseModel):
    """Verifier agent output for a candidate set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[VerificationCandidateResult, ...] = Field(default_factory=tuple)
    summary: str = Field(min_length=1)
    rationale: str = Field(
        min_length=1,
        description="Concise public summary of how verification was judged.",
    )
    issues: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def salvage_candidate_results(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        warnings = list(normalized.get("warnings") or [])
        candidates, salvage_warnings = _salvage_model_items(
            normalized.get("candidates"),
            VerificationCandidateResult,
            item_name="verification candidate",
        )
        if salvage_warnings:
            warnings.extend(salvage_warnings)
            normalized["warnings"] = warnings
        if candidates is not None:
            normalized["candidates"] = candidates
        return normalized

    @model_validator(mode="after")
    def validate_output(self) -> "VerificationOutput":
        _reject_blank_string("summary", self.summary)
        _reject_blank_string("rationale", self.rationale)
        _reject_blank_sequence("issues", self.issues)
        _reject_blank_sequence("warnings", self.warnings)
        _reject_duplicate_track_ids(tuple(candidate.track_id for candidate in self.candidates))
        return self


class CritiqueOutput(BaseModel):
    """Critic agent output for group-level result quality."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issues: tuple[str, ...] = Field(default_factory=tuple)
    should_revise: bool = False
    revision_focus: tuple[str, ...] = Field(default_factory=tuple)
    summary: str = Field(min_length=1)
    rationale: str = Field(
        min_length=1,
        description="Concise public summary of why the critique decision was made.",
    )
    confidence: AgentConfidence = "medium"
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_critique(self) -> "CritiqueOutput":
        _reject_blank_string("summary", self.summary)
        _reject_blank_string("rationale", self.rationale)
        _reject_blank_sequence("issues", self.issues)
        _reject_blank_sequence("revision_focus", self.revision_focus)
        _reject_blank_sequence("warnings", self.warnings)
        if self.should_revise and not (self.issues or self.revision_focus):
            raise ValueError("should_revise requires issues or revision_focus.")
        return self


class RevisionOutput(BaseModel):
    """Revision agent output for the bounded retry loop."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    should_retry: bool
    revised_retrieval_plan: RetrievalStrategyOutput | None = None
    rationale: str = Field(
        min_length=1,
        description="Concise public summary of why retry is or is not needed.",
    )
    issues: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_revision(self) -> "RevisionOutput":
        _reject_blank_string("rationale", self.rationale)
        _reject_blank_sequence("issues", self.issues)
        _reject_blank_sequence("warnings", self.warnings)
        if self.should_retry and self.revised_retrieval_plan is None:
            raise ValueError("revised_retrieval_plan is required when should_retry is true.")
        if not self.should_retry and self.revised_retrieval_plan is not None:
            raise ValueError("revised_retrieval_plan cannot be supplied when should_retry is false.")
        return self


class ExplanationCandidate(BaseModel):
    """Grounded explanation for one final recommendation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    explanation: str = Field(min_length=1)
    supporting_fields: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Dataset columns used as explanation evidence.",
    )
    supporting_tool_outputs: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tool outputs used as explanation evidence.",
    )
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def normalize_supporting_tool_outputs(cls, data: object) -> object:
        if isinstance(data, dict):
            normalized = dict(data)
            if "supporting_tool_outputs" in normalized:
                normalized["supporting_tool_outputs"] = _normalize_tool_output_names(
                    normalized.get("supporting_tool_outputs")
                )
            return normalized
        return data

    @model_validator(mode="after")
    def validate_explanation(self) -> "ExplanationCandidate":
        _reject_blank_string("track_id", self.track_id)
        _reject_blank_string("explanation", self.explanation)
        _validate_dataset_fields(self.supporting_fields)
        _validate_tool_output_names(self.supporting_tool_outputs, "supporting_tool_outputs")
        _reject_blank_sequence("warnings", self.warnings)
        return self


class ExplanationOutput(BaseModel):
    """Explanation agent output for final recommendations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recommendations: tuple[ExplanationCandidate, ...] = Field(default_factory=tuple)
    overall_summary: str | None = None
    rationale: str = Field(
        min_length=1,
        description="Concise public summary of why these explanation choices were made.",
    )
    issues: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def salvage_recommendations(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        warnings = list(normalized.get("warnings") or [])
        recommendations, salvage_warnings = _salvage_model_items(
            normalized.get("recommendations"),
            ExplanationCandidate,
            item_name="explanation recommendation",
        )
        if salvage_warnings:
            warnings.extend(salvage_warnings)
            normalized["warnings"] = warnings
        if recommendations is not None:
            normalized["recommendations"] = recommendations
        return normalized

    @model_validator(mode="after")
    def validate_output(self) -> "ExplanationOutput":
        _reject_optional_blank_string("overall_summary", self.overall_summary)
        _reject_blank_string("rationale", self.rationale)
        _reject_blank_sequence("issues", self.issues)
        _reject_blank_sequence("warnings", self.warnings)
        _reject_duplicate_track_ids(tuple(candidate.track_id for candidate in self.recommendations))
        return self


AGENT_OUTPUT_SCHEMAS: dict[AgentOutputSchemaName, type[BaseModel]] = {
    "PreferenceExtractionOutput": PreferenceExtractionOutput,
    "RetrievalStrategyOutput": RetrievalStrategyOutput,
    "VerificationOutput": VerificationOutput,
    "CritiqueOutput": CritiqueOutput,
    "RevisionOutput": RevisionOutput,
    "ExplanationOutput": ExplanationOutput,
}


def agent_output_schema_prompt_specs() -> dict[str, dict[str, object]]:
    """Return JSON schemas for LLM structured-output prompts."""

    return {
        name: {
            "schema_name": name,
            "json_schema": schema.model_json_schema(),
            "constraints": _schema_constraints(name),
        }
        for name, schema in AGENT_OUTPUT_SCHEMAS.items()
    }


def _schema_constraints(name: AgentOutputSchemaName) -> dict[str, object]:
    base_constraints: dict[str, object] = {
        "no_external_data": True,
        "no_invented_dataset_columns": True,
        "required_intermediate_fields": ["rationale", "issues", "warnings"],
        "rationale_policy": (
            "Use concise public rationale summaries. Do not include hidden chain-of-thought "
            "or private scratchpad reasoning."
        ),
    }
    if name == "PreferenceExtractionOutput":
        return {
            **base_constraints,
            "categorical_metadata_fields": list(CATEGORICAL_METADATA_COLUMNS),
            "full_text_metadata_fields": list(FULL_TEXT_METADATA_COLUMNS),
            "numeric_feature_fields": list(FEATURE_COLUMNS),
            "note": (
                "Lyric intent is free-form text extracted from the user query. "
                "Do not invent categorical values outside the retrieval prompt config."
            ),
        }
    if name == "RetrievalStrategyOutput":
        return {
            **base_constraints,
            "retrieval_modes": ["lyric_retrieval", "metadata_retrieval", "feature_filter"],
            "accepted_retrieval_mode_aliases": RETRIEVAL_MODE_ALIASES,
            "requires_tool_ready_requests": True,
        }
    if name == "VerificationOutput":
        return {
            **base_constraints,
            "evidence_sources": list(TOOL_OUTPUT_NAMES),
            "accepted_evidence_source_aliases": TOOL_OUTPUT_NAME_ALIASES,
            "note": "Prefer exact tool output names. Common aliases are normalized before validation.",
        }
    if name == "ExplanationOutput":
        return {
            **base_constraints,
            "supporting_fields_must_be_dataset_columns": True,
            "dataset_columns": list(REQUIRED_COLUMNS),
            "tool_outputs": list(TOOL_OUTPUT_NAMES),
            "accepted_tool_output_aliases": TOOL_OUTPUT_NAME_ALIASES,
        }
    return base_constraints


def _reject_blank_string(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} cannot be blank.")


def _reject_optional_blank_string(field_name: str, value: str | None) -> None:
    if value is not None and not value.strip():
        raise ValueError(f"{field_name} cannot be blank.")


def _looks_like_prompt_artifact(value: str) -> bool:
    normalized = value.casefold()
    artifact_markers = (
        "prompt config",
        "retrieval prompt config",
        "excluded from this prompt",
        "intentionally excluded",
    )
    return any(marker in normalized for marker in artifact_markers)


def _reject_blank_sequence(field_name: str, values: tuple[str, ...]) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} cannot contain blank values.")


def _reject_duplicate_track_ids(track_ids: tuple[str, ...]) -> None:
    cleaned = [track_id.strip() for track_id in track_ids]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("track_id values must be unique.")


def _validate_dataset_fields(fields: tuple[str, ...]) -> None:
    _reject_blank_sequence("supporting_fields", fields)
    unsupported = [field for field in fields if field not in REQUIRED_COLUMNS]
    if unsupported:
        raise ValueError(f"Unsupported dataset evidence fields: {', '.join(unsupported)}")


def _validate_tool_output_names(values: tuple[str, ...], field_name: str) -> None:
    _reject_blank_sequence(field_name, values)
    unsupported = [value for value in values if value not in TOOL_OUTPUT_NAMES]
    if unsupported:
        raise ValueError(f"Unsupported tool output names for {field_name}: {', '.join(unsupported)}")


def _normalize_retrieval_strategy_payload(data: object) -> object:
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    if "primary_mode" in normalized:
        normalized["primary_mode"] = _normalize_retrieval_mode(normalized["primary_mode"])
    if "modes" in normalized and isinstance(normalized["modes"], (list, tuple)):
        normalized["modes"] = [
            _normalize_retrieval_mode(mode)
            for mode in normalized["modes"]
        ]
    top_k_final = normalized.get("top_k_final")
    if isinstance(top_k_final, int) and top_k_final > MAX_FINAL_TOP_K:
        normalized["top_k_final"] = MAX_FINAL_TOP_K
    return normalized


def _normalize_retrieval_mode(value: object) -> object:
    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    alias_key = cleaned.casefold().replace(" ", "_").replace("-", "_")
    return RETRIEVAL_MODE_ALIASES.get(alias_key, cleaned)


def _normalize_tool_output_names(values: object) -> object:
    normalized, _warnings = _normalize_tool_output_names_with_warnings(values)
    return normalized


def _normalize_tool_output_names_with_warnings(
    values: object,
    *,
    warning_prefix: str = "Moved unsupported tool output to warnings",
) -> tuple[object, list[str]]:
    if values is None:
        return values, []
    if not isinstance(values, (list, tuple)):
        return values, []

    normalized: list[object] = []
    warnings: list[str] = []
    for value in values:
        if not isinstance(value, str):
            normalized.append(value)
            continue
        cleaned = value.strip()
        alias_key = cleaned.casefold().replace(" ", "_").replace("-", "_")
        mapped = TOOL_OUTPUT_NAME_ALIASES.get(alias_key, cleaned)
        if mapped not in TOOL_OUTPUT_NAMES and _looks_like_sentence(cleaned):
            warnings.append(f"{warning_prefix}: {cleaned}")
            continue
        normalized.append(mapped)
    return normalized, warnings


def _looks_like_sentence(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return False
    return any(mark in cleaned for mark in ".!?") or len(cleaned.split()) >= 5


def _salvage_model_items(
    values: object,
    model: type[BaseModel],
    *,
    item_name: str,
) -> tuple[list[dict[str, object]] | None, list[str]]:
    if values is None:
        return None, []
    if not isinstance(values, (list, tuple)):
        return values, []  # type: ignore[return-value]

    salvaged: list[dict[str, object]] = []
    warnings: list[str] = []
    seen_track_ids: set[str] = set()
    for index, value in enumerate(values):
        try:
            item = model.model_validate(value)
        except ValidationError as exc:
            warnings.append(f"Dropped invalid {item_name} at index {index}: {_compact_validation_error(exc)}")
            continue

        dumped = item.model_dump(mode="json")
        track_id = str(dumped.get("track_id", "")).strip()
        if track_id and track_id in seen_track_ids:
            warnings.append(f"Dropped duplicate {item_name} for track_id: {track_id}")
            continue
        if track_id:
            seen_track_ids.add(track_id)
        salvaged.append(dumped)
    return salvaged, warnings


def _compact_validation_error(exc: ValidationError) -> str:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first_error.get("loc", ())) or "item"
    message = str(first_error.get("msg", "validation failed"))
    return f"{location}: {message}"
