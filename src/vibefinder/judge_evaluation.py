"""Human and LLM-as-judge evaluation helpers."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vibefinder.llm import LLMClient


JudgeType = Literal["human", "llm"]
Winner = Literal["A", "B", "tie"]
JudgeConfidence = Literal["low", "medium", "high"]
ComponentName = Literal["critic_revision", "lyric_retrieval"]
FailureFlag = Literal[
    "wrong_language",
    "wrong_genre",
    "wrong_artist_or_album",
    "missing_lyric_theme",
    "weak_lyric_evidence",
    "weak_audio_feature_fit",
    "weak_metadata_match",
    "poor_ranking",
    "weak_explanation",
    "overconfident",
    "too_many_warnings",
    "empty_or_too_few_results",
]


COMPONENT_ABLATION_VARIANTS: dict[ComponentName, str] = {
    "critic_revision": "no_critic_revision",
    "lyric_retrieval": "no_lyric_retriever",
}

JUDGE_SCORE_FIELDS: tuple[str, ...] = (
    "query_relevance",
    "constraint_satisfaction",
    "lyric_theme_fit",
    "ranking_quality",
    "explanation_grounding",
    "warning_honesty",
    "overall_usefulness",
)

JUDGE_TASK_SCHEMA = "vibefinder_judge_task_v1"
JUDGE_TASK_KEY_SCHEMA = "vibefinder_judge_task_key_v1"
JUDGEMENT_SCHEMA = "vibefinder_judgement_v1"
JUDGE_REPORT_SCHEMA = "vibefinder_judge_report_v1"


class JudgeRecommendation(BaseModel):
    """One recommendation shown to a human or LLM judge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1)
    track_name: str | None = None
    track_artist: str | None = None
    track_album_name: str | None = None
    playlist_genre: str | None = None
    playlist_subgenre: str | None = None
    language: str | None = None
    audio_features: dict[str, float | int | None] = Field(default_factory=dict)
    lyric_preview: str | None = None
    explanation: str | None = None


class JudgeSystemOutput(BaseModel):
    """One blinded system output in a pairwise judge task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: Literal["A", "B"]
    recommendations: tuple[JudgeRecommendation, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class JudgeTask(BaseModel):
    """Blinded pairwise task for human or LLM judges."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_name: Literal["vibefinder_judge_task_v1"] = Field(default=JUDGE_TASK_SCHEMA, alias="schema")
    task_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    group: str = Field(min_length=1)
    comparison: ComponentName
    rubric: tuple[str, ...]
    system_a: JudgeSystemOutput
    system_b: JudgeSystemOutput

    @model_validator(mode="after")
    def validate_labels(self) -> "JudgeTask":
        if self.system_a.label != "A" or self.system_b.label != "B":
            raise ValueError("JudgeTask system labels must be A and B.")
        return self


class JudgeTaskKey(BaseModel):
    """Private answer key for a blinded judge task."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_name: Literal["vibefinder_judge_task_key_v1"] = Field(default=JUDGE_TASK_KEY_SCHEMA, alias="schema")
    task_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    group: str = Field(min_length=1)
    comparison: ComponentName
    baseline_variant: str = Field(min_length=1)
    system_a_variant: str = Field(min_length=1)
    system_b_variant: str = Field(min_length=1)


class JudgeScores(BaseModel):
    """Seven rubric scores for one blinded system."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_relevance: int = Field(ge=1, le=5)
    constraint_satisfaction: int = Field(ge=1, le=5)
    lyric_theme_fit: int = Field(ge=1, le=5)
    ranking_quality: int = Field(ge=1, le=5)
    explanation_grounding: int = Field(ge=1, le=5)
    warning_honesty: int = Field(ge=1, le=5)
    overall_usefulness: int = Field(ge=1, le=5)

    def average(self) -> float:
        values = [getattr(self, field) for field in JUDGE_SCORE_FIELDS]
        return round(sum(values) / len(values), 4)


class PairwiseScores(BaseModel):
    """Score breakdown for both blinded systems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system_a: JudgeScores
    system_b: JudgeScores


class Judgement(BaseModel):
    """Human or LLM-as-judge label for one pairwise task."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_name: Literal["vibefinder_judgement_v1"] = Field(default=JUDGEMENT_SCHEMA, alias="schema")
    task_id: str = Field(min_length=1)
    judge_id: str = Field(min_length=1)
    judge_type: JudgeType
    winner: Winner
    confidence: JudgeConfidence
    scores: PairwiseScores
    system_a_failure_flags: tuple[FailureFlag, ...] = Field(default_factory=tuple)
    system_b_failure_flags: tuple[FailureFlag, ...] = Field(default_factory=tuple)
    rationale: str = Field(min_length=1)
    raw_response: str | None = None


def build_judge_tasks(
    *,
    evaluation_report: dict[str, Any],
    songs: pd.DataFrame | None = None,
    seed: int = 17,
    top_k: int = 10,
) -> tuple[tuple[JudgeTask, ...], tuple[JudgeTaskKey, ...]]:
    """Build blinded full-vs-ablation judge tasks from an evaluation report."""

    results = evaluation_report.get("results")
    if not isinstance(results, list):
        raise ValueError("Evaluation report must contain a results list.")
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        if isinstance(result, dict):
            by_query[str(result.get("query_id"))][str(result.get("variant"))] = result

    dataset_lookup = _dataset_lookup(songs)
    tasks: list[JudgeTask] = []
    keys: list[JudgeTaskKey] = []
    for query_id in sorted(by_query):
        variants = by_query[query_id]
        full = variants.get("full")
        if full is None:
            continue
        for comparison, baseline_variant in COMPONENT_ABLATION_VARIANTS.items():
            baseline = variants.get(baseline_variant)
            if baseline is None:
                continue
            task_id = f"{query_id}__full_vs_{baseline_variant}"
            full_output = _judge_system_output(
                result=full,
                label="A",
                dataset_lookup=dataset_lookup,
                top_k=top_k,
            )
            baseline_output = _judge_system_output(
                result=baseline,
                label="B",
                dataset_lookup=dataset_lookup,
                top_k=top_k,
            )
            rng = random.Random(f"{seed}:{task_id}")
            full_on_a = rng.choice((True, False))
            system_a = full_output if full_on_a else baseline_output.model_copy(update={"label": "A"})
            system_b = baseline_output if full_on_a else full_output.model_copy(update={"label": "B"})
            tasks.append(
                JudgeTask(
                    task_id=task_id,
                    query_id=str(full.get("query_id")),
                    query=str(full.get("query")),
                    group=str(full.get("group") or "general"),
                    comparison=comparison,
                    rubric=default_judge_rubric(),
                    system_a=system_a,
                    system_b=system_b,
                )
            )
            keys.append(
                JudgeTaskKey(
                    task_id=task_id,
                    query_id=str(full.get("query_id")),
                    group=str(full.get("group") or "general"),
                    comparison=comparison,
                    baseline_variant=baseline_variant,
                    system_a_variant="full" if full_on_a else baseline_variant,
                    system_b_variant=baseline_variant if full_on_a else "full",
                )
            )
    return tuple(tasks), tuple(keys)


def default_judge_rubric() -> tuple[str, ...]:
    """Return concise judge instructions used by both human and LLM judges."""

    return (
        "Judge which system better satisfies the user's music request.",
        "Prioritize explicit constraints such as language, genre, artist, album, and audio-feature words.",
        "For lyric/theme requests, use the short lyric previews and explanations without requiring exact keyword matches.",
        "Reward useful ranking: the best matches should appear near the top.",
        "Reward explanations that are grounded in shown evidence and do not overclaim.",
        "Reward honest warnings when the match is weak or uncertain.",
        "Choose tie only when both systems are similarly useful or similarly weak.",
    )


def judge_task_prompt(task: JudgeTask) -> str:
    """Build the prompt for an LLM judge."""

    public_task = task.model_dump(mode="json", by_alias=True)
    return (
        "You are evaluating two blinded music recommendation outputs for the same user query.\n"
        "Use only the provided task data. Do not use outside music knowledge.\n"
        "Do not assume full lyrics beyond the short previews shown.\n"
        "Return one JSON object that validates against the requested Judgement schema.\n"
        f"Set task_id to \"{task.task_id}\", judge_id to \"llm_judge\", and judge_type to \"llm\".\n"
        "Use a short public rationale, not hidden chain-of-thought.\n\n"
        "Failure flags may include: wrong_language, wrong_genre, wrong_artist_or_album, "
        "missing_lyric_theme, weak_lyric_evidence, weak_audio_feature_fit, weak_metadata_match, "
        "poor_ranking, weak_explanation, overconfident, too_many_warnings, empty_or_too_few_results.\n\n"
        "Task:\n"
        f"{json.dumps(public_task, indent=2, ensure_ascii=True)}"
    )


def run_llm_judge_task(
    *,
    task: JudgeTask,
    llm_client: LLMClient,
    judge_id: str,
) -> Judgement:
    """Run one LLM-as-judge task through the provider-agnostic LLM layer."""

    output = llm_client.complete_json_model(
        prompt=judge_task_prompt(task),
        schema_name="Judgement",
        schema_model=Judgement,
        constraints={
            "judge_task_id": task.task_id,
            "judge_type": "llm",
            "winner_values": ["A", "B", "tie"],
            "score_range": "1 to 5, where 5 is best",
            "blind_evaluation": True,
        },
    )
    output = {**output, "task_id": task.task_id, "judge_id": judge_id, "judge_type": "llm"}
    return Judgement.model_validate(output)


def aggregate_judgements(
    *,
    judgements: tuple[Judgement, ...],
    task_keys: tuple[JudgeTaskKey, ...],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate human or LLM judge labels into component-level reports."""

    keys_by_task = {key.task_id: key for key in task_keys}
    rows = []
    for judgement in judgements:
        key = keys_by_task.get(judgement.task_id)
        if key is None:
            continue
        winner_variant = None
        if judgement.winner == "A":
            winner_variant = key.system_a_variant
        elif judgement.winner == "B":
            winner_variant = key.system_b_variant
        full_scores = judgement.scores.system_a if key.system_a_variant == "full" else judgement.scores.system_b
        baseline_scores = judgement.scores.system_b if key.system_a_variant == "full" else judgement.scores.system_a
        rows.append(
            {
                "task_id": judgement.task_id,
                "query_id": key.query_id,
                "group": key.group,
                "comparison": key.comparison,
                "baseline_variant": key.baseline_variant,
                "winner": judgement.winner,
                "winner_variant": winner_variant,
                "full_won": winner_variant == "full",
                "baseline_won": winner_variant == key.baseline_variant,
                "tied": judgement.winner == "tie",
                "confidence": judgement.confidence,
                "full_overall_usefulness": full_scores.overall_usefulness,
                "baseline_overall_usefulness": baseline_scores.overall_usefulness,
                "overall_usefulness_delta": full_scores.overall_usefulness - baseline_scores.overall_usefulness,
                "full_average_score": full_scores.average(),
                "baseline_average_score": baseline_scores.average(),
                "average_score_delta": round(full_scores.average() - baseline_scores.average(), 4),
                "full_failure_flags": (
                    judgement.system_a_failure_flags
                    if key.system_a_variant == "full"
                    else judgement.system_b_failure_flags
                ),
                "baseline_failure_flags": (
                    judgement.system_b_failure_flags
                    if key.system_a_variant == "full"
                    else judgement.system_a_failure_flags
                ),
            }
        )

    return {
        "schema": JUDGE_REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "component_summary": _aggregate_rows_by(rows, "comparison"),
        "group_summary": _aggregate_rows_by(rows, "group"),
        "judgement_count": len(judgements),
        "matched_judgement_count": len(rows),
        "unmatched_judgement_count": len(judgements) - len(rows),
        "rows": _json_safe(rows),
    }


def judge_report_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report for judged evaluation."""

    metadata = report.get("metadata", {})
    lines = [
        "# VibeFinder Judged Evaluation",
        "",
        f"- Generated at: {report.get('generated_at', 'unknown')}",
        f"- Judge mode: {metadata.get('judge_mode', 'unknown')}",
        f"- Judge id: {metadata.get('judge_id', 'unknown')}",
        f"- Judgements: {report.get('matched_judgement_count', 0)}",
        "",
        "## Component Results",
        "",
        "| Component | Tasks | Full Win | Ablation Win | Tie | Avg Overall Delta | Avg Score Delta | Top Ablation Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for component, summary in sorted(report.get("component_summary", {}).items()):
        lines.append(
            "| {name} | {count} | {full} | {baseline} | {tie} | {overall} | {score} | {flags} |".format(
                name=component,
                count=summary.get("count", 0),
                full=_format_rate(summary.get("full_win_rate")),
                baseline=_format_rate(summary.get("baseline_win_rate")),
                tie=_format_rate(summary.get("tie_rate")),
                overall=_format_number(summary.get("avg_overall_usefulness_delta")),
                score=_format_number(summary.get("avg_average_score_delta")),
                flags=", ".join(summary.get("top_baseline_failure_flags", ())[:3]) or "-",
            )
        )
    lines.extend(["", "## Query Group Results", ""])
    lines.extend(
        [
            "| Group | Tasks | Full Win | Ablation Win | Tie | Avg Overall Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, summary in sorted(report.get("group_summary", {}).items()):
        lines.append(
            "| {name} | {count} | {full} | {baseline} | {tie} | {overall} |".format(
                name=group,
                count=summary.get("count", 0),
                full=_format_rate(summary.get("full_win_rate")),
                baseline=_format_rate(summary.get("baseline_win_rate")),
                tie=_format_rate(summary.get("tie_rate")),
                overall=_format_number(summary.get("avg_overall_usefulness_delta")),
            )
        )
    return "\n".join(lines) + "\n"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a JSONL file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _judge_system_output(
    *,
    result: dict[str, Any],
    label: Literal["A", "B"],
    dataset_lookup: dict[str, dict[str, Any]],
    top_k: int,
) -> JudgeSystemOutput:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    explanation_lookup = {
        str(item.get("track_id")): item
        for item in summary.get("final_explanations", ())
        if isinstance(item, dict) and item.get("track_id")
    }
    recommendations = []
    for candidate in tuple(summary.get("final_candidates", ()))[:top_k]:
        if not isinstance(candidate, dict):
            continue
        track_id = str(candidate.get("track_id") or "")
        row = dataset_lookup.get(track_id, {})
        explanation = explanation_lookup.get(track_id, {}).get("explanation")
        recommendations.append(
            JudgeRecommendation(
                rank=int(candidate.get("rank") or len(recommendations) + 1),
                track_name=_optional_text(candidate.get("track_name") or row.get("track_name")),
                track_artist=_optional_text(candidate.get("track_artist") or row.get("track_artist")),
                track_album_name=_optional_text(
                    candidate.get("track_album_name") or row.get("track_album_name")
                ),
                playlist_genre=_optional_text(candidate.get("playlist_genre") or row.get("playlist_genre")),
                playlist_subgenre=_optional_text(
                    candidate.get("playlist_subgenre") or row.get("playlist_subgenre")
                ),
                language=_optional_text(candidate.get("language") or row.get("language")),
                audio_features=_audio_features(row),
                lyric_preview=_lyric_preview(row.get("lyrics")),
                explanation=_optional_text(explanation),
            )
        )
    warnings = tuple(str(item) for item in summary.get("warnings", ())[:5])
    return JudgeSystemOutput(label=label, recommendations=tuple(recommendations), warnings=warnings)


def _dataset_lookup(songs: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if songs is None or "track_id" not in songs.columns:
        return {}
    return {
        str(row["track_id"]): row
        for row in songs.to_dict(orient="records")
        if row.get("track_id") is not None
    }


def _audio_features(row: dict[str, Any]) -> dict[str, float | int | None]:
    feature_names = (
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    )
    values = {}
    for name in feature_names:
        value = row.get(name)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(value, int):
            values[name] = value
        else:
            try:
                values[name] = round(float(value), 6)
            except (TypeError, ValueError):
                continue
    return values


def _lyric_preview(value: Any, max_chars: int = 240) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _aggregate_rows_by(rows: list[dict[str, Any]], key_name: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key_name])].append(row)
    return {
        key: {
            "count": len(values),
            "full_win_count": sum(1 for row in values if row["full_won"]),
            "baseline_win_count": sum(1 for row in values if row["baseline_won"]),
            "tie_count": sum(1 for row in values if row["tied"]),
            "full_win_rate": _ratio(sum(1 for row in values if row["full_won"]), len(values)),
            "baseline_win_rate": _ratio(sum(1 for row in values if row["baseline_won"]), len(values)),
            "tie_rate": _ratio(sum(1 for row in values if row["tied"]), len(values)),
            "avg_overall_usefulness_delta": _average(
                row["overall_usefulness_delta"] for row in values
            ),
            "avg_average_score_delta": _average(row["average_score_delta"] for row in values),
            "top_baseline_failure_flags": _top_flags(
                flag for row in values for flag in row["baseline_failure_flags"]
            ),
            "top_full_failure_flags": _top_flags(
                flag for row in values for flag in row["full_failure_flags"]
            ),
        }
        for key, values in grouped.items()
    }


def _top_flags(flags: Any) -> list[str]:
    counts: dict[str, int] = {}
    for flag in flags:
        counts[str(flag)] = counts.get(str(flag), 0) + 1
    return [flag for flag, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _average(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 4)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _format_rate(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
