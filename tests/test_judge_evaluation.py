from __future__ import annotations

import pandas as pd

from vibefinder.judge_evaluation import (
    JudgeTask,
    Judgement,
    aggregate_judgements,
    build_judge_tasks,
    judge_report_markdown,
    run_llm_judge_task,
)


def test_build_judge_tasks_blinds_full_vs_ablations():
    tasks, keys = build_judge_tasks(
        evaluation_report=_report(),
        songs=_songs(),
        seed=7,
        top_k=1,
    )

    assert len(tasks) == 2
    assert len(keys) == 2
    assert {task.comparison for task in tasks} == {"critic_revision", "lyric_retrieval"}
    assert all(key.system_a_variant in {"full", key.baseline_variant} for key in keys)
    assert all(key.system_b_variant in {"full", key.baseline_variant} for key in keys)
    assert tasks[0].system_a.label == "A"
    assert tasks[0].system_b.label == "B"
    assert tasks[0].system_a.recommendations or tasks[0].system_b.recommendations
    recommendations = [
        item
        for task in tasks
        for system in (task.system_a, task.system_b)
        for item in system.recommendations
    ]
    assert any(item.lyric_preview == "short lyric evidence" for item in recommendations)
    assert any(item.audio_features.get("energy") == 0.9 for item in recommendations)


def test_aggregate_judgements_counts_full_wins_against_private_keys():
    tasks, keys = build_judge_tasks(evaluation_report=_report(), songs=_songs(), seed=7, top_k=1)
    labels = []
    for task, key in zip(tasks, keys):
        winner = "A" if key.system_a_variant == "full" else "B"
        labels.append(_judgement(task.task_id, winner=winner))

    report = aggregate_judgements(judgements=tuple(labels), task_keys=keys, metadata={"judge_mode": "human"})
    rendered = judge_report_markdown(report)

    assert report["matched_judgement_count"] == 2
    assert report["component_summary"]["lyric_retrieval"]["full_win_rate"] == 1.0
    assert "Lyric retrieval" not in rendered
    assert "lyric_retrieval" in rendered


def test_run_llm_judge_task_uses_generic_llm_schema():
    task = build_judge_tasks(evaluation_report=_report(), songs=_songs(), seed=7, top_k=1)[0][0]
    client = _FakeJudgeClient()

    judgement = run_llm_judge_task(task=task, llm_client=client, judge_id="fake")

    assert judgement.task_id == task.task_id
    assert judgement.judge_id == "fake"
    assert judgement.judge_type == "llm"
    assert client.schema_name == "Judgement"


def _judgement(task_id: str, winner: str = "A") -> Judgement:
    return Judgement.model_validate(
        {
            "schema": "vibefinder_judgement_v1",
            "task_id": task_id,
            "judge_id": "judge-1",
            "judge_type": "human",
            "winner": winner,
            "confidence": "high",
            "scores": {
                "system_a": {
                    "query_relevance": 5,
                    "constraint_satisfaction": 5,
                    "lyric_theme_fit": 4,
                    "ranking_quality": 5,
                    "explanation_grounding": 4,
                    "warning_honesty": 4,
                    "overall_usefulness": 5,
                },
                "system_b": {
                    "query_relevance": 3,
                    "constraint_satisfaction": 3,
                    "lyric_theme_fit": 2,
                    "ranking_quality": 3,
                    "explanation_grounding": 2,
                    "warning_honesty": 3,
                    "overall_usefulness": 3,
                },
            },
            "system_a_failure_flags": [],
            "system_b_failure_flags": ["missing_lyric_theme"],
            "rationale": "System A is more useful for the request.",
        }
    )


class _FakeJudgeClient:
    schema_name: str | None = None

    def complete_json(self, prompt: str, schema_name: str) -> dict:
        raise NotImplementedError

    def complete_json_model(self, prompt: str, schema_name: str, schema_model: type, constraints=None) -> dict:
        self.schema_name = schema_name
        return _judgement(task_id="placeholder", winner="tie").model_dump(mode="json", by_alias=True)


def _report() -> dict:
    return {
        "results": [
            _result("full", "Song A", "Artist A", "song-a"),
            _result("no_critic_revision", "Song B", "Artist B", "song-b"),
            _result("no_lyric_retriever", "Song B", "Artist B", "song-b"),
        ]
    }


def _result(variant: str, track_name: str, artist: str, track_id: str) -> dict:
    return {
        "query_id": "sad-song",
        "query": "sad English songs about betrayal",
        "group": "lyric_theme",
        "variant": variant,
        "success": True,
        "summary": {
            "warnings": ["one warning"],
            "final_candidates": [
                {
                    "rank": 1,
                    "track_id": track_id,
                    "track_name": track_name,
                    "track_artist": artist,
                    "track_album_name": "Album",
                    "playlist_genre": "pop",
                    "playlist_subgenre": "dance pop",
                    "language": "en",
                }
            ],
            "final_explanations": [
                {
                    "rank": 1,
                    "track_id": track_id,
                    "explanation": "Grounded explanation.",
                    "warnings": [],
                }
            ],
        },
    }


def _songs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "song-a",
                "lyrics": "short lyric evidence",
                "energy": 0.9,
                "danceability": 0.7,
            },
            {
                "track_id": "song-b",
                "lyrics": "other lyric evidence",
                "energy": 0.2,
                "danceability": 0.4,
            },
        ]
    )
