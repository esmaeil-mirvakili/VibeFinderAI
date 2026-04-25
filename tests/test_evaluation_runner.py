from __future__ import annotations

import json

from scripts import run_evaluation as runner
from vibefinder.evaluation import EvaluationRunResult
from vibefinder.llm import LLMSettings


def test_run_result_file_round_trip_and_resume_context(tmp_path):
    result = EvaluationRunResult(
        query_id="sad-pop",
        query="sad pop songs",
        group="mixed",
        variant="full",
        elapsed_seconds=1.2,
        success=True,
        summary={"final_track_ids": ["song-a"]},
        metrics={"automatic_constraint_score": 0.8},
    )
    metadata = runner.run_metadata(
        evaluation_query=_Query(),
        variant_name="full",
        llm_settings=LLMSettings(provider="ollama", model="qwen2.5:14b"),
        include_state=False,
        top_k=10,
        max_revision_count=1,
    )
    path = runner.run_result_path(tmp_path, "sad-pop", "full")

    runner.write_run_result_file(
        path=path,
        result=result,
        status="completed",
        metadata=metadata,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == runner.RUN_RESULT_SCHEMA
    assert payload["status"] == "completed"
    loaded = runner.load_resumable_result(
        path,
        query_id="sad-pop",
        query="sad pop songs",
        variant="full",
        expected_metadata=metadata,
    )
    assert loaded is not None
    assert loaded.query_id == "sad-pop"

    stale_metadata = {**metadata, "top_k": 5}
    assert (
        runner.load_resumable_result(
            path,
            query_id="sad-pop",
            query="sad pop songs",
            variant="full",
            expected_metadata=stale_metadata,
        )
        is None
    )


def test_failed_run_result_is_not_resumed(tmp_path):
    result = EvaluationRunResult(
        query_id="sad-pop",
        query="sad pop songs",
        group="mixed",
        variant="full",
        elapsed_seconds=1.2,
        success=False,
        error={"type": "RuntimeError", "message": "LLM failed"},
    )
    metadata = runner.run_metadata(
        evaluation_query=_Query(),
        variant_name="full",
        llm_settings=LLMSettings(provider="ollama", model="qwen2.5:14b"),
        include_state=False,
        top_k=10,
        max_revision_count=1,
    )
    path = runner.run_result_path(tmp_path, "sad-pop", "full")

    runner.write_run_result_file(path=path, result=result, status="failed", metadata=metadata)

    assert (
        runner.load_resumable_result(
            path,
            query_id="sad-pop",
            query="sad pop songs",
            variant="full",
            expected_metadata=metadata,
        )
        is None
    )


def test_evaluation_status_payload_marks_incomplete_runs():
    results = [
        EvaluationRunResult(
            query_id="ok",
            query="ok query",
            group="metadata",
            variant="full",
            elapsed_seconds=1.0,
            success=True,
        ),
        EvaluationRunResult(
            query_id="bad",
            query="bad query",
            group="metadata",
            variant="full",
            elapsed_seconds=1.0,
            success=False,
            error={"type": "RuntimeError", "message": "LLM failed"},
        ),
    ]

    status = runner.evaluation_status_payload(
        metadata={"result_schema": "test"},
        results=results,
        run_statuses=[],
        expected_total=3,
    )

    assert status["completed_count"] == 1
    assert status["failed_count"] == 1
    assert status["missing_count"] == 1
    assert status["all_complete"] is False
    assert status["failed_runs"][0]["query_id"] == "bad"


class _Query:
    id = "sad-pop"
    query = "sad pop songs"
