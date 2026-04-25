"""Run reproducible benchmark queries across VibeFinder ablation variants."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vibefinder.config import load_retrieval_prompt_config
from vibefinder.data import load_songs_dataset
from vibefinder.embeddings import get_default_embedding_provider
from vibefinder.evaluation import (
    build_evaluation_report,
    evaluation_variant_configs,
    failed_run_result,
    load_evaluation_queries,
    markdown_summary,
    result_from_dict,
    result_to_dict,
    summarize_pipeline_state,
)
from vibefinder.graph import run_recommendation
from vibefinder.graph_runtime import create_graph_runtime_context
from vibefinder.llm import LLMSettings, create_llm_client
from vibefinder.tools import ensure_lyric_index

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "evaluation" / "benchmark_queries.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "evaluation.log"
RUN_RESULT_SCHEMA = "vibefinder_evaluation_run_v1"
STATUS_SCHEMA = "vibefinder_evaluation_status_v1"


def configure_logging(log_file: Path | None) -> None:
    """Configure concise console logs and optional file logs."""

    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message} | {extra}"
    logger.add(
        sys.stderr, level="INFO", format=log_format, diagnose=False, backtrace=False
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="INFO",
            format=log_format,
            rotation="2 MB",
            retention=5,
            diagnose=False,
            backtrace=False,
        )


def load_env_file(path: Path) -> int:
    """Load simple KEY=VALUE lines from .env without overriding exported variables."""

    if not path.exists():
        return 0
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def run_evaluation(args: argparse.Namespace) -> dict:
    """Run every selected query/variant pair and write report files."""

    project_root = Path(args.project_root).expanduser().resolve()
    loaded_env_count = load_env_file(project_root / ".env") if args.load_env else 0
    logger.info("evaluation_env_loaded", loaded_count=loaded_env_count)

    queries_path = _resolve_project_path(args.queries, project_root)
    data_path = _resolve_project_path(args.data_path, project_root)
    config_path = _resolve_project_path(args.config_path, project_root)

    queries = load_evaluation_queries(queries_path)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    variants = evaluation_variant_configs(_split_csv(args.variants))

    logger.info(
        "evaluation_started",
        query_count=len(queries),
        variants=[variant.name for variant in variants],
        data_path=str(data_path),
        config_path=str(config_path),
    )

    songs = load_songs_dataset(path=data_path)
    retrieval_prompt_config = load_retrieval_prompt_config(config_path)
    llm_settings = LLMSettings.from_env()
    llm_client = create_llm_client(llm_settings)

    lyric_index = None
    lyric_embedder = None
    if any(variant.use_lyric_retriever for variant in variants):
        lyric_embedder = get_default_embedding_provider()
        lyric_index = ensure_lyric_index(
            songs=songs,
            root=project_root,
            embedder=lyric_embedder,
        )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_results_dir = output_dir / "runs"
    run_results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    run_statuses: list[dict] = []
    total_runs = len(queries) * len(variants)
    with tqdm(
        total=total_runs, desc="Evaluation runs", unit="run", dynamic_ncols=True
    ) as progress:
        for query_index, evaluation_query in enumerate(queries):
            for variant in variants:
                run_path = run_result_path(
                    run_results_dir, evaluation_query.id, variant.name
                )
                current_run_metadata = run_metadata(
                    evaluation_query=evaluation_query,
                    variant_name=variant.name,
                    llm_settings=llm_settings,
                    include_state=args.include_state,
                    top_k=args.top_k,
                    max_revision_count=args.max_revision_count,
                )
                if args.resume and not args.force:
                    existing_result = load_resumable_result(
                        run_path,
                        query_id=evaluation_query.id,
                        query=evaluation_query.query,
                        variant=variant.name,
                        expected_metadata=current_run_metadata,
                    )
                    if existing_result is not None:
                        results.append(existing_result)
                        run_statuses.append(
                            {
                                "query_id": evaluation_query.id,
                                "variant": variant.name,
                                "status": "skipped_completed",
                                "success": True,
                                "path": str(run_path),
                            }
                        )
                        logger.info(
                            "evaluation_run_skipped_completed",
                            query_id=evaluation_query.id,
                            variant=variant.name,
                            path=str(run_path),
                        )
                        progress.update(1)
                        if args.query_cooldown_seconds > 0 and not (
                            query_index == len(queries) - 1 and variant == variants[-1]
                        ):
                            logger.info(
                                "evaluation_run_cooldown_started",
                                query_id=evaluation_query.id,
                                variant=variant.name,
                                cooldown_seconds=args.query_cooldown_seconds,
                            )
                            time.sleep(args.query_cooldown_seconds)
                        continue

                started = time.perf_counter()
                try:
                    context = create_graph_runtime_context(
                        songs=songs,
                        retrieval_prompt_config=retrieval_prompt_config,
                        llm_client=llm_client,
                        variant_config=variant,
                        lyric_index=lyric_index,
                        lyric_embedder=lyric_embedder,
                        max_revision_count=args.max_revision_count,
                    )
                    state = run_recommendation(evaluation_query.query, context)
                    elapsed = time.perf_counter() - started
                    result = summarize_pipeline_state(
                        state=state,
                        songs=songs,
                        evaluation_query=evaluation_query,
                        variant_name=variant.name,
                        elapsed_seconds=elapsed,
                        top_k=args.top_k,
                        include_state=args.include_state,
                    )
                    logger.info(
                        "evaluation_run_finished",
                        query_id=evaluation_query.id,
                        variant=variant.name,
                        elapsed_seconds=round(elapsed, 4),
                        confidence=result.summary.get("confidence_label"),
                        final_count=len(result.summary.get("final_track_ids", ())),
                    )
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    logger.exception(
                        "evaluation_run_failed",
                        query_id=evaluation_query.id,
                        variant=variant.name,
                    )
                    result = failed_run_result(
                        evaluation_query=evaluation_query,
                        variant_name=variant.name,
                        elapsed_seconds=elapsed,
                        error=exc,
                    )
                    write_run_result_file(
                        path=run_path,
                        result=result,
                        status="failed",
                        metadata=current_run_metadata,
                    )
                    if args.fail_fast:
                        raise
                else:
                    write_run_result_file(
                        path=run_path,
                        result=result,
                        status="completed",
                        metadata=current_run_metadata,
                    )
                results.append(result)
                run_statuses.append(
                    {
                        "query_id": evaluation_query.id,
                        "variant": variant.name,
                        "status": "completed" if result.success else "failed",
                        "success": result.success,
                        "path": str(run_path),
                    }
                )
                progress.update(1)
                if args.query_cooldown_seconds > 0 and not (
                    query_index == len(queries) - 1
                ):
                    logger.info(
                        "evaluation_run_cooldown_started",
                        query_id=evaluation_query.id,
                        variant=variant.name,
                        cooldown_seconds=args.query_cooldown_seconds,
                    )
                    time.sleep(args.query_cooldown_seconds)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_count": len(queries),
        "variants": [variant.name for variant in variants],
        "top_k": args.top_k,
        "max_revision_count": args.max_revision_count,
        "data_path": str(data_path),
        "config_path": str(config_path),
        "llm_provider": llm_settings.provider,
        "llm_model": llm_settings.model,
        "embedding_model": lyric_embedder.name if lyric_embedder is not None else None,
        "result_schema": "vibefinder_evaluation_v1",
        "run_result_schema": RUN_RESULT_SCHEMA,
        "run_results_dir": str(run_results_dir),
        "resume": args.resume,
        "query_cooldown_seconds": args.query_cooldown_seconds,
    }
    stem = datetime.now().strftime("evaluation_%Y%m%d_%H%M%S")
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    latest_json_path = output_dir / "latest.json"
    latest_markdown_path = output_dir / "latest.md"
    status_path = output_dir / "latest_status.json"

    status_payload = evaluation_status_payload(
        metadata=metadata,
        results=results,
        run_statuses=run_statuses,
        expected_total=total_runs,
    )
    write_json_atomic(status_path, status_payload)

    incomplete_count = int(status_payload["failed_count"]) + int(
        status_payload["missing_count"]
    )
    if incomplete_count and not args.allow_partial_report:
        logger.warning(
            "evaluation_incomplete_report_skipped",
            completed_count=status_payload["completed_count"],
            failed_count=status_payload["failed_count"],
            missing_count=status_payload["missing_count"],
            status_path=str(status_path),
        )
        return {
            "report": None,
            "json_path": None,
            "markdown_path": None,
            "latest_json_path": None,
            "latest_markdown_path": None,
            "status_path": str(status_path),
            "results": [result_to_dict(result) for result in results],
        }

    report_metadata = {
        **metadata,
        "partial_report": bool(incomplete_count),
        "status_path": str(status_path),
    }
    report = build_evaluation_report(results=tuple(results), metadata=report_metadata)

    json_text = json.dumps(report, indent=2, sort_keys=True)
    markdown_text = markdown_summary(report)
    write_text_atomic(json_path, json_text + "\n")
    write_text_atomic(markdown_path, markdown_text)
    write_text_atomic(latest_json_path, json_text + "\n")
    write_text_atomic(latest_markdown_path, markdown_text)

    logger.info(
        "evaluation_finished",
        result_count=len(results),
        successful_runs=sum(1 for result in results if result.success),
        failed_runs=sum(1 for result in results if not result.success),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        status_path=str(status_path),
    )
    return {
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "latest_json_path": str(latest_json_path),
        "latest_markdown_path": str(latest_markdown_path),
        "status_path": str(status_path),
        "results": [result_to_dict(result) for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root. Relative paths are resolved against this location.",
    )
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES_PATH),
        help="Benchmark query JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON and Markdown evaluation reports.",
    )
    parser.add_argument(
        "--data-path",
        default=".",
        help="Dataset CSV or directory. Defaults to the project root dataset.",
    )
    parser.add_argument(
        "--config-path",
        default=str(PROJECT_ROOT / "config" / "retrieval_prompt_config.json"),
        help="Retrieval prompt config JSON path.",
    )
    parser.add_argument(
        "--variants",
        default="full,no_critic_revision,no_lyric_retriever",
        help="Comma-separated variants to run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top ranked candidates to score in metrics.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit for quick smoke runs.",
    )
    parser.add_argument(
        "--max-revision-count", type=int, default=1, help="Bounded revision loop count."
    )
    parser.add_argument(
        "--query-cooldown-seconds",
        type=float,
        default=30.0,
        help="Cooldown after each query/variant run slot, including skipped completed runs.",
    )
    parser.add_argument(
        "--include-state",
        action="store_true",
        help="Include compact JSON-safe graph state.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed query/variant run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected query/variant pairs even when completed per-run files exist.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not skip completed per-run result files.",
    )
    parser.add_argument(
        "--allow-partial-report",
        action="store_true",
        help="Write aggregate latest reports even when some per-run results failed.",
    )
    parser.add_argument(
        "--no-load-env", dest="load_env", action="store_false", help="Do not load .env."
    )
    parser.set_defaults(load_env=True, resume=True)
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="Evaluation log file path. Use an empty string to disable file logging.",
    )
    return parser.parse_args()


def _split_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or None


def _resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def run_result_path(run_results_dir: Path, query_id: str, variant_name: str) -> Path:
    """Return the per query/variant result file path."""

    return (
        run_results_dir
        / f"{safe_filename(query_id)}__{safe_filename(variant_name)}.json"
    )


def safe_filename(value: str) -> str:
    """Return a compact filesystem-safe identifier."""

    cleaned = "".join(
        character if character.isalnum() else "_" for character in value.strip().lower()
    )
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or "item"


def load_resumable_result(
    path: Path,
    *,
    query_id: str,
    query: str,
    variant: str,
    expected_metadata: dict,
) -> object | None:
    """Load a completed per-run result when it matches the selected query/variant."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("evaluation_run_result_unreadable", path=str(path))
        return None

    if (
        payload.get("schema") != RUN_RESULT_SCHEMA
        or payload.get("status") != "completed"
    ):
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    if (
        result.get("query_id") != query_id
        or result.get("variant") != variant
        or result.get("query") != query
    ):
        logger.info(
            "evaluation_run_result_stale",
            path=str(path),
            query_id=query_id,
            variant=variant,
        )
        return None
    if result.get("success") is not True:
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not _metadata_matches_resume_context(
        metadata, expected_metadata
    ):
        logger.info(
            "evaluation_run_result_context_mismatch",
            path=str(path),
            query_id=query_id,
            variant=variant,
        )
        return None
    return result_from_dict(result)


def write_run_result_file(
    *,
    path: Path,
    result: object,
    status: str,
    metadata: dict,
) -> None:
    """Persist one query/variant result immediately."""

    payload = {
        "schema": RUN_RESULT_SCHEMA,
        "status": status,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "result": result_to_dict(result),
    }
    write_json_atomic(path, payload)


def run_metadata(
    *,
    evaluation_query: object,
    variant_name: str,
    llm_settings: LLMSettings,
    include_state: bool,
    top_k: int,
    max_revision_count: int,
) -> dict:
    """Build metadata stored with each per-run result file."""

    return {
        "query_id": evaluation_query.id,
        "query": evaluation_query.query,
        "variant": variant_name,
        "llm_provider": llm_settings.provider,
        "llm_model": llm_settings.model,
        "include_state": include_state,
        "top_k": top_k,
        "max_revision_count": max_revision_count,
    }


def _metadata_matches_resume_context(current: dict, expected: dict) -> bool:
    """Return whether an existing result was produced by the current run context."""

    keys = (
        "query_id",
        "query",
        "variant",
        "llm_provider",
        "llm_model",
        "include_state",
        "top_k",
        "max_revision_count",
    )
    return all(current.get(key) == expected.get(key) for key in keys)


def evaluation_status_payload(
    *,
    metadata: dict,
    results: list,
    run_statuses: list[dict],
    expected_total: int,
) -> dict:
    """Build the resumable evaluation status file."""

    completed_count = sum(1 for result in results if result.success)
    failed_count = sum(1 for result in results if not result.success)
    missing_count = max(0, expected_total - len(results))
    return {
        "schema": STATUS_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "expected_total": expected_total,
        "recorded_total": len(results),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "missing_count": missing_count,
        "all_complete": completed_count == expected_total
        and failed_count == 0
        and missing_count == 0,
        "run_statuses": run_statuses,
        "failed_runs": [
            {
                "query_id": result.query_id,
                "variant": result.variant,
                "error": result.error,
            }
            for result in results
            if not result.success
        ],
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON with a replace step so resume never sees partial files."""

    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    """Write text with a replace step so resume never sees partial files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file).expanduser() if args.log_file else None
    configure_logging(log_file)
    run_evaluation(args)


if __name__ == "__main__":
    main()
