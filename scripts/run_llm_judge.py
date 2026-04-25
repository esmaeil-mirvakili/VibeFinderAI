"""Run LLM-as-judge over blinded judge tasks."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from loguru import logger
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vibefinder.judge_evaluation import JudgeTask, read_jsonl, run_llm_judge_task, write_jsonl
from vibefinder.llm import LLMSettings, create_llm_client


DEFAULT_TASKS_PATH = PROJECT_ROOT / "evaluation" / "judgements" / "tasks.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "judgements" / "llm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS_PATH), help="Blinded judge tasks JSONL path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for labels.")
    parser.add_argument("--judge-id", default=None, help="Judge id. Defaults to provider/model.")
    parser.add_argument("--force", action="store_true", help="Rerun tasks with existing labels.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional task limit for smoke runs.")
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=30.0,
        help="Cooldown between each LLM judge call, including skipped completed tasks.",
    )
    parser.add_argument("--load-env", action="store_true", default=True, help="Load .env before creating the LLM.")
    parser.add_argument("--no-load-env", dest="load_env", action="store_false", help="Do not load .env.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.load_env:
        _load_env_file(PROJECT_ROOT / ".env")
    settings = LLMSettings.from_env()
    judge_id = args.judge_id or f"{settings.provider}:{settings.model}"
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir / _safe_filename(judge_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.jsonl"
    failures_path = output_dir / "failures.jsonl"

    tasks_path = Path(args.tasks).expanduser()
    if not tasks_path.is_absolute():
        tasks_path = PROJECT_ROOT / tasks_path
    tasks = [JudgeTask.model_validate(row) for row in read_jsonl(tasks_path)]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    existing_labels = [] if args.force or not labels_path.exists() else read_jsonl(labels_path)
    completed = {str(row.get("task_id")) for row in existing_labels}
    labels = list(existing_labels)
    failures = [] if args.force or not failures_path.exists() else read_jsonl(failures_path)
    llm_client = create_llm_client(settings)

    for index, task in enumerate(tqdm(tasks, desc="LLM judge tasks", unit="task", dynamic_ncols=True)):
        if task.task_id in completed:
            if args.cooldown_seconds > 0 and index < len(tasks) - 1:
                logger.info(
                    "llm_judge_cooldown_started",
                    task_id=task.task_id,
                    cooldown_seconds=args.cooldown_seconds,
                )
                time.sleep(args.cooldown_seconds)
            continue
        try:
            judgement = run_llm_judge_task(task=task, llm_client=llm_client, judge_id=judge_id)
        except Exception as exc:
            logger.exception("llm_judge_task_failed", task_id=task.task_id)
            failures.append(
                {
                    "task_id": task.task_id,
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                }
            )
            write_jsonl(failures_path, failures)
            if args.cooldown_seconds > 0 and index < len(tasks) - 1:
                logger.info(
                    "llm_judge_cooldown_started",
                    task_id=task.task_id,
                    cooldown_seconds=args.cooldown_seconds,
                )
                time.sleep(args.cooldown_seconds)
            continue
        labels.append(judgement.model_dump(mode="json", by_alias=True))
        completed.add(task.task_id)
        write_jsonl(labels_path, labels)
        if args.cooldown_seconds > 0 and index < len(tasks) - 1:
            logger.info(
                "llm_judge_cooldown_started",
                task_id=task.task_id,
                cooldown_seconds=args.cooldown_seconds,
            )
            time.sleep(args.cooldown_seconds)

    write_jsonl(labels_path, labels)
    write_jsonl(failures_path, failures)
    print(f"Wrote {len(labels)} labels to {labels_path}")
    print(f"Wrote {len(failures)} failures to {failures_path}")


def _load_env_file(path: Path) -> int:
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


def _safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in value.strip().lower())
    return "_".join(part for part in cleaned.split("_") if part) or "judge"


if __name__ == "__main__":
    main()
