"""Aggregate human or LLM judge labels into a component report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vibefinder.judge_evaluation import (
    JudgeTaskKey,
    Judgement,
    aggregate_judgements,
    judge_report_markdown,
    read_jsonl,
)
from vibefinder.judge_visualization import create_judge_visualizations


DEFAULT_KEYS_PATH = PROJECT_ROOT / "evaluation" / "judgements" / "task_keys.jsonl"
DEFAULT_TASKS_PATH = PROJECT_ROOT / "evaluation" / "judgements" / "tasks.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "judgements" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="Human or LLM labels JSONL path.")
    parser.add_argument("--keys", default=str(DEFAULT_KEYS_PATH), help="Private task keys JSONL path.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS_PATH), help="Public judge tasks JSONL path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Report output directory.")
    parser.add_argument("--visual-output-dir", default=None, help="Visualization output directory.")
    parser.add_argument("--judge-mode", choices=("human", "llm", "mixed"), default="llm")
    parser.add_argument("--judge-id", default="unknown")
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Write only JSON/Markdown reports and skip chart generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = _project_path(args.labels)
    keys_path = _project_path(args.keys)
    tasks_path = _project_path(args.tasks)
    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    judgements = tuple(Judgement.model_validate(row) for row in read_jsonl(labels_path))
    keys = tuple(JudgeTaskKey.model_validate(row) for row in read_jsonl(keys_path))
    report = aggregate_judgements(
        judgements=judgements,
        task_keys=keys,
        metadata={
            "judge_mode": args.judge_mode,
            "judge_id": args.judge_id,
            "labels_path": str(labels_path),
            "keys_path": str(keys_path),
        },
    )
    json_path = output_dir / "latest_judge_report.json"
    markdown_path = output_dir / "latest_judge_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(judge_report_markdown(report), encoding="utf-8")
    print(f"Wrote judge report JSON to {json_path}")
    print(f"Wrote judge report Markdown to {markdown_path}")
    if args.skip_visualizations:
        return

    visual_output_dir = _project_path(args.visual_output_dir) if args.visual_output_dir else output_dir
    paths = create_judge_visualizations(
        report_path=json_path,
        output_dir=visual_output_dir,
        tasks_path=tasks_path,
        labels_path=labels_path,
        keys_path=keys_path,
    )
    visual_report_path = paths.get("visual_report_md")
    if visual_report_path and visual_report_path.exists():
        combined_markdown = markdown_path.read_text(encoding="utf-8").rstrip() + "\n\n---\n\n"
        combined_markdown += visual_report_path.read_text(encoding="utf-8")
        markdown_path.write_text(combined_markdown, encoding="utf-8")
        visual_report_path.unlink()
        paths.pop("visual_report_md", None)
    print("Wrote judge visualizations:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    main()
