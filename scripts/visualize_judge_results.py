"""Create seaborn visualizations for judged ablation results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vibefinder.judge_visualization import create_judge_visualizations


DEFAULT_REPORT_PATH = PROJECT_ROOT / "evaluation" / "judgements" / "reports" / "latest_judge_report.json"
DEFAULT_TASKS_PATH = PROJECT_ROOT / "evaluation" / "judgements" / "tasks.jsonl"
DEFAULT_KEYS_PATH = PROJECT_ROOT / "evaluation" / "judgements" / "task_keys.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Aggregated judge report JSON path.")
    parser.add_argument("--output-dir", default=None, help="Visualization output directory.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS_PATH), help="Public judge tasks JSONL path.")
    parser.add_argument("--labels", default=None, help="Optional judge labels JSONL path for examples.")
    parser.add_argument("--keys", default=str(DEFAULT_KEYS_PATH), help="Private task keys JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = create_judge_visualizations(
        report_path=_project_path(args.report),
        output_dir=_project_path(args.output_dir) if args.output_dir else None,
        tasks_path=_project_path(args.tasks) if args.tasks else None,
        labels_path=_project_path(args.labels) if args.labels else None,
        keys_path=_project_path(args.keys) if args.keys else None,
    )
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
