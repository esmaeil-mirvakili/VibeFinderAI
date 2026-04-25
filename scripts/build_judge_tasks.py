"""Build blinded human/LLM judge tasks from saved evaluation results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vibefinder.data import load_songs_dataset
from vibefinder.judge_evaluation import build_judge_tasks, write_jsonl


DEFAULT_REPORT_PATH = PROJECT_ROOT / "evaluation" / "results" / "latest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "judgements"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Evaluation report JSON path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--data-path", default=".", help="Dataset CSV or directory for audio fields and lyric previews.")
    parser.add_argument("--seed", type=int, default=17, help="Deterministic A/B randomization seed.")
    parser.add_argument("--top-k", type=int, default=10, help="Recommendations per system shown to judges.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report).expanduser()
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    data_path = Path(args.data_path).expanduser()
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    report = json.loads(report_path.read_text(encoding="utf-8"))
    songs = load_songs_dataset(path=data_path)
    tasks, keys = build_judge_tasks(
        evaluation_report=report,
        songs=songs,
        seed=args.seed,
        top_k=args.top_k,
    )
    write_jsonl(output_dir / "tasks.jsonl", [task.model_dump(mode="json", by_alias=True) for task in tasks])
    write_jsonl(output_dir / "task_keys.jsonl", [key.model_dump(mode="json", by_alias=True) for key in keys])
    print(f"Wrote {len(tasks)} judge tasks to {output_dir / 'tasks.jsonl'}")
    print(f"Wrote {len(keys)} private task keys to {output_dir / 'task_keys.jsonl'}")


if __name__ == "__main__":
    main()
