"""Visualization helpers for judged ablation results."""

from __future__ import annotations

import csv
import json
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


COMPONENT_LABELS = {
    "critic_revision": "Critic / Revision",
    "lyric_retrieval": "Lyric RAG",
}

GROUP_LABELS = {
    "audio_feature": "Audio Feature",
    "hard_constraints": "Hard Constraints",
    "lyric_theme": "Lyric Theme",
    "metadata": "Metadata",
    "metadata_text": "Metadata Text",
    "mixed": "Mixed",
}

FAILURE_LABELS = {
    "empty_or_too_few_results": "Empty / Too Few Results",
    "missing_lyric_theme": "Missing Lyric Theme",
    "overconfident": "Overconfident",
    "poor_ranking": "Poor Ranking",
    "too_many_warnings": "Too Many Warnings",
    "weak_audio_feature_fit": "Weak Audio Feature Fit",
    "weak_explanation": "Weak Explanation",
    "weak_lyric_evidence": "Weak Lyric Evidence",
    "weak_metadata_match": "Weak Metadata Match",
    "wrong_artist_or_album": "Wrong Artist / Album",
    "wrong_genre": "Wrong Genre",
    "wrong_language": "Wrong Language",
}


def create_judge_visualizations(
    *,
    report_path: str | Path,
    output_dir: str | Path | None = None,
    tasks_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    keys_path: str | Path | None = None,
) -> dict[str, Path]:
    """Create charts and tables for an aggregated judge report."""

    report_file = Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    out_dir = Path(output_dir) if output_dir else report_file.parent / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    component_df = component_results_frame(report)
    group_df = group_results_frame(report)
    failure_df = failure_reason_frame(report)
    examples_df = representative_examples_frame(
        report=report,
        tasks_path=tasks_path,
        labels_path=labels_path or report.get("metadata", {}).get("labels_path"),
        keys_path=keys_path or report.get("metadata", {}).get("keys_path"),
    )

    paths = {
        "component_results_csv": out_dir / "component_results.csv",
        "query_group_results_csv": out_dir / "query_group_results.csv",
        "failure_reason_counts_csv": out_dir / "failure_reason_counts.csv",
        "representative_examples_csv": out_dir / "representative_examples.csv",
        "component_outcomes_png": out_dir / "component_outcomes.png",
        "component_avg_delta_png": out_dir / "component_avg_delta.png",
        "query_group_heatmap_png": out_dir / "query_group_heatmap.png",
        "failure_reason_counts_png": out_dir / "failure_reason_counts.png",
        "visual_report_md": out_dir / "judge_visual_report.md",
    }

    component_df.to_csv(paths["component_results_csv"], index=False)
    group_df.to_csv(paths["query_group_results_csv"], index=False)
    failure_df.to_csv(paths["failure_reason_counts_csv"], index=False)
    examples_df.to_csv(paths["representative_examples_csv"], index=False, quoting=csv.QUOTE_MINIMAL)

    _plot_component_outcomes(component_df, paths["component_outcomes_png"])
    _plot_component_delta(component_df, paths["component_avg_delta_png"])
    _plot_group_heatmap(group_df, paths["query_group_heatmap_png"])
    _plot_failure_reasons(failure_df, paths["failure_reason_counts_png"])
    paths["visual_report_md"].write_text(
        _visual_report_markdown(report, paths, component_df, group_df, failure_df, examples_df),
        encoding="utf-8",
    )
    return paths


def component_results_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Return display-ready component summary rows."""

    rows = []
    for component, summary in report.get("component_summary", {}).items():
        count = int(summary.get("count") or 0)
        rows.append(
            {
                "component": component,
                "Component": _component_label(component),
                "Tasks": count,
                "Full System Win %": _pct(summary.get("full_win_rate")),
                "Full System Win Count": int(summary.get("full_win_count") or 0),
                "Ablated System Win %": _pct(summary.get("baseline_win_rate")),
                "Ablated System Win Count": int(summary.get("baseline_win_count") or 0),
                "Tie %": _pct(summary.get("tie_rate")),
                "Tie Count": int(summary.get("tie_count") or 0),
                "Mean Judge Preference Delta": _num(summary.get("avg_overall_usefulness_delta")),
                "Mean Score Margin": _num(summary.get("avg_average_score_delta")),
                "Most Common Failure Modes": ", ".join(
                    _failure_label(flag) for flag in summary.get("top_baseline_failure_flags", ())[:5]
                ),
                "Full System Win": _rate_with_count(summary.get("full_win_rate"), summary.get("full_win_count"), count),
                "Ablated System Win": _rate_with_count(
                    summary.get("baseline_win_rate"), summary.get("baseline_win_count"), count
                ),
                "Tie": _rate_with_count(summary.get("tie_rate"), summary.get("tie_count"), count),
            }
        )
    return pd.DataFrame(rows).sort_values("Mean Judge Preference Delta", ascending=False)


def group_results_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Return display-ready query group summary rows."""

    rows = []
    for group, summary in report.get("group_summary", {}).items():
        count = int(summary.get("count") or 0)
        rows.append(
            {
                "group": group,
                "Query Group": GROUP_LABELS.get(group, group.replace("_", " ").title()),
                "Tasks": count,
                "Full System Win %": _pct(summary.get("full_win_rate")),
                "Full System Win Count": int(summary.get("full_win_count") or 0),
                "Ablated System Win %": _pct(summary.get("baseline_win_rate")),
                "Ablated System Win Count": int(summary.get("baseline_win_count") or 0),
                "Tie %": _pct(summary.get("tie_rate")),
                "Tie Count": int(summary.get("tie_count") or 0),
                "Mean Judge Preference Delta": _num(summary.get("avg_overall_usefulness_delta")),
                "Mean Score Margin": _num(summary.get("avg_average_score_delta")),
                "Full System Win": _rate_with_count(summary.get("full_win_rate"), summary.get("full_win_count"), count),
                "Ablated System Win": _rate_with_count(
                    summary.get("baseline_win_rate"), summary.get("baseline_win_count"), count
                ),
                "Tie": _rate_with_count(summary.get("tie_rate"), summary.get("tie_count"), count),
            }
        )
    return pd.DataFrame(rows).sort_values("Mean Judge Preference Delta", ascending=False)


def failure_reason_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Count ablated-system failure flags, focused on rows where full won."""

    counts: Counter[tuple[str, str]] = Counter()
    global_counts: Counter[str] = Counter()
    for row in report.get("rows", ()):
        if not row.get("full_won"):
            continue
        component = str(row.get("comparison") or "unknown")
        for flag in row.get("baseline_failure_flags", ()):
            flag = str(flag)
            counts[(component, flag)] += 1
            global_counts[flag] += 1

    rows = [
        {
            "Component": _component_label(component),
            "component": component,
            "Failure Reason": _failure_label(flag),
            "failure_reason": flag,
            "Count": count,
            "Global Count": global_counts[flag],
        }
        for (component, flag), count in counts.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["Component", "component", "Failure Reason", "failure_reason", "Count", "Global Count"])
    return pd.DataFrame(rows).sort_values(["Global Count", "Count", "Failure Reason"], ascending=[False, False, True])


def representative_examples_frame(
    *,
    report: dict[str, Any],
    tasks_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    keys_path: str | Path | None = None,
    examples_per_component: int = 3,
) -> pd.DataFrame:
    """Pick useful full-system-win examples with judge rationale."""

    task_lookup = _read_jsonl_lookup(tasks_path, "task_id") if tasks_path else {}
    label_lookup = _read_jsonl_lookup(labels_path, "task_id") if labels_path else {}
    key_lookup = _read_jsonl_lookup(keys_path, "task_id") if keys_path else {}
    selected: list[dict[str, Any]] = []
    seen_by_component: defaultdict[str, int] = defaultdict(int)
    sorted_rows = sorted(
        report.get("rows", ()),
        key=lambda row: (
            -float(row.get("overall_usefulness_delta") or 0),
            -float(row.get("average_score_delta") or 0),
            str(row.get("task_id") or ""),
        ),
    )
    for row in sorted_rows:
        if not row.get("full_won"):
            continue
        component = str(row.get("comparison") or "")
        if seen_by_component[component] >= examples_per_component:
            continue
        task_id = str(row.get("task_id") or "")
        label = label_lookup.get(task_id, {})
        rationale = str(label.get("rationale") or "").strip()
        if not rationale:
            continue
        rationale = _replace_blinded_system_labels(rationale, key_lookup.get(task_id, {}))
        task = task_lookup.get(task_id, {})
        selected.append(
            {
                "Component": _component_label(component),
                "Prompt": str(task.get("query") or row.get("query_id") or ""),
                "Winner": "Full System",
                "Judge Evidence": rationale,
                "Mean Judge Preference Delta": _num(row.get("overall_usefulness_delta")),
                "Mean Score Margin": _num(row.get("average_score_delta")),
                "Task ID": task_id,
            }
        )
        seen_by_component[component] += 1
    return pd.DataFrame(
        selected,
        columns=[
            "Component",
            "Prompt",
            "Winner",
            "Judge Evidence",
            "Mean Judge Preference Delta",
            "Mean Score Margin",
            "Task ID",
        ],
    )


def _plot_component_outcomes(df: pd.DataFrame, output_path: Path) -> None:
    _configure_matplotlib_cache(output_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plot_df = df.sort_values("Mean Judge Preference Delta", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    left = [0.0] * len(plot_df)
    segments = [
        ("Full System Win %", "#2a9d8f", "Full System Win"),
        ("Ablated System Win %", "#e76f51", "Ablated System Win"),
        ("Tie %", "#9aa0a6", "Tie"),
    ]
    y = list(plot_df["Component"])
    for column, color, label in segments:
        values = list(plot_df[column])
        ax.barh(y, values, left=left, color=color, label=label)
        for i, value in enumerate(values):
            if value >= 9:
                ax.text(left[i] + value / 2, i, f"{value:.1f}%", va="center", ha="center", color="white", fontsize=8)
        left = [base + value for base, value in zip(left, values)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("Judge outcome share")
    ax.set_ylabel("Component")
    ax.set_title("Full System vs Ablated System Outcomes")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.text(
        0.01,
        0.01,
        "Win means the blinded judge preferred that system for the same query and ablation comparison.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_component_delta(df: pd.DataFrame, output_path: Path) -> None:
    _configure_matplotlib_cache(output_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plot_df = df.sort_values("Mean Judge Preference Delta", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=plot_df, y="Component", x="Mean Judge Preference Delta", ax=ax, color="#457b9d")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Mean overall usefulness delta")
    ax.set_ylabel("Component")
    ax.set_title("Average Effect Size by Component")
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(width + 0.03, patch.get_y() + patch.get_height() / 2, f"{width:.2f}", va="center", fontsize=9)
    fig.text(
        0.01,
        0.01,
        "Delta is Full System score minus Ablated System score on the judge overall usefulness field.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_group_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    _configure_matplotlib_cache(output_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white")
    if df.empty:
        _empty_plot(output_path, "Query Group Results")
        return
    plot_df = df.set_index("Query Group")[
        ["Full System Win %", "Ablated System Win %", "Tie %", "Mean Judge Preference Delta"]
    ]
    fig_height = max(4.5, 0.55 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Metric value"}, ax=ax)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Query Group")
    ax.set_title("Query-Group Judge Results")
    fig.text(
        0.01,
        0.01,
        "Rows group blinded full-vs-ablation comparisons by query type; percentages are shown on a 0-100 scale.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_failure_reasons(df: pd.DataFrame, output_path: Path) -> None:
    _configure_matplotlib_cache(output_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    if df.empty:
        _empty_plot(output_path, "Failure Reason Frequency")
        return
    global_df = (
        df.groupby("Failure Reason", as_index=False)["Count"]
        .sum()
        .sort_values("Count", ascending=False)
        .head(12)
    )
    fig_height = max(4.5, 0.42 * len(global_df) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.barplot(data=global_df, y="Failure Reason", x="Count", ax=ax, color="#6d597a")
    ax.set_xlabel("Count in full-system wins")
    ax.set_ylabel("Ablated-system failure reason")
    ax.set_title("Failure Reasons When Full System Won")
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(width + 0.05, patch.get_y() + patch.get_height() / 2, f"{int(width)}", va="center", fontsize=9)
    fig.text(
        0.01,
        0.01,
        "Counts use ablated-system failure flags only for comparisons where the judge preferred the full system.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _empty_plot(output_path: Path, title: str) -> None:
    _configure_matplotlib_cache(output_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _visual_report_markdown(
    report: dict[str, Any],
    paths: dict[str, Path],
    component_df: pd.DataFrame,
    group_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    examples_df: pd.DataFrame,
) -> str:
    metadata = report.get("metadata", {})
    lines = [
        "# Judge Visualization Report",
        "",
        f"- Generated from: `{metadata.get('labels_path', 'unknown labels')}`",
        f"- Judge mode: {metadata.get('judge_mode', 'unknown')}",
        f"- Judge id: {metadata.get('judge_id', 'unknown')}",
        f"- Matched judgements: {report.get('matched_judgement_count', 0)}",
        "",
        "## Component-Level Results",
        "",
        f"![Component outcomes]({paths['component_outcomes_png'].name})",
        "",
        "Blinded judge outcome share for full-system recommendations versus ablated recommendations.",
        "",
        f"![Component average delta]({paths['component_avg_delta_png'].name})",
        "",
        _markdown_table(
            component_df[
                [
                    "Component",
                    "Tasks",
                    "Full System Win",
                    "Ablated System Win",
                    "Tie",
                    "Mean Judge Preference Delta",
                    "Mean Score Margin",
                    "Most Common Failure Modes",
                ]
            ]
        ),
        "",
        "## Query-Group Results",
        "",
        f"![Query group heatmap]({paths['query_group_heatmap_png'].name})",
        "",
        _markdown_table(
            group_df[
                [
                    "Query Group",
                    "Tasks",
                    "Full System Win",
                    "Ablated System Win",
                    "Tie",
                    "Mean Judge Preference Delta",
                    "Mean Score Margin",
                ]
            ]
        ),
        "",
        "## Failure Analysis",
        "",
        f"![Failure reason counts]({paths['failure_reason_counts_png'].name})",
        "",
        _markdown_table(failure_df[["Component", "Failure Reason", "Count"]].head(20))
        if not failure_df.empty
        else "No failure flags were available for full-system wins.",
        "",
        "## Representative Examples",
        "",
        _markdown_table(examples_df[["Component", "Prompt", "Winner", "Judge Evidence"]])
        if not examples_df.empty
        else "No rationale-backed examples were available.",
        "",
    ]
    return "\n".join(lines)


def _read_jsonl_lookup(path: str | Path | None, key: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    rows = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return {str(row.get(key)): row for row in rows if row.get(key)}


def _replace_blinded_system_labels(text: str, key: dict[str, Any]) -> str:
    system_a = _variant_label(str(key.get("system_a_variant") or "System A"))
    system_b = _variant_label(str(key.get("system_b_variant") or "System B"))
    return text.replace("System A", system_a).replace("System B", system_b)


def _variant_label(variant: str) -> str:
    if variant == "full":
        return "Full System"
    if variant.startswith("no_") or variant in {"System A", "System B"}:
        return "Ablated System" if variant.startswith("no_") else variant
    return variant.replace("_", " ").title()


def _configure_matplotlib_cache(output_path: Path) -> None:
    cache_dir = Path("/tmp") / "vibefinder-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir = cache_dir / "xdg"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def _markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.to_dict(orient="records"):
        values = [_markdown_cell(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text


def _component_label(value: str) -> str:
    return COMPONENT_LABELS.get(value, value.replace("_", " ").title())


def _failure_label(value: str) -> str:
    return FAILURE_LABELS.get(value, value.replace("_", " ").title())


def _pct(value: Any) -> float:
    if value is None:
        return 0.0
    return round(float(value) * 100, 2)


def _num(value: Any) -> float:
    if value is None:
        return 0.0
    return round(float(value), 4)


def _rate_with_count(rate: Any, count: Any, total: int) -> str:
    if rate is None:
        return "-"
    count_value = int(count or 0)
    return f"{float(rate) * 100:.1f}% ({count_value}/{total})"


def wrap_text(value: str, width: int = 80) -> str:
    """Wrap long table text for optional callers."""

    return "\n".join(textwrap.wrap(value, width=width)) if value else ""
