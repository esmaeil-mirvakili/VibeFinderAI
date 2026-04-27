"""Streamlit UI for blinded human evaluation of VibeFinder judge tasks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vibefinder.judge_evaluation import JUDGE_SCORE_FIELDS, JudgeTask, Judgement, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_TASKS_PATH = PROJECT_ROOT / "evaluation" / "judgements" / "tasks.jsonl"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "evaluation" / "judgements" / "human"
DEFAULT_DRAFTS_DIR = DEFAULT_LABELS_DIR / "drafts"
DEFAULT_JUDGE_ID = "judge_001"
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
FAILURE_FLAGS = tuple(FAILURE_LABELS)
SCORE_FIELD_LABELS = {
    "query_relevance": "Query relevance",
    "constraint_satisfaction": "Constraint satisfaction",
    "lyric_theme_fit": "Lyric/theme fit",
    "ranking_quality": "Ranking quality",
    "explanation_grounding": "Explanation grounding",
    "warning_honesty": "Warning honesty",
    "overall_usefulness": "Overall usefulness",
}


def main() -> None:
    st.set_page_config(page_title="VibeFinder Human Evaluation", layout="wide")
    _render_styles()
    st.title("VibeFinder Human Evaluation")

    tasks_path = DEFAULT_TASKS_PATH
    labels_dir = DEFAULT_LABELS_DIR
    drafts_dir = DEFAULT_DRAFTS_DIR
    drafts_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.header("Session")
        judge_id = st.text_input("Judge ID", value=DEFAULT_JUDGE_ID).strip() or DEFAULT_JUDGE_ID
        labels_path = labels_dir / f"labels_{judge_id}.jsonl"
        draft_path = drafts_dir / f"{judge_id}.json"
        if st.button("Reload from disk", use_container_width=True):
            _clear_loaded_state()
            st.rerun()
        st.caption(f"Draft file: `{draft_path.relative_to(PROJECT_ROOT)}`")
        st.caption(f"Final labels: `{labels_path.relative_to(PROJECT_ROOT)}`")

    tasks = _load_tasks(tasks_path)
    saved_entries = _load_saved_entries(draft_path=draft_path, labels_path=labels_path)
    judgments = _session_entries(judge_id=judge_id, tasks=tasks, saved_entries=saved_entries)
    completed_count = sum(1 for task in tasks if _is_complete_entry(judgments.get(task.task_id, {})))

    with st.sidebar:
        st.metric("Completed", f"{completed_count}/{len(tasks)}")
        if not tasks:
            st.error("No judge tasks were found.")
            st.stop()
        if "selected_task_index" not in st.session_state or not (0 <= st.session_state["selected_task_index"] < len(tasks)):
            st.session_state["selected_task_index"] = 0
        current_index = int(st.session_state["selected_task_index"])
        nav_cols = st.columns(2, gap="small")
        with nav_cols[0]:
            if st.button("Previous", use_container_width=True, disabled=current_index == 0):
                previous_index = max(0, current_index - 1)
                st.session_state["selected_task_index"] = previous_index
                st.rerun()
        with nav_cols[1]:
            if st.button("Next", use_container_width=True, disabled=current_index == len(tasks) - 1):
                next_index = min(len(tasks) - 1, current_index + 1)
                st.session_state["selected_task_index"] = next_index
                st.rerun()
        selected_task_index = st.selectbox(
            "Tasks",
            options=list(range(len(tasks))),
            index=current_index,
            format_func=lambda idx: (
                f"{'✅' if _is_complete_entry(judgments.get(tasks[int(idx)].task_id, {})) else '⬜'} "
                f"Task-{int(idx) + 1}"
            ),
            label_visibility="collapsed",
        )
        st.session_state["selected_task_index"] = int(selected_task_index)
        st.session_state["selected_task_id"] = tasks[int(selected_task_index)].task_id

    selected_task_index = int(st.session_state["selected_task_index"])
    selected_task = tasks[selected_task_index]
    entry = dict(judgments.get(selected_task.task_id, _blank_entry(selected_task.task_id, judge_id)))

    st.subheader("Task")
    st.write(f"**Task label:** `Task-{selected_task_index + 1}`")
    st.write(f"**Query:** {selected_task.query}")
    st.write(
        f"**Group:** `{selected_task.group}`  |  **Comparison:** `{selected_task.comparison}`  |  "
        f"**Task ID:** `{selected_task.task_id}`"
    )

    with st.expander("Judging rubric", expanded=False):
        for item in selected_task.rubric:
            st.write(f"- {item}")

    _render_system_output_comparison(selected_task)

    st.subheader("Judgement")
    with st.form(key=f"judgement_form_{selected_task.task_id}", clear_on_submit=False):
        winner = st.radio(
            "Winner",
            options=["A", "B", "tie"],
            horizontal=True,
            index=["A", "B", "tie"].index(str(entry.get("winner", "tie"))),
        )
        confidence = st.radio(
            "Confidence",
            options=["low", "medium", "high"],
            horizontal=True,
            index=["low", "medium", "high"].index(str(entry.get("confidence", "medium"))),
        )

        score_cols = st.columns(2, gap="large")
        system_a_scores = _render_score_inputs(
            parent=score_cols[0],
            system_label="System A",
            prefix=f"{selected_task.task_id}_system_a",
            values=entry.get("scores", {}).get("system_a", {}),
        )
        system_b_scores = _render_score_inputs(
            parent=score_cols[1],
            system_label="System B",
            prefix=f"{selected_task.task_id}_system_b",
            values=entry.get("scores", {}).get("system_b", {}),
        )

        failure_cols = st.columns(2, gap="large")
        system_a_flags = failure_cols[0].multiselect(
            "System A failure flags",
            options=FAILURE_FLAGS,
            default=list(entry.get("system_a_failure_flags", [])),
            format_func=lambda flag: FAILURE_LABELS.get(flag, flag),
            key=f"{selected_task.task_id}_flags_a",
        )
        system_b_flags = failure_cols[1].multiselect(
            "System B failure flags",
            options=FAILURE_FLAGS,
            default=list(entry.get("system_b_failure_flags", [])),
            format_func=lambda flag: FAILURE_LABELS.get(flag, flag),
            key=f"{selected_task.task_id}_flags_b",
        )

        rationale = st.text_area(
            "Rationale",
            value=str(entry.get("rationale", "")),
            height=120,
            help="Short public explanation for the judgement.",
        )

        save_clicked = st.form_submit_button("Save task judgement", use_container_width=True)
        if save_clicked:
            updated_entry = {
                "schema": "vibefinder_judgement_v1",
                "task_id": selected_task.task_id,
                "judge_id": judge_id,
                "judge_type": "human",
                "winner": winner,
                "confidence": confidence,
                "scores": {
                    "system_a": system_a_scores,
                    "system_b": system_b_scores,
                },
                "system_a_failure_flags": system_a_flags,
                "system_b_failure_flags": system_b_flags,
                "rationale": rationale.strip(),
                "raw_response": None,
            }
            try:
                Judgement.model_validate(updated_entry)
            except Exception as exc:
                st.error(f"Could not save this task yet: {exc}")
            else:
                judgments[selected_task.task_id] = updated_entry
                _write_draft(draft_path=draft_path, judge_id=judge_id, entries=judgments)
                st.session_state["human_eval_entries"] = judgments
                st.success("Task judgement saved to draft.")
                st.rerun()

    st.divider()
    st.subheader("Finalize")
    st.write(
        "Final submission writes the JSONL label file only when every task has a valid saved judgement. "
        "You can keep editing tasks after that and submit again to overwrite the final file."
    )
    submit_disabled = completed_count != len(tasks)
    if submit_disabled:
        st.info(f"Complete all tasks before final submit. Remaining: {len(tasks) - completed_count}")
    if st.button("Submit all completed judgements", type="primary", disabled=submit_disabled):
        rows = [judgments[task.task_id] for task in tasks]
        try:
            validated_rows = [Judgement.model_validate(row).model_dump(mode="json", by_alias=True) for row in rows]
        except Exception as exc:
            st.error(f"Submission failed validation: {exc}")
        else:
            write_jsonl(labels_path, validated_rows)
            st.success(f"Saved {len(validated_rows)} judgements to `{labels_path.relative_to(PROJECT_ROOT)}`.")


@st.cache_data(show_spinner=False)
def _load_tasks(tasks_path: Path) -> tuple[JudgeTask, ...]:
    return tuple(JudgeTask.model_validate(row) for row in read_jsonl(tasks_path))


def _load_saved_entries(*, draft_path: Path, labels_path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if labels_path.exists():
        for row in read_jsonl(labels_path):
            task_id = str(row.get("task_id") or "")
            if task_id:
                entries[task_id] = row
    if draft_path.exists():
        payload = json.loads(draft_path.read_text(encoding="utf-8"))
        for row in payload.get("entries", []):
            task_id = str(row.get("task_id") or "")
            if task_id:
                entries[task_id] = row
    return entries


def _session_entries(
    *,
    judge_id: str,
    tasks: tuple[JudgeTask, ...],
    saved_entries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    cached_judge_id = st.session_state.get("human_eval_judge_id")
    if "human_eval_entries" not in st.session_state or cached_judge_id != judge_id:
        st.session_state["human_eval_entries"] = {
            task.task_id: dict(saved_entries.get(task.task_id, _blank_entry(task.task_id, judge_id)))
            for task in tasks
        }
        st.session_state["human_eval_judge_id"] = judge_id
    return st.session_state["human_eval_entries"]


def _blank_entry(task_id: str, judge_id: str) -> dict[str, Any]:
    return {
        "schema": "vibefinder_judgement_v1",
        "task_id": task_id,
        "judge_id": judge_id,
        "judge_type": "human",
        "winner": "tie",
        "confidence": "medium",
        "scores": {
            "system_a": {field: 3 for field in JUDGE_SCORE_FIELDS},
            "system_b": {field: 3 for field in JUDGE_SCORE_FIELDS},
        },
        "system_a_failure_flags": [],
        "system_b_failure_flags": [],
        "rationale": "",
        "raw_response": None,
    }


def _is_complete_entry(entry: dict[str, Any]) -> bool:
    if not entry:
        return False
    try:
        Judgement.model_validate(entry)
    except Exception:
        return False
    return True


def _task_label(task: JudgeTask, *, completed: bool) -> str:
    prefix = "✅" if completed else "⬜"
    return f"{prefix} {task.query}"


def _render_system_output_comparison(task: JudgeTask) -> None:
    header_cols = st.columns(2, gap="large")
    with header_cols[0]:
        st.markdown("### System A")
        _render_warning_block(task.system_a.warnings)
    with header_cols[1]:
        st.markdown("### System B")
        _render_warning_block(task.system_b.warnings)

    max_len = max(len(task.system_a.recommendations), len(task.system_b.recommendations))
    for index in range(max_len):
        pair_cols = st.columns(2, gap="large")
        rec_a = task.system_a.recommendations[index] if index < len(task.system_a.recommendations) else None
        rec_b = task.system_b.recommendations[index] if index < len(task.system_b.recommendations) else None
        _render_recommendation_card(pair_cols[0], rec_a)
        _render_recommendation_card(pair_cols[1], rec_b)


def _render_warning_block(warnings: tuple[str, ...]) -> None:
    if warnings:
        with st.expander("Warnings", expanded=False):
            for warning in warnings:
                st.write(f"- {warning}")


def _render_recommendation_card(parent: Any, recommendation: Any | None) -> None:
    with parent:
        with st.container(border=True):
            if recommendation is None:
                st.caption("No recommendation at this rank.")
                return
            st.markdown(
                f"**{recommendation.rank}. {recommendation.track_name or 'Unknown track'}**"
                f"  \n{recommendation.track_artist or 'Unknown artist'}"
            )
            meta_bits = [
                recommendation.track_album_name,
                recommendation.playlist_genre,
                recommendation.playlist_subgenre,
                recommendation.language,
            ]
            meta_text = " | ".join(bit for bit in meta_bits if bit)
            if meta_text:
                st.caption(meta_text)
            feature_df = _feature_table_frame(recommendation.audio_features or {})
            if not feature_df.empty:
                st.write("**Audio features**")
                st.table(feature_df)
            if recommendation.lyric_preview:
                st.write("**Lyric preview**")
                st.write(recommendation.lyric_preview)
            st.write("**Explanation**")
            if recommendation.explanation:
                st.write(recommendation.explanation)
            else:
                st.write("This system did not produce any explanation.")


def _feature_table_frame(audio_features: dict[str, Any]) -> pd.DataFrame:
    preferred_order = (
        "danceability",
        "energy",
        "valence",
        "tempo",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "liveness",
        "duration_ms",
    )
    rows = []
    for key in preferred_order:
        if key not in audio_features or audio_features[key] is None:
            continue
        value = audio_features[key]
        display_value = f"{float(value):.3f}" if isinstance(value, float) else str(value)
        rows.append({"Feature": key, "Value": display_value})
    return pd.DataFrame(rows)


def _render_score_inputs(
    *,
    parent: Any,
    system_label: str,
    prefix: str,
    values: dict[str, Any],
) -> dict[str, int]:
    results: dict[str, int] = {}
    with parent:
        st.markdown(f"#### {system_label} scores")
        for field in JUDGE_SCORE_FIELDS:
            default_value = int(values.get(field, 3))
            results[field] = st.slider(
                SCORE_FIELD_LABELS[field],
                min_value=1,
                max_value=5,
                value=default_value,
                step=1,
                key=f"{prefix}_{field}",
            )
    return results


def _write_draft(*, draft_path: Path, judge_id: str, entries: dict[str, dict[str, Any]]) -> None:
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "judge_id": judge_id,
        "entries": [entries[task_id] for task_id in sorted(entries)],
    }
    draft_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clear_loaded_state() -> None:
    for key in ("human_eval_entries", "human_eval_judge_id", "selected_task_id", "selected_task_index"):
        st.session_state.pop(key, None)


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            max-width: 100%;
        }
        div[data-testid="stSidebar"] .stRadio label p {
            font-size: 0.92rem;
            line-height: 1.25rem;
        }
        div[data-testid="stVerticalBlock"] div[data-testid="stCodeBlock"] code {
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
