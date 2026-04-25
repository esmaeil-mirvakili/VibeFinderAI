from __future__ import annotations

from vibefinder.judge_visualization import (
    component_results_frame,
    failure_reason_frame,
    group_results_frame,
)


def test_judge_visualization_frames_format_report_summaries():
    report = {
        "component_summary": {
            "lyric_retrieval": {
                "count": 2,
                "full_win_count": 1,
                "baseline_win_count": 0,
                "tie_count": 1,
                "full_win_rate": 0.5,
                "baseline_win_rate": 0.0,
                "tie_rate": 0.5,
                "avg_overall_usefulness_delta": 1.25,
                "avg_average_score_delta": 0.75,
                "top_baseline_failure_flags": ["poor_ranking", "weak_explanation"],
            }
        },
        "group_summary": {
            "lyric_theme": {
                "count": 2,
                "full_win_count": 1,
                "baseline_win_count": 0,
                "tie_count": 1,
                "full_win_rate": 0.5,
                "baseline_win_rate": 0.0,
                "tie_rate": 0.5,
                "avg_overall_usefulness_delta": 1.25,
                "avg_average_score_delta": 0.75,
            }
        },
        "rows": [
            {
                "comparison": "lyric_retrieval",
                "full_won": True,
                "baseline_failure_flags": ["poor_ranking", "weak_explanation", "poor_ranking"],
            },
            {
                "comparison": "lyric_retrieval",
                "full_won": False,
                "baseline_failure_flags": ["wrong_genre"],
            },
        ],
    }

    component_df = component_results_frame(report)
    group_df = group_results_frame(report)
    failure_df = failure_reason_frame(report)

    assert component_df.iloc[0]["Component"] == "Lyric RAG"
    assert component_df.iloc[0]["Full System Win"] == "50.0% (1/2)"
    assert "Poor Ranking" in component_df.iloc[0]["Most Common Failure Modes"]
    assert group_df.iloc[0]["Query Group"] == "Lyric Theme"
    assert set(failure_df["Failure Reason"]) == {"Poor Ranking", "Weak Explanation"}
    assert failure_df.loc[failure_df["Failure Reason"] == "Poor Ranking", "Count"].iloc[0] == 2
