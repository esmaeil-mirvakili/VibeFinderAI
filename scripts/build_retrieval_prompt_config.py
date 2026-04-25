"""Build dataset-derived retrieval prompt config for LLM agents."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vibefinder.data import NUMERIC_COLUMNS, load_songs_dataset


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "config" / "retrieval_prompt_config.json"

# Retrieval columns from design.md, excluding lyrics.
CATEGORICAL_METADATA_COLUMNS = (
    "playlist_genre",
    "playlist_subgenre",
    "language",
)
FULL_TEXT_METADATA_COLUMNS = (
    "playlist_name",
    "track_artist",
    "track_album_name",
)
METADATA_RETRIEVAL_COLUMNS = CATEGORICAL_METADATA_COLUMNS + FULL_TEXT_METADATA_COLUMNS
FEATURE_RETRIEVAL_COLUMNS = (
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
    "valence",
    "tempo",
    "speechiness",
    "liveness",
    "duration_ms",
    "loudness",
    "track_popularity",
)
RETRIEVAL_PROMPT_COLUMNS = METADATA_RETRIEVAL_COLUMNS + FEATURE_RETRIEVAL_COLUMNS

FEATURE_PREFERENCE_FIELDS = (
    "energy",
    "valence",
    "tempo",
    "danceability",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness",
    "duration_ms",
    "loudness",
    "track_popularity",
)
FEATURE_DIRECTIONS = ("low", "medium", "high")
RETRIEVAL_MODES = ("lyric", "metadata", "feature", "hybrid")


def build_retrieval_prompt_config(data_path: str | Path) -> dict[str, Any]:
    songs = load_songs_dataset(path=data_path)

    columns: dict[str, dict[str, Any]] = {}
    numeric_ranges: dict[str, dict[str, float | int | None]] = {}
    categorical_values: dict[str, list[str]] = {}
    full_text_search_columns: dict[str, dict[str, Any]] = {}

    for column in RETRIEVAL_PROMPT_COLUMNS:
        series = songs[column]
        null_count = int(series.isna().sum())
        non_empty = series.dropna()

        if column in NUMERIC_COLUMNS:
            minimum = _json_number(non_empty.min()) if not non_empty.empty else None
            maximum = _json_number(non_empty.max()) if not non_empty.empty else None
            column_config = {
                "type": "numeric",
                "min": minimum,
                "max": maximum,
                "null_count": null_count,
            }
            numeric_ranges[column] = {
                "min": minimum,
                "max": maximum,
            }
        elif column in CATEGORICAL_METADATA_COLUMNS:
            values = sorted(
                (str(value) for value in non_empty.astype(str).unique() if str(value) != ""),
                key=str.casefold,
            )
            column_config = {
                "type": "categorical",
                "unique_count": len(values),
                "values": values,
            }
            categorical_values[column] = values
        elif column in FULL_TEXT_METADATA_COLUMNS:
            unique_count = int(non_empty.astype(str).nunique())
            column_config = {
                "type": "full_text",
                "search_method": "full_text_search",
                "unique_count": unique_count,
                "null_count": null_count,
                "values_included": False,
                "values": [],
                "note": "Free-form text field. Do not ask the LLM to choose from an enum list.",
            }
            full_text_search_columns[column] = {
                "search_method": "full_text_search",
                "unique_count": unique_count,
            }
        else:
            raise ValueError(f"Unhandled retrieval prompt column: {column}")

        columns[column] = column_config

    config = {
        "metadata": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "dataset_path": str(data_path),
            "row_count": len(songs),
            "retrieval_prompt_column_count": len(RETRIEVAL_PROMPT_COLUMNS),
            "source": "design.md retrieval columns excluding lyrics",
        },
        "retrieval_columns": {
            "metadata": {
                "categorical": list(CATEGORICAL_METADATA_COLUMNS),
                "full_text": list(FULL_TEXT_METADATA_COLUMNS),
            },
            "feature": list(FEATURE_RETRIEVAL_COLUMNS),
            "excluded": ["lyrics"],
        },
        "columns": columns,
        "llm_prompt_constraints": {
            "retrieval_columns": list(RETRIEVAL_PROMPT_COLUMNS),
            "numeric_ranges": numeric_ranges,
            "categorical_values": categorical_values,
            "full_text_search_columns": full_text_search_columns,
            "preference_fields": {
                "lyric_concepts": {
                    "type": "free_text_terms",
                    "source_columns": ["lyrics"],
                    "note": "lyrics are retrieved by the Lyric RAG tool and are intentionally excluded from this prompt config",
                },
                "genre_terms": {
                    "type": "categorical_terms",
                    "source_columns": ["playlist_genre", "playlist_subgenre"],
                },
                "metadata_text_terms": {
                    "type": "free_text_terms",
                    "source_columns": list(FULL_TEXT_METADATA_COLUMNS),
                    "search_method": "full_text_search",
                },
                "language": {
                    "type": "categorical_value",
                    "source_column": "language",
                },
                **{
                    field: {
                        "type": "feature_direction",
                        "valid_values": list(FEATURE_DIRECTIONS),
                        "source_column": field,
                    }
                    for field in FEATURE_PREFERENCE_FIELDS
                },
                "exclusions": {
                    "type": "free_text_terms",
                    "source_columns": list(RETRIEVAL_PROMPT_COLUMNS),
                },
            },
            "retrieval_strategy_fields": {
                "modes": {
                    "type": "enum_list",
                    "valid_values": list(RETRIEVAL_MODES),
                },
                "primary_mode": {
                    "type": "enum",
                    "valid_values": list(RETRIEVAL_MODES),
                },
                "strictness": {
                    "type": "enum",
                    "valid_values": ["strict", "balanced", "broad"],
                },
            },
        },
    }
    return config


def write_config(config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("retrieval_prompt_config_written", output_path=str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=".", help="Dataset CSV or folder. Defaults to project root.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON config path.",
    )
    return parser.parse_args()


def _json_number(value: Any) -> float | int:
    as_float = float(value)
    if as_float.is_integer():
        return int(as_float)
    return as_float


def main() -> None:
    args = parse_args()
    logger.info(
        "retrieval_prompt_config_build_start",
        data_path=args.data_path,
        output=args.output,
    )
    config = build_retrieval_prompt_config(data_path=args.data_path)
    write_config(config, Path(args.output))
    logger.info("retrieval_prompt_config_build_finished", output=args.output)


if __name__ == "__main__":
    main()
