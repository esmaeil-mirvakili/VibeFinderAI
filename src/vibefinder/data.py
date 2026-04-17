"""Dataset loading and schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from vibefinder.dataset import DatasetLocation, ensure_dataset_downloaded


REQUIRED_COLUMNS: tuple[str, ...] = (
    "track_id",
    "track_name",
    "track_artist",
    "lyrics",
    "track_popularity",
    "track_album_id",
    "track_album_name",
    "track_album_release_date",
    "playlist_name",
    "playlist_id",
    "playlist_genre",
    "playlist_subgenre",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "language",
)

TEXT_COLUMNS: tuple[str, ...] = (
    "track_id",
    "track_name",
    "track_artist",
    "lyrics",
    "track_album_id",
    "track_album_name",
    "track_album_release_date",
    "playlist_name",
    "playlist_id",
    "playlist_genre",
    "playlist_subgenre",
    "language",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "track_popularity",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
)


def load_songs_dataset(path: str | Path | None = ".") -> pd.DataFrame:
    """Load the approved Spotify songs dataset columns into memory."""

    location = ensure_dataset_downloaded(path=path)
    csv_path = location.csv_files[0]
    logger.info("songs_dataset_read_start", csv_path=str(csv_path), source=location.source)
    df = pd.read_csv(csv_path)
    validate_song_schema(df, csv_path=csv_path)
    cleaned = clean_song_dataframe(df)
    logger.info(
        "songs_dataset_read_finished",
        csv_path=str(csv_path),
        row_count=len(cleaned),
        column_count=len(cleaned.columns),
    )
    return cleaned


def validate_song_schema(df: pd.DataFrame, csv_path: str | Path | None = None) -> None:
    """Validate that the dataset contains the approved project columns."""

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        logger.error(
            "songs_dataset_schema_invalid",
            csv_path=str(csv_path) if csv_path else None,
            missing_columns=missing,
        )
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def clean_song_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return only approved columns with basic null and type cleanup."""

    cleaned = df.loc[:, REQUIRED_COLUMNS].copy()

    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].fillna("").astype(str)

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    return cleaned


def dataset_summary(df: pd.DataFrame, location: DatasetLocation | None = None) -> dict[str, object]:
    """Return non-sensitive summary metadata for logs or UI display."""

    summary: dict[str, object] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }
    if location:
        summary["dataset_path"] = str(location.path)
        summary["csv_files"] = [str(csv_file) for csv_file in location.csv_files]
        summary["source"] = location.source
    return summary
