from __future__ import annotations

import pandas as pd
import pytest

from vibefinder import data as data_module
from vibefinder.data import (
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TEXT_COLUMNS,
    clean_song_dataframe,
    dataset_summary,
    load_songs_dataset,
    validate_song_schema,
)
from vibefinder.dataset import DatasetLocation


def test_validate_song_schema_accepts_required_columns():
    df = _raw_song_dataframe()

    validate_song_schema(df)


def test_validate_song_schema_rejects_missing_required_columns():
    df = _raw_song_dataframe().drop(columns=["lyrics", "tempo"])

    with pytest.raises(ValueError, match="lyrics, tempo"):
        validate_song_schema(df, csv_path="songs.csv")


def test_clean_song_dataframe_keeps_only_required_columns_and_normalizes_types():
    df = _raw_song_dataframe()

    cleaned = clean_song_dataframe(df)

    assert tuple(cleaned.columns) == REQUIRED_COLUMNS
    assert "extra_column" not in cleaned.columns
    for column in TEXT_COLUMNS:
        assert cleaned[column].isna().sum() == 0
        assert isinstance(cleaned.loc[0, column], str)
    for column in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(cleaned[column])
    assert cleaned.loc[0, "lyrics"] == ""
    assert pd.isna(cleaned.loc[0, "tempo"])
    assert cleaned.loc[0, "energy"] == 0.8


def test_load_songs_dataset_uses_resolved_dataset_location(monkeypatch, tmp_path):
    csv_path = tmp_path / "spotify_songs.csv"
    _raw_song_dataframe().to_csv(csv_path, index=False)
    location = DatasetLocation(path=tmp_path, csv_files=(csv_path,), source="configured")
    calls: list[object] = []

    def fake_ensure_dataset_downloaded(path):
        calls.append(path)
        return location

    monkeypatch.setattr(data_module, "ensure_dataset_downloaded", fake_ensure_dataset_downloaded)

    loaded = load_songs_dataset(path=tmp_path)

    assert calls == [tmp_path]
    assert tuple(loaded.columns) == REQUIRED_COLUMNS
    assert len(loaded) == 1
    assert loaded.loc[0, "track_id"] == "track-1"
    assert loaded.loc[0, "track_popularity"] == 80


def test_load_songs_dataset_raises_for_invalid_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([{"track_id": "track-1"}]).to_csv(csv_path, index=False)
    location = DatasetLocation(path=tmp_path, csv_files=(csv_path,), source="configured")
    monkeypatch.setattr(data_module, "ensure_dataset_downloaded", lambda path: location)

    with pytest.raises(ValueError, match="Dataset is missing required columns"):
        load_songs_dataset(path=tmp_path)


def test_dataset_summary_includes_location_when_provided(tmp_path):
    df = clean_song_dataframe(_raw_song_dataframe())
    csv_path = tmp_path / "spotify_songs.csv"
    location = DatasetLocation(path=tmp_path, csv_files=(csv_path,), source="configured")

    summary = dataset_summary(df, location=location)

    assert summary["row_count"] == 1
    assert summary["column_count"] == len(REQUIRED_COLUMNS)
    assert summary["columns"] == list(REQUIRED_COLUMNS)
    assert summary["dataset_path"] == str(tmp_path)
    assert summary["csv_files"] == [str(csv_path)]
    assert summary["source"] == "configured"


def test_dataset_summary_without_location_has_only_dataframe_metadata():
    df = clean_song_dataframe(_raw_song_dataframe())

    summary = dataset_summary(df)

    assert summary == {
        "row_count": 1,
        "column_count": len(REQUIRED_COLUMNS),
        "columns": list(REQUIRED_COLUMNS),
    }


def _raw_song_dataframe() -> pd.DataFrame:
    row = {
        "track_id": "track-1",
        "track_name": "Song",
        "track_artist": "Artist",
        "lyrics": None,
        "track_popularity": "80",
        "track_album_id": "album-1",
        "track_album_name": "Album",
        "track_album_release_date": "2020-01-01",
        "playlist_name": "Playlist",
        "playlist_id": "playlist-1",
        "playlist_genre": "pop",
        "playlist_subgenre": "dance pop",
        "danceability": "0.7",
        "energy": "0.8",
        "key": "5",
        "loudness": "-6.5",
        "mode": "1",
        "speechiness": "0.05",
        "acousticness": "0.2",
        "instrumentalness": "0.0",
        "liveness": "0.1",
        "valence": "0.6",
        "tempo": "bad-number",
        "duration_ms": "210000",
        "language": "en",
        "extra_column": "ignored",
    }
    return pd.DataFrame([row])
