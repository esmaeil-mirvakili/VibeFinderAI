from __future__ import annotations

from pathlib import Path

import pytest

from vibefinder import dataset as dataset_module
from vibefinder.dataset import (
    DATASET_SLUG,
    DEFAULT_DATASET_FILENAME,
    ensure_dataset_downloaded,
)


def test_ensure_dataset_downloaded_returns_configured_csv_without_download(tmp_path, monkeypatch):
    csv_path = tmp_path / "spotify_songs.csv"
    csv_path.write_text("track_id\n1\n", encoding="utf-8")

    def fail_download(_: str) -> str:
        raise AssertionError("download should not be called")

    monkeypatch.setattr(dataset_module.kagglehub, "dataset_download", fail_download)

    location = ensure_dataset_downloaded(path=csv_path)

    assert location.path == csv_path
    assert location.csv_files == (csv_path,)
    assert location.source == "configured"


def test_ensure_dataset_downloaded_finds_direct_csv_in_configured_directory(tmp_path, monkeypatch):
    nested = tmp_path / "data"
    nested.mkdir()
    csv_path = nested / "songs.csv"
    csv_path.write_text("track_id\n1\n", encoding="utf-8")
    ignored = nested / "__pycache__"
    ignored.mkdir()
    (ignored / "ignored.csv").write_text("track_id\nbad\n", encoding="utf-8")

    def fail_download(_: str) -> str:
        raise AssertionError("download should not be called")

    monkeypatch.setattr(dataset_module.kagglehub, "dataset_download", fail_download)

    location = ensure_dataset_downloaded(path=nested)

    assert location.path == nested
    assert location.csv_files == (csv_path,)
    assert location.source == "configured"


def test_ensure_dataset_downloaded_prefers_default_dataset_filename_in_configured_directory(tmp_path, monkeypatch):
    csv_path = tmp_path / DEFAULT_DATASET_FILENAME
    csv_path.write_text("track_id\n1\n", encoding="utf-8")
    other_csv = tmp_path / "component_results.csv"
    other_csv.write_text("x,y\n1,2\n", encoding="utf-8")

    def fail_download(_: str) -> str:
        raise AssertionError("download should not be called")

    monkeypatch.setattr(dataset_module.kagglehub, "dataset_download", fail_download)

    location = ensure_dataset_downloaded(path=tmp_path)

    assert location.path == tmp_path
    assert location.csv_files == (csv_path,)
    assert location.source == "configured"


def test_ensure_dataset_downloaded_does_not_scan_nested_csvs_for_configured_directory(tmp_path, monkeypatch):
    nested = tmp_path / "nested"
    nested.mkdir()
    csv_path = nested / DEFAULT_DATASET_FILENAME
    csv_path.write_text("track_id\n1\n", encoding="utf-8")

    def fake_download(_: str) -> str:
        raise RuntimeError("downloaded instead of using nested csv")

    monkeypatch.setattr(dataset_module.kagglehub, "dataset_download", fake_download)

    with pytest.raises(RuntimeError, match="downloaded instead of using nested csv"):
        ensure_dataset_downloaded(path=tmp_path)


def test_ensure_dataset_downloaded_uses_environment_path(tmp_path, monkeypatch):
    csv_path = tmp_path / "env_songs.csv"
    csv_path.write_text("track_id\n1\n", encoding="utf-8")
    monkeypatch.setenv(dataset_module.DATASET_PATH_ENV, str(csv_path))

    def fail_download(_: str) -> str:
        raise AssertionError("download should not be called")

    monkeypatch.setattr(dataset_module.kagglehub, "dataset_download", fail_download)

    location = ensure_dataset_downloaded(path=None)

    assert location.csv_files == (csv_path,)
    assert location.source == "configured"


def test_ensure_dataset_downloaded_downloads_when_no_configured_path(monkeypatch, tmp_path):
    downloaded = tmp_path / "kaggle"
    downloaded.mkdir()
    csv_path = downloaded / "downloaded.csv"
    csv_path.write_text("track_id\n1\n", encoding="utf-8")
    calls: list[str] = []

    def fake_download(slug: str) -> str:
        calls.append(slug)
        return str(downloaded)

    monkeypatch.setattr(dataset_module.kagglehub, "dataset_download", fake_download)

    location = ensure_dataset_downloaded(path=None)

    assert calls == [DATASET_SLUG]
    assert location.path == downloaded
    assert location.csv_files == (csv_path,)
    assert location.source == "kagglehub"


def test_ensure_dataset_downloaded_copies_download_to_configured_directory(monkeypatch, tmp_path):
    configured = tmp_path / "project_root"
    downloaded = tmp_path / "kaggle"
    downloaded.mkdir()
    source_csv = downloaded / "source.csv"
    source_csv.write_text("track_id\n1\n", encoding="utf-8")

    def fake_download(_: str) -> str:
        return str(downloaded)

    monkeypatch.setattr(dataset_module.kagglehub, "dataset_download", fake_download)

    location = ensure_dataset_downloaded(path=configured)

    target_csv = configured / source_csv.name
    assert location.path == configured
    assert location.csv_files == (target_csv,)
    assert location.source == "project_copy"
    assert target_csv.read_text(encoding="utf-8") == "track_id\n1\n"


def test_ensure_dataset_downloaded_copies_download_to_configured_csv_path(monkeypatch, tmp_path):
    target_csv = tmp_path / DEFAULT_DATASET_FILENAME
    downloaded = tmp_path / "kaggle"
    downloaded.mkdir()
    source_csv = downloaded / "source.csv"
    source_csv.write_text("track_id\n1\n", encoding="utf-8")

    def fake_download(_: str) -> str:
        return str(downloaded)

    monkeypatch.setattr(dataset_module.kagglehub, "dataset_download", fake_download)

    location = ensure_dataset_downloaded(path=target_csv)

    assert location.path == tmp_path
    assert location.csv_files == (target_csv,)
    assert location.source == "project_copy"
    assert target_csv.read_text(encoding="utf-8") == "track_id\n1\n"


def test_ensure_dataset_downloaded_raises_when_download_has_no_csv(monkeypatch, tmp_path):
    downloaded = tmp_path / "empty_download"
    downloaded.mkdir()

    def fake_download(_: str) -> str:
        return str(downloaded)

    monkeypatch.setattr(dataset_module.kagglehub, "dataset_download", fake_download)

    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        ensure_dataset_downloaded(path=None)


def test_ensure_dataset_downloaded_reraises_download_error(monkeypatch):
    def fake_download(_: str) -> str:
        raise RuntimeError("kaggle unavailable")

    monkeypatch.setattr(dataset_module.kagglehub, "dataset_download", fake_download)

    with pytest.raises(RuntimeError, match="kaggle unavailable"):
        ensure_dataset_downloaded(path=None)


def test_find_csv_files_sorts_and_skips_hidden_and_ignored_dirs(tmp_path):
    visible = tmp_path / "visible"
    visible.mkdir()
    second_csv = visible / "b.csv"
    first_csv = visible / "a.csv"
    second_csv.write_text("track_id\n2\n", encoding="utf-8")
    first_csv.write_text("track_id\n1\n", encoding="utf-8")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "hidden.csv").write_text("track_id\nhidden\n", encoding="utf-8")
    ignored = tmp_path / "__pycache__"
    ignored.mkdir()
    (ignored / "ignored.csv").write_text("track_id\nignored\n", encoding="utf-8")

    csv_files = dataset_module._find_csv_files(tmp_path)

    assert csv_files == (first_csv, second_csv)


def test_find_csv_files_prefers_project_spotify_dataset_over_generated_csvs(tmp_path):
    reports = tmp_path / "evaluation" / "judgements" / "reports" / "visualizations"
    reports.mkdir(parents=True)
    generated_csv = reports / "component_results.csv"
    generated_csv.write_text("Component,Tasks\nx,1\n", encoding="utf-8")
    dataset_csv = tmp_path / DEFAULT_DATASET_FILENAME
    dataset_csv.write_text("track_id\n1\n", encoding="utf-8")

    csv_files = dataset_module._find_csv_files(tmp_path)

    assert csv_files[0] == dataset_csv
    assert generated_csv in csv_files
