"""Dataset download helpers for app startup."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import kagglehub
from loguru import logger

DATASET_SLUG = "imuhammad/audio-features-and-lyrics-of-spotify-songs"
DATASET_PATH_ENV = "VIBEFINDER_DATA_PATH"
DEFAULT_DATASET_FILENAME = "spotify_songs.csv"
IGNORED_DATASET_SEARCH_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "env",
    "node_modules",
    "venv",
}


@dataclass(frozen=True)
class DatasetLocation:
    """Resolved local dataset location."""

    path: Path
    csv_files: tuple[Path, ...]
    source: Literal["configured", "kagglehub", "project_copy"]


def ensure_dataset_downloaded(path: str | Path | None = None) -> DatasetLocation:
    """Return a local dataset path, downloading through kagglehub if needed.

    Startup code can pass a known dataset path or set `VIBEFINDER_DATA_PATH`.
    If that location exists and contains at least one CSV file, this function
    returns it without downloading. Otherwise it calls `kagglehub` with the
    dataset slug documented in `dataset.md`; kagglehub handles its own cache.
    """

    configured_path = path or os.getenv(DATASET_PATH_ENV)
    if configured_path:
        logger.info("dataset_configured_path_check", path=str(configured_path))
        location = _resolve_existing_dataset(Path(configured_path), source="configured")
        if location:
            logger.info(
                "dataset_configured_path_resolved",
                path=str(location.path),
                csv_count=len(location.csv_files),
            )
            return location
        logger.warning("dataset_configured_path_missing_csv", path=str(configured_path))

    logger.info("dataset_download_start", dataset_slug=DATASET_SLUG)
    try:
        downloaded_path = Path(kagglehub.dataset_download(DATASET_SLUG)).expanduser()
    except Exception:
        logger.exception("dataset_download_failed", dataset_slug=DATASET_SLUG)
        raise

    logger.info("dataset_download_finished", path=str(downloaded_path))
    location = _resolve_existing_dataset(downloaded_path, source="kagglehub")
    if location:
        if configured_path:
            target_location = _copy_dataset_to_configured_path(location.csv_files, Path(configured_path))
            logger.info(
                "dataset_project_copy_resolved",
                path=str(target_location.path),
                csv_count=len(target_location.csv_files),
            )
            return target_location
        logger.info(
            "dataset_download_resolved",
            path=str(location.path),
            csv_count=len(location.csv_files),
        )
        return location

    logger.error("dataset_download_missing_csv", path=str(downloaded_path))
    raise FileNotFoundError(f"No CSV files found after downloading {DATASET_SLUG}.")


def _resolve_existing_dataset(
    path: Path,
    source: Literal["configured", "kagglehub", "project_copy"],
) -> DatasetLocation | None:
    resolved = path.expanduser()
    if resolved.is_file() and resolved.suffix.lower() == ".csv":
        return DatasetLocation(path=resolved, csv_files=(resolved,), source=source)
    if resolved.is_dir():
        csv_files = _find_csv_files(resolved)
        if csv_files:
            return DatasetLocation(path=resolved, csv_files=csv_files, source=source)
    return None


def _copy_dataset_to_configured_path(
    csv_files: tuple[Path, ...],
    configured_path: Path,
) -> DatasetLocation:
    target = configured_path.expanduser()
    source_csv = csv_files[0]

    if target.suffix.lower() == ".csv":
        target_csv = target
        target_dir = target.parent
    else:
        target_dir = target
        target_csv = target_dir / source_csv.name
        if not target_csv.name:
            target_csv = target_dir / DEFAULT_DATASET_FILENAME

    target_dir.mkdir(parents=True, exist_ok=True)
    if source_csv.resolve() != target_csv.resolve():
        logger.info("dataset_copy_start", source=str(source_csv), target=str(target_csv))
        shutil.copy2(source_csv, target_csv)
        logger.info("dataset_copy_finished", target=str(target_csv))

    return DatasetLocation(path=target_dir, csv_files=(target_csv,), source="project_copy")


def _find_csv_files(root: Path) -> tuple[Path, ...]:
    csv_files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in IGNORED_DATASET_SEARCH_DIRS and not dirname.startswith(".")
        ]
        current_path = Path(current_root)
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                csv_files.append(current_path / filename)
    return tuple(sorted(csv_files))
