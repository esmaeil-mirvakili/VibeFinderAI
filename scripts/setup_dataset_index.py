"""Download the dataset and build the lyric FAISS index for local setup."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vibefinder.data import load_songs_dataset
from vibefinder.dataset import ensure_dataset_downloaded
from vibefinder.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingProvider,
)
from vibefinder.tools import (
    build_lyric_index,
    lyric_index_files,
    load_lyric_index,
    save_lyric_index,
)


DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "setup_dataset_index.log"


def configure_logging(log_file: Path | None) -> None:
    """Configure concise console logs and optional file logs."""

    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message} | {extra}"
    logger.add(sys.stderr, level="INFO", format=log_format, diagnose=False, backtrace=False)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="INFO",
            format=log_format,
            rotation="2 MB",
            retention=5,
            diagnose=False,
            backtrace=False,
        )


def run_setup(args: argparse.Namespace) -> None:
    """Run setup steps with progress feedback."""

    project_root = Path(args.project_root).expanduser().resolve()
    data_path = Path(args.data_path).expanduser()
    if not data_path.is_absolute():
        data_path = project_root / data_path

    embedding_model = args.embedding_model or os.getenv("VIBEFINDER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedding_batch_size = args.embedding_batch_size or int(
        os.getenv("VIBEFINDER_EMBEDDING_BATCH_SIZE", "32")
    )
    embedding_device = args.embedding_device
    if embedding_device is None:
        embedding_device = os.getenv("VIBEFINDER_EMBEDDING_DEVICE") or None

    logger.info(
        "setup_started",
        project_root=str(project_root),
        data_path=str(data_path),
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        embedding_device=embedding_device,
    )

    with stage_progress("Download/check dataset") as progress:
        location = ensure_dataset_downloaded(path=data_path)
        logger.info(
            "setup_dataset_ready",
            path=str(location.path),
            csv_files=[str(csv_file) for csv_file in location.csv_files],
            source=location.source,
        )
        progress.update(1)

    with stage_progress("Load and validate dataset") as progress:
        songs = load_songs_dataset(path=data_path)
        logger.info(
            "setup_dataset_loaded",
            row_count=len(songs),
            column_count=len(songs.columns),
        )
        progress.update(1)

    with stage_progress("Check lyric FAISS index") as progress:
        index_files = lyric_index_files(project_root)
        persisted_index_loaded = False
        try:
            lyric_index = load_lyric_index(
                root=project_root,
                expected_embedding_model=embedding_model,
            )
            persisted_index_loaded = True
            logger.info(
                "setup_lyric_index_exists",
                index_path=str(index_files.index_path),
                metadata_path=str(index_files.metadata_path),
                indexed_count=len(lyric_index.track_ids),
                dimension=lyric_index.dimension,
                embedding_model=lyric_index.embedding_model,
            )
        except FileNotFoundError:
            logger.info(
                "setup_lyric_index_missing",
                index_path=str(index_files.index_path),
                metadata_path=str(index_files.metadata_path),
            )
        except Exception as exc:
            logger.warning(
                "setup_lyric_index_invalid",
                index_path=str(index_files.index_path),
                metadata_path=str(index_files.metadata_path),
                error_type=exc.__class__.__name__,
                error=str(exc),
            )
        progress.update(1)

    if not persisted_index_loaded:
        embedder = SentenceTransformerEmbeddingProvider(
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            device=embedding_device,
            show_progress_bar=not args.no_embedding_progress,
        )

        with stage_progress("Load embedding model weights") as progress:
            embedder.load_model()
            logger.info(
                "setup_embedding_model_loaded",
                embedding_model=embedder.name,
                embedding_batch_size=embedding_batch_size,
                embedding_device=embedding_device,
            )
            progress.update(1)

        with stage_progress("Embed lyrics and build FAISS index") as progress:
            lyric_index = build_lyric_index(
                songs=songs,
                embedder=embedder,
            )
            progress.update(1)

        with stage_progress("Save lyric FAISS index") as progress:
            save_lyric_index(
                lyric_index=lyric_index,
                root=project_root,
            )
            progress.update(1)

        logger.info(
            "setup_lyric_index_built",
            index_path=str(index_files.index_path),
            metadata_path=str(index_files.metadata_path),
            indexed_count=len(lyric_index.track_ids),
            dimension=lyric_index.dimension,
            embedding_model=lyric_index.embedding_model,
        )
    else:
        with stage_progress("Lyric FAISS index ready") as progress:
            logger.info(
                "setup_lyric_index_reused",
                index_path=str(index_files.index_path),
                metadata_path=str(index_files.metadata_path),
                indexed_count=len(lyric_index.track_ids),
                dimension=lyric_index.dimension,
                embedding_model=lyric_index.embedding_model,
            )
            progress.update(1)

    logger.info(
        "setup_finished",
        dataset_csv=str(location.csv_files[0]),
        index_path=str(index_files.index_path),
        metadata_path=str(index_files.metadata_path),
    )


def stage_progress(description: str):
    """Create one visible progress bar for a single setup stage."""

    return tqdm(
        total=1,
        desc=description,
        unit="stage",
        leave=True,
        dynamic_ncols=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root where lyric_faiss.index and lyric_faiss_metadata.json are stored.",
    )
    parser.add_argument(
        "--data-path",
        default=".",
        help="Dataset CSV or folder. Relative paths are resolved against --project-root.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"SentenceTransformers model name. Defaults to env var or {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Batch size for lyric embedding. Defaults to env var or 32.",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Optional SentenceTransformers device override, such as cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--no-embedding-progress",
        action="store_true",
        help="Disable the SentenceTransformers embedding progress bar.",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="Setup log file path. Use an empty string to disable file logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file).expanduser() if args.log_file else None
    configure_logging(log_file)
    run_setup(args)


if __name__ == "__main__":
    main()
