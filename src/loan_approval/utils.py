import logging
from pathlib import Path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
