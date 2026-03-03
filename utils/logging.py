from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(level: str = 'INFO', logs_dir: str = 'outputs/logs') -> None:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    rotating = logging.handlers.RotatingFileHandler(
        Path(logs_dir) / 'app.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
    )
    rotating.setFormatter(formatter)

    root_logger.addHandler(console)
    root_logger.addHandler(rotating)
