
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Project logging directory
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "sem_analysis.log"

# Log message format
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(filename)s:%(lineno)d | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Global logger
logger = logging.getLogger("my_application")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Prevent duplicate handlers when this module is imported again
if not logger.handlers:
    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler: 5 MB per file, keep 5 backups
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
