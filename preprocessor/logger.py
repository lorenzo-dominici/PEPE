"""Simple logging configuration helpers for the preprocessor.

Provides a small wrapper around the standard `logging` module so other
modules can request a logger and the CLI can configure global logging
behaviour from command-line options.
"""
from __future__ import annotations

# at top of preprocessor/logger.py
import logging
from typing import Optional
from colorama import Fore, Style, init as colorama_init

LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}

class ColorFormatter(logging.Formatter):
    def __init__(self, datefmt: Optional[str] = None):
        # base formatter only renders the message and exception text;
        # we will prepend time/level/name ourselves to avoid duplication.
        super().__init__(fmt="%(message)s", datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        color = LEVEL_COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL
        time = self.formatTime(record, self.datefmt)
        level = record.levelname.ljust(8)
        prefix = f"{time} {color}{level}{reset} {record.name}:"
        # super().format will use the base fmt ("%(message)s") and include exception text
        message = super().format(record)
        return f"{prefix} {message}"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with a single StreamHandler and a
    compact, readable format.

    Level is a case-insensitive string like "DEBUG", "INFO", etc.
    """
    colorama_init()  # enables ANSI color handling on Windows
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    formatter = ColorFormatter(datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(lvl)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger instance for the requested name.

    Use this in modules as: logger = get_logger(__name__)
    """
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
