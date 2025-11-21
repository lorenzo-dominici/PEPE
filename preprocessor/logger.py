"""Simple logging configuration helpers for the preprocessor.

Provides a small wrapper around the standard `logging` module so other
modules can request a logger and the CLI can configure global logging
behaviour from command-line options.
"""
from __future__ import annotations

# at top of preprocessor/logger.py
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Optional, Any
from colorama import Fore, Style, init as colorama_init

LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}

_task_var: ContextVar[str] = ContextVar("preprocessor_task", default="idle")


def set_task_label(task: str) -> None:
    """Set the current task label for this process."""
    _task_var.set(task)


@contextmanager
def task_context(task: str):
    """Context manager to temporarily set the task label."""
    token = _task_var.set(task)
    try:
        yield
    finally:
        _task_var.reset(token)


def _current_task() -> str:
    return _task_var.get()


class TaskFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.task = _current_task()
        return True


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
        task = getattr(record, "task", "idle")
        prefix = f"{time} [{task}] {color}{level}{reset} {record.name}:"
        # super().format will use the base fmt ("%(message)s") and include exception text
        message = super().format(record)
        return f"{prefix} {message}"

def _build_handler(log_file: str | Path | None) -> logging.Handler:
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(task)s] %(levelname)s %(name)s %(message)s"
        )
    else:
        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = ColorFormatter(datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    return handler


def configure_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    queue: Any | None = None,
    start_listener: bool = False,
) -> QueueListener | None:
    """Configure logging either directly or through a multiprocessing queue.

    When ``queue`` is provided, a ``QueueHandler`` is attached to the root
    logger and, if ``start_listener`` is True, a ``QueueListener`` is created
    with the appropriate output handler. The listener (if started) is returned
    so callers can stop it at shutdown.
    """
    colorama_init()  # enables ANSI color handling on Windows
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    listener: QueueListener | None = None
    if queue is not None:
        queue_handler = QueueHandler(queue)
        queue_handler.addFilter(TaskFilter())
        root.addHandler(queue_handler)
        if start_listener:
            target_handler = _build_handler(log_file)
            listener = QueueListener(queue, target_handler)
            listener.start()
    else:
        handler = _build_handler(log_file)
        handler.addFilter(TaskFilter())
        root.addHandler(handler)

    root.setLevel(lvl)
    return listener


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger instance for the requested name.

    Use this in modules as: logger = get_logger(__name__)
    """
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger", "set_task_label", "task_context"]
