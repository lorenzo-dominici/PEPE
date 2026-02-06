"""logger.py

Simple logging configuration helpers for the preprocessor.

Provides a small wrapper around the standard `logging` module so other
modules can request a logger and the CLI can configure global logging
behaviour from command-line options.

This module supports:
- Colored console output using colorama
- File-based logging with timestamps
- Multiprocessing-safe logging via QueueHandler/QueueListener
- Task context labels for tracking parallel worker execution

Classes:
    TaskFilter: Logging filter that adds task labels to records.
    ColorFormatter: Formatter with ANSI color codes for console output.

Functions:
    set_task_label: Set the current task label for the process.
    task_context: Context manager for temporary task labels.
    configure_logging: Set up logging handlers and level.
    get_logger: Get a logger instance by name.

Example:
    >>> from preprocessor.logger import configure_logging, get_logger
    >>> configure_logging("DEBUG", log_file="output.log")
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Optional, Any
from colorama import Fore, Style, init as colorama_init

# Color mapping for log levels (used by ColorFormatter)
LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}

# Context variable to track current task label across async/parallel execution
_task_var: ContextVar[str] = ContextVar("preprocessor_task", default="idle")


def set_task_label(task: str) -> None:
    """Set the current task label for this process.
    
    The task label appears in log messages to identify which task
    or worker produced the log entry.
    
    Args:
        task: A short descriptive label for the current task.
    """
    _task_var.set(task)


@contextmanager
def task_context(task: str):
    """Context manager to temporarily set the task label.
    
    Automatically restores the previous task label when exiting the context.
    
    Args:
        task: A short descriptive label for the task within this context.
    
    Yields:
        None
    
    Example:
        >>> with task_context("worker:node_1"):
        ...     logger.info("Processing node")  # logs with [worker:node_1]
    """
    token = _task_var.set(task)
    try:
        yield
    finally:
        _task_var.reset(token)


def _current_task() -> str:
    """Get the current task label from context.
    
    Returns:
        The current task label string, or "idle" if not set.
    """
    return _task_var.get()


class TaskFilter(logging.Filter):
    """Logging filter that injects the current task label into log records.
    
    Adds a 'task' attribute to each LogRecord, which can be referenced
    in formatter strings as %(task)s.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add task label to the record and allow it through.
        
        Args:
            record: The log record to process.
        
        Returns:
            True (always allows the record).
        """
        record.task = _current_task()
        return True


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds ANSI color codes to log level names.
    
    Produces colored console output for better visual distinction
    between log levels. Colors are defined in LEVEL_COLORS mapping.
    
    Attributes:
        datefmt: Date format string for timestamps.
    """
    
    def __init__(self, datefmt: Optional[str] = None):
        """Initialize the color formatter.
        
        Args:
            datefmt: Optional strftime format string for timestamps.
        """
        # Base formatter only renders the message and exception text;
        # we will prepend time/level/name ourselves to avoid duplication.
        super().__init__(fmt="%(message)s", datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with colors and task context.
        
        Args:
            record: The log record to format.
        
        Returns:
            Formatted log string with ANSI color codes.
        """
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
    """Create an appropriate logging handler based on output destination.
    
    Creates a FileHandler for file output or StreamHandler for console.
    File handlers use plain text formatting; console handlers use colors.
    
    Args:
        log_file: Path to log file, or None for console output.
    
    Returns:
        Configured logging Handler instance.
    """
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
    
    Sets up the root logger with appropriate handlers. When used with
    multiprocessing, logs can be safely collected via a Queue.
    
    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file. If None, logs to stderr.
        queue: Optional multiprocessing Queue for collecting logs from workers.
        start_listener: If True and queue is provided, start a QueueListener.
    
    Returns:
        QueueListener instance if start_listener is True, otherwise None.
        Caller is responsible for calling listener.stop() at shutdown.
    
    Example:
        >>> # Main process setup with listener
        >>> queue = mp.Queue()
        >>> listener = configure_logging("DEBUG", queue=queue, start_listener=True)
        >>> # ... do work ...
        >>> listener.stop()
        
        >>> # Worker process setup
        >>> configure_logging("DEBUG", queue=queue)
    """
    colorama_init()  # enables ANSI color handling on Windows
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    
    # Remove existing handlers to avoid duplicate output
    for h in list(root.handlers):
        root.removeHandler(h)

    listener: QueueListener | None = None
    if queue is not None:
        # Multiprocessing mode: send logs to queue
        queue_handler = QueueHandler(queue)
        queue_handler.addFilter(TaskFilter())
        root.addHandler(queue_handler)
        if start_listener:
            # Main process: create listener to drain queue
            target_handler = _build_handler(log_file)
            listener = QueueListener(queue, target_handler)
            listener.start()
    else:
        # Single-process mode: log directly
        handler = _build_handler(log_file)
        handler.addFilter(TaskFilter())
        root.addHandler(handler)

    root.setLevel(lvl)
    return listener


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger instance for the requested name.
    
    Use this in modules to get a named logger that inherits
    the root logger's configuration.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
    
    Returns:
        A logging.Logger instance.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger", "set_task_label", "task_context"]
