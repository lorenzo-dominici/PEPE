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
import threading
from contextlib import contextmanager
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

# Thread-local storage for the current task label.
# Using threading.local() instead of ContextVar for reliable behaviour
# across all platforms when used inside multiprocessing workers spawned
# via the 'spawn' start method.  ContextVar values are not guaranteed to
# propagate correctly in spawned worker processes on Linux, whereas a
# plain thread-local (or module-level global accessed from a single
# thread) works identically everywhere.
_task_local: threading.local = threading.local()


def set_task_label(task: str) -> None:
    """Set the current task label for this process/thread.
    
    The task label appears in log messages to identify which task
    or worker produced the log entry.
    
    Args:
        task: A short descriptive label for the current task.
    """
    _task_local.task = task


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
    previous = getattr(_task_local, "task", "idle")
    _task_local.task = task
    try:
        yield
    finally:
        _task_local.task = previous


def _current_task() -> str:
    """Get the current task label.
    
    Returns:
        The current task label string, or "idle" if not set.
    """
    return getattr(_task_local, "task", "idle")


class TaskFilter(logging.Filter):
    """Logging filter that injects the current task label into log records.
    
    Adds a 'task' attribute to each LogRecord, which can be referenced
    in formatter strings as %(task)s.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add task label to the record if not already present.
        
        Only sets the task attribute when the record does not already
        carry one.  This allows worker-originated records (which have
        their task set before being sent through the multiprocessing
        queue) to retain their original label when the filter runs
        again on the listener side.
        
        Args:
            record: The log record to process.
        
        Returns:
            True (always allows the record).
        """
        if not hasattr(record, 'task'):
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
    
    # Remove existing handlers and filters to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)
    for f in list(root.filters):
        root.removeFilter(f)

    listener: QueueListener | None = None
    if queue is not None:
        if start_listener:
            # Main-process mode: log directly to the output handler so
            # the main process never touches the multiprocessing Queue's
            # write lock (_wlock).  Worker log records arrive via the
            # queue and are drained by a QueueListener into the *same*
            # handler; the handler's internal threading.RLock serialises
            # writes from the main thread and the listener thread safely.
            target_handler = _build_handler(log_file)
            # TaskFilter is placed on the *handler* so it runs for every
            # record regardless of which logger emitted it (child-logger
            # records propagate to the root's handlers without passing
            # through root-level filters).  The filter is non-destructive:
            # it only sets 'task' when the attribute is absent, so worker
            # records that already carry a 'task' attribute are untouched.
            target_handler.addFilter(TaskFilter())
            root.addHandler(target_handler)
            listener = QueueListener(queue, target_handler)
            listener.start()
        else:
            # Worker mode: send all logs through the queue
            queue_handler = QueueHandler(queue)
            queue_handler.addFilter(TaskFilter())
            root.addHandler(queue_handler)
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
