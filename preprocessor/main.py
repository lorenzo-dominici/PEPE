"""main.py

Command-line entry point for the preprocessor.

This module orchestrates the complete preprocessing pipeline:
1. Parse command-line arguments
2. Load and validate configuration and data files
3. Distribute template processing across worker processes
4. Collect results and optionally join output files

The module uses multiprocessing for parallel template processing,
with progress tracking via tqdm and centralized logging via a queue.

Architecture:
    - Main process: Logs directly to file/console (no queue).
    - Worker processes: Log via QueueHandler -> mp.Queue -> QueueListener
      in the main process.
    - The QueueListener thread is started AFTER mp.Pool forks workers
      so that no thread locks are copied into children (Linux fork-safety).

Functions:
    _worker_task: Entry point for worker process task execution.
    _worker_init: Initialize logging in worker processes.
    run: Core preprocessing logic (can be called programmatically).
    main: CLI entry point with argument parsing.

Usage:
    python -m preprocessor --config config.json --data data.json
    python -m preprocessor --data data_dir/ --log-level DEBUG --log-file out.log

Exit Codes:
    0: Success
    1: Validation error in config or data files
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp

from logging.handlers import QueueListener
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from .load import load_data, load_data_dir, load_config
from .process import process_instance
from .store import write_file, ensure_dir, join_files, delete_files
from .models import Config, PreprocessorData, JoinMode
from .logger import (
    configure_logging,
    get_logger,
    task_context,
    set_task_label,
    _build_handler,
    TaskFilter,
)
from tqdm import tqdm


def _worker_task(args_tuple: tuple[Config, str, Dict[str, Any], Dict[str, Any]]):
    """Execute a single template processing task in a worker process.

    This function is the entry point called by each worker in the process pool.
    It unpacks the task arguments, sets up task context for logging, and
    delegates to process_instance for actual template processing.

    Args:
        args_tuple: Tuple of (config, template_path, consts, instance).

    Returns:
        Tuple of (output_filename, generated_content) from process_instance.
    """
    cfg, template_base, consts, instance = args_tuple
    task_label = f"{Path(template_base).stem}:{instance.get('name', 'unnamed')}"
    logger = get_logger("preprocessor.worker")
    logger.debug(f"Worker starting task: {task_label}")

    with task_context(task_label):
        result = process_instance(cfg, template_base, consts, instance)
        logger.debug(f"Worker completed task: {task_label}")
        return result


def _worker_init(log_level: str, log_file: str | None, log_queue: Any | None):
    """Initialize a worker process with logging configuration.

    Called once when each worker process starts. Sets up logging to send
    all log records to the main process via the provided queue.

    Args:
        log_level: Logging level string (DEBUG, INFO, etc.).
        log_file: Path to log file (unused here; actual output goes via queue).
        log_queue: Multiprocessing queue for sending logs to main process.
    """
    configure_logging(log_level, log_file, queue=log_queue)
    logger = get_logger("preprocessor.worker")
    logger.debug(f"Worker process initialized with log_level={log_level}")


def run(
    config_path: str | Path,
    data_path: str | Path,
    log_level: str = "INFO",
    log_file: str | None = None,
    log_queue: Any | None = None,
) -> List[Path]:
    """Execute the preprocessing pipeline.

    Loads configuration and data, builds a task list, processes templates
    in parallel using a forked process pool, and optionally joins output.

    IMPORTANT (Linux fork-safety):
        ``log_queue`` must be a freshly-created ``mp.Queue`` that has never
        had ``.put()`` called on it.  The first ``.put()`` spawns an internal
        feeder thread; if that thread exists before ``fork()`` its locks will
        be copied in a locked state into child processes -> deadlock.

        The main process therefore logs **directly** (not via the queue).
        Only forked workers use the queue.  A ``QueueListener`` is started
        **after** ``mp.Pool`` forks workers so no thread is present at fork.

    Args:
        config_path: Path to the preprocessor configuration JSON file.
        data_path: Path to data JSON file or directory of data files.
        log_level: Logging level for worker processes.
        log_file: Optional path for log file output.
        log_queue: Optional multiprocessing queue for centralized logging.

    Returns:
        List of Path objects for all generated output files.
    """
    set_task_label("main")
    logger = get_logger("preprocessor.main")
    logger.debug(f"run() called with config_path={config_path}, data_path={data_path}")
    logger.debug(f"log_level={log_level}, log_file={log_file}")

    # Load and validate data and config
    is_dir = Path(data_path).is_dir()
    logger.debug(f"Data path is directory: {is_dir}")
    input_data: List[PreprocessorData] = (
        load_data_dir(data_path) if is_dir else [load_data(data_path)]
    )
    logger.debug(f"Loaded {len(input_data)} data file(s)")

    config = load_config(config_path)
    logger.debug(
        f"Config loaded: output_dir={config.output_dir}, "
        f"jobs={config.jobs}, join_mode={config.join_mode}"
    )

    ensure_dir(config.output_dir)

    # Build task list
    tasks: List[Tuple[Config, str, Dict[str, Any], Dict[str, Any]]] = []
    for data in input_data:
        for item in data.items:
            for instance in item.instances:
                tasks.append((config, item.template, data.consts, instance))

    logger.debug(f"Built {len(tasks)} tasks for processing")

    results: List[Path] = []

    cpu_count = mp.cpu_count()
    pool_size = (
        config.jobs
        if config and config.jobs and config.jobs > 0
        else max(1, cpu_count - 1)
    )
    logger.info(f"Starting processing: {len(tasks)} tasks, pool size={pool_size}")

    # -- Fork-safe pool + listener lifecycle ----------------------------
    # 1. Create Pool  -> fork() happens here  (no threads yet = safe)
    # 2. Start QueueListener thread          (children already forked)
    # 3. Process tasks
    # 4. Close Pool
    # 5. Stop QueueListener
    # -------------------------------------------------------------------
    listener: QueueListener | None = None
    with mp.Pool(
        pool_size,
        initializer=_worker_init,
        initargs=(log_level, log_file, log_queue),
    ) as pool:

        # Now that workers are forked, start the listener thread safely.
        if log_queue is not None:
            target_handler = _build_handler(log_file)
            target_handler.addFilter(TaskFilter())
            listener = QueueListener(log_queue, target_handler)
            listener.start()

        logger.debug("Pool created, listener started")
        processed_count = 0

        for out_name, content in tqdm(
            pool.imap_unordered(_worker_task, tasks),
            total=len(tasks),
            desc="Processing",
            unit="item",
            disable=log_file is None,
        ):
            target = config.output_dir / out_name
            write_file(target, content)
            results.append(target)
            processed_count += 1
            logger.debug(f"Task {processed_count}/{len(tasks)} completed: {out_name}")

        logger.debug("All tasks completed, closing pool")

        # Gracefully shut down workers so their Queue feeder threads can
        # flush remaining log records to the pipe.  The context manager's
        # __exit__ calls pool.terminate() which SIGTERM-kills workers,
        # potentially leaving partial pickled objects in the pipe that
        # would make QueueListener.get() block forever.
        pool.close()
        pool.join()

    # Pool is closed/joined; stop the listener so it flushes remaining records.
    if listener is not None:
        listener.stop()

    # Post-processing: join files if configured
    logger.debug(f"Join mode: {config.join_mode}, results count: {len(results)}")
    if results and config.join_mode != JoinMode.none:
        joined_path = config.output_dir / config.joined_file
        logger.debug(f"Joining {len(results)} files into {joined_path}")
        join_files(results, joined_path)
        logger.info(f"Joined output written to: {joined_path}")

        if config.join_mode == JoinMode.clean_join:
            logger.debug(f"Clean join: deleting {len(results)} individual files")
            delete_files(results)

        results.append(joined_path)

    logger.info(f"Processing complete, generated {len(results)} output files")
    return results


def main(argv: List[str] | None = None) -> int:
    """Command-line interface entry point.

    Parses arguments, configures logging, and executes the pipeline.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    ap = argparse.ArgumentParser(prog="preprocessor")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args(argv)

    # -- Main-process logging (direct, no queue) ------------------------
    # The main process MUST NOT use a QueueHandler because mp.Queue.put()
    # spawns an internal feeder thread; that thread's locks would be
    # copied (in a held state) into forked children -> deadlock.
    # Instead the main process writes directly to file/console.
    # -------------------------------------------------------------------
    configure_logging(args.log_level, args.log_file)        # direct handler
    set_task_label("main")
    logger = get_logger("preprocessor.cli")
    logger.debug(f"CLI args: {vars(args)}")
    logger.debug(f"Python version: {__import__('sys').version}")
    logger.debug(f"Working directory: {Path.cwd()}")

    # Create the queue that workers will use.  It MUST remain empty (no
    # .put() calls) until after mp.Pool forks, to avoid spawning the
    # internal feeder thread before the fork.
    log_queue: Queue[Any] = mp.Queue()

    try:
        generated = run(
            args.config, args.data,
            args.log_level, args.log_file,
            log_queue=log_queue,
        )
    except ValidationError as exc:
        exc_any: Any = exc
        logger.error(f"Validation error: {exc_any}")
        raise SystemExit(f"Error parsing input or config: {exc_any}")

    print("Generated files:")
    for p in generated:
        print(" - ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
