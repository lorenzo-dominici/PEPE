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
    - Main process: Loads config/data, manages worker pool, writes output
    - Worker processes: Execute template processing (macros + placeholders)
    - Logging: All workers send logs to main process via queue

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
import multiprocessing as mp
 
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from .load import load_data, load_data_dir, load_config
from .process import process_instance
from .store import write_file, ensure_dir, join_files, delete_files
from .models import Config, PreprocessorData, JoinMode
from .logger import configure_logging, get_logger, task_context, set_task_label
from tqdm import tqdm


def _worker_task(args_tuple: tuple[Config, str, Dict[str, Any], Dict[str, Any]]):
    """Execute a single template processing task in a worker process.
    
    This function is the entry point called by each worker in the process pool.
    It unpacks the task arguments, sets up task context for logging, and
    delegates to process_instance for actual template processing.
    
    Note:
        On Windows, multiprocessing requires top-level callables.
        This wrapper provides a simple interface accepting a single tuple.
    
    Args:
        args_tuple: Tuple of (config, template_path, consts, instance).
    
    Returns:
        Tuple of (output_filename, generated_content) from process_instance.
    """
    # Unpack task arguments
    cfg, template_base, consts, instance = args_tuple
    task_label = f"{Path(template_base).stem}:{instance.get('name', 'unnamed')}"
    logger = get_logger("preprocessor.worker")
    logger.debug(f"Worker starting task: {task_label}")
    
    # Process within task context for proper log labeling
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
        log_file: Path to log file (for reference, actual output goes to queue).
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
    
    This is the main entry point for programmatic use. It loads configuration
    and data, builds a task list, processes templates in parallel, and
    optionally joins the output files.
    
    Args:
        config_path: Path to the preprocessor configuration JSON file.
        data_path: Path to data JSON file or directory of data files.
        log_level: Logging level for worker processes.
        log_file: Optional path for log file output.
        log_queue: Optional multiprocessing queue for centralized logging.
    
    Returns:
        List of Path objects for all generated output files.
    
    Raises:
        FileNotFoundError: If config or data files don't exist.
        pydantic.ValidationError: If config or data validation fails.
    
    Example:
        >>> from preprocessor.main import run
        >>> files = run("config.json", "data.json")
        >>> print(f"Generated {len(files)} files")
    """
    set_task_label("main")
    logger = get_logger("preprocessor.main")
    logger.debug(f"run() called with config_path={config_path}, data_path={data_path}")
    logger.debug(f"log_level={log_level}, log_file={log_file}")
    
    # Load and validate data and config using the loader helpers
    is_dir = Path(data_path).is_dir()
    logger.debug(f"Data path is directory: {is_dir}")
    input: List[PreprocessorData] = load_data_dir(data_path) if is_dir else [load_data(data_path)]
    logger.debug(f"Loaded {len(input)} data file(s)")
    
    config = load_config(config_path)
    logger.debug(f"Config loaded: output_dir={config.output_dir}, jobs={config.jobs}, join_mode={config.join_mode}")

    ensure_dir(config.output_dir)
    logger.debug(f"Output directory ensured: {config.output_dir}")

    # Build a list of (config, template_path, consts, instance) tuples for parallel processing
    tasks: List[Tuple[Config, str, Dict[str, Any], Dict[str, Any]]] = []
    for data in input:
        logger.debug(f"Processing data with {len(data.items)} items")
        for item in data.items:
            logger.debug(f"Item '{item.name}': template={item.template}, {len(item.instances)} instances")
            for instance in item.instances:
                # Include config in the task tuple so workers can be simple top-level callables
                tasks.append((config, item.template, data.consts, instance))
    
    logger.debug(f"Built {len(tasks)} tasks for processing")

    results: List[Path] = []
    
    # Calculate pool size: use config.jobs if set, otherwise CPU count - 1
    cpu_count = mp.cpu_count()
    pool_size = (config.jobs if config and config.jobs and config.jobs > 0 else max(1, cpu_count - 1))
    logger.debug(f"CPU count: {cpu_count}, calculated pool_size: {pool_size}")
    logger.info(f"Starting processing: {len(tasks)} tasks, pool size={pool_size}")

    # Process tasks in parallel using a process pool
    with mp.Pool(
        pool_size,
        initializer=_worker_init,
        initargs=(log_level, log_file, log_queue),
    ) as pool:
        logger.debug("Multiprocessing pool created")
        processed_count = 0
        
        # Use imap_unordered for efficient streaming of results with progress bar
        # Progress bar is disabled when logging to file to avoid output conflicts
        for out_name, content in tqdm(pool.imap_unordered(_worker_task, tasks), total=len(tasks), desc="Processing", unit="item", disable=log_file is None):
            target = config.output_dir / out_name
            write_file(target, content)
            results.append(target)
            processed_count += 1
            logger.debug(f"Task {processed_count}/{len(tasks)} completed: {out_name}")
        logger.debug("All tasks completed, closing pool")

    # Post-processing: join files if configured
    logger.debug(f"Join mode: {config.join_mode}, results count: {len(results)}")
    if results and config.join_mode != JoinMode.none:
        joined_path = config.output_dir / config.joined_file
        logger.debug(f"Joining {len(results)} files into {joined_path}")
        join_files(results, joined_path)
        logger.info(f"Joined output written to: {joined_path}")
        
        # In clean_join mode, delete individual files after joining
        if config.join_mode == JoinMode.clean_join:
            logger.debug(f"Clean join mode: deleting {len(results)} individual files")
            delete_files(results)

        results.append(joined_path)

    logger.info(f"Processing complete, generated {len(results)} output files")
    return results


def main(argv: List[str] | None = None) -> int:
    """Command-line interface entry point.
    
    Parses command-line arguments, sets up logging, and executes the
    preprocessing pipeline. Handles errors gracefully with appropriate
    exit codes.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv if None).
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    
    Command-line Arguments:
        --config: Path to preprocessor configuration file.
        --data: Path to data JSON file or directory (required).
        --log-level: Logging level (DEBUG, INFO, WARNING, ERROR).
        --log-file: Path for log output; defaults to stderr.
    """
    # Set up argument parser
    ap = argparse.ArgumentParser(prog="preprocessor")
    ap.add_argument("--config", required=True, help="Path to preprocessor configuration file")
    ap.add_argument("--data", required=True, help="Path to preprocessor JSON data file or directory")
    ap.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    ap.add_argument("--log-file", default=None, help="Path for log output; defaults to stderr")
    args = ap.parse_args(argv)

    # Set up centralized logging with queue for multiprocessing
    log_queue: Queue[Any] = mp.Queue()
    listener = configure_logging(args.log_level, args.log_file, queue=log_queue, start_listener=True)
    set_task_label("main")
    logger = get_logger("preprocessor.cli")
    logger.debug(f"CLI args: config={args.config}, data={args.data}, log_level={args.log_level}, log_file={args.log_file}")
    logger.debug(f"Python version: {__import__('sys').version}")
    logger.debug(f"Working directory: {Path.cwd()}")

    try:
        logger.debug("Starting preprocessor run")
        generated = run(args.config, args.data, args.log_level, args.log_file, log_queue=log_queue)
        logger.debug(f"Preprocessor run completed, generated {len(generated)} files")
    except ValidationError as exc:
        exc_any: Any = exc
        logger.error(f"Validation error: {exc_any}")
        raise SystemExit(f"Error parsing input or config: {exc_any}")
    finally:
        # Always stop the log listener to flush remaining messages
        logger.debug("Stopping log listener")
        if listener:
            listener.stop()

    # Print summary of generated files
    print("Generated files:")
    for p in generated:
        print(" - ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
