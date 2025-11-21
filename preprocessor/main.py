"""main.py

Command-line entry point for the preprocessor.

This module wires together the loader, processor and store. It uses the
Pydantic models in `models.py` to validate input data and config files.
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
from .store import write_file, ensure_dir
from .models import Config, PreprocessorData
from .logger import configure_logging, get_logger, task_context, set_task_label
from tqdm import tqdm

# Use imap_unordered so we can update progress as results come in.
    # On Windows multiprocessing, top-level callables are safer; provide a small wrapper that
    # accepts a single tuple per task.
def _worker_task(args_tuple: tuple[Config, str, Dict[str, Any], Dict[str, Any]]):
    # args_tuple is (config, template_base, consts, instance)
    cfg, template_base, consts, instance = args_tuple
    task_label = f"{Path(template_base).stem}:{instance.get('name', 'unnamed')}"
    with task_context(task_label):
        return process_instance(cfg, template_base, consts, instance)

def _worker_init(log_level: str, log_file: str | None, log_queue: Any | None):
    configure_logging(log_level, log_file, queue=log_queue)

def run(
    config_path: str | Path,
    data_path: str | Path,
    log_level: str = "INFO",
    log_file: str | None = None,
    log_queue: Any | None = None,
) -> List[Path]:
    set_task_label("main")
    # Load and validate data and config using the loader helpers
    input: List[PreprocessorData] = load_data_dir(data_path) if Path(data_path).is_dir() else [load_data(data_path)]
    config = load_config(config_path)

    ensure_dir(config.output_dir)

    # Build a list of (template_path, instance) pairs where instance is a mapping
    tasks: List[Tuple[Config, str, Dict[str, Any], Dict[str, Any]]] = []
    for data in input:
        for item in data.items:
            for instance in item.instances:
                # include config in the task tuple so workers can be simple top-level callables
                tasks.append((config, item.template, data.consts, instance))

    results: List[Path] = []
    # Use multiprocessing to process instances in parallel and show progress
    cpu_count = mp.cpu_count()
    pool_size = (config.jobs if config and config.jobs and config.jobs > 0 else max(1, cpu_count - 1))
    logger = get_logger("preprocessor.main")
    logger.info(f"Starting processing: {len(tasks)} tasks, pool size={pool_size}")

    with mp.Pool(
        pool_size,
        initializer=_worker_init,
        initargs=(log_level, log_file, log_queue),
    ) as pool:
        for out_name, content in tqdm(pool.imap_unordered(_worker_task, tasks), total=len(tasks), desc="Processing", unit="item"):
            target = config.output_dir / out_name
            write_file(target, content)
            results.append(target)

    logger.info(f"Processing complete, generated {len(results)} files")
    return results


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="preprocessor")
    ap.add_argument("--config", default=None, help="Path to preprocessor configuration file")
    ap.add_argument("--data", required=True, help="Path to preprocessor JSON data file or directory")
    ap.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    ap.add_argument("--log-file", default=None, help="Path for log output; defaults to stderr")
    args = ap.parse_args(argv)

    log_queue: Queue[Any] = mp.Queue()
    listener = configure_logging(args.log_level, args.log_file, queue=log_queue, start_listener=True)
    set_task_label("main")
    logger = get_logger("preprocessor.cli")
    logger.debug(f"CLI args: {args}")

    try:
        generated = run(args.config, args.data, args.log_level, args.log_file, log_queue=log_queue)
    except ValidationError as exc:
        exc_any: Any = exc
        raise SystemExit(f"Error parsing input or config: {exc_any}")
    finally:
        if listener:
            listener.stop()

    for p in generated:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
