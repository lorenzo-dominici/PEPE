"""main.py

Command-line entry point for the preprocessor.

Usage (simple):

  python -m preprocessor.main --data examples/preprocessor.json --out generated

The input JSON should have the following (minimal) shape:

{
  "consts": { ... },
  "items": [
    { "name": "id", "template": "templates/foo.pre", "instances": [ {...}, ... ] },
    ...
  ]
}

Each `item` defines a template and a list of instances. Each instance is a
mapping merged with `consts` and passed to the template processor.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .load import load_json, load_config
from .process import process_instance
from .store import write_file, ensure_dir


def _process_item(config: Dict[str, Any], template_base: str, consts: Dict[str, Any], instance: Dict[str, Any]):
    # Helper wrapper for Pool.map - calls process_instance and returns target path and content.
    out_name, content = process_instance(config, template_base, instance, consts,)
    return out_name, content


def run(config_path: str | Path, data_file: str | Path) -> List[Path]:
    data = load_json(data_file)
    config = load_config(config_path) if config_path else {}
    consts: Dict[str, Any] = data.get("consts") or {}
    items: List[Any] = data.get("items") or []

    out_dir = Path(config.get("output_dir") or "./")
    ensure_dir(out_dir)

    # Build a list of (template_path, name, instance) pairs
    tasks: List[Tuple[str, str, Any]] = []
    for item in items:
        template = item.get("template")
        name = item.get("name")
        if not template or not name:
            continue
        instances: List[Any] = item.get("instances") or []
        for inst in instances:
            tasks.append((template, name, inst))

    results: List[Path] = []
    # Use multiprocessing to process instances in parallel
    cpu_count = mp.cpu_count()
    jobs = config.get("jobs")
    pool_size = jobs if jobs and jobs > 0 else max(1, cpu_count - 1)
    with mp.Pool(pool_size) as pool:
        func = partial(_process_item, config=config, consts=consts)
        for out_name, content in pool.starmap(func, tasks):
            target = out_dir / out_name
            write_file(target, content)
            results.append(target)

    return results


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="preprocessor")
    ap.add_argument("--config", default=None, help="Path to preprocessor configuration file")
    ap.add_argument("--data", required=True, help="Path to preprocessor JSON data file or directory")
    args = ap.parse_args(argv)

    generated = run(args.config, args.data)
    for p in generated:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
