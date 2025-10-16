"""load.py

Utilities to load JSON configuration/data files used by the preprocessor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: str | Path) -> Dict[str, Any]:
	"""Load and return a JSON object from a file.

	Raises FileNotFoundError or json.JSONDecodeError on failure.
	"""
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"JSON file not found: {p}")
	with p.open("r", encoding="utf-8") as fh:
		return json.load(fh)


def load_config(path: str | Path) -> Dict[str, Any]:
	"""Alias for load_json for clarity when reading preprocessor config files."""
	return load_json(path)
