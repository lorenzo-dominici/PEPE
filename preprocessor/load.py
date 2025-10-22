"""load.py

Utilities to load JSON configuration/data files used by the preprocessor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .models import PreprocessorData, Config
from .logger import get_logger

logger = get_logger(__name__)


def load_json(path: str | Path) -> Dict[str, Any]:
	"""Load and return a JSON object from a file."""
	p = Path(path)
	if not p.exists():
		logger.error("JSON file not found: %s", p)
		raise FileNotFoundError(f"JSON file not found: {p}")
	with p.open("r", encoding="utf-8") as fh:
		return json.load(fh)


def load_data(path: str | Path) -> PreprocessorData:
	"""Load a preprocessor data JSON file and parse into PreprocessorData."""
	raw = load_json(path)
	logger.debug("Loaded data file: %s", path)
	return PreprocessorData.model_validate(raw)


def load_data_dir(path: str | Path) -> List[PreprocessorData]:
	"""Load all JSON files in a directory as PreprocessorData objects."""
	p = Path(path)
	if not p.is_dir():
		logger.error("Expected directory for data files: %s", p)
		raise NotADirectoryError(f"Expected a directory for data files: {p}")
	data_list: List[PreprocessorData] = []
	for json_file in p.glob("*.json"):
		raw = load_json(json_file)
		data = PreprocessorData.model_validate(raw)
		logger.debug("Loaded data from %s", json_file)
		data_list.append(data)
	return data_list


def load_config(path: str | Path) -> Config:
	"""Load and parse a preprocessor config file."""
	raw = load_json(path)
	logger.debug("Loaded config file: %s", path)
	return Config.model_validate(raw)


def load_template(path: str | Path) -> str:
	"""Load and return the content of a template file as a string."""
	p = Path(path)
	if not p.exists():
		logger.error("Template file not found: %s", p)
		raise FileNotFoundError(f"Template file not found: {p}")
	with p.open("r", encoding="utf-8") as fh:
		content = fh.read()
		logger.debug("Loaded template %s (size=%d bytes)", p, len(content))
		return content