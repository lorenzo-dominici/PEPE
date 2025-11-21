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
	logger.info(f"Loading JSON file: {path}")
	p = Path(path)
	if not p.exists():
		logger.error(f"JSON file not found: {p}")
		raise FileNotFoundError(f"JSON file not found: {p}")
	with p.open("r", encoding="utf-8") as fh:
		return json.load(fh)


def load_data(path: str | Path) -> PreprocessorData:
	"""Load a preprocessor data JSON file and parse into PreprocessorData."""
	logger.info(f"Loading data file: {path}")
	raw = load_json(path)
	logger.info(f"Loaded data file: {path}")
	return PreprocessorData.model_validate(raw)


def load_data_dir(path: str | Path) -> List[PreprocessorData]:
	"""Load all JSON files in a directory as PreprocessorData objects."""
	logger.info(f"Loading data directory: {path}")
	p = Path(path)
	if not p.is_dir():
		logger.error(f"Expected directory for data files: {p}")
		raise NotADirectoryError(f"Expected a directory for data files: {p}")
	data_list: List[PreprocessorData] = []
	for json_file in p.glob("*.json"):
		raw = load_json(json_file)
		data = PreprocessorData.model_validate(raw)
		logger.info(f"Loaded data from {json_file}")
		data_list.append(data)
	return data_list


def load_config(path: str | Path) -> Config:
	"""Load and parse a preprocessor config file."""
	logger.info(f"Loading config file: {path}")
	raw = load_json(path)
	logger.info(f"Loaded config file: {path}")
	return Config.model_validate(raw)


def load_template(path: str | Path) -> str:
	"""Load and return the content of a template file as a string."""
	logger.info(f"Loading template file: {path}")
	p = Path(path)
	if not p.exists():
		logger.error(f"Template file not found: {p}")
		raise FileNotFoundError(f"Template file not found: {p}")
	with p.open("r", encoding="utf-8") as fh:
		content = fh.read()
		logger.info(f"Loaded template {p} (size={len(content)} bytes)")
		return content