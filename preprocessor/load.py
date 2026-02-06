"""load.py

Utilities to load JSON configuration/data files used by the preprocessor.

This module provides functions to load and validate:
- JSON data files containing template instances
- Configuration files with preprocessor settings
- Template files for text generation

All loading functions include comprehensive logging for debugging and
use Pydantic models for validation.

Functions:
    load_json: Load raw JSON from a file path.
    load_data: Load and validate a preprocessor data file.
    load_data_dir: Load all data files from a directory.
    load_config: Load and validate a preprocessor config file.
    load_template: Load a template file as a string.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .models import PreprocessorData, Config
from .logger import get_logger

logger = get_logger(__name__)


def load_json(path: str | Path) -> Dict[str, Any]:
	"""Load and return a JSON object from a file.
	
	Args:
		path: Path to the JSON file (string or Path object).
	
	Returns:
		Parsed JSON data as a dictionary.
	
	Raises:
		FileNotFoundError: If the file does not exist.
		json.JSONDecodeError: If the file contains invalid JSON.
	"""
	logger.info(f"Loading JSON file: {path}")
	p = Path(path)
	logger.debug(f"Resolved path: {p.resolve()}")
	if not p.exists():
		logger.error(f"JSON file not found: {p}")
		raise FileNotFoundError(f"JSON file not found: {p}")
	logger.debug(f"File size: {p.stat().st_size} bytes")
	with p.open("r", encoding="utf-8") as fh:
		data = json.load(fh)
		logger.debug(f"Parsed JSON with {len(data)} top-level keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
		return data


def load_data(path: str | Path) -> PreprocessorData:
	"""Load a preprocessor data JSON file and parse into PreprocessorData.
	
	Args:
		path: Path to the data JSON file.
	
	Returns:
		Validated PreprocessorData model instance.
	
	Raises:
		FileNotFoundError: If the file does not exist.
		pydantic.ValidationError: If the JSON structure is invalid.
	"""
	logger.info(f"Loading data file: {path}")
	raw = load_json(path)
	logger.debug(f"Validating PreprocessorData model from raw data")
	data = PreprocessorData.model_validate(raw)
	logger.debug(f"Validated data: {len(data.items)} items, {len(data.consts)} consts")
	logger.info(f"Loaded data file: {path}")
	return data


def load_data_dir(path: str | Path) -> List[PreprocessorData]:
	"""Load all JSON files in a directory as PreprocessorData objects.
	
	Scans the directory for *.json files and loads each as a PreprocessorData
	instance. Files are processed in glob order (not guaranteed to be sorted).
	
	Args:
		path: Path to the directory containing data JSON files.
	
	Returns:
		List of validated PreprocessorData model instances.
	
	Raises:
		NotADirectoryError: If the path is not a directory.
		pydantic.ValidationError: If any JSON structure is invalid.
	"""
	logger.info(f"Loading data directory: {path}")
	p = Path(path)
	logger.debug(f"Resolved directory path: {p.resolve()}")
	if not p.is_dir():
		logger.error(f"Expected directory for data files: {p}")
		raise NotADirectoryError(f"Expected a directory for data files: {p}")
	json_files = list(p.glob("*.json"))
	logger.debug(f"Found {len(json_files)} JSON files in directory")
	data_list: List[PreprocessorData] = []
	for json_file in json_files:
		logger.debug(f"Processing file: {json_file.name}")
		raw = load_json(json_file)
		data = PreprocessorData.model_validate(raw)
		logger.info(f"Loaded data from {json_file}")
		data_list.append(data)
	logger.debug(f"Loaded {len(data_list)} PreprocessorData objects from directory")
	return data_list


def load_config(path: str | Path) -> Config:
	"""Load and parse a preprocessor config file.
	
	The config file specifies output directory, parallelism, separator
	tokens, join mode, and other preprocessor settings.
	
	Args:
		path: Path to the configuration JSON file.
	
	Returns:
		Validated Config model instance.
	
	Raises:
		FileNotFoundError: If the file does not exist.
		pydantic.ValidationError: If the JSON structure is invalid.
	"""
	logger.info(f"Loading config file: {path}")
	raw = load_json(path)
	logger.debug(f"Validating Config model from raw data")
	config = Config.model_validate(raw)
	logger.debug(f"Config: output_dir={config.output_dir}, jobs={config.jobs}, join_mode={config.join_mode}")
	logger.debug(f"Separators: macro_open='{config.separators.macro_open}', placeholder_open='{config.separators.placeholder_open}'")
	logger.info(f"Loaded config file: {path}")
	return config


def load_template(path: str | Path) -> str:
	"""Load and return the content of a template file as a string.
	
	Template files contain text with placeholders and macros that will
	be resolved during preprocessing.
	
	Args:
		path: Path to the template file.
	
	Returns:
		The raw template content as a string.
	
	Raises:
		FileNotFoundError: If the template file does not exist.
	"""
	logger.info(f"Loading template file: {path}")
	p = Path(path)
	logger.debug(f"Resolved template path: {p.resolve()}")
	if not p.exists():
		logger.error(f"Template file not found: {p}")
		raise FileNotFoundError(f"Template file not found: {p}")
	logger.debug(f"Template file size: {p.stat().st_size} bytes")
	with p.open("r", encoding="utf-8") as fh:
		content = fh.read()
		logger.debug(f"Template contains {content.count(chr(10))} lines")
		logger.info(f"Loaded template {p} (size={len(content)} bytes)")
		return content