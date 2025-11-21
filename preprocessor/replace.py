from __future__ import annotations

from typing import Any, Dict, List

from preprocessor.models import Separators
from .logger import get_logger

logger = get_logger(__name__)


def resolve_key(data: Dict[str, Any], key: str) -> List[Dict[str, Any]] | Dict[str, Any] | Any:
	"""Resolve a JSON-path-like key against a nested dict/list structure.

	Supported features:
	- Dot operator for object members: a.b.c
	- Array indexing with brackets: a.b[0].c
	- Quoted keys to allow dots or brackets in keys: a."complex.key"['weird[0]']

	The function walks the `data` following the path and returns the value or
	raises KeyError/IndexError/TypeError when the path cannot be resolved.
	"""
	if key == "":
		return data

	i = 0
	n = len(key)
	current = data

	def read_identifier() -> str:
		nonlocal i
		start = i
		while i < n and key[i] not in ".[":
			i += 1
		return key[start:i]

	def read_quoted() -> str:
		nonlocal i
		quote = key[i]
		i += 1  # skip opening quote
		parts: List[str] = []
		while i < n:
			ch = key[i]
			if ch == "\\":
				# escape next char
				if i + 1 < n:
					parts.append(key[i + 1])
					i += 2
					continue
			if ch == quote:
				i += 1
				break
			parts.append(ch)
			i += 1
		return "".join(parts)

	while i < n:
		if key[i] == ".":
			i += 1
			continue

		# read next token: either quoted key, identifier, or bracketed index/key
		if key[i] in ('"', "'"):
			token = read_quoted()
			# access mapping with token
			if not isinstance(current, dict):
				logger.error(f"Cannot access key '{token}' on non-dict {type(current).__name__}") # type: ignore
				raise TypeError(f"Cannot access key '{token}' on non-dict {type(current).__name__}") # type: ignore
			if token not in current:
				logger.error(f"Key '{token}' not found in dict")
				raise KeyError(token)
			current = current[token] # type: ignore
			continue

		if key[i] == "[":
			# bracket expression: could be an index or a quoted key
			i += 1
			if i < n and key[i] in ('"', "'"):
				token = read_quoted()
				# expect closing ]
				if i >= n or key[i] != "]":
					logger.error("Missing closing bracket for quoted key")
					raise ValueError("Missing closing bracket for quoted key")
				i += 1
				if not isinstance(current, dict):
					logger.error(f"Cannot access key '{token}' on non-dict {type(current).__name__}") # type: ignore
					raise TypeError(f"Cannot access key '{token}' on non-dict {type(current).__name__}") # type: ignore
				if token not in current:
					logger.error(f"Key '{token}' not found in dict")
					raise KeyError(token)
				current = current[token] # type: ignore
			else:
				# read number (possibly negative)
				start = i
				while i < n and key[i] != "]":
					i += 1
				if i >= n:
					logger.error("Missing closing bracket for index")
					raise ValueError("Missing closing bracket for index")
				idx_str = key[start:i].strip()
				i += 1  # skip ]
				if idx_str == "":
					logger.error("Empty index in brackets")
					raise ValueError("Empty index in brackets")
				try:
					idx = int(idx_str)
				except ValueError:
					logger.error(f"Invalid list index: {idx_str}")
					raise ValueError(f"Invalid list index: {idx_str}")
				if not isinstance(current, (list, tuple)):
					logger.error(f"Cannot index non-list {type(current).__name__}") # type: ignore
					raise TypeError(f"Cannot index non-list {type(current).__name__}") # type: ignore
				try:
					current = current[idx] # type: ignore
				except IndexError:
					logger.error(f"List index out of range: {idx}")
					raise
			continue

		# identifier
		ident = read_identifier()
		if ident == "":
			# stray character
			logger.error(f"Unexpected token at position {i} in key: '{key}'")
			raise ValueError(f"Unexpected token at position {i} in key: '{key}'")
		if not isinstance(current, dict):
			logger.error(f"Cannot access key '{ident}' on non-dict {type(current).__name__}") # type: ignore
			raise TypeError(f"Cannot access key '{ident}' on non-dict {type(current).__name__}") # type: ignore
		if ident not in current:
			logger.error(f"Key '{ident}' not found in dict")
			raise KeyError(ident)
		current = current[ident] # type: ignore

	return current # type: ignore



def replace_placeholders(text: str, data: Dict[str, Any], seps: Separators) -> str:
	"""Replace all placeholders in the text with values from data.

	Placeholders are delimited by seps.placeholder_open and seps.placeholder_close.
	The keys inside placeholders are resolved using resolve_key against the data dict.

	Example:
	  If seps.placeholder_open is "{{" and seps.placeholder_close is "}}",
	  then a placeholder looks like "{{ key.subkey[0] }}".

	Args:
		text: The input text containing placeholders.
		data: The data dictionary to resolve keys against.
		seps: The Separators object defining placeholder delimiters.

	Returns:
		The text with all placeholders replaced by their corresponding values.
	"""
	open_tok = seps.placeholder_open
	close_tok = seps.placeholder_close
	result_parts: List[str] = []
	pos = 0
	n = len(text)

	while pos < n:
		start_idx = text.find(open_tok, pos)
		if start_idx == -1:
			# No more placeholders
			result_parts.append(text[pos:])
			break
		end_idx = text.find(close_tok, start_idx + len(open_tok))
		if end_idx == -1:
			# No closing token found; treat as error
			logger.error(f"Unclosed placeholder starting at position {start_idx}")
			raise ValueError(f"Unclosed placeholder starting at position {start_idx}")

		# Append text before placeholder
		result_parts.append(text[pos:start_idx])

		# Extract key inside placeholder
		key_start = start_idx + len(open_tok)
		key_end = end_idx
		key = text[key_start:key_end].strip()

		# Resolve key and convert to string
		try:
			value = resolve_key(data, key)
			replacement = str(value)
		except Exception:
			logger.error(f"Failed to resolve key '{key}' in placeholder")
			raise

		# Append replacement
		result_parts.append(replacement)

		# Move position past the closing token
		pos = end_idx + len(close_tok)

	return "".join(result_parts)