"""replace.py

Placeholder resolution and variable substitution for the preprocessor.

This module handles the second phase of template processing: replacing
placeholder tokens with actual values from the data dictionary. It supports
a JSON-path-like syntax for accessing nested data structures.

Key Path Syntax:
    - Dot notation: parent.child.grandchild
    - Array indexing: items[0], items[-1]
    - Quoted keys: data."key.with.dots"['key[with]brackets']
    - Mixed: users[0].address.street

Placeholder Format:
    Placeholders use configurable delimiters (typically {{ and }}).
    Example: {{user.name}} -> "John Doe"
             {{items[0].price}} -> "29.99"

Functions:
    resolve_key: Resolve a JSON-path key against nested data.
    replace_placeholders: Find and replace all placeholders in text.

Example:
    >>> from preprocessor.replace import replace_placeholders
    >>> from preprocessor.models import Separators
    >>> seps = Separators(placeholder_open="{{", placeholder_close="}}", ...)
    >>> data = {"user": {"name": "Alice", "scores": [95, 87, 92]}}
    >>> text = "Hello {{user.name}}, your first score was {{user.scores[0]}}."
    >>> replace_placeholders(text, data, seps)
    'Hello Alice, your first score was 95.'
"""
from __future__ import annotations

from typing import Any, Dict, List

from preprocessor.models import Separators
from .logger import get_logger

logger = get_logger(__name__)


def resolve_key(data: Dict[str, Any], key: str) -> List[Dict[str, Any]] | Dict[str, Any] | Any:
	"""Resolve a JSON-path-like key against a nested dict/list structure.
	
	Parses the key string and traverses the data structure step by step,
	following object member access and array indexing operations.
	
	Supported Path Syntax:
	    - Dot operator for object members: a.b.c
	    - Array indexing with brackets: a.b[0].c, items[-1]
	    - Quoted keys for special characters: a."complex.key"['weird[0]']
	    - Escape sequences in quoted keys: a."key\\"quote"
	
	Args:
		data: The root data dictionary to resolve against.
		key: The path string to resolve (e.g., "user.profile.name").
	
	Returns:
		The value at the specified path. Can be any type (dict, list,
		string, number, etc.).
	
	Raises:
		KeyError: If a dictionary key doesn't exist.
		IndexError: If a list index is out of range.
		TypeError: If trying to access a key on non-dict or index on non-list.
		ValueError: If the key syntax is invalid.
	
	Example:
		>>> data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
		>>> resolve_key(data, "users[0].name")
		'Alice'
		>>> resolve_key(data, "users[1].name")
		'Bob'
	"""
	logger.debug(f"Resolving key: '{key}'")
	if key == "":
		logger.debug("Empty key, returning entire data")
		return data

	i = 0
	n = len(key)
	current = data

	def read_identifier() -> str:
		"""Read an unquoted identifier (alphanumeric key name)."""
		nonlocal i
		start = i
		while i < n and key[i] not in ".[":
			i += 1
		return key[start:i]

	def read_quoted() -> str:
		"""Read a quoted string, handling escape sequences."""
		nonlocal i
		quote = key[i]
		i += 1  # skip opening quote
		parts: List[str] = []
		while i < n:
			ch = key[i]
			if ch == "\\":
				# Escape: include next character literally
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

	# Parse and traverse the key path
	while i < n:
		# Skip dot separators
		if key[i] == ".":
			i += 1
			continue

		# Handle quoted key (with single or double quotes)
		if key[i] in ('"', "'"):
			token = read_quoted()
			# Access mapping with quoted token
			if not isinstance(current, dict):
				logger.error(f"Cannot access key '{token}' on non-dict {type(current).__name__}") # type: ignore
				raise TypeError(f"Cannot access key '{token}' on non-dict {type(current).__name__}") # type: ignore
			if token not in current:
				logger.error(f"Key '{token}' not found in dict")
				raise KeyError(token)
			current = current[token] # type: ignore
			continue

		# Handle bracket expression (array index or quoted key)
		if key[i] == "[":
			i += 1
			if i < n and key[i] in ('"', "'"):
				# Quoted key inside brackets: obj["key"]
				token = read_quoted()
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
				# Numeric index: arr[0] or arr[-1]
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

		# Handle unquoted identifier
		ident = read_identifier()
		if ident == "":
			logger.error(f"Unexpected token at position {i} in key: '{key}'")
			raise ValueError(f"Unexpected token at position {i} in key: '{key}'")
		if not isinstance(current, dict):
			logger.error(f"Cannot access key '{ident}' on non-dict {type(current).__name__}") # type: ignore
			raise TypeError(f"Cannot access key '{ident}' on non-dict {type(current).__name__}") # type: ignore
		if ident not in current:
			logger.error(f"Key '{ident}' not found in dict")
			raise KeyError(ident)
		current = current[ident] # type: ignore

	logger.debug(f"Resolved key '{key}' to value of type {type(current).__name__}")

	return current


def replace_placeholders(text: str, data: Dict[str, Any], seps: Separators) -> str:
	"""Replace all placeholders in text with resolved values from data.
	
	Scans the text for placeholder delimiters, extracts the key path from
	each placeholder, resolves it against the data dictionary, and replaces
	the entire placeholder with the string representation of the value.
	
	Args:
		text: The input text containing placeholders.
		data: The data dictionary to resolve keys against.
		seps: Separator configuration with placeholder_open and placeholder_close.
	
	Returns:
		The text with all placeholders replaced by their resolved values.
	
	Raises:
		ValueError: If a placeholder is not properly closed.
		KeyError: If a placeholder key cannot be resolved.
		TypeError: If key path traversal encounters type mismatches.
	
	Example:
		>>> seps = Separators(placeholder_open="{{", placeholder_close="}}", ...)
		>>> data = {"name": "World", "count": 42}
		>>> text = "Hello, {{name}}! Count: {{count}}"
		>>> replace_placeholders(text, data, seps)
		'Hello, World! Count: 42'
	"""
	logger.debug(f"Replacing placeholders in text of length {len(text)}")
	open_tok = seps.placeholder_open
	close_tok = seps.placeholder_close
	logger.debug(f"Using delimiters: open='{open_tok}', close='{close_tok}'")
	result_parts: List[str] = []
	pos = 0
	n = len(text)
	placeholder_count = 0

	while pos < n:
		# Find next placeholder opening
		start_idx = text.find(open_tok, pos)
		if start_idx == -1:
			# No more placeholders, append remaining text
			result_parts.append(text[pos:])
			break
		
		# Find corresponding closing delimiter
		end_idx = text.find(close_tok, start_idx + len(open_tok))
		if end_idx == -1:
			logger.error(f"Unclosed placeholder starting at position {start_idx}")
			raise ValueError(f"Unclosed placeholder starting at position {start_idx}")

		# Append text before placeholder
		result_parts.append(text[pos:start_idx])

		# Extract and resolve key
		key_start = start_idx + len(open_tok)
		key_end = end_idx
		key = text[key_start:key_end].strip()
		logger.debug(f"Found placeholder with key: '{key}'")

		try:
			value = resolve_key(data, key)
			replacement = str(value)
			logger.debug(f"Placeholder '{key}' resolved to: '{replacement[:50]}{'...' if len(replacement) > 50 else ''}'")
		except Exception:
			logger.error(f"Failed to resolve key '{key}' in placeholder")
			raise

		# Append resolved value
		result_parts.append(replacement)
		placeholder_count += 1

		# Move past closing delimiter
		pos = end_idx + len(close_tok)

	logger.debug(f"Replaced {placeholder_count} placeholders")
	return "".join(result_parts)