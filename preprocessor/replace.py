"""replace.py

Placeholder replacement utilities with configurable delimiters.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Dict, Any, Tuple


def _build_placeholder_re(open_tok: str, close_tok: str) -> re.Pattern[str]:
    open = re.escape(open_tok)
    close = re.escape(close_tok)
    pattern =  rf"{open}[^({open})({close})]*{close}"
    return re.compile(pattern)


def replace_placeholders(text: str, mapping: Dict[str, Any], tokens: Tuple[str, str]) -> str:
	"""Replace placeholders using mapping.

	tokens: tuple (open_tok, close_tok).
	If a key is not found in mapping, the placeholder is left unchanged.
	"""
	open_tok, close_tok = tokens
	placeholder_re = _build_placeholder_re(open_tok, close_tok)

	def _lookup(key: str) -> str:
		parts = key.split(".")
		current = mapping
		try:
			for part in parts:
				if isinstance(current, Mapping):
					current = current[part]  # may raise KeyError
				else:
					return f"{open_tok}{key}{close_tok}"
			return str(current)
		except (KeyError, TypeError):
			return f"{open_tok}{key}{close_tok}"

	return placeholder_re.sub(lambda m: _lookup(m.group(1)), text)
