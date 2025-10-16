"""store.py

Functions to write generated PRISM files to disk and manage output paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def write_file(path: str | Path, content: str, encoding: str = "utf-8") -> None:
	p = Path(path)
	ensure_dir(p.parent)
	with p.open("w", encoding=encoding) as fh:
		fh.write(content)


def write_files(files: Iterable[tuple[str | Path, str]]) -> None:
	for path, content in files:
		write_file(path, content)
