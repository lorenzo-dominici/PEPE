"""store.py

Functions to write generated PRISM files to disk and manage output paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from .logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
	p = Path(path)
	if not p.exists():
		logger.info(f"Creating directory: {p}")
		p.mkdir(parents=True, exist_ok=True)
	return p


def write_file(path: str | Path, content: str, encoding: str = "utf-8") -> None:
	logger.info(f"Writing file: {path}")
	p = Path(path)
	ensure_dir(p.parent)
	with p.open("w", encoding=encoding) as fh:
		fh.write(content)
	logger.info(f"Wrote file: {p} (bytes={len(content.encode(encoding))})")


def write_files(files: Iterable[tuple[str | Path, str]]) -> None:
	for path, content in files:
		write_file(path, content)


def join_to_prism(
	file_paths: Iterable[str | Path],
	output_path: str | Path,
	encoding: str = "utf-8",
) -> Path:
	"""Join multiple files into a single .prism file with comment separators.
	
	Reads and appends files one at a time for memory efficiency with large file counts.
	Each file's content is preceded by a comment indicating the source filename.
	
	Args:
		file_paths: Iterable of file paths to join.
		output_path: Path for the joined .prism output file.
		encoding: File encoding (default: utf-8).
	
	Returns:
		Path to the written joined file.
	"""
	out = Path(output_path)
	ensure_dir(out.parent)
	
	logger.info(f"Joining files into: {out}")
	file_count = 0
	
	with out.open("w", encoding=encoding) as out_fh:
		for file_path in file_paths:
			p = Path(file_path)
			if not p.exists():
				logger.warning(f"File not found, skipping: {p}")
				continue
			
			# Write comment header with source filename
			out_fh.write(f"// ========== Source: {p.name} ==========\n")
			
			# Read and write content in one pass
			with p.open("r", encoding=encoding) as in_fh:
				for line in in_fh:
					out_fh.write(line)
			
			# Ensure newline separator between files
			out_fh.write("\n\n")
			file_count += 1
	
	logger.info(f"Joined {file_count} files into {out}")
	return out


def delete_files(paths: Iterable[str | Path]) -> int:
	"""Delete multiple files.
	
	Args:
		paths: Iterable of file paths to delete.
	
	Returns:
		Number of files successfully deleted.
	"""
	deleted = 0
	for path in paths:
		p = Path(path)
		if p.exists():
			try:
				p.unlink()
				deleted += 1
			except OSError as e:
				logger.warning(f"Failed to delete {p}: {e}")
		else:
			logger.warning(f"File not found for deletion: {p}")
	logger.info(f"Deleted {deleted} files")
	return deleted

