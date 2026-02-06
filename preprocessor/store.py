"""store.py

File I/O operations for the preprocessor.

This module handles all file system operations including:
- Writing generated output files to disk
- Creating output directories
- Joining multiple files into a single output
- Cleaning up intermediate files

All operations include comprehensive logging for debugging and support
configurable file encoding (default: UTF-8).

Functions:
    ensure_dir: Create a directory if it doesn't exist.
    write_file: Write content to a single file.
    write_files: Write multiple files from an iterable.
    join_files: Concatenate multiple files into one with separators.
    delete_files: Remove multiple files from disk.

Example:
    >>> from preprocessor.store import write_file, join_files, delete_files
    >>> write_file("output/model.cfg", "// Generated model")
    >>> files = ["a.cfg", "b.cfg", "c.cfg"]
    >>> join_files(files, "output/combined.cfg")
    >>> delete_files(files)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from .logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
	"""Create a directory and all parent directories if they don't exist.
	
	Works like `mkdir -p` on Unix systems. If the directory already exists,
	this function does nothing.
	
	Args:
		path: Path to the directory to create.
	
	Returns:
		Path object for the directory.
	
	Example:
		>>> ensure_dir("output/models/v2")
		PosixPath('output/models/v2')
	"""
	p = Path(path)
	logger.debug(f"Ensuring directory exists: {p}")
	if not p.exists():
		logger.info(f"Creating directory: {p}")
		p.mkdir(parents=True, exist_ok=True)
	else:
		logger.debug(f"Directory already exists: {p}")
	return p


def write_file(path: str | Path, content: str, encoding: str = "utf-8") -> None:
	"""Write string content to a file, creating parent directories as needed.
	
	Args:
		path: Destination file path.
		content: String content to write.
		encoding: File encoding (default: utf-8).
	
	Example:
		>>> write_file("output/model.prism", "module main { }")
	"""
	logger.info(f"Writing file: {path}")
	p = Path(path)
	logger.debug(f"Resolved path: {p.resolve()}")
	ensure_dir(p.parent)
	logger.debug(f"Writing {len(content)} characters to file")
	with p.open("w", encoding=encoding) as fh:
		fh.write(content)
	logger.info(f"Wrote file: {p} (bytes={len(content.encode(encoding))})")


def write_files(files: Iterable[tuple[str | Path, str]]) -> None:
	"""Write multiple files from an iterable of (path, content) tuples.
	
	Convenience function for batch file writing.
	
	Args:
		files: Iterable of (file_path, content) tuples.
	
	Example:
		>>> files = [("a.txt", "content a"), ("b.txt", "content b")]
		>>> write_files(files)
	"""
	logger.debug("Writing multiple files")
	count = 0
	for path, content in files:
		write_file(path, content)
		count += 1
	logger.debug(f"Wrote {count} files")


def join_files(
	file_paths: Iterable[str | Path],
	output_path: str | Path,
	encoding: str = "utf-8",
) -> Path:
	"""Join multiple files into a single file with comment separators.
	
	Concatenates all source files into one output file. Each source file's
	content is preceded by a comment line indicating its origin. Files are
	processed one at a time for memory efficiency.
	
	Output Format:
		// ========== Source: file1.cfg ==========
		<content of file1>
		
		// ========== Source: file2.cfg ==========
		<content of file2>
		...
	
	Args:
		file_paths: Iterable of source file paths to join.
		output_path: Destination path for the joined file.
		encoding: File encoding (default: utf-8).
	
	Returns:
		Path to the written joined file.
	
	Note:
		Files that don't exist are skipped with a warning logged.
	
	Example:
		>>> files = ["node1.cfg", "node2.cfg", "node3.cfg"]
		>>> join_files(files, "output/model.cfg")
		PosixPath('output/model.cfg')
	"""
	out = Path(output_path)
	logger.debug(f"Output path resolved to: {out.resolve()}")
	ensure_dir(out.parent)
	
	logger.info(f"Joining files into: {out}")
	file_count = 0
	total_bytes = 0
	
	with out.open("w", encoding=encoding) as out_fh:
		for file_path in file_paths:
			p = Path(file_path)
			logger.debug(f"Processing source file: {p}")
			if not p.exists():
				logger.warning(f"File not found, skipping: {p}")
				continue
			
			# Write comment header identifying source file
			out_fh.write(f"// ========== Source: {p.name} ==========\n")
			
			# Stream content line by line for memory efficiency
			file_bytes = 0
			with p.open("r", encoding=encoding) as in_fh:
				for line in in_fh:
					out_fh.write(line)
					file_bytes += len(line)
			
			logger.debug(f"Appended {file_bytes} bytes from {p.name}")
			total_bytes += file_bytes
			
			# Add blank lines between files for readability
			out_fh.write("\n\n")
			file_count += 1
	
	logger.debug(f"Total bytes written: {total_bytes}")
	logger.info(f"Joined {file_count}/{len(file_paths)} files into {out}")
	return out


def delete_files(paths: list[str | Path]) -> None:
	"""Delete multiple files using pop logic.
	
	Removes files from the filesystem, popping items from the provided list.
	After completion, the paths list will be empty. Files that don't exist
	or can't be deleted are logged as warnings.
	
	Note:
		The paths list is modified in place (emptied) as files are processed.
		This allows the caller to see which files remain if deletion is
		interrupted.
	
	Args:
		paths: List of file paths to delete. Modified in place (emptied).
	
	Example:
		>>> files = ["temp1.cfg", "temp2.cfg", "temp3.cfg"]
		>>> delete_files(files)
		>>> print(files)
		[]
	"""
	num_files = len(paths)
	logger.debug(f"Starting deletion of {num_files} files")
	deleted = 0
	
	# Pop files from list one at a time
	while paths:
		path = paths.pop()
		p = Path(path)
		logger.debug(f"Attempting to delete: {p}")
		if p.exists():
			try:
				p.unlink()
				deleted += 1
				logger.debug(f"Successfully deleted: {p}")
			except OSError as e:
				logger.warning(f"Failed to delete {p}: {e}")
		else:
			logger.warning(f"File not found for deletion: {p}")

	logger.info(f"Deleted {deleted}/{num_files} files")


