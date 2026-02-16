"""
store.py — Persistence layer for serialising generated network data to disk.

Provides a single utility function that writes an arbitrary Python dictionary
(typically the network descriptor or the sessions summary) to a JSON file,
creating any missing parent directories on the fly.

Output format
-------------
* Encoding : UTF-8
* Indentation : 4 spaces (human-readable, diff-friendly)
* Non-ASCII policy : characters are preserved as-is (ensure_ascii=False)
* Write mode : always overwrite — no append/merge semantics
"""

import json
from pathlib import Path


def store_network(network: dict, file_path: str) -> None:
    """Serialise *network* as pretty-printed JSON and write it to *file_path*.

    Parameters
    ----------
    network : dict
        The data structure to persist.  Accepted shapes include the full
        network descriptor produced by ``NetworkGenerator._generate_json()``
        and the sessions summary from ``_generate_sessions_summary()``.
    file_path : str
        Absolute or relative path for the target file.  Intermediate
        directories are created automatically when they do not exist.

    Notes
    -----
    * The call is intentionally idempotent: running it twice with the same
      arguments simply overwrites the file with identical content.
    * ``Path.write_text`` opens, writes and closes the file atomically,
      avoiding the need for an explicit context manager.
    """
    path = Path(file_path)
    # Ensure the full directory tree exists before writing
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    # Serialise and write — always overwrite the target file
    path.write_text(
        json.dumps(network, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )