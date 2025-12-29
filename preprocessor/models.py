from __future__ import annotations
from enum import Enum
from pathlib import Path

from pydantic import BaseModel
from typing import Any, Dict, List


class JoinMode(str, Enum):
    """Controls how generated files are joined.
    
    - none: Generate all files but don't join them.
    - join: Generate all files and join them into a single .prism file.
    - clean_join: Generate all files, join them, then remove individual files.
    """
    none = "none"
    join = "join"
    clean_join = "clean_join"

class Item(BaseModel):
    name: str
    template: str
    instances: List[Dict[str, Any]]

class PreprocessorData(BaseModel):
    consts: Dict[str, Any] = {}
    items: List[Item]

class Separators(BaseModel):
    macro_open: str
    macro_separator: str
    macro_close: str
    match_default: str
    placeholder_open: str
    placeholder_close: str

class Config(BaseModel):
    output_dir: Path
    jobs: int
    separators: Separators
    join_mode: JoinMode = JoinMode.none
    joined_file: Path = Path("joined.prism")


__all__ = ["Item", "PreprocessorData", "Config", "JoinMode"]
