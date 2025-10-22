from __future__ import annotations
from pathlib import Path

from pydantic import BaseModel
from typing import Any, Dict, List

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


__all__ = ["Item", "PreprocessorData", "Config"]
