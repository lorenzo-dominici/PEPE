"""models.py

Pydantic models for preprocessor configuration and data validation.

This module defines the data structures used throughout the preprocessor:
- Configuration settings (output paths, parallelism, syntax tokens)
- Input data format (templates, instances, constants)
- Separator/delimiter definitions for macro and placeholder syntax

All models use Pydantic for automatic validation, ensuring that configuration
and data files conform to expected schemas before processing begins.

Classes:
    JoinMode: Enum for output file joining behavior.
    Item: A template with its list of instances to generate.
    PreprocessorData: Top-level data structure with constants and items.
    Separators: Token definitions for macro and placeholder syntax.
    Config: Complete preprocessor configuration.

Example Config JSON:
    {
        "output_dir": "./output",
        "jobs": 4,
        "join_mode": "clean_join",
        "joined_file": "model.cfg",
        "separators": {
            "macro_open": "{%",
            "macro_close": "%}",
            "macro_separator": "||",
            "match_default": "_",
            "placeholder_open": "{{",
            "placeholder_close": "}}"
        }
    }

Example Data JSON:
    {
        "consts": {"version": "1.0"},
        "items": [
            {
                "name": "nodes",
                "template": "templates/node.pre",
                "instances": [
                    {"name": "node_1.cfg", "type": "server"},
                    {"name": "node_2.cfg", "type": "client"}
                ]
            }
        ]
    }
"""
from __future__ import annotations
from enum import Enum
from pathlib import Path

from pydantic import BaseModel
from typing import Any, Dict, List


class JoinMode(str, Enum):
    """Controls how generated files are joined after processing.
    
    This enum determines the post-processing behavior for output files.
    
    Attributes:
        none: Generate all files individually, no joining.
        join: Generate all files and also create a joined .prism file.
        clean_join: Generate files, join them, then delete individual files.
    """
    none = "none"
    join = "join"
    clean_join = "clean_join"


class Item(BaseModel):
    """A template definition with instances to generate.
    
    Each Item represents a template file and a list of data instances.
    The preprocessor generates one output file per instance.
    
    Attributes:
        name: Descriptive name for this item group.
        template: Path to the template file (.pre format).
        instances: List of data dictionaries, each generating one output file.
                   Each instance must have a 'name' key for output filename.
    
    Example:
        {
            "name": "network_nodes",
            "template": "templates/node.pre",
            "instances": [
                {"name": "router_1.cfg", "ip": "10.0.0.1"},
                {"name": "router_2.cfg", "ip": "10.0.0.2"}
            ]
        }
    """
    name: str
    template: str
    instances: List[Dict[str, Any]]


class PreprocessorData(BaseModel):
    """Top-level data structure for preprocessor input.
    
    Contains shared constants and a list of items to process.
    Constants are merged with instance data, with instance values
    taking precedence on conflicts.
    
    Attributes:
        consts: Shared key-value pairs available to all instances.
        items: List of Item definitions to process.
    
    Example:
        {
            "consts": {
                "version": "2.0",
                "author": "QESM Team"
            },
            "items": [...]
        }
    """
    consts: Dict[str, Any] = {}
    items: List[Item]


class Separators(BaseModel):
    """Token definitions for macro and placeholder syntax.
    
    Defines the delimiter strings used to identify macros and placeholders
    in template files. These are configurable to avoid conflicts with
    template content.
    
    Attributes:
        macro_open: Opening delimiter for macros (e.g., "{%").
        macro_separator: Field separator within macros (e.g., "||").
        macro_close: Closing delimiter for macros (e.g., "%}").
        match_default: Token for default/fallback branch in match macros (e.g., "_").
        placeholder_open: Opening delimiter for placeholders (e.g., "{{").
        placeholder_close: Closing delimiter for placeholders (e.g., "}}").
    
    Example:
        Macro syntax:    {%match||key||val1||out1||_||default%}
        Placeholder:     {{variable.path}}
    """
    macro_open: str
    macro_separator: str
    macro_close: str
    match_default: str
    placeholder_open: str
    placeholder_close: str


class Config(BaseModel):
    """Complete preprocessor configuration.
    
    Defines all settings for a preprocessing run, including output
    destination, parallelism, syntax tokens, and file joining behavior.
    
    Attributes:
        output_dir: Directory path for generated output files.
        jobs: Number of parallel worker processes (0 = auto).
        separators: Syntax token definitions.
        join_mode: How to handle output file joining (default: none).
        joined_file: Filename for joined output (default: joined.prism).
    
    Example:
        {
            "output_dir": "./generated",
            "jobs": 8,
            "separators": {...},
            "join_mode": "clean_join",
            "joined_file": "complete_model.prism"
        }
    """
    output_dir: Path
    jobs: int
    separators: Separators
    join_mode: JoinMode = JoinMode.none
    joined_file: Path = Path("joined.prism")


__all__ = ["Item", "PreprocessorData", "Config", "JoinMode", "Separators"]
