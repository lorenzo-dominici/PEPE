"""process.py

Process a single template file with provided instance data.

This module is the core of template processing. It takes a template file
and instance data, then applies the two-phase transformation:
1. Macro resolution: Evaluate control structures (match, loop)
2. Placeholder replacement: Substitute variable references with values

The module exposes a single `process_instance` function designed to be
called by worker processes in the parallel processing pool.

Processing Pipeline:
    1. Load template file content
    2. Merge constants with instance data (instance takes precedence)
    3. Resolve all macros (conditionals, loops)
    4. Replace all placeholders with data values
    5. Return generated content with output filename

Functions:
    process_instance: Main entry point for template processing.

Example:
    >>> from preprocessor.process import process_instance
    >>> from preprocessor.models import Config
    >>> config = Config.model_validate({...})
    >>> filename, content = process_instance(
    ...     config,
    ...     "templates/node.pre",
    ...     {"version": "1.0"},
    ...     {"name": "node_1.cfg", "ip": "10.0.0.1"}
    ... )
    >>> print(f"Generated {filename}: {len(content)} bytes")
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

from preprocessor.load import load_template
from preprocessor.macro import resolve_macros

from .replace import replace_placeholders
from .models import Config
from .logger import get_logger

logger = get_logger(__name__)


def process_instance(
    config: Config,
    template_path: str | Path,
    consts: Dict[str, Any],
    instance: Dict[str, Any]
) -> Tuple[str, str]:
    """Process one template instance through macro resolution and placeholder substitution.
    
    This function performs the complete template-to-output transformation for
    a single instance. It's designed to be called from worker processes and
    is fully self-contained (no shared state).
    
    Processing Steps:
        1. Load the template file from disk
        2. Merge consts and instance data (instance values override consts)
        3. Resolve macros: Evaluate match/loop control structures
        4. Replace placeholders: Substitute {{key}} with data values
        5. Generate output filename from instance 'name' field
    
    Args:
        config: Preprocessor configuration with separator definitions.
        template_path: Path to the template file.
        consts: Shared constant values available to all instances.
        instance: Instance-specific data (must include 'name' key).
    
    Returns:
        Tuple of (output_filename, generated_text):
        - output_filename: Suggested filename based on instance name
        - generated_text: Fully processed template content
    
    Raises:
        FileNotFoundError: If template file doesn't exist.
        ValueError: If instance is missing required 'name' key.
        ValueError: If macro syntax errors occur during resolution.
        KeyError: If placeholder references missing data keys.
    
    Example:
        >>> filename, content = process_instance(
        ...     config,
        ...     "node.pre",
        ...     {"max_nodes": 10},
        ...     {"name": "router.cfg", "type": "gateway"}
        ... )
        >>> # Returns: ("router.cfg", "<processed content>")
    """
    instance_name = instance.get('name', 'unnamed')
    logger.info(f"Processing instance name={instance_name} of template={template_path}")
    logger.debug(f"Instance data keys: {list(instance.keys())}")
    logger.debug(f"Consts keys: {list(consts.keys())}")
    
    # Step 1: Load template content
    template = load_template(template_path)
    logger.debug(f"Template loaded, size={len(template)} bytes")
    
    # Step 2: Merge consts and instance data (instance overrides consts)
    data: Dict[str, Any] = dict(consts)
    data.update(instance)
    logger.debug(f"Merged data has {len(data)} keys")

    try:
        # Step 3: Resolve macros first (they may inject repeated blocks containing placeholders)
        logger.debug(f"Starting macro resolution for instance '{instance_name}'")
        text = resolve_macros(template, data, config.separators)
        logger.debug(f"Macro resolution complete, text size={len(text)} bytes")
        
        # Step 4: Replace placeholders with data values
        logger.debug(f"Starting placeholder replacement for instance '{instance_name}'")
        generated = replace_placeholders(text, data, config.separators)
        logger.debug(f"Placeholder replacement complete, generated size={len(generated)} bytes")
    except Exception as e:
        logger.error(f"Failed to process instance {instance_name}: {type(e).__name__}: {e}")
        raise

    # Step 5: Determine output filename from instance name
    out_name = data.get("name")

    if not out_name:
        logger.error(f"Instance missing 'name' key: {instance}")
        raise ValueError("Instance data must contain a 'name' key for output filename generation.")
    
    out_filename = f"{out_name}"
    logger.debug(f"Generated output filename: {out_filename}")
    logger.info(f"Processed instance {instance_name} of {template_path}")
    return out_filename, generated
