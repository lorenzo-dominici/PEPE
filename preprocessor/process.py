"""process.py

Process a single template file with provided instance data. This module
exposes a single `process_instance` function that returns the generated text
and a suggested output filename.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

from preprocessor.load import load_template
from preprocessor.macro import resolve_template

from .replace import replace_placeholders
from .models import Config
from .logger import get_logger

logger = get_logger(__name__)


def process_instance(config: Config, template_path: str | Path, consts: Dict[str, Any], instance: Dict[str, Any]) -> Tuple[str, str]:
    """Process one instance: load template, resolve macros, substitute placeholders.

    Returns a tuple (output_filename, generated_text). The suggested output
    filename is taken from instance['name'] or falls back to the template
    basename with a suffix.
    """
    logger.debug("Processing instance for template=%s name=%s", template_path, instance.get("name"))
    template = load_template(template_path)
    data: Dict[str, Any] = dict(consts)
    data.update(instance)

    try:
        # Resolve macros first (they may inject repeated blocks containing placeholders)
        text = resolve_template(template, data, config.separators)
        # Then replace placeholders
        generated = replace_placeholders(text, data, config.separators)
    except Exception:
        logger.exception("Error processing instance %s", instance.get("name"))
        raise

    # Determine output filename
    out_name = data.get("name")

    if not out_name:
        logger.error("Instance missing 'name' key: %s", instance)
        raise ValueError("Instance data must contain a 'name' key for output filename generation.")
    
    out_filename = f"{out_name}.prism"
    logger.debug("Generated file %s (len=%d)", out_filename, len(generated))
    return out_filename, generated
