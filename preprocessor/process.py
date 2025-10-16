"""process.py

Process a single template file with provided instance data. This module
exposes a single `process_instance` function that returns the generated text
and a suggested output filename.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

from .replace import replace_placeholders
from .macro import resolve_macros


def _read_template(template_path: str | Path) -> str:
    p = Path(template_path)
    with p.open("r", encoding="utf-8") as fh:
        return fh.read()


def process_instance(config: Dict[str, Any],
                     template_path: str | Path, instance: Dict[str, Any],
                     consts: Dict[str, Any] | None = None) -> Tuple[str, str]:
    """Process one instance: load template, resolve macros, substitute placeholders.

    Returns a tuple (output_filename, generated_text). The suggested output
    filename is taken from `instance.get('name')` or falls back to the template
    basename with a suffix.
    """
    template = _read_template(template_path)
    data = dict(consts or {})
    data.update(instance or {})
    separators: Dict[str, str] = config.get("separators") or {}
    

    # If separators include 'open'/'close' those are used for placeholders
    ph_tokens = None
    if separators:
        open_tok = separators.get("placeholder_open") or separators.get("open")
        close_tok = separators.get("placeholder_close") or separators.get("close")
        if open_tok and close_tok:
            ph_tokens = (open_tok, close_tok)
        else:
            raise ValueError("Both 'placeholder_open' and 'placeholder_close' must be defined in separators")
    else:
        raise ValueError("No separators defined in config or data")

    # Resolve macros first (they may inject repeated blocks containing placeholders)
    with_macros = resolve_macros(template, data, separators)

    # Then replace placeholders
    generated = replace_placeholders(with_macros, data, ph_tokens)

    # Determine output filename
    out_name = instance.get("name")
    if not out_name:
        out_name = f"{Path(template_path).stem}_generated"
    out_filename = f"{out_name}.prism"
    return out_filename, generated
