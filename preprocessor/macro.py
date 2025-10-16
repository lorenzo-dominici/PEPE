
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

from .replace import replace_placeholders


def _compile_macro_regex(open_tok: str, sep_tok: str, close_tok: str, default_tok: str) -> Tuple[re.Pattern[str], re.Pattern[str]]:
    """Compile regexes for loop and match macros using provided tokens.

    Returns (loop_re, match_re). Regexes are case-insensitive for the keywords
    'loop' and 'match'
    """
    open = re.escape(open_tok)
    sep = re.escape(sep_tok)
    close = re.escape(close_tok)
    default = re.escape(default_tok)
    loop_pat = rf"(?<loop>{open}\s*loop\s*{sep}(?<iterator>[^{sep}]*){sep}(?<content>[^{sep}]*(?:{open}[\s\S]*?{close}[^{sep}]*)*){close})"
    match_pat = rf"(?<match>{open}\s*match\s*{sep}(?<key>[^{sep}]*)(?<branch>{sep}(?<value>(?(?!\s*{default}\s*)[^{sep}]*|\s*{default}[^{sep}]*[^\s{sep}][^{sep}]*)){sep}(?<content>[^{sep}]*(?:{open}(?:[\s\S]*?)*{close}[^{sep}]*)*))+(?<default>{sep}{default}{sep}(?<default_content>[^{sep}]*))?{close})"
    return re.compile(loop_pat, re.S | re.I), re.compile(match_pat, re.S | re.I)


def _resolve_path(mapping: Dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    current = mapping
    for p in parts:
        if isinstance(current, Mapping) and p in current:
            current = current[p]
        else:
            return None
    return current


def _find_first_macro(text: str, loop_re: re.Pattern[str], match_re: re.Pattern[str]) -> Optional[Tuple[str, re.Match[str]]]:
    """Return the first-occurring macro match as (kind, match) or None."""
    m_loop = loop_re.search(text)
    m_match = match_re.search(text)
    candidates: List[Tuple[str, re.Match[str]]] = []
    if m_loop:
        candidates.append(("loop", m_loop))
    if m_match:
        candidates.append(("match", m_match))
    if not candidates:
        return None
    # choose the match with smallest start index
    candidates.sort(key=lambda kv: kv[1].start())
    return candidates[0]


def _process_loop_at(match: re.Match[str], data: Dict[str, Any], loop_re: re.Pattern[str], match_re: re.Pattern[str], placeholder_tokens: Tuple[str, str]) -> str:
    array_path = match.group("iterator")
    block = match.group("content")
    lst: List[Dict[str, Any] | Any] | str= _resolve_path(data, array_path)
    if not isinstance(lst, list):
        raise ValueError(f"Loop macro expects a list at path '{array_path}', got: {lst}")
    parts: List[str] = []
    for item in lst:
        # recursively process macros inside the block
        processed_block = _process_macros_in_text(block, data, loop_re, match_re, placeholder_tokens)
        # now that nested macros are resolved, replace placeholders in this block using local
        processed_block = replace_placeholders(processed_block, item, placeholder_tokens)
        parts.append(processed_block)
    return "".join(parts)


def _process_match_at(match: re.Match[str], data: Dict[str, Any], loop_re: re.Pattern[str], match_re: re.Pattern[str], sep_tok: str, placeholder_tokens: Tuple[str, str]) -> str:
    keypath = match.group("key")
    rest = match.group("branches")
    keyval = _resolve_path(data, keypath)
    chosen: Optional[str] = None
    default = ""
    parts = rest.split(sep_tok)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            if k == "default":
                default = v
            else:
                if str(k) == str(keyval):
                    chosen = v
                    break
        else:
            if str(part) == str(keyval):
                chosen = part
                break
    replacement = chosen if chosen is not None else default
    # recursively resolve macros inside the chosen replacement
    processed = _process_macros_in_text(replacement, data, loop_re, match_re, placeholder_tokens)
    # replace placeholders in processed text using outer data
    processed = replace_placeholders(processed, data, placeholder_tokens)
    return processed


def _process_macros_in_text(text: str, data: Dict[str, Any], loop_re: re.Pattern[str], match_re: re.Pattern[str], placeholder_tokens: Tuple[str, str]) -> str:
    """Process macros in text following the requested workflow.

    - find first macro; if none, return text unchanged
    - otherwise resolve that macro; for its inner block, recursively process nested macros
      before replacing placeholders inside the block
    - substitute the processed macro content into the outer text and continue
    """
    cur = text
    while True:
        found = _find_first_macro(cur, loop_re, match_re)
        if not found:
            return cur
        kind, m = found
        if kind == "loop":
            replacement = _process_loop_at(m, data, loop_re, match_re, placeholder_tokens)
        else:
            replacement = _process_match_at(m, data, loop_re, match_re, match_re.pattern, placeholder_tokens)
        cur = cur[: m.start()] + replacement + cur[m.end():]


def resolve_macros(text: str, data: Dict[str, Any], separators: Dict[str, str]) -> str:
    """Resolve macros then placeholders according to the specified workflow.

    separators: optional dict with keys 'open', 'sep', 'close', and optional
    'placeholder_open'/'placeholder_close'. Defaults: '^', '|', '$', placeholders '{{','}}'.
    """
    open_tok = separators.get("macro_open")
    sep_tok = separators.get("macro_separator")
    close_tok = separators.get("macro_close")
    default_tok = separators.get("macro_default")
    ph_open = separators.get("placeholder_open")
    ph_close = separators.get("placeholder_close")
    
    if not (open_tok and sep_tok and close_tok and default_tok and ph_open and ph_close):
        raise ValueError("All macro and placeholder separators must be provided.")

    loop_re, match_re = _compile_macro_regex(open_tok, sep_tok, close_tok, default_tok)

    # Process macros top-level, resolving nested macros as described
    processed = _process_macros_in_text(text, data, loop_re, match_re, (ph_open, ph_close))

    # Finally, replace any placeholders left outside macros using the global data
    processed = replace_placeholders(processed, data, (ph_open, ph_close))
    return processed
