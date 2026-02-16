"""macro.py

Macro resolution engine for the preprocessor.

This module handles the parsing and evaluation of macros embedded in templates.
Macros are control structures that allow conditional content (match) and
repeated content (loop) based on input data.

Macro Syntax:
    Macros are delimited by configurable open/close tokens (e.g., {%...%}).
    Fields within macros are separated by a configurable separator (e.g., ||).

Supported Macro Types:
    match: Conditional branching based on a data value.
           Format: {%match||key||value1||output1||value2||output2||...||default||fallback%}
    
    loop:  Repeat content for each item in a list.
           Format: {%loop||list_key||body_template%}
           Use list_key[] in body to reference current item.

Features:
    - Nested macros: Macros can contain other macros
    - Lazy evaluation: Only selected branches are evaluated
    - Depth tracking: Proper handling of nested delimiters

Functions:
    find_matching_close: Find closing delimiter for a macro.
    find_outermost_macro: Locate the first top-level macro.
    resolve_macros: Main entry point for macro resolution.
    get_macro_fields_nested: Parse macro fields respecting nesting.
    resolve_match: Evaluate a match (conditional) macro.
    resolve_loop: Evaluate a loop (iteration) macro.

Example:
    >>> from preprocessor.macro import resolve_macros
    >>> from preprocessor.models import Separators
    >>> seps = Separators(macro_open="{%", macro_close="%}", ...)
    >>> text = "{%match||type||a||Type A||b||Type B||_||Unknown%}"
    >>> data = {"type": "a"}
    >>> resolve_macros(text, data, seps)
    'Type A'
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

from preprocessor.models import Separators
from preprocessor.replace import resolve_key
from preprocessor.logger import get_logger

logger = get_logger(__name__)


def find_matching_close(text: str, start: int, seps: Separators) -> int:
    """Find the matching close delimiter for a macro starting at 'start'.
    
    Uses depth tracking to handle nested macros correctly. Scans forward
    from the opening delimiter, incrementing depth on opens and decrementing
    on closes until depth reaches zero.
    
    Args:
        text: The full text containing macros.
        start: Index of the opening delimiter.
        seps: Separator configuration with macro_open and macro_close.
    
    Returns:
        Index of the matching close delimiter, or -1 if not found.
    """
    logger.debug(f"Finding matching close from position {start}")
    depth = 1
    pos = start + len(seps.macro_open)
    
    while depth > 0 and pos < len(text):
        next_open = text.find(str(seps.macro_open), pos)
        next_close = text.find(str(seps.macro_close), pos)
        
        if next_close == -1:
            logger.debug(f"No matching close found, depth={depth}")
            return -1  # No matching close found
        
        if next_open != -1 and next_open < next_close:
            # Found a nested open before the next close
            depth += 1
            pos = next_open + len(seps.macro_open)
        else:
            # Found a close
            depth -= 1
            if depth == 0:
                logger.debug(f"Found matching close at position {next_close}")
                return next_close
            pos = next_close + len(seps.macro_close)
    
    logger.debug(f"No matching close found after exhausting text")
    return -1


def find_outermost_macro(text: str, seps: Separators) -> Optional[Tuple[int, int]]:
    """Find the first outermost macro in the text.
    
    Locates the first macro opening delimiter and finds its matching
    closing delimiter using depth-aware scanning.
    
    Args:
        text: The text to search for macros.
        seps: Separator configuration.
    
    Returns:
        Tuple (start, end) with indices of open and close delimiters,
        or None if no macro is found.
    
    Raises:
        ValueError: If macro delimiters are mismatched.
    """
    start = text.find(str(seps.macro_open))
    if start == -1:
        logger.debug("No macro found in text")
        return None
    
    logger.debug(f"Found macro open at position {start}")
    end = find_matching_close(text, start, seps)
    if end == -1:
        logger.error("Mismatched macro delimiters in template")
        raise ValueError("Mismatched macro delimiters in template")
    
    logger.debug(f"Outermost macro spans positions {start} to {end}")
    return (start, end)


def resolve_macros(text: str, data: Dict[str, Any], seps: Separators) -> str:
    """Resolve all macros in the text using the provided data.
    
    Processes macros iteratively from outermost to innermost. Each macro
    is evaluated and replaced with its result before proceeding to the next.
    Nested macros in selected branches are resolved recursively.
    
    Args:
        text: Template text containing macros.
        data: Dictionary of data values for variable substitution.
        seps: Separator configuration defining macro syntax.
    
    Returns:
        Text with all macros resolved and replaced.
    
    Raises:
        ValueError: If an unknown macro type is encountered or delimiters mismatch.
    """
    logger.debug(f"Resolving macros in text of length {len(text)}")
    iteration = 0
    while True:
        macro_pos = find_outermost_macro(text, seps)
        if macro_pos is None:
            logger.debug(f"Macro resolution complete after {iteration} iterations")
            break
        
        iteration += 1
        start, end = macro_pos
        inner_content = text[start + len(seps.macro_open):end]
        logger.debug(f"Iteration {iteration}: processing macro at {start}-{end}")
        
        # Parse the macro type and fields, respecting nested macros
        fields = get_macro_fields_nested(inner_content, seps)
        macro_type = fields[0].strip()
        logger.debug(f"Macro type: '{macro_type}', field count: {len(fields)}")
        
        # Dispatch to appropriate resolver based on macro type
        result: str
        match macro_type:
            case "match":
                result = resolve_match(fields, data, seps)
            case "loop":
                result = resolve_loop(fields, data, seps)
            case _:
                logger.error(f"Unknown macro type '{macro_type}'")
                raise ValueError(f"Unknown macro type '{macro_type}'")
        
        logger.debug(f"Macro resolved to {len(result)} chars")
        text = text[:start] + result + text[end + len(seps.macro_close):]
    
    return text  


def get_macro_fields_nested(content: str, seps: Separators) -> List[str]:
    """Extract fields from macro content, respecting nested macros.
    
    Splits the macro content on the field separator, but only at depth 0
    (not inside nested macros). This ensures nested macro delimiters don't
    interfere with field parsing.
    
    Args:
        content: The inner content of a macro (between open/close delimiters).
        seps: Separator configuration.
    
    Returns:
        List of field strings extracted from the macro.
    
    Raises:
        ValueError: If there are unbalanced macro delimiters.
    """
    logger.debug(f"Extracting fields from macro content of length {len(content)}")
    fields: List[str] = []
    current_field_start = 0
    depth = 0
    pos = 0
    
    while pos < len(content):
        # Check for macro open - increase nesting depth
        if content[pos:pos + len(seps.macro_open)] == seps.macro_open:
            depth += 1
            pos += len(seps.macro_open)
            continue
        
        # Check for macro close - decrease nesting depth
        if content[pos:pos + len(seps.macro_close)] == seps.macro_close:
            depth -= 1
            if depth < 0:
                logger.error("Unexpected macro close delimiter found (no matching open)")
                raise ValueError("Unexpected macro close delimiter found (no matching open)")
            pos += len(seps.macro_close)
            continue
        
        # Check for separator at depth 0 - this is a field boundary
        if depth == 0 and content[pos:pos + len(seps.macro_separator)] == seps.macro_separator:
            fields.append(content[current_field_start:pos])
            current_field_start = pos + len(seps.macro_separator)
            pos += len(seps.macro_separator)
            continue
        
        pos += 1
    
    # Add the last field (after the last separator)
    fields.append(content[current_field_start:])
    
    logger.debug(f"Extracted {len(fields)} fields from macro content")
    return fields


def get_macro_fields(start: int, end: int, text: str, seps: Separators) -> List[str]:
    """Extract fields from a macro span using simple splitting.
    
    Note:
        This is kept for backward compatibility but get_macro_fields_nested
        should be preferred for proper nested macro handling.
    
    Args:
        start: Start index of the macro content.
        end: End index of the macro content.
        text: The full text.
        seps: Separator configuration.
    
    Returns:
        List of field strings (may incorrectly split nested macros).
    """
    return text[start:end].split(seps.macro_separator)


def resolve_match(fields: List[str], data: Dict[str, Any], seps: Separators) -> str:
    """Resolve a 'match' macro for conditional content selection.
    
    Match macros compare a data value against multiple branches and return
    the content of the first matching branch. Uses lazy evaluation - only
    the selected branch is processed for nested macros.
    
    Format: match||key||value1||output1||value2||output2||...[||default||fallback]
    
    Args:
        fields: Parsed macro fields [type, key, val1, out1, val2, out2, ...].
        data: Data dictionary for key resolution.
        seps: Separator configuration (includes match_default token).
    
    Returns:
        The output string from the matched branch.
    
    Raises:
        ValueError: If field count is wrong or no branch matches.
        TypeError: If the match key resolves to a non-primitive value.
    """
    logger.debug(f"Resolving match macro with {len(fields)} fields")
    if len(fields) % 2 != 0:
        logger.error("Match macro must have an even number of fields (pairs of value and output)")
        raise ValueError("Match macro must have an even number of fields (pairs of value and output)")
    
    match_key = fields[1].strip()
    logger.debug(f"Match key: '{match_key}'")
    match_value = resolve_key(data, match_key)

    if match_value is None:
        logger.error(f"Match key '{match_key}' not found in data")
        raise ValueError(f"Match key '{match_key}' not found in data")
    
    if isinstance(match_value, (dict, list)):
        logger.error(f"Match key '{match_key}' must resolve to a primitive value, got {type(match_value).__name__}") # type: ignore
        raise TypeError(f"Match key '{match_key}' must resolve to a primitive value, got {type(match_value).__name__}") # type: ignore
    
    # Build list of (value, output) branch pairs
    branches = [(fields[i].strip(), fields[i+1]) for i in range(2, len(fields), 2)]
    logger.debug(f"Match value resolved to: '{match_value}', checking {len(branches)} branches")

    # Find first matching branch or default
    for idx, (value, output) in enumerate(branches):
        if value == seps.match_default or str(match_value) == value:
            logger.debug(f"Match found at branch {idx}: value='{value}'")
            # Only resolve macros in the selected branch (lazy evaluation)
            return resolve_macros(output, data, seps)
    
    logger.error(f"No matching branch found for key '{match_key}' with value '{match_value}'")
    raise ValueError(f"No matching branch found for key '{match_key}' with value '{match_value}'")


def resolve_loop(fields: List[str], data: Dict[str, Any], seps: Separators) -> str:
    """Resolve a 'loop' macro for repeated content generation.
    
    Loop macros iterate over a list in the data and repeat the body template
    for each item. The placeholder list_key[] in the body is replaced with
    list_key[i] for each iteration index.
    
    Format: loop|list_key|body_template
    
    Args:
        fields: Parsed macro fields [type, list_key, body].
        data: Data dictionary containing the list to iterate.
        seps: Separator configuration.
    
    Returns:
        Concatenated output from all loop iterations.
    
    Raises:
        ValueError: If field count is wrong or list key not found.
        TypeError: If the list key doesn't resolve to a list.
    """
    logger.debug(f"Resolving loop macro with {len(fields)} fields")
    if len(fields) != 3:
        logger.error("Loop macro must have exactly three fields: 'loop', list_key, body_template")
        raise ValueError("Loop macro must have exactly three fields: 'loop', list_key, body_template")
    
    list_key: str = fields[1].strip()
    logger.debug(f"Loop list key: '{list_key}'")
    loop_list = resolve_key(data, list_key)

    if loop_list is None:
        logger.error(f"Loop key '{list_key}' not found in data")
        raise ValueError(f"Loop key '{list_key}' not found in data")
    
    if not isinstance(loop_list, list):
        logger.error(f"Loop key '{list_key}' must resolve to a list, got {type(loop_list).__name__}")
        raise TypeError(f"Loop key '{list_key}' must resolve to a list, got {type(loop_list).__name__}")

    body = fields[-1]
    logger.debug(f"Loop will iterate over {len(loop_list)} items")

    # Generate output for each iteration
    result_parts: List[str] = []
    for i in range(0, len(loop_list)): # type: ignore
        logger.debug(f"Loop iteration {i}/{len(loop_list)-1}")
        iter_body = str(body)
        # Replace generic index placeholder with specific index
        iter_body = iter_body.replace(f"{list_key}[]", f"{list_key}[{i}]")
        # Recursively resolve macros in each iteration's body
        iter_body = resolve_macros(iter_body, data, seps)
        result_parts.append(iter_body)
    
    logger.debug(f"Loop completed, generated {len(result_parts)} parts")
    return "".join(result_parts)