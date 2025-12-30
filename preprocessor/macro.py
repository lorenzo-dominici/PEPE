from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

from preprocessor.models import Separators
from preprocessor.replace import resolve_key
from preprocessor.logger import get_logger

logger = get_logger(__name__)

def find_matching_close(text: str, start: int, seps: Separators) -> int:
    """Find the matching close delimiter for a macro starting at 'start'.
    
    Returns the index of the matching close delimiter, or -1 if not found.
    """
    depth = 1
    pos = start + len(seps.macro_open)
    
    while depth > 0 and pos < len(text):
        next_open = text.find(str(seps.macro_open), pos)
        next_close = text.find(str(seps.macro_close), pos)
        
        if next_close == -1:
            return -1  # No matching close found
        
        if next_open != -1 and next_open < next_close:
            depth += 1
            pos = next_open + len(seps.macro_open)
        else:
            depth -= 1
            if depth == 0:
                return next_close
            pos = next_close + len(seps.macro_close)
    
    return -1

def find_outermost_macro(text: str, seps: Separators) -> Optional[Tuple[int, int]]:
    """Find the first outermost macro in the text.
    
    Returns a tuple (start, end) where start is the index of the open delimiter
    and end is the index of the close delimiter, or None if no macro found.
    """
    start = text.find(str(seps.macro_open))
    if start == -1:
        return None
    
    end = find_matching_close(text, start, seps)
    if end == -1:
        logger.error("Mismatched macro delimiters in template")
        raise ValueError("Mismatched macro delimiters in template")
    
    return (start, end)

def resolve_macros(text: str, data: Dict[str, Any], seps: Separators) -> str:
    """Resolve macros in the text using the provided data and separators.
    
    Processes outermost macros first, then recursively resolves nested macros
    only in the parts that need evaluation (e.g., selected match branches).
    """
    while True:
        macro_pos = find_outermost_macro(text, seps)
        if macro_pos is None:
            break
        
        start, end = macro_pos
        inner_content = text[start + len(seps.macro_open):end]
        
        # Parse the macro type and fields, respecting nested macros
        fields = get_macro_fields_nested(inner_content, seps)
        
        result: str
        match fields[0].strip():
            case "match":
                result = resolve_match(fields, data, seps)
            case "loop":
                result = resolve_loop(fields, data, seps)
            case _:
                logger.error(f"Unknown macro type '{fields[0].strip()}'")
                raise ValueError(f"Unknown macro type '{fields[0].strip()}'")
        
        text = text[:start] + result + text[end + len(seps.macro_close):]
    
    return text  

def get_macro_fields_nested(content: str, seps: Separators) -> List[str]:
    """Extract fields from macro content, respecting nested macros.
    
    Only splits on separators that are at depth 0 (not inside nested macros).
    """
    fields: List[str] = []
    current_field_start = 0
    depth = 0
    pos = 0
    
    while pos < len(content):
        # Check for macro open
        if content[pos:pos + len(seps.macro_open)] == seps.macro_open:
            depth += 1
            pos += len(seps.macro_open)
            continue
        
        # Check for macro close
        if content[pos:pos + len(seps.macro_close)] == seps.macro_close:
            depth -= 1
            if depth < 0:
                logger.error("Unexpected macro close delimiter found (no matching open)")
                raise ValueError("Unexpected macro close delimiter found (no matching open)")
            pos += len(seps.macro_close)
            continue
        
        # Check for separator at depth 0
        if depth == 0 and content[pos:pos + len(seps.macro_separator)] == seps.macro_separator:
            fields.append(content[current_field_start:pos])
            current_field_start = pos + len(seps.macro_separator)
            pos += len(seps.macro_separator)
            continue
        
        pos += 1
    
    # Add the last field
    fields.append(content[current_field_start:])
    
    return fields

def get_macro_fields(start: int, end: int, text: str, seps: Separators) -> List[str]:
    """Extract fields from a macro span in the text using the provided separators.
    
    Note: This is kept for backward compatibility but get_macro_fields_nested
    should be preferred for proper nested macro handling.
    """
    return text[start:end].split(seps.macro_separator)
            
def resolve_match(fields: List[str], data: Dict[str, Any], seps: Separators) -> str:
    """Resolve a 'match' macro in the text using the provided data and separators.
    
    Only recursively resolves macros in the selected branch.
    """
    if len(fields) % 2 != 0:
        logger.error("Match macro must have an even number of fields (pairs of value and output)")
        raise ValueError("Match macro must have an even number of fields (pairs of value and output)")
    
    match_key = fields[1].strip()
    match_value = resolve_key(data, match_key)

    if match_value is None:
        logger.error(f"Match key '{match_key}' not found in data")
        raise ValueError(f"Match key '{match_key}' not found in data")
    
    if isinstance(match_value, (dict, list)):
        logger.error(f"Match key '{match_key}' must resolve to a primitive value, got {type(match_value).__name__}") # type: ignore
        raise TypeError(f"Match key '{match_key}' must resolve to a primitive value, got {type(match_value).__name__}") # type: ignore
    
    branches = [(fields[i].strip(), fields[i+1]) for i in range(2, len(fields), 2)]

    for value, output in branches:
        if value == seps.match_default or str(match_value) == value:
            # Only resolve macros in the selected branch
            return resolve_macros(output, data, seps)
    
    logger.error(f"No matching branch found for key '{match_key}' with value '{match_value}'")
    raise ValueError(f"No matching branch found for key '{match_key}' with value '{match_value}'")

def resolve_loop(fields: List[str], data: Dict[str, Any], seps: Separators) -> str:
    """Resolve a 'loop' macro in the text using the provided data and separators.
    
    Expands the loop first, then recursively resolves macros in the expanded content.
    """
    if len(fields) != 3:
        logger.error("Loop macro must have exactly three fields: 'loop', list_key, body_template")
        raise ValueError("Loop macro must have exactly three fields: 'loop', list_key, body_template")
    
    list_key: str = fields[1].strip()
    loop_list = resolve_key(data, list_key)

    if loop_list is None:
        logger.error(f"Loop key '{list_key}' not found in data")
        raise ValueError(f"Loop key '{list_key}' not found in data")
    
    if not isinstance(loop_list, list):
        logger.error(f"Loop key '{list_key}' must resolve to a list, got {type(loop_list).__name__}")
        raise TypeError(f"Loop key '{list_key}' must resolve to a list, got {type(loop_list).__name__}")

    body = fields[-1]

    result_parts: List[str] = []
    for i in range(0, len(loop_list)): # type: ignore
        iter_body = str(body)
        iter_body = iter_body.replace(f"{list_key}[]", f"{list_key}[{i}]")
        # Recursively resolve macros in each iteration's body
        iter_body = resolve_macros(iter_body, data, seps)
        result_parts.append(iter_body)
    
    return "".join(result_parts)