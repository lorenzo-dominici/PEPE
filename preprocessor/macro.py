from __future__ import annotations
from typing import Any, Dict, List

from preprocessor.models import Separators
from preprocessor.replace import resolve_key
from preprocessor.logger import get_logger

logger = get_logger(__name__)

def resolve_macros(text: str, data: Dict[str, Any], seps: Separators) -> str:
    """Resolve a macro in the text using the provided data and separators."""
    finished = False
    stack: List[int] = []

    while not finished:
        open: int = text.find(str(seps.macro_open), 0 if len(stack) == 0 else stack[-1] + len(seps.macro_open))
        close: int = text.find(str(seps.macro_close), 0 if len(stack) == 0 else stack[-1] + len(seps.macro_open))
        match (open, close, len(stack)):
            case (-1, -1, 0):
                finished = True
            case (o, c, _) if o < c and o != -1:
                stack.append(o)
            case (o, c, n) if (c < o or (c != -1 and o == -1)) and n > 0:
                start = stack.pop()
                fields = get_macro_fields(start + len(seps.macro_open), close, text, seps)
                result: str
                match fields[0].strip():
                    case "match":
                        result = resolve_match(fields, data, seps)
                    case "loop":
                        result = resolve_loop(fields, data, seps)
                    case _:
                        logger.error(f"Unknown macro type '{fields[0].strip()}'")
                        raise ValueError(f"Unknown macro type '{fields[0].strip()}'")
                text = text[:start] + result + text[close + len(seps.macro_close):]
            case _:
                logger.error("Mismatched macro delimiters in template")
                raise ValueError("Mismatched macro delimiters in template")
    
    return text  

def get_macro_fields(start: int, end: int, text: str, seps: Separators) -> List[str]:
    """Extract fields from a macro span in the text using the provided separators."""
    return text[start:end].split(seps.macro_separator)
            
def resolve_match(fields: List[str], data: Dict[str, Any], seps: Separators) -> str:
    """Resolve a 'match' macro in the text using the provided data and separators."""
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
        if value == seps.match_default or match_value == value:
            return output
    
    logger.error(f"No matching branch found for key '{match_key}' with value '{match_value}'")
    raise ValueError(f"No matching branch found for key '{match_key}' with value '{match_value}'")

def resolve_loop(fields: List[str], data: Dict[str, Any], seps: Separators) -> str:
    """Resolve a 'loop' macro in the text using the provided data and separators."""
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
        result_parts.append(iter_body)
    
    return "".join(result_parts)