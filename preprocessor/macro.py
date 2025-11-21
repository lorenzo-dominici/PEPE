from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterator, Tuple

from preprocessor.models import Separators
from preprocessor.replace import resolve_key
from preprocessor.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Macro:
    """A simple dataclass representing a macro span."""
    type: str
    start: int
    end: int

    def contains(self, pos: int) -> bool:
        return self.start <= pos < self.end

    def encloses(self, other: Macro) -> bool:
        return self.start <= other.start and other.end <= self.end
    
    def __repr__(self) -> str:
        return f"Macro(type={self.type!r}, start={self.start}, end={self.end})"

class MacroNode:
    """
    Represents one macro span in the text.
    - type: string identifier of macro (e.g. "if", "include", "expr", ...)
    - start: inclusive start index in the text
    - end: exclusive end index in the text
    - children: nested macro nodes
    - parent: optional parent node (set when added to a tree)
    """
    macro: Macro
    children: List[MacroNode]
    parent: Optional[MacroNode]

    def __init__(self, macro: Macro):
        self.macro = macro
        self.children = []
        self.parent = None

    @property
    def type(self) -> str:
        return self.macro.type

    @property
    def start(self) -> int:
        return self.macro.start
    
    @property
    def end(self) -> int:
        return self.macro.end
    
    def contains(self, pos: int) -> bool:
        return self.macro.contains(pos)
    
    def encloses(self, other: MacroNode) -> bool:
        return self.macro.encloses(other.macro)

    def add_child(self, child: MacroNode) -> None:
        """Add child and set its parent. No automatic reordering/validation."""
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: MacroNode) -> None:
        self.children.remove(child)
        child.parent = None

    def traverse(self) -> Iterator[MacroNode]:
        """Yield self and all descendants in preorder."""
        yield self
        for c in self.children:
            yield from c.traverse()

    def find_deepest_containing(self, pos: int) -> Optional[MacroNode]:
        """Return the deepest descendant node that contains pos, or None."""
        if not self.macro.contains(pos):
            return None
        for c in self.children:
            found = c.find_deepest_containing(pos)
            if found is not None:
                return found
        return self

    def __repr__(self) -> str:
        return f"MacroNode(type={self.macro.type!r}, start={self.macro.start}, end={self.macro.end})"


class MacroTree:
    """
    A tree of MacroNode objects. The tree has a single root node that
    represents the entire document (type='root').
    """

    def __init__(self, doc_length: int = 0):
        self.root = MacroNode(Macro("root", 0, max(0, doc_length)))

    def populate(self, text: str, separators: Separators):
        """Parse the text and populate the tree with macro nodes."""
        open_s = separators.macro_open
        sep_s = separators.macro_separator
        close_s = separators.macro_close

        # nothing to do if separators missing or empty
        if not open_s or not close_s or not sep_s:
            logger.error("Macro separators must be defined and non-empty")
            raise ValueError("Macro separators must be defined and non-empty")

        text_len = len(text)
        # ensure root covers the whole text
        self.root.macro.start = 0
        self.root.macro.end = text_len

        # collect open/close events in text order
        events: List[Tuple[str, int]] = []
        offset = 0
        while (pos := text.find(open_s, offset)) != -1:
            events.append(("open", pos))
            offset = pos + len(open_s)

        offset = 0
        while (pos := text.find(close_s, offset)) != -1:
            events.append(("close", pos))
            offset = pos + len(close_s)

        # sort by position to process tokens as they appear
        events.sort(key=lambda e: e[1])

        # match opens and closes using a stack; build spans
        stack: List[int] = []
        spans: List[Macro] = []
        for typ, pos in events:
            if typ == "open":
                stack.append(pos)
                logger.debug(f"MACRO {text[pos:pos+10]}")
            else:  # close
                if not stack:
                    logger.error(f"Closing separator at position {pos} without matching open")
                    raise ValueError(f"Closing separator at position {pos} without matching open")
                start = stack.pop()
                end = pos + len(close_s)
                if end <= start:
                    logger.error(f"Invalid macro span for open at {start} and close at {pos}")
                    raise ValueError(f"Invalid macro span for open at {start} and close at {pos}")
                # derive a simple type name from the inner content (first token) or default to "macro"
                inner = text[start + len(open_s) : end - len(close_s)]
                mtype = inner.split(sep_s)[0].strip()
                spans.append(Macro(mtype, start, end))

        if stack:
            logger.error("Unclosed macro opens at positions: " + ", ".join(map(str, stack)))
            raise ValueError("Unclosed macro open(s) at positions: " + ", ".join(map(str, stack)))

        # build nodes and insert into tree ensuring proper nesting (no overlaps)
        nodes: List[MacroNode] = [MacroNode(m) for m in spans]
        # sort by start ascending and end descending so that potential parents come before children with same start
        nodes.sort(key=lambda n: (n.start, -n.end))

        for node in nodes:
            # find deepest parent that fully encloses this node
            parent = self.root
            while True:
                found_child = None
                for child in parent.children:
                    if child.start <= node.start and node.end <= child.end:
                        found_child = child
                        break
                if found_child is None:
                    break
                parent = found_child

            # ensure no overlapping with existing siblings
            for child in parent.children:
                if not (child.end <= node.start or child.start >= node.end):
                    logger.error(f"Macro spans overlap or are not properly nested: {child} vs {node}")
                    raise ValueError(f"Macro spans overlap or are not properly nested: {child} vs {node}")

            parent.add_child(node)
            

    def traverse(self) -> Iterator[MacroNode]:
        yield from self.root.traverse()

    def __repr__(self) -> str:
        return f"MacroTree(root={self.root!r})"

def resolve_template(text: str, data: Dict[str, Any], seps: Separators) -> str:
    """Resolve macros in the text using the provided data and separators."""
    tree = MacroTree(len(text))
    tree.populate(text, seps)

    return resolve_macro(tree.root, text, data, seps)

def resolve_macro(macro: MacroNode, text: str, data: Dict[str, Any], seps: Separators) -> str:
    """Resolve a macro in the text using the provided data and separators."""
    match macro.type:
        case "root":
            for child in macro.children:
                text = resolve_macro(child, text, data, seps)
            return text
        case "match":
            return resolve_match(macro, text, data, seps)
        case "loop":
            return resolve_loop(macro, text, data, seps)
        case _:
            logger.error(f"Unknown macro type: {macro.type}")
            raise ValueError(f"Unknown macro type: {macro.type}")

def get_macro_fields(macro: MacroNode, text: str, seps: Separators) -> List[str]:
    """Extract fields from a macro span in the text using the provided separators."""
    chunks = text[macro.start + len(seps.macro_open) : macro.end - len(seps.macro_close)].split(seps.macro_separator)
    fields: List[str] = []
    buffer = ""
    nested = 0
    for chunk in chunks:
        nested += chunk.count(seps.macro_open)
        nested -= chunk.count(seps.macro_close)
        if nested == 0:
            fields.append(buffer + chunk)
            buffer = ""
        elif nested > 0:
            buffer += chunk + seps.macro_separator
        else:
            logger.error("Unbalanced macro separators in macro")
            raise ValueError("Unbalanced macro separators in macro")

    if len(fields) == 0:
        logger.error("Macro contains no fields")
        raise ValueError("Macro contains no fields")
    
    return fields
            
def resolve_match(macro: MacroNode, text: str, data: Dict[str, Any], seps: Separators) -> str:
    """Resolve a 'match' macro in the text using the provided data and separators."""
    fields: List[str] = get_macro_fields(macro, text, seps)

    if len(fields) % 2 != 0:
        logger.error("Match macro must have an even number of fields (pairs of value and output)")
        raise ValueError("Match macro must have an even number of fields (pairs of value and output)")
    
    if fields[0].strip() != "match":
        logger.error("First field of match macro must be 'match'")
        raise ValueError("First field of match macro must be 'match'")
    
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
            text = text[:macro.start] + output + text[macro.end:]
            for child in macro.children:
                text = resolve_macro(child, text, data, seps)
            return text
    
    logger.error(f"No matching branch found for key '{match_key}' with value '{match_value}'")
    raise ValueError(f"No matching branch found for key '{match_key}' with value '{match_value}'")

def resolve_loop(macro: MacroNode, text: str, data: Dict[str, Any], seps: Separators) -> str:
    """Resolve a 'loop' macro in the text using the provided data and separators."""
    fields: List[str] = get_macro_fields(macro, text, seps)

    if len(fields) != 3:
        logger.error("Loop macro must have exactly three fields: 'loop', list_key, body_template")
        raise ValueError("Loop macro must have exactly three fields: 'loop', list_key, body_template")
    
    if fields[0].strip() != "loop":
        logger.error("First field of loop macro must be 'loop'")
        raise ValueError("First field of loop macro must be 'loop'")
    
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
        for child in macro.children:
            iter_body = resolve_macro(child, iter_body, data, seps)
        result_parts.append(iter_body)
    
    result = "".join(result_parts)
    text = text[:macro.start] + result + text[macro.end:]
    return text