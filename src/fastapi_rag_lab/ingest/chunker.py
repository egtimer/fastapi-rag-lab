"""Parent-child chunker with heading-aware markdown splitting."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import tiktoken
from markdown_it import MarkdownIt

logger = logging.getLogger(__name__)

PARENT_TOKEN_CEILING = 1024
CHILD_TOKEN_TARGET = 256
CHILD_OVERLAP_TOKENS = 32
EMBEDDING_CONTEXT_LIMIT = 2048

_encoder = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_encoder.encode(text))


@dataclass(frozen=True)
class Parent:
    parent_id: str
    text: str
    source_file: str
    heading_path: list[str] = field(default_factory=list)
    token_count: int = 0

    def __hash__(self):
        return hash(self.parent_id)


@dataclass(frozen=True)
class Child:
    child_id: str
    parent_id: str
    text: str
    child_index: int
    token_count: int = 0

    def __hash__(self):
        return hash(self.child_id)


def chunk_markdown(
    source_file: str, markdown_text: str
) -> tuple[list[Parent], list[Child]]:
    """Split a markdown file into parent and child chunks."""
    sections = _split_into_sections(markdown_text)
    parents_out: list[Parent] = []
    children_out: list[Child] = []

    grouped = _group_sections_into_parents(sections)

    for parent_index, (heading_path, parent_text) in enumerate(grouped):
        parent_tokens = token_count(parent_text)
        pid = _deterministic_id(source_file, f"parent-{parent_index}")

        parent = Parent(
            parent_id=pid,
            text=parent_text,
            source_file=source_file,
            heading_path=heading_path,
            token_count=parent_tokens,
        )
        parents_out.append(parent)

        children = _split_parent_into_children(
            parent_text, pid, source_file, len(children_out)
        )
        children_out.extend(children)

    return parents_out, children_out


@dataclass
class _Section:
    heading: str
    level: int
    content: str
    tokens: int = 0


def _split_into_sections(markdown_text: str) -> list[_Section]:
    md = MarkdownIt()
    tokens = md.parse(markdown_text)

    sections: list[_Section] = []
    current_heading = ""
    current_level = 0
    content_parts: list[str] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            if content_parts or current_heading:
                text = "".join(content_parts).strip()
                sections.append(
                    _Section(
                        heading=current_heading,
                        level=current_level,
                        content=text,
                        tokens=token_count(text) if text else 0,
                    )
                )
                content_parts = []

            level = int(tok.tag[1])
            heading_text = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                heading_text = tokens[i + 1].content or ""
            current_heading = heading_text
            current_level = level
            # skip heading_open, inline, heading_close
            i += 3
            continue

        if tok.type == "inline":
            content_parts.append(tok.content or "")
        elif tok.type == "fence":
            lang = tok.info.strip() if tok.info else ""
            fence_mark = "```"
            if lang:
                content_parts.append(
                    f"\n{fence_mark}{lang}\n{tok.content}{fence_mark}\n"
                )
            else:
                content_parts.append(f"\n{fence_mark}\n{tok.content}{fence_mark}\n")
        elif tok.type == "code_block":
            content_parts.append(f"\n```\n{tok.content}```\n")
        elif tok.type in ("paragraph_open", "paragraph_close"):
            if tok.type == "paragraph_close":
                content_parts.append("\n\n")
        elif tok.type in (
            "bullet_list_open",
            "bullet_list_close",
            "ordered_list_open",
            "ordered_list_close",
            "list_item_open",
            "list_item_close",
        ):
            pass
        elif tok.type == "html_block":
            content_parts.append(tok.content or "")

        i += 1

    # flush last section
    text = "".join(content_parts).strip()
    if text or current_heading:
        sections.append(
            _Section(
                heading=current_heading,
                level=current_level,
                content=text,
                tokens=token_count(text) if text else 0,
            )
        )

    return sections


def _group_sections_into_parents(
    sections: list[_Section],
) -> list[tuple[list[str], str]]:
    if not sections:
        return []

    # Try H2 boundaries first, fall back to H3 if H2 parents are too large
    h2_groups = _group_at_level(sections, 2)
    oversized = sum(
        1 for _, text in h2_groups if token_count(text) > PARENT_TOKEN_CEILING
    )

    if oversized > len(h2_groups) * 0.3 and any(s.level == 3 for s in sections):
        return _group_at_level(sections, 3)

    return h2_groups


def _group_at_level(
    sections: list[_Section], split_level: int
) -> list[tuple[list[str], str]]:
    groups: list[tuple[list[str], str]] = []
    heading_stack: list[str] = []
    current_parts: list[str] = []

    def _flush():
        text = "\n\n".join(p for p in current_parts if p).strip()
        if text:
            groups.append((list(heading_stack), text))

    for section in sections:
        if section.level > 0 and section.level <= split_level:
            _flush()
            current_parts = []

            # maintain heading stack
            while len(heading_stack) >= section.level:
                heading_stack.pop()
            heading_stack.append(section.heading)

            if section.content:
                current_parts.append(section.content)
        else:
            if section.content:
                current_parts.append(section.content)

    _flush()
    return groups


def _split_parent_into_children(
    parent_text: str,
    parent_id: str,
    source_file: str,
    global_child_offset: int,
) -> list[Child]:
    encoded = _encoder.encode(parent_text)
    total_tokens = len(encoded)

    if total_tokens <= CHILD_TOKEN_TARGET + CHILD_OVERLAP_TOKENS:
        child_id = _deterministic_id(source_file, f"child-{global_child_offset}")
        return [
            Child(
                child_id=child_id,
                parent_id=parent_id,
                text=parent_text,
                child_index=global_child_offset,
                token_count=total_tokens,
            )
        ]

    code_block_ranges = _find_code_block_token_ranges(parent_text, encoded)
    children: list[Child] = []
    start = 0

    while start < total_tokens:
        end = min(start + CHILD_TOKEN_TARGET, total_tokens)

        # Don't split inside a code block
        for block_start, block_end in code_block_ranges:
            if block_start < end <= block_end:
                # extend to include the whole code block
                end = min(block_end, total_tokens)
                break
            if start < block_start < end < block_end:
                # code block extends beyond this chunk — shrink
                end = block_start
                break

        if end <= start:
            end = min(start + CHILD_TOKEN_TARGET, total_tokens)

        child_text = _encoder.decode(encoded[start:end]).strip()
        child_tokens = end - start
        child_index = global_child_offset + len(children)

        if child_text:
            _check_embedding_limit(child_text, child_tokens, source_file)
            child_id = _deterministic_id(source_file, f"child-{child_index}")
            children.append(
                Child(
                    child_id=child_id,
                    parent_id=parent_id,
                    text=child_text,
                    child_index=child_index,
                    token_count=child_tokens,
                )
            )

        if end >= total_tokens:
            break

        start = max(end - CHILD_OVERLAP_TOKENS, start + 1)

    return children


def _find_code_block_token_ranges(
    text: str, encoded: list[int]
) -> list[tuple[int, int]]:
    ranges = []
    in_block = False
    block_char_start = 0
    char_pos = 0

    for line in text.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("```"):
            if not in_block:
                in_block = True
                block_char_start = char_pos
            else:
                in_block = False
                block_char_end = char_pos + len(line)
                tok_start = len(_encoder.encode(text[:block_char_start]))
                tok_end = len(_encoder.encode(text[:block_char_end]))
                ranges.append((tok_start, tok_end))

        char_pos += len(line) + 1  # +1 for the newline

    return ranges


def _check_embedding_limit(text: str, tokens: int, source_file: str):
    if tokens > EMBEDDING_CONTEXT_LIMIT:
        raise ValueError(
            f"Child chunk from {source_file} has {tokens} tokens, "
            f"exceeding the {EMBEDDING_CONTEXT_LIMIT}-token embedding limit for "
            f"nomic-embed-text. This likely means a code block is too large. "
            f"Text preview: {text[:200]}..."
        )


def _deterministic_id(source_file: str, suffix: str) -> str:
    return hashlib.sha256(f"{source_file}:{suffix}".encode()).hexdigest()[:16]
