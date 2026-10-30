"""Loading, parsing and chunking.

Two paths on purpose:
  naive:      what most first RAG builds do. Read raw text, cut every N characters.
  structured: parse the document's real structure (headings, tables), chunk along it,
              and prefix each chunk with where it came from (a cheap, deterministic
              form of contextual retrieval).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


def load_manifest() -> list[dict]:
    return json.loads((CORPUS / "manifest.json").read_text())["documents"]


# ---------------------------------------------------------------- naive path

def _naive_text(path: Path) -> str:
    if path.suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text()


def fixed_size_chunks(text: str, size: int = 500, overlap: int = 0) -> list[str]:
    """Character splitter: the default in countless tutorials."""
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def naive_chunks(size: int = 500, overlap: int = 0) -> list[Chunk]:
    out = []
    for doc in load_manifest():
        text = _naive_text(CORPUS / doc["file"])
        for i, piece in enumerate(fixed_size_chunks(text, size, overlap)):
            out.append(Chunk(f"{doc['doc_id']}#n{i}", doc["doc_id"], piece, dict(doc)))
    return out


# ----------------------------------------------------------- structured path

def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split on ## headings. Returns (heading, body) pairs; the preamble keeps the title."""
    sections, heading, buf = [], "Overview", []
    for line in text.splitlines():
        if line.startswith("## "):
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        elif line.startswith("# "):
            continue
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def _split_long(body: str, max_chars: int) -> list[str]:
    """Split an oversized section on paragraph boundaries, never mid-sentence."""
    if len(body) <= max_chars:
        return [body]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", body):
        if cur and len(cur) + len(para) > max_chars:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _pdf_structured(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Return (prose sections, table row records). Tables become one record per row,
    with every column header repeated next to its value."""
    import pdfplumber

    prose, rows = [], []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            tables = page.find_tables()
            for t in tables:
                data = t.extract()
                header, body = data[0], data[1:]
                for r in body:
                    rows.append(" | ".join(f"{h}: {v}" for h, v in zip(header, r)))
            # prose = page text with table regions removed
            outside = page
            for t in tables:
                outside = outside.outside_bbox(t.bbox)
            text = (outside.extract_text() or "").strip()
            if text:
                prose.append(("Overview", text))
    return prose, rows


def structured_chunks(max_chars: int = 1200, contextual_header: bool = True) -> list[Chunk]:
    out = []
    for doc in load_manifest():
        path = CORPUS / doc["file"]
        head = f"[{doc['title']} | {doc['doc_id']}]"
        if path.suffix == ".pdf":
            sections, rows = _pdf_structured(path)
            for i, rec in enumerate(rows):
                text = f"{head} Fee table row. {rec}" if contextual_header else rec
                out.append(Chunk(f"{doc['doc_id']}#row{i}", doc["doc_id"], text, {**doc, "section": "table"}))
        else:
            sections = _markdown_sections(path.read_text())
        for s_i, (heading, body) in enumerate(sections):
            for p_i, piece in enumerate(_split_long(body, max_chars)):
                text = f"{head} Section: {heading}\n{piece}" if contextual_header else piece
                out.append(Chunk(f"{doc['doc_id']}#s{s_i}.{p_i}", doc["doc_id"], text, {**doc, "section": heading}))
    return out
