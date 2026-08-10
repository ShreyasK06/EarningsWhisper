"""Section-aware, sentence-packed chunking for parsed earnings press releases.

Do NOT use fixed-size windows: these documents mix qualitative narrative
(headline, executive quotes, business outlook) with dense financial tables.
Blindly windowing would glue a half-sentence of commentary to a column of
numbers. Instead:

  1. Split the document into sections using regex on known headers
     (e.g. "Business Outlook", "Conference Call Information",
     "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS").
  2. Narrative sections are sentence-tokenized (nltk) and packed into
     ~300-word chunks with ~50-word overlap.
  3. Financial-statement table sections are NOT sentence-tokenized (their
     "sentences" are just numbers) -- their raw lines are packed into
     chunks instead, tagged with section="FinancialTables" so retrieval/
     eval code can filter them out of qualitative queries if desired.
  4. A chunk never crosses a section boundary.

Reads:  data/processed/{TICKER}_{accession}.json  (from parse.py)
Writes: data/processed/chunks.jsonl
"""

import json
import re
import sys
from pathlib import Path

import nltk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

PROCESSED_DIR = config.PROCESSED_DIR
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

WORDS_PER_CHUNK = 230  # ~300 tokens
OVERLAP_WORDS = 40  # ~50 tokens

# (regex matched against a stripped line, section label, is_table)
HEADER_PATTERNS = [
    (r"^condensed consolidated statements? of operations", "FinancialTables", True),
    (r"^condensed consolidated balance sheets?", "FinancialTables", True),
    (r"^condensed consolidated statements? of cash flows?", "FinancialTables", True),
    (r"^segment (information|results|highlights|detail)", "FinancialTables", True),
    (r"^reconciliation", "FinancialTables", True),
    (r"^business (outlook|highlights)", "BusinessOutlook", False),
    (r"^(quarterly )?highlights$", "BusinessOutlook", False),
    (r"^q[1-4]\s+(fiscal\s+)?\d{4}\s+highlights", "BusinessOutlook", False),
    (r"^conference call information", "ConferenceCall", False),
    (r"^webcast", "ConferenceCall", False),
    (r"^non-gaap financial measures", "NonGAAP", False),
    (r"^use of non-gaap", "NonGAAP", False),
    (r"^forward-looking statements", "ForwardLooking", False),
    (r"^safe harbor statement", "ForwardLooking", False),
    (r"^about (apple|microsoft|nvidia|amazon|alphabet|google)", "About", False),
    (r"^press contact", "Contacts", False),
    (r"^investor relations contact", "Contacts", False),
]
COMPILED_HEADERS = [(re.compile(p, re.IGNORECASE), label, is_table) for p, label, is_table in HEADER_PATTERNS]

DEFAULT_SECTION = "PreparedRemarks"


def classify_line(line: str):
    """Return (section_label, is_table) if this line is a section header, else None."""
    stripped = line.strip()
    if len(stripped) > 80:
        return None
    for pattern, label, is_table in COMPILED_HEADERS:
        if pattern.search(stripped):
            return label, is_table
    return None


def split_into_sections(text: str):
    """Walk lines, tagging each contiguous run with a section label."""
    sections = []  # list of {"label": str, "is_table": bool, "lines": [str]}
    current_label = DEFAULT_SECTION
    current_is_table = False
    current_lines = []

    for line in text.split("\n"):
        header = classify_line(line)
        if header:
            if current_lines:
                sections.append(
                    {"label": current_label, "is_table": current_is_table, "lines": current_lines}
                )
            current_label, current_is_table = header
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append({"label": current_label, "is_table": current_is_table, "lines": current_lines})

    return sections


def pack_words(units: list[str], words_per_chunk: int, overlap_words: int):
    """Pack a list of text units (sentences or raw lines) into overlapping
    word-count-bounded chunks, without splitting a unit across chunks."""
    chunks = []
    current_units: list[str] = []
    current_word_count = 0

    for unit in units:
        unit_words = len(unit.split())
        if current_units and current_word_count + unit_words > words_per_chunk:
            chunks.append(" ".join(current_units))
            # start next chunk with overlap: keep trailing units totalling ~overlap_words
            overlap_units = []
            overlap_count = 0
            for u in reversed(current_units):
                w = len(u.split())
                if overlap_count + w > overlap_words:
                    break
                overlap_units.insert(0, u)
                overlap_count += w
            current_units = overlap_units
            current_word_count = overlap_count

        current_units.append(unit)
        current_word_count += unit_words

    if current_units:
        chunks.append(" ".join(current_units))

    return chunks


def chunk_section(section: dict):
    lines = [l.strip() for l in section["lines"] if l.strip()]
    if not lines:
        return []

    if section["is_table"]:
        units = lines
    else:
        text_block = " ".join(lines)
        units = nltk.sent_tokenize(text_block)

    return pack_words(units, WORDS_PER_CHUNK, OVERLAP_WORDS)


def calendar_quarter(filing_date: str) -> str:
    if not filing_date:
        return ""
    year, month, _ = filing_date.split("-")
    q = (int(month) - 1) // 3 + 1
    return f"{year}Q{q}"


def chunk_all():
    all_chunks = []
    for doc_path in sorted(PROCESSED_DIR.glob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        ticker = doc["ticker"]
        fiscal_quarter = calendar_quarter(doc["filing_date"])
        sections = split_into_sections(doc["text"])

        seq = 0
        for section in sections:
            chunk_texts = chunk_section(section)
            for chunk_text in chunk_texts:
                chunk = {
                    "chunk_id": f"{ticker}_{fiscal_quarter}_{section['label']}_{seq:03d}",
                    "ticker": ticker,
                    "fiscal_quarter": fiscal_quarter,
                    "filing_date": doc["filing_date"],
                    "filed_datetime": doc["filed_datetime"],
                    "accession": doc["accession"],
                    "section": section["label"],
                    "speaker": None,
                    "text": chunk_text,
                }
                all_chunks.append(chunk)
                seq += 1

        print(f"  {doc_path.name}: {seq} chunks across {len(sections)} sections")

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    chunk_all()
