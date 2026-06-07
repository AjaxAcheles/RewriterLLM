"""
01_segment — segment source books into 300-700 word excerpts.

Pipeline position: M2 — Data pipeline, stage 1 of 4
Depends on: data/raw_texts/ (UTF-8 .txt files, one book per file)
Produces:   data/raw_excerpts.jsonl

Each book is split into prose excerpts of 300-700 words. Splitting happens at paragraph
or scene boundaries (blank lines, chapter breaks) — never mid-sentence — so that each
excerpt is a coherent scene fragment with a single setting and continuous POV. Token-based
splitting is explicitly avoided: it produces mid-sentence cuts that break coherence and
confuse the outline extractor in M3.

IDs are a 12-character SHA-256 prefix of the excerpt text. This makes segmentation
idempotent — re-running on the same corpus produces the same IDs. M3, M4, and M5 all
use the excerpt ID as their join key, so ID stability is load-bearing.

Output schema (one JSON object per line):
  {"id": str, "source": str, "text": str, "word_count": int}

Usage:
    python scripts/01_segment.py [--input data/raw_texts] [--output data/raw_excerpts.jsonl]

Key implementation notes:
  - Split on blank lines first; then merge short paragraphs forward until >= 300 words,
    then cut when the running total would exceed 700 words at the next paragraph boundary.
  - A paragraph that is itself > 700 words (rare in fiction) should be split at the last
    sentence boundary before the 700-word limit — never at a word boundary.
  - Strip gutenberg boilerplate (lines matching "Project Gutenberg", "CHAPTER", etc.) before
    segmenting, or it ends up in the excerpts and trains the model on header text.
  - Target >= 5,000 excerpts total. With ~500 books averaging ~80k words each and ~500-word
    excerpts, the expected yield is ~80,000 excerpts — sample down if needed, but never up.
"""

import hashlib
import json
from pathlib import Path


def excerpt_id(text):
    """Stable 12-char ID from the SHA-256 of the excerpt text (UTF-8)."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def segment_book(text, min_words=300, max_words=700):
    """Split a book's full text into boundary-aligned excerpts.

    Yields dicts with keys: text, word_count.
    TODO: implement paragraph-boundary splitting logic.
    """
    raise NotImplementedError


def main(input_dir="data/raw_texts", output_path="data/raw_excerpts.jsonl"):
    """Read all .txt files in input_dir, segment, and write to output_path."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
