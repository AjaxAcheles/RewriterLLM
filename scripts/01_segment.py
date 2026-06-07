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

IDs are a 12-character MD5 prefix of the excerpt text. This makes segmentation
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

import argparse
import hashlib
import json
import re
from pathlib import Path

GUT_START = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
GUT_END   = re.compile(r"\*\*\*\s*END OF.*?\*\*\*",   re.IGNORECASE | re.DOTALL)


def strip_boilerplate(text):
    s = GUT_START.search(text)
    if s:
        text = text[s.end():]
    e = GUT_END.search(text)
    if e:
        text = text[:e.start()]
    return text.strip()


def printable_ratio(s):
    if not s:
        return 0.0
    good = sum(c.isalpha() or c.isspace() or c in ".,;:!?'\"-" for c in s)
    return good / len(s)


def excerpt_id(text):
    """Stable 12-char ID from the MD5 of the excerpt text (UTF-8)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def segment_book(text, min_words=300, max_words=700):
    """Split a book's full text into boundary-aligned excerpts.

    Yields dicts with keys: text, word_count.
    """
    cleaned = strip_boilerplate(text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    out, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if w > max_words and not cur:
            continue
        if n + w > max_words and n >= min_words:
            excerpt = " ".join(cur)
            if printable_ratio(excerpt) >= 0.9:
                yield {"text": excerpt, "word_count": n}
            cur, n = [p], w
        else:
            cur.append(p)
            n += w
    if n >= min_words:
        excerpt = " ".join(cur)
        if printable_ratio(excerpt) >= 0.9:
            yield {"text": excerpt, "word_count": n}


def main(input_dir="data/raw_texts", output_path="data/raw_excerpts.jsonl"):
    """Read all .txt files in input_dir, segment, and write to output_path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    with open(output_path, "w") as out:
        for path in sorted(Path(input_dir).glob("**/*.txt")):
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            for seg in segment_book(text):
                uid = excerpt_id(seg["text"])
                if uid in seen:
                    continue
                seen.add(uid)
                out.write(json.dumps({"id": uid, "source": path.name,
                                      "text": seg["text"],
                                      "word_count": seg["word_count"]}) + "\n")
                n += 1
    print(f"Wrote {n} excerpts from {input_dir} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw_texts")
    parser.add_argument("--output", default="data/raw_excerpts.jsonl")
    args = parser.parse_args()
    main(args.input, args.output)
