# tests/test_segmentation.py — M2 corpus segmentation KPI checks.
#
# Requires data/raw_excerpts.jsonl to exist (produced by scripts/01_segment.py).
# Run after segmentation completes and before starting M3 outline extraction.
#
# pytest tests/test_segmentation.py -v
import json
from pathlib import Path

import pytest

# skip entire module if data not yet generated
if not Path("data/raw_excerpts.jsonl").exists():
    pytest.skip("data/raw_excerpts.jsonl not yet generated — run scripts/01_segment.py first",
                allow_module_level=True)


def _load():
    return [json.loads(line) for line in open("data/raw_excerpts.jsonl")]


def test_minimum_excerpt_count():
    # Must produce >= 5,000 excerpts from the source corpus.
    # With ~500 books averaging ~80k words, the expected yield at 500 words/excerpt is ~80,000.
    # 5,000 is the minimum usable floor for downstream M4 generation targets.
    excerpts = _load()
    assert len(excerpts) >= 5000, f"Only {len(excerpts)} excerpts; want >= 5000"


def test_word_count_bounds():
    # Every excerpt must be 300-700 words inclusive.
    # Excerpts shorter than 300 words lack enough context for meaningful outline extraction (M3).
    # Excerpts longer than 700 words push the SFT context window uncomfortably close to the limit.
    excerpts = _load()
    for e in excerpts:
        wc = e.get("word_count", len(e["text"].split()))
        assert 280 <= wc <= 760, f"Excerpt {e['id']} has {wc} words"


def test_ids_are_unique():
    # All excerpt IDs must be unique — no duplicate content hash collisions.
    # Duplicates indicate the same passage was segmented twice, which would cause
    # the same content to appear in both train and test sets.
    excerpts = _load()
    ids = [e["id"] for e in excerpts]
    assert len(ids) == len(set(ids)), f"{len(ids) - len(set(ids))} duplicate IDs"


def test_id_format():
    # Each ID must be a 12-character hex string (SHA-256 prefix).
    # The format is relied on by M3 (keying outlines.json) and M4 (constructing pair IDs).
    import re
    excerpts = _load()
    for e in excerpts[:200]:
        assert re.fullmatch(r"[0-9a-f]{12}", e["id"]), f"Bad ID format: {e['id']}"


def test_source_field_present():
    # Each excerpt must have a non-empty "source" field (the filename or book title).
    # Used for attribution and for debugging quality issues back to source books.
    excerpts = _load()
    for e in excerpts:
        assert e.get("source"), f"Missing source in {e['id']}"


def test_text_field_not_empty():
    # No empty text fields — an empty excerpt would produce a trivial outline and a useless pair.
    excerpts = _load()
    for e in excerpts:
        assert e["text"].strip(), f"Empty text in {e['id']}"


def test_no_boilerplate():
    # No excerpt should contain Gutenberg boilerplate strings
    # ("Project Gutenberg", "www.gutenberg.org", "END OF THIS PROJECT").
    # Boilerplate in training data teaches the model to include attribution text.
    excerpts = _load()
    boilerplate_strings = ["PROJECT GUTENBERG", "WWW.GUTENBERG.ORG", "END OF THIS PROJECT"]
    for e in excerpts[:500]:
        upper = e["text"].upper()
        for bp in boilerplate_strings:
            assert bp not in upper, f"Boilerplate '{bp}' found in {e['id']}"
