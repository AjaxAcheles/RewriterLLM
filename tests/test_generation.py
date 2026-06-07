# tests/test_generation.py — M4 sloppy pair generation KPI checks.
#
# Requires data/raw_pairs.jsonl to exist (produced by scripts/02_generate_pairs.py).
# Run after generation and before M5 filtering. The keep rate in 03_filter_pairs.py is the
# stronger quality signal; these checks verify structural validity and scale targets.
#
# pytest tests/test_generation.py -v
import json
from pathlib import Path
import pytest

if not Path("data/raw_pairs.jsonl").exists():
    pytest.skip(
        "data/raw_pairs.jsonl not yet generated — run scripts/02_generate_pairs.py first",
        allow_module_level=True,
    )


def _load():
    return [json.loads(line) for line in open("data/raw_pairs.jsonl")]


def test_minimum_raw_pair_count():
    pairs = _load()
    assert len(pairs) >= 4500, (
        f"Only {len(pairs)} raw pairs; need >= 4500 so >= 1,800 survive M5 curation. "
        "Generate more excerpts or add another teacher."
    )


def test_pair_id_format():
    for p in _load():
        assert "_" in p["id"], (
            f"Pair ID '{p['id']}' has no underscore — base_id() in 05_format_dataset.py "
            "will treat it as its own base, breaking the leakage-free split."
        )


def test_multiple_teachers_present():
    teachers = {p["teacher"] for p in _load()}
    assert len(teachers) >= 2, (
        f"Only {teachers}; need >= 2 teacher families for slop diversity. "
        "Add a second teacher GGUF and re-run generation."
    )


def test_clean_and_sloppy_fields():
    for p in _load():
        assert p.get("clean", "").strip(), f"Empty 'clean' field in pair {p['id']}"
        assert p.get("sloppy", "").strip(), f"Empty 'sloppy' field in pair {p['id']}"


def test_sloppy_differs_from_clean():
    pairs = _load()
    identical = sum(1 for p in pairs if p["clean"] == p["sloppy"])
    frac_identical = identical / len(pairs) if pairs else 0
    assert frac_identical < 0.10, (
        f"{frac_identical:.1%} of raw pairs are identical (clean == sloppy). "
        "Raise temperature or strengthen slop injection in the teacher prompt."
    )


def test_checkpoint_resumes_correctly():
    pairs = _load()
    ids = [p["id"] for p in pairs]
    assert len(ids) == len(set(ids)), (
        f"{len(ids) - len(set(ids))} duplicate pair IDs found. "
        "The checkpoint/resume logic is not deduplicating correctly."
    )
