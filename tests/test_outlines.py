# tests/test_outlines.py — M3 outline extraction KPI checks.
#
# Requires data/outlines.json to exist (produced by scripts/extract_outlines.py).
# Run after extraction and before the manual reconstruction gate.
# The manual gate (50-sample, >= 90% reconstructability) is a human step and cannot be
# automated here — this file checks structural validity only.
#
# pytest tests/test_outlines.py -v
import json
import pytest
from pathlib import Path


def _load():
    if not Path("data/outlines.json").exists():
        pytest.skip("data/outlines.json not yet generated — run scripts/extract_outlines.py first")
    return json.loads(Path("data/outlines.json").read_text())


def _load_excerpt_ids():
    if not Path("data/raw_excerpts.jsonl").exists():
        pytest.skip("data/raw_excerpts.jsonl not yet generated")
    return {json.loads(l)["id"] for l in open("data/raw_excerpts.jsonl")}


def test_all_excerpts_have_outlines():
    # Every excerpt ID in data/raw_excerpts.jsonl must have a corresponding entry in outlines.json.
    # Missing outlines mean M4 cannot generate constrained pairs for those excerpts.
    outlines = _load()
    excerpt_ids = _load_excerpt_ids()
    missing = excerpt_ids - set(outlines.keys())
    coverage = 1 - (len(missing) / len(excerpt_ids)) if excerpt_ids else 1.0
    assert coverage >= 0.95, f"Only {coverage:.1%} coverage; missing {len(missing)} IDs"


def test_outline_schema():
    # Every outline must have all four required fields: characters, setting, events, pov_state.
    # A missing field means the teacher in M4 will have incomplete constraints and may drift.
    outlines = _load()
    required = {"characters", "setting", "events", "pov_state"}
    bad = []
    for eid, entry in list(outlines.items())[:200]:
        outline = entry.get("outline", entry)
        if not required.issubset(outline.keys()):
            bad.append(eid)
    assert len(bad) / max(len(outlines), 1) < 0.05, f"{len(bad)} outlines missing sections"


def test_characters_is_list():
    # The characters field must be a list (possibly empty for scene-only excerpts).
    # A string or None causes a crash in 02_generate_pairs.py when building the prompt.
    outlines = _load()
    for eid, entry in list(outlines.items())[:200]:
        outline = entry.get("outline", entry)
        assert isinstance(outline.get("characters"), list), f"characters not list in {eid}"


def test_events_is_nonempty_list():
    # The events field must be a non-empty list — at least one plot beat per excerpt.
    # An empty events list gives the M4 teacher no structural constraints to honour.
    outlines = _load()
    for eid, entry in list(outlines.items())[:200]:
        outline = entry.get("outline", entry)
        assert isinstance(outline.get("events"), list), f"events not list in {eid}"
        assert len(outline.get("events", [])) >= 1, f"events empty in {eid}"


def test_pov_state_is_string():
    # pov_state must be a non-empty string describing whose POV and their emotional state.
    # An empty pov_state allows the teacher to shift perspective, breaking content preservation.
    outlines = _load()
    for eid, entry in list(outlines.items())[:200]:
        outline = entry.get("outline", entry)
        pov = outline.get("pov_state", "")
        assert isinstance(pov, str) and pov.strip(), f"pov_state empty/invalid in {eid}"


def test_coverage_fraction():
    # At least 95% of excerpts should have successfully extracted outlines.
    # A lower rate indicates the extraction prompt is failing for certain text styles
    # (e.g. non-standard formatting, archaic language) and needs adjustment.
    outlines = _load()
    excerpt_ids = _load_excerpt_ids()
    coverage = len(set(outlines.keys()) & excerpt_ids) / max(len(excerpt_ids), 1)
    assert coverage >= 0.95, f"Coverage {coverage:.1%} below 95% threshold"
