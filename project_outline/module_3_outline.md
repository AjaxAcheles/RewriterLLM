# Module 3 — Outline Extraction Integration & Validation

| Field | Value |
|---|---|
| **Phase** | Data pipeline (stage 2 of 4) |
| **Depends on** | M2 |
| **Blocks** | M4 |
| **Critical path** | Yes |
| **Owner effort** | 1–2 days (includes the manual validation gate) |
| **Runtime budget** | sample extraction minutes · full pool depends on extractor speed |

---

## 1. Primary Objective

Wrap the existing outline extractor behind one uniform interface, run it across the excerpt pool
to produce one structured outline per excerpt, and **prove on a manual sample** that each outline
carries enough structure for a different model to faithfully reproduce the excerpt's events.
Outputs: `data/outlines.json` keyed by excerpt ID, plus a signed-off validation log.

---

## 2. Core Concepts — Deep Dive

### 2.1 The outline is the alignment contract

The project uses inverse injection: take clean human text and have a teacher LLM produce a
*sloppy* version (Module 4). The danger is that a free-running teacher writes something
thematically similar but factually different — a different character acts, the scene moves, a new
minor character appears, the ending resolves differently. That yields a pair differing in
**content**, not just **style**, and a model trained on it learns to *rewrite content* — exactly
the forbidden behavior. The outline prevents this by constraining the teacher: before any prose
is written, the extractor distills the clean excerpt into its skeleton (who/what/where/when/why,
in order), and Module 4 hands that skeleton to the teacher as a hard requirement. The teacher's
only freedom is *prose style*. Result: a sloppy version matching the clean original in substance
while differing in style — precisely the (input, target) relationship the editor must learn.

### 2.2 Why the failure this guards against is silent

A lossy outline (one that drops a causal link or a character) produces a teacher output that
omits or invents around the missing information. If the drift is subtle, Module 5's filters may
pass it, and it only surfaces as degraded NER-preservation in Module 6 — days later, with the
root cause buried under several stages. A human reading 50 outlines and confirming
reconstructability costs an afternoon and catches the problem at its source. This is the cheapest
possible interception point, so the manual gate is **mandatory** and blocks Module 4 below 90%.

### 2.3 Why the schema mirrors the evaluation criteria

The four required fields — characters, location/time, ordered events with causal links, POV
knowledge/state as facts — are the same dimensions Module 9 later checks for preservation
(NER = characters/places; hallucination rate = no new entities/events; semantic similarity = same
content). **The outline defines what "preserved" means.** A dimension the outline captures can be
enforced downstream; a dimension it omits is unprotected. Hence schema completeness is a hard
requirement.

### 2.4 Emotion as fact, not prose — and why

The POV-state section records emotional facts ("grief over the lost letter") rather than prose
("a wave of sorrow washed over her"). Two reasons: (1) the *fact* must be preserved through the
pipeline so the editor keeps the emotional beat; (2) supplying *prose* would leak the clean
excerpt's style into the teacher's slop generation, contaminating the pair's style contrast.

### 2.5 Sampling theory behind the 50-sample gate

A 50-sample binomial estimate of the true pass rate has a standard error around ±6–7 points near
90%. That is precise enough to distinguish "≈90% and acceptable" from "≈75% and broken," while
small enough to review by hand in one sitting. If the observed rate sits right at the boundary,
extend the sample rather than guess.

---

## 3. Inputs & Outputs Contract

**Input:** `data/raw_excerpts.jsonl` (from M2).

**Outputs:**
- `data/outlines.json` — `{excerpt_id: outline_text}`, 100% coverage of excerpt IDs.
- `data/outlines_sample.json` + `..._originals.json` — the 50-sample review set.
- `data/outline_validation_log.csv` — `excerpt_id,reviewer,verdict,lost_information_note`.

**Interface other modules rely on:** `extract_outline(excerpt: str) -> str` emitting the
four-section schema (§4). Module 4 injects the returned string verbatim.

---

## 4. The Outline Schema

```
CHARACTERS:
  - <name>: <role in this scene, 1 line>
SETTING:
  Location: <where>
  Time: <when, relative or absolute>
EVENTS (in order):
  1. <event> [because <cause> / leading to <effect>]
  2. ...
POV STATE:
  Narrator/POV: <which character>
  Knows: <key facts the POV holds at this point>
  Feels: <emotional fact, NOT prose — "grief over X", not "sorrow washed over her">
```

---

## 5. Common Challenges & Solutions

**Challenge 1 — The manual gate stalls below 90%.**
*Why:* the extractor systematically drops a class of information (commonly causal links). *Detect:*
cluster the FAIL notes in the validation log; a repeated phrase ("dropped the reason…") reveals
the pattern. *Solve:* strengthen the EVENTS section to require explicit because/leading-to
annotations; re-sample and re-review before scaling.

**Challenge 2 — Outlines drift toward near-prose.**
*Why:* the extractor copies sentences instead of distilling facts. *Detect:* outlines are long and
read like summaries. *Solve:* constrain output to bullet facts; cap section lengths; post-process
to strip adjectives/adverbs.

**Challenge 3 — Clean style leaks via the emotion field.**
*Why:* "Feels" is written as prose. *Detect:* evocative sentences in POV STATE. *Solve:* enforce
the fact-form rule; reject/rewrite emotion lines containing figurative language.

**Challenge 4 — Missing schema sections.**
*Why:* the extractor's output format is unstable across inputs. *Detect:* `has_all_sections`
flags a high malformed rate. *Solve:* pin the four-section template in the prompt; validate format
and retry on malformed outputs.

**Challenge 5 — Coverage gaps (some excerpts have no outline).**
*Why:* the extractor crashed on certain inputs (very long, odd characters). *Detect:*
`test_full_coverage` fails. *Solve:* wrap extraction in try/except with logging; collect failed
IDs and re-run them.

**Challenge 6 — Reviewer subjectivity skews the gate.**
*Why:* "reconstructable" is judged inconsistently. *Detect:* two reviewers disagree on the same
samples. *Solve:* define a concrete rubric (all named characters present, all events present and
ordered, the inciting cause present) and have the reviewer write the one-line reconstruction
*before* seeing the original.

---

## 6. Step-by-Step Implementation Guide

**Step 1 — Conform the extractor.** Adapt your implementation to
`extract_outline(excerpt: str) -> str` emitting the §4 schema. Document the schema in its
docstring (Module 4 depends on the format).

**Step 2 — Extract the validation sample.**
```bash
python scripts/extract_outlines.py sample
# writes data/outlines_sample.json + data/outlines_sample_originals.json (50 each)
```

**Step 3 — Run the human gate.** For each of the 50 samples: read *only the outline*, write a
one-line reconstruction, then compare to the original. Mark PASS/FAIL with a note. Record in
`data/outline_validation_log.csv`.

**Step 4 — Gate decision.** Compute the PASS rate.
- ≥ 90% → proceed to Step 5.
- < 90% → cluster FAIL notes, fix the extractor (Challenge 1), re-sample, re-review.

**Step 5 — Full extraction.**
```bash
python scripts/extract_outlines.py
# expect: "Wrote N outlines; M missing >=1 section"  (M small)
```

**Step 6 — Coverage + format check, then verify.**
```bash
pytest tests/test_outline_extractor.py -v
```

---

## 7. Reference Implementation — `scripts/extract_outlines.py`

```python
# scripts/extract_outlines.py
"""Batch-run the outline extractor across the excerpt pool.
Input: data/raw_excerpts.jsonl  →  Output: data/outlines.json ({id: outline})"""
import json, random
from your_outline_module import extract_outline   # conform to: (str) -> str, §4 schema

REQUIRED = ["CHARACTERS:", "SETTING:", "EVENTS", "POV STATE:"]


def has_all_sections(o):
    return all(s in o for s in REQUIRED)


def extract_sample(excerpts="data/raw_excerpts.jsonl",
                   out="data/outlines_sample.json", n=50, seed=42):
    items = [json.loads(l) for l in open(excerpts)]
    random.seed(seed)
    sample = random.sample(items, n)
    json.dump({e["id"]: extract_outline(e["text"]) for e in sample},
              open(out, "w"), indent=2)
    json.dump({e["id"]: e["text"] for e in sample},
              open("data/outlines_sample_originals.json", "w"), indent=2)
    print(f"Wrote {n}-sample outlines to {out}")


def extract_full(excerpts="data/raw_excerpts.jsonl", out="data/outlines.json"):
    items = [json.loads(l) for l in open(excerpts)]
    outlines, malformed, failed = {}, 0, []
    for e in items:
        try:
            o = extract_outline(e["text"])
        except Exception as ex:               # never let one bad excerpt halt the run
            failed.append((e["id"], str(ex))); continue
        if not has_all_sections(o):
            malformed += 1
        outlines[e["id"]] = o
    json.dump(outlines, open(out, "w"))
    if failed:
        json.dump(failed, open("data/outline_failures.json", "w"), indent=2)
    print(f"Wrote {len(outlines)} outlines; {malformed} missing >=1 section; "
          f"{len(failed)} failed")


if __name__ == "__main__":
    import sys
    extract_sample() if (len(sys.argv) > 1 and sys.argv[1] == "sample") else extract_full()
```

**Validation log — `data/outline_validation_log.csv`:**
```csv
excerpt_id,reviewer,verdict,lost_information_note
a1b2c3d4e5f6,AJ,PASS,
b2c3d4e5f6a1,AJ,FAIL,"dropped the reason the letter was burned (inciting cause)"
```

---

## 8. Configuration & Parameters

| Parameter | Value | Rationale | If you change it |
|---|---|---|---|
| sample size | 50 | ±~7% estimate; reviewable in one sitting | Smaller → noisier gate |
| pass gate | ≥ 90% | Below → too much M4 drift to filter cleanly | Lower → more bad pairs survive |
| sections | 4 | Mirror M9 preservation dimensions | Fewer → unprotected dimensions |
| emotion form | fact | Prevent clean-style leakage | Prose form → contaminated pairs |

---

## 9. Alternatives Considered & Rejected

| Alternative | Why considered | Why rejected |
|---|---|---|
| Skip outlines; constrain M4 by prompt alone | Less work | Teachers drift without a concrete skeleton; pairs misalign |
| Auto-validate outlines with an LLM judge | Scales past 50 | Judge shares model blind spots; the silent-loss failure needs a human |
| Embedding-similarity check vs. excerpt | Cheap, automatic | Measures topical overlap, not event/causal completeness |
| Free-form summary instead of fixed schema | Simpler extractor | Unenforceable downstream; can't test section coverage |

---

## 10. KPIs / Test File

**Test file:** `tests/test_outline_extractor.py`

```python
# tests/test_outline_extractor.py
import json
from pathlib import Path
import pytest

OUTLINES, EXCERPTS = "data/outlines.json", "data/raw_excerpts.jsonl"
VALID_LOG = "data/outline_validation_log.csv"
REQUIRED = ["CHARACTERS:", "SETTING:", "EVENTS", "POV STATE:"]

@pytest.fixture(scope="module")
def outlines():
    assert Path(OUTLINES).exists(), "Run scripts/extract_outlines.py first"
    return json.load(open(OUTLINES))

def test_returns_nonempty():
    from your_outline_module import extract_outline
    e = next(json.loads(l) for l in open(EXCERPTS))
    out = extract_outline(e["text"])
    assert isinstance(out, str) and out.strip()

def test_sections_present(outlines):
    bad = [i for i, o in outlines.items() if not all(s in o for s in REQUIRED)]
    assert len(bad) / max(len(outlines), 1) < 0.05

def test_full_coverage(outlines):
    ids = {json.loads(l)["id"] for l in open(EXCERPTS)}
    assert not (ids - set(outlines))

def test_manual_gate_passed():
    assert Path(VALID_LOG).exists(), "Manual validation log missing"
    rows = [r.strip().split(",") for r in open(VALID_LOG) if r.strip()][1:]
    verdicts = [r[2].upper() for r in rows]
    assert len(verdicts) >= 50
    assert verdicts.count("PASS") / len(verdicts) >= 0.90
```

---

## 11. Definition of Done

- `data/outlines.json` covers 100% of excerpt IDs; < 5% missing any section.
- Manual gate recorded over ≥ 50 samples at ≥ 90% PASS.
- `pytest tests/test_outline_extractor.py` passes.
