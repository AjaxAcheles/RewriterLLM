# Module 2 — Source Corpus Acquisition & Segmentation

| Field | Value |
|---|---|
| **Phase** | Data pipeline (stage 1 of 4) |
| **Depends on** | M1 |
| **Blocks** | M3, M5; serves M7 as the human baseline |
| **Critical path** | Yes |
| **Owner effort** | 0.5–1 day + download time |
| **Runtime budget** | download 500 books ~10–30 min · segmentation ~2–5 min |

---

## 1. Primary Objective

Convert a library of clean, human-written, rights-cleared fiction into a normalized pool of
300–700-word excerpts, each with a stable unique ID, stored as one JSONL record per excerpt.
This pool is the *clean target* side of every training pair and doubles as the human baseline for
slop profiling in Module 7.

---

## 2. Core Concepts — Deep Dive

### 2.1 The corpus sets the project's quality ceiling

A model trained by imitation cannot exceed the quality of what it imitates, and it regresses
toward the corpus *average*. So corpus selection is a quality decision, not logistics. The single
most important invariant: **every clean target is genuine, well-written human prose** — never
AI-generated, never AI-"cleaned," never machine-translated. If AI text contaminates the clean
side, the model learns to imitate AI cleaning, reintroducing the very slop the project removes.
This is why public-domain human fiction (Project Gutenberg) is the default source.

### 2.2 Why the 300–700-word window is a three-way compromise

- **Lower bound (300):** below this an excerpt has too little internal causality and character
  action to be a meaningful preservation test — there is nothing structural to preserve, so the
  pair teaches nothing about preservation.
- **Upper bound (700):** both the sloppy input *and* the clean output are tokenized into context
  during training. 700 words ≈ 900–1,000 tokens; input + output ≈ 2,000 tokens, plus system
  prompt, chat template, and (in Module 8) a style-reference block. That fits the 8,192 context
  budget with headroom. Larger excerpts risk overflow on the biggest examples.
- **Extractor fit (Module 3):** outline extraction is most reliable on scene-sized chunks. Whole
  chapters yield sprawling outlines; single paragraphs yield trivial ones.

### 2.3 Why boundary-based segmentation, not fixed token counts

A fixed token cut routinely severs a scene mid-action, producing an excerpt that opens with a
dangling reference ("She handed it to him") to entities introduced in the prior chunk. Such an
excerpt is *impossible to preserve faithfully* because the referents are absent. Segmenting on
blank-line (paragraph/scene) boundaries keeps each excerpt internally coherent; the variable
length that results is exactly what the 300–700 window absorbs.

### 2.4 Why a stable content-hash ID

The ID threads the whole pipeline: M3 keys outlines by it, M4 tags sloppy versions `{id}_{teacher}`,
M5 reassembles pairs and enforces a leakage-free split *by base ID*. A content hash (not a
counter) makes segmentation **idempotent** — re-running on the same source yields the same IDs,
so previously generated outlines and pairs are never orphaned. This is what lets the pipeline be
re-run safely after a partial failure.

### 2.5 Dual use as the Module 7 baseline

Module 7's slop profiler measures the post-SFT model's word/pattern frequencies *relative to
human text*. These same source files are that human reference. Hence the raw `.txt` files are
retained on disk after segmentation rather than discarded.

---

## 3. Inputs & Outputs Contract

**Input:** `data/raw_texts/*.txt` — clean human fiction (≥ ~500 files recommended).

**Output:** `data/raw_excerpts.jsonl` — one object per line:
```json
{"id": "a1b2c3d4e5f6", "source": "00042.txt", "text": "<300-700 words>"}
```

**Invariants:** `id` unique; 300 ≤ word count ≤ 700 (small tolerance on file-final excerpt);
`text` non-empty; ≥ 200 distinct `source` values. Raw texts retained for M7.

---

## 4. Common Challenges & Solutions

**Challenge 1 — License boilerplate leaks into excerpts.**
*Why:* Gutenberg headers/footers weren't stripped. *Detect:* `test_no_gutenberg_boilerplate`
fails; excerpts contain "PROJECT GUTENBERG". *Solve:* trim on the `*** START OF ... ***` /
`*** END OF ... ***` markers before segmenting; verify the regex matches your source's exact
delimiter format.

**Challenge 2 — Many excerpts fall below 300 words.**
*Why:* dialogue-heavy or short-paragraph source produces small accumulation units. *Detect:* low
total excerpt count; word-count histogram skews low. *Solve:* accumulate across more paragraphs
before cutting; optionally lower `min_words` to 250 for such sources.

**Challenge 3 — Low source diversity.**
*Why:* too few books, or one prolific author dominates. *Detect:* `test_source_diversity` fails;
< 200 distinct sources. *Solve:* download more books; cap excerpts-per-source so no single book
floods the pool.

**Challenge 4 — OCR garbage / encoding artifacts.**
*Why:* scanned-source texts contain non-text noise. *Detect:* spot-check shows gibberish; high
ratio of non-alphabetic characters. *Solve:* add a printable-character-ratio filter (drop
excerpts below ~0.9 alphabetic+space); prefer born-digital sources.

**Challenge 5 — Duplicate excerpts inflate the count.**
*Why:* the same passage appears in multiple books (anthologies, reprints). *Detect:* duplicate
IDs would collide (already deduped by content hash). *Solve:* the content-hash dedup in the
segmenter handles this; confirm the dedup set is active.

**Challenge 6 — A single pathological paragraph exceeds the window.**
*Why:* a wall-of-text paragraph > 700 words. *Detect:* segmenter would otherwise emit an
oversized excerpt. *Solve:* the segmenter skips an over-length paragraph when it would start a
new excerpt; for frequent cases, add intra-paragraph sentence splitting.

---

## 5. Step-by-Step Implementation Guide

**Step 1 — Acquire sources.** Either drop `.txt` files into `data/raw_texts/`, or run the
downloader:
```bash
python scripts/fetch_gutenberg.py            # writes ~500 .txt files
ls data/raw_texts | wc -l                    # expect ~500
```

**Step 2 — Implement `segment_text()`** with boundary-based accumulation and the 300–700 window
(see §6). Unit-test it on a synthetic multi-paragraph string before running at scale.

**Step 3 — Run segmentation.**
```bash
python scripts/01_segment.py
# expect: "Wrote N excerpts from data/raw_texts to data/raw_excerpts.jsonl"  (N >= 5000)
```

**Step 4 — Inspect the distribution.**
```bash
python - <<'PY'
import json
wc=[len(json.loads(l)["text"].split()) for l in open("data/raw_excerpts.jsonl")]
print("count",len(wc),"min",min(wc),"max",max(wc),"mean",sum(wc)//len(wc))
PY
```
Confirm the window holds and the count clears 5,000 (headroom for downstream attrition).

**Step 5 — Manual spot-check.** Read 10 random excerpts; each should be coherent,
self-contained, and free of boilerplate/OCR noise.

**Step 6 — Verify.**
```bash
pytest tests/test_segmentation.py -v
```

---

## 6. Reference Implementation

### `scripts/01_segment.py`

```python
# scripts/01_segment.py
"""Segment clean source texts into 300-700 word excerpts.
Input: data/raw_texts/*.txt  →  Output: data/raw_excerpts.jsonl (id, source, text)"""
import re, json, hashlib
from pathlib import Path

GUT_START = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
GUT_END   = re.compile(r"\*\*\*\s*END OF.*?\*\*\*",   re.IGNORECASE | re.DOTALL)


def strip_boilerplate(text):
    s = GUT_START.search(text)
    if s: text = text[s.end():]
    e = GUT_END.search(text)
    if e: text = text[:e.start()]
    return text.strip()


def printable_ratio(s):
    if not s: return 0.0
    good = sum(c.isalpha() or c.isspace() or c in ".,;:!?'\"-" for c in s)
    return good / len(s)


def segment_text(text, min_words=300, max_words=700):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if w > max_words and not cur:        # skip pathological single paragraph
            continue
        if n + w > max_words and n >= min_words:
            out.append(" ".join(cur)); cur, n = [p], w
        else:
            cur.append(p); n += w
    if n >= min_words:
        out.append(" ".join(cur))
    return out


def excerpt_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def main(input_dir="data/raw_texts", output_file="data/raw_excerpts.jsonl"):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    with open(output_file, "w") as out:
        for path in sorted(Path(input_dir).glob("**/*.txt")):
            cleaned = strip_boilerplate(path.read_text(errors="ignore"))
            for ex in segment_text(cleaned):
                if printable_ratio(ex) < 0.9:          # drop OCR garbage
                    continue
                uid = excerpt_id(ex)
                if uid in seen:                         # dedup identical excerpts
                    continue
                seen.add(uid)
                out.write(json.dumps({"id": uid, "source": path.name, "text": ex}) + "\n")
                n += 1
    print(f"Wrote {n} excerpts from {input_dir} to {output_file}")


if __name__ == "__main__":
    main()
```

### `scripts/fetch_gutenberg.py`

```python
# scripts/fetch_gutenberg.py
from pathlib import Path
from datasets import load_dataset

def _text_field(row):
    # Handle schema variants (TEXT/text/content) so a renamed column doesn't crash the run.
    for k in ("TEXT", "text", "content", "body"):
        if isinstance(row.get(k), str):
            return k
    str_keys = [k for k, v in row.items() if isinstance(v, str)]
    return max(str_keys, key=lambda k: len(row[k])) if str_keys else None

def main(n_books=500, out_dir="data/raw_texts", dataset="sedthh/gutenberg_english"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset, split="train", streaming=True)
    field, i = None, 0
    for i, row in enumerate(ds):
        if i >= n_books:
            break
        field = field or _text_field(row)
        if not field:
            raise SystemExit(f"No text field found; row keys = {list(row)}")
        Path(f"{out_dir}/{i:05d}.txt").write_text(row[field], errors="ignore")
    print(f"Downloaded {min(i + 1, n_books)} books to {out_dir} (field='{field}')")

if __name__ == "__main__":
    main()
```

---

## 7. Configuration & Parameters

| Parameter | Value | Rationale | If you change it |
|---|---|---|---|
| `min_words` | 300 | Floor for meaningful structure | Lower → more but thinner excerpts |
| `max_words` | 700 | Fits input+output+style in 8,192 ctx | Raise → overflow risk in M6 |
| ID scheme | MD5[:12] | Idempotent, collision-safe at scale | — |
| printable ratio | 0.9 | Drops OCR noise | Lower → more garbage admitted |
| target books | ≥ 500 | Author/genre/era diversity | Fewer → homogenization risk |

---

## 8. Alternatives Considered & Rejected

| Alternative | Why considered | Why rejected |
|---|---|---|
| Fixed token-count chunks | Uniform sizes, simpler | Severs scenes → dangling references → unpreservable excerpts |
| Sentence-window sliding chunks | Smooth coverage | Overlapping excerpts duplicate content and complicate leakage-free splits |
| Modern copyrighted fiction | Higher prose quality | Rights risk; public-domain is the safe, sufficient default |
| Sequential integer IDs | Simple | Not idempotent — re-segmentation orphans outlines/pairs |
| LLM-cleaned web prose | Abundant | Reintroduces AI slop into the clean target — defeats the project |

---

## 9. KPIs / Test File

**Test file:** `tests/test_segmentation.py`

```python
# tests/test_segmentation.py
import json
from pathlib import Path
import pytest

F = "data/raw_excerpts.jsonl"

@pytest.fixture(scope="module")
def excerpts():
    assert Path(F).exists(), "Run scripts/01_segment.py first"
    return [json.loads(l) for l in open(F)]

def test_minimum_count(excerpts):
    assert len(excerpts) >= 5000, f"Only {len(excerpts)}; want >= 5000 for headroom"

def test_fields(excerpts):
    for e in excerpts: assert set(e) >= {"id", "source", "text"}

def test_word_window(excerpts):
    for e in excerpts:
        wc = len(e["text"].split())
        assert 280 <= wc <= 760, f"{e['id']} has {wc} words"

def test_no_empty(excerpts):
    for e in excerpts: assert e["text"].strip()

def test_unique_ids(excerpts):
    ids = [e["id"] for e in excerpts]
    assert len(ids) == len(set(ids))

def test_source_diversity(excerpts):
    assert len({e["source"] for e in excerpts}) >= 200

def test_no_boilerplate(excerpts):
    for e in excerpts[:200]:
        assert "PROJECT GUTENBERG" not in e["text"].upper()
```

---

## 10. Definition of Done

- `data/raw_excerpts.jsonl` has ≥ 5,000 excerpts, all in-window, unique IDs, ≥ 200 sources.
- No license boilerplate / OCR garbage in a 10-excerpt manual spot-check.
- `data/raw_texts/` retained for Module 7.
- `pytest tests/test_segmentation.py` passes.
