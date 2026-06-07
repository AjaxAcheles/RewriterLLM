# Module 5 — Pair Filtering & Dataset Curation

| Field | Value |
|---|---|
| **Phase** | Data pipeline (stage 4 of 4) |
| **Depends on** | M2, M4 |
| **Blocks** | M6 |
| **Critical path** | Yes |
| **Owner effort** | 0.5–1 day |
| **Runtime budget** | filtering ~minutes–1 hr (embeddings + NER over thousands of pairs) |

---

## 1. Primary Objective

Reduce the raw pair corpus to a high-quality, content-aligned training set; inject null-edit
pairs; and emit formatted, leakage-free train/validation/test splits ready for SFT. Deliverables:
`data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`, and the unformatted `data/test_raw.jsonl`
used by the evaluation harness.

---

## 2. Core Concepts — Deep Dive

### 2.1 This module is the quality gate

Module 4 generates at scale but imperfectly — some teachers drift, some under-inject slop, some
hallucinate. Training on bad pairs teaches bad behavior. This module stands between raw generation
and the expensive training run, and its mandate is to be *ruthless*: discarding a questionable
pair is far cheaper than letting it poison the model. The governing principle, confirmed across
narrow-task fine-tuning work, is that **a smaller set of clean, diverse pairs beats a larger noisy
set.** Target ~2,000 curated pairs, not "as many as survive."

### 2.2 The three filters and the failure each catches

- **NER overlap ≥ 0.90.** Extract named entities (PERSON/GPE/LOC/ORG/FAC) from both sides; require
  the sloppy side to retain ≥ 90% of the clean side's entities. This is the primary defense
  against content drift — a dropped character or invented place is a hard signal the pair teaches
  content rewriting.
- **Semantic similarity ≥ 0.80 (lower bound).** Catches meaning drift: different events, emphasis,
  or resolution.
- **Semantic similarity ≤ 0.97 (upper bound).** The non-obvious one: *too high* means the teacher
  barely changed anything, so there is little slop to remove. A near-identical pair teaches the
  model nothing (or that the right edit is "do nothing" even amid slop). If the keep rate exceeds
  ~90%, the upper bound is too loose — tighten to 0.94.

### 2.3 Null-edit pairs — the over-edit antidote

Trained only on (sloppy→clean), a model learns an implicit rule: "always rewrite heavily." On
already-clean prose it then over-edits, damaging good writing. Null-edit pairs (input identical to
output, both clean human prose) break that rule — they teach that **clean input → unchanged
output.** At 10–15% of the set they instill restraint without diluting the slop-removal signal.
Their effect is measured directly by Module 9's over-edit adversarial test.

### 2.4 Leakage-free splitting — why split by base ID

The test set estimates performance on unseen data. Module 4 produced multiple sloppy versions per
clean excerpt (one per teacher), all sharing a base excerpt ID. If different versions of the same
base land in different splits, the test set has effectively seen the training content and the score
is inflated by memorization. Therefore the split operates on **base excerpt IDs**, not individual
pairs — and because it does, leakage becomes impossible *by construction*, not by a post-hoc check.

### 2.5 Why the test set is saved twice

Formatted splits wrap each pair in the chat template (needed for training symmetry/inspection).
Module 9 instead needs the *raw* sloppy input (to feed the model fresh) and the *raw* clean target
(to measure against). So the test set is written formatted (`test.jsonl`) and raw
(`test_raw.jsonl`, the evaluation input).

### 2.6 The system prompt is a frozen contract

The system prompt encodes the preservation rules in natural language and must be **byte-identical**
at training (here) and inference (Module 9). A mismatch means the model is asked at inference for
behavior it wasn't trained under. It is therefore defined exactly once in `scripts/prompt_config.py`
and imported by every consumer (this module, M6, M8, M9). The chat structure is likewise rendered
through the *model's own tokenizer template* rather than hand-written markup, so the formatted
training text and the inference-time prompt are produced by identical code — eliminating subtle
template drift (extra newlines, a default system turn, reasoning scaffolding) as a failure source.

---

## 3. Inputs & Outputs Contract

**Inputs:** `data/raw_pairs.jsonl` (M4), `data/raw_excerpts.jsonl` (M2).

**Outputs:**
- `data/filtered_pairs.jsonl` — kept pairs + `ner_overlap`, `sem_sim` fields.
- `data/null_pairs.jsonl` — identical-input/output pairs, `teacher="null"`.
- `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` — formatted `{ "text": "<templated>" }`.
- `data/test_raw.jsonl` — unformatted held-out pairs for M9.

**Invariants:** kept pairs satisfy thresholds; nulls byte-identical and 10–15% of total; splits
80/10/10 by base ID with zero cross-split base overlap; every formatted record contains the
system prompt + assistant turn.

---

## 4. Common Challenges & Solutions

**Challenge 1 — Keep rate below 40%.**
*Why:* widespread M4 drift or weak slop. *Detect:* the script prints a low keep rate. *Solve:*
return to Module 4 — strengthen the teacher prompt, re-check M3 outline quality; do not loosen
filters to compensate.

**Challenge 2 — Keep rate above 90%.**
*Why:* `max_sem` too loose — degenerate near-identical pairs slipping through. *Detect:* the script
warns. *Solve:* tighten `max_sem` to 0.94 and re-filter.

**Challenge 3 — NER filter over-drops fantasy/invented names.**
*Why:* spaCy mislabels coined names (common in SFF). *Detect:* manual review of dropped pairs shows
valid pairs rejected. *Solve:* inspect a sample; consider lowering `min_ner` to 0.85, or augment
NER with a capitalized-token heuristic.

**Challenge 4 — Model over-edits later (Module 6).**
*Why:* too few null pairs. *Detect:* M9 over-edit test fails. *Solve:* raise null fraction toward
0.15 and re-split.

**Challenge 5 — Test scores look suspiciously high (Module 6+).**
*Why:* ID leakage. *Detect:* near-perfect metrics. *Solve:* confirm the split-by-base-ID logic ran;
emit per-split base-ID manifests and assert disjointness.

**Challenge 6 — Filtering is slow.**
*Why:* re-encoding every pair with the embedder one at a time. *Detect:* long runtime. *Solve:*
batch-encode with `embedder.encode([...])`; cache NER results; run on GPU if available.

---

## 5. Step-by-Step Implementation Guide

**Step 1 — Filter.**
```bash
python scripts/03_filter_pairs.py
# reads raw_pairs.jsonl → filtered_pairs.jsonl; prints "Kept X | Dropped Y | Keep rate Z%"
```
Act on the keep rate per Challenges 1–2 before proceeding.

**Step 2 — Add null-edit pairs.**
```bash
python scripts/04_add_nulls.py
# writes null_pairs.jsonl at ~12% of filtered count
```

**Step 3 — Format and split by base ID.**
```bash
python scripts/05_format_dataset.py
# writes train/val/test.jsonl + test_raw.jsonl; prints split sizes "by N base excerpts"
```

**Step 4 — Confirm zero leakage (independent backstop).**
```bash
python - <<'PY'
import json
def bases(p):
    s=set()
    for l in open(p):
        d=json.loads(l); i=d["id"]
        s.add(i[5:] if i.startswith("null_") else i.rsplit("_",1)[0])
    return s
# test_raw carries ids; ensure none of its bases reappear in filtered training source
print("test bases:", len(bases("data/test_raw.jsonl")))
PY
```

**Step 5 — Verify.**
```bash
pytest tests/test_filtering.py -v
```

---

## 6. Reference Implementation

### `scripts/03_filter_pairs.py`

```python
# scripts/03_filter_pairs.py
import json, spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
LABELS = {"PERSON", "GPE", "LOC", "ORG", "FAC", "NORP"}

def ents(t): return {e.text.lower() for e in nlp(t).ents if e.label_ in LABELS}
def ner_overlap(c, s):
    ce = ents(c); return 1.0 if not ce else len(ce & ents(s)) / len(ce)
def sem(a, b):
    return float(util.cos_sim(embedder.encode(a, convert_to_tensor=True),
                              embedder.encode(b, convert_to_tensor=True)))

def main(raw="data/raw_pairs.jsonl", out="data/filtered_pairs.jsonl",
         min_ner=0.90, min_sem=0.80, max_sem=0.97):
    kept = dropped = 0
    with open(raw) as fin, open(out, "w") as fout:
        for line in fin:
            p = json.loads(line)
            n, s = ner_overlap(p["clean"], p["sloppy"]), sem(p["clean"], p["sloppy"])
            if n >= min_ner and min_sem <= s <= max_sem:
                p["ner_overlap"], p["sem_sim"] = round(n, 3), round(s, 3)
                fout.write(json.dumps(p) + "\n"); kept += 1
            else:
                dropped += 1
    total = kept + dropped
    print(f"Kept {kept} | Dropped {dropped} | Keep rate {kept/total:.1%}")
    if total and kept/total > 0.90: print("WARNING: keep>90% — tighten max_sem to 0.94")
    if total and kept/total < 0.40: print("WARNING: keep<40% — revisit Module 4")

if __name__ == "__main__":
    main()
```

### `scripts/04_add_nulls.py`

```python
# scripts/04_add_nulls.py
import json, random

def main(filtered="data/filtered_pairs.jsonl", excerpts="data/raw_excerpts.jsonl",
         out="data/null_pairs.jsonl", fraction=0.12, seed=42):
    n_filtered = sum(1 for _ in open(filtered))
    pool = [json.loads(l) for l in open(excerpts)]
    random.seed(seed)
    sample = random.sample(pool, min(int(n_filtered * fraction), len(pool)))
    with open(out, "w") as f:
        for e in sample:
            f.write(json.dumps({"id": f"null_{e['id']}", "clean": e["text"],
                                "sloppy": e["text"], "teacher": "null"}) + "\n")
    print(f"Wrote {len(sample)} null pairs ({fraction:.0%} of {n_filtered})")

if __name__ == "__main__":
    main()
```

### `scripts/prompt_config.py` — the frozen prompt contract

The system prompt and chat structure must be byte-identical at training and inference, so they live
in one importable module that every consumer (M5 formatting, M6/M8 training, M9 evaluation) reads
from. Rendering the chat template with the **model's own tokenizer** — rather than hand-writing
ChatML — guarantees the training text matches exactly what the model sees at inference, and gives
the SFT completion-only collator (M6) a reliable response boundary. `enable_thinking=False` is set
because the chosen base (Qwen3) otherwise injects reasoning scaffolding the editing task neither has
nor wants.

```python
# scripts/prompt_config.py
"""Single source of truth for the editor's prompt contract. Imported by M5, M6, M8, M9."""

EDITOR_SYSTEM_PROMPT = (
    "You are a prose editor. Your job is to remove AI slop patterns from the excerpt. "
    "Preserve every character, event, location, causal link, and plot beat exactly. "
    "Do not add new content. Do not improve the plot. Only improve the prose.")

# Style mode (M8) extends — does not replace — the base contract.
STYLE_SYSTEM_PROMPT = EDITOR_SYSTEM_PROMPT + (
    " Adapt the prose toward the style shown in the style reference.")

USER_TEMPLATE = "Edit this excerpt:\n\n{sloppy}"
STYLE_USER_TEMPLATE = "<style_reference>\n{reference}\n</style_reference>\n\n" + USER_TEMPLATE

# Canonical tokenizer renders the chat template identically everywhere. Proto (4B) and final (7B)
# share the same ChatML template, so either produces identical structure.
CANONICAL_TOKENIZER = "unsloth/Qwen3-7B-bnb-4bit"
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"   # completion-only loss boundary (Qwen ChatML)


def build_messages(sloppy, clean=None, style_reference=None):
    system = STYLE_SYSTEM_PROMPT if style_reference else EDITOR_SYSTEM_PROMPT
    user = (STYLE_USER_TEMPLATE.format(reference=style_reference, sloppy=sloppy)
            if style_reference else USER_TEMPLATE.format(sloppy=sloppy))
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if clean is not None:
        msgs.append({"role": "assistant", "content": clean})
    return msgs


def render(tokenizer, msgs, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
        enable_thinking=False)
```

### `scripts/05_format_dataset.py`

```python
# scripts/05_format_dataset.py
import json, random
from transformers import AutoTokenizer
from prompt_config import CANONICAL_TOKENIZER, build_messages, render

_tok = AutoTokenizer.from_pretrained(CANONICAL_TOKENIZER)

def format_pair(sloppy, clean):
    # Render with the model's own template so training text == inference input exactly.
    return {"text": render(_tok, build_messages(sloppy, clean))}

def base_id(pid):
    return pid[5:] if pid.startswith("null_") else pid.rsplit("_", 1)[0]

def main(filtered="data/filtered_pairs.jsonl", nulls="data/null_pairs.jsonl",
         seed=42, val_frac=0.10, test_frac=0.10):
    pairs = [json.loads(l) for l in open(filtered)] + [json.loads(l) for l in open(nulls)]
    by_base = {}
    for p in pairs: by_base.setdefault(base_id(p["id"]), []).append(p)
    bases = list(by_base); random.seed(seed); random.shuffle(bases)
    n = len(bases); n_test = int(n*test_frac); n_val = int(n*val_frac)
    test_b, val_b = set(bases[:n_test]), set(bases[n_test:n_test+n_val])
    train, val, test = [], [], []
    for b, grp in by_base.items():
        (test if b in test_b else val if b in val_b else train).extend(grp)
    def write_fmt(recs, path):
        with open(path, "w") as f:
            for p in recs: f.write(json.dumps(format_pair(p["sloppy"], p["clean"])) + "\n")
    write_fmt(train, "data/train.jsonl"); write_fmt(val, "data/val.jsonl")
    write_fmt(test, "data/test.jsonl")
    with open("data/test_raw.jsonl", "w") as f:
        for p in test: f.write(json.dumps(p) + "\n")
    print(f"train={len(train)} val={len(val)} test={len(test)} (by {n} base excerpts)")

if __name__ == "__main__":
    main()
```

---

## 7. Configuration & Parameters

| Parameter | Value | Rationale | If you change it |
|---|---|---|---|
| `min_ner` | 0.90 | Catch dropped/invented entities | Lower → more drift admitted |
| `min_sem` | 0.80 | Catch meaning drift | Lower → looser content match |
| `max_sem` | 0.97 (→0.94) | Reject no-slop near-duplicates | Higher → degenerate pairs slip in |
| null fraction | 0.12 | Restraint without diluting signal | Higher → less aggressive editor |
| split | 80/10/10 by base ID | Standard + leakage-proof | — |
| embedder | all-MiniLM-L6-v2 | Fast, adequate | Larger → slower, marginal gain |

---

## 8. Alternatives Considered & Rejected

| Alternative | Why considered | Why rejected |
|---|---|---|
| No filtering, keep all pairs | Maximize data | Noise poisons the model; quality beats quantity |
| Split by individual pair | Simple shuffle | Leaks same-base versions across splits → inflated scores |
| LLM judge to score pairs | Richer than NER+sim | Slow, costly, shares model blind spots; NER+sim is sufficient and cheap |
| Only lower similarity bound | Simpler | Misses degenerate no-slop pairs the upper bound catches |
| Skip null pairs | Less work | Reintroduces the over-edit failure mode |

---

## 9. KPIs / Test File

**Test file:** `tests/test_filtering.py`

```python
# tests/test_filtering.py
import json
from pathlib import Path

FILTERED, NULLS = "data/filtered_pairs.jsonl", "data/null_pairs.jsonl"
TRAIN, VAL, TEST, TEST_RAW = ("data/train.jsonl", "data/val.jsonl",
                              "data/test.jsonl", "data/test_raw.jsonl")
def L(p): return [json.loads(l) for l in open(p)]

def test_filtered_count():
    assert len(L(FILTERED)) >= 1800

def test_thresholds():
    for p in L(FILTERED):
        assert p["ner_overlap"] >= 0.90 and 0.80 <= p["sem_sim"] <= 0.97

def test_nulls_identical():
    for p in L(NULLS): assert p["clean"] == p["sloppy"]

def test_null_fraction():
    frac = sum(1 for _ in open(NULLS)) / sum(1 for _ in open(FILTERED))
    assert 0.08 <= frac <= 0.18

def test_split_sizes():
    n = sum(len(L(p)) for p in (TRAIN, VAL, TEST))
    assert 0.06 <= len(L(VAL))/n <= 0.14
    assert 0.06 <= len(L(TEST))/n <= 0.14

def test_test_set_unique_ids():
    ids = [p["id"] for p in L(TEST_RAW)]
    assert len(ids) == len(set(ids))

def test_template_present():
    from scripts.prompt_config import EDITOR_SYSTEM_PROMPT, RESPONSE_TEMPLATE
    for p in L(TRAIN)[:50]:
        assert EDITOR_SYSTEM_PROMPT in p["text"]    # the frozen prompt is actually embedded
        assert RESPONSE_TEMPLATE in p["text"]        # assistant response boundary present
```

> The strongest leakage guarantee is structural (`05_format_dataset.py` splits by base ID). For a
> fully independent check, emit per-split base-ID manifests and assert pairwise-disjoint sets.

---

## 10. Definition of Done

- ≥ 1,800 filtered pairs; nulls 10–15% and byte-identical.
- 80/10/10 split by base ID with zero leakage (by construction).
- Every formatted record carries the system prompt + assistant turn.
- `data/test_raw.jsonl` exists for Module 9.
- `pytest tests/test_filtering.py` passes.
