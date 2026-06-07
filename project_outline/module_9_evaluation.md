# Module 9 — Evaluation Harness

| Field | Value |
|---|---|
| **Phase** | Cross-cutting (build early, use throughout) |
| **Depends on** | M1 |
| **Blocks** | M6, M7, M8 completion gates |
| **Critical path** | Yes — must exist before M6 finishes |
| **Owner effort** | 2–3 days (build in parallel with M2–M5) |
| **Runtime budget** | full test-set eval ~20–60 min · in-loop subset ~1–2 min |

---

## 1. Primary Objective

Provide the single source of truth for "is the model good": a reusable metric library, an automated
test-set runner with tiered pass/warn/fail thresholds, and the human-preference and adversarial
protocols that calibrate and stress-test the automated numbers. Deliverables:
`scripts/eval_metrics.py`, `scripts/run_eval.py`, `scripts/run_adversarial.py`, adversarial
fixtures, and documented human-eval protocols.

---

## 2. Core Concepts — Deep Dive

### 2.1 Why this module defines success for three others

Modules 6, 7, 8 each declare completion in metric terms ("slop_delta > 1.0", "slop down without
regression", "no core regression"). Those statements are meaningless until this module computes
them. The harness is the instrument that turns "the run finished" into "the run *worked*." That is
why the dependency map pulls it onto the critical path *early*: a project that writes the evaluator
after the trainer cannot know whether the trainer succeeded, and will block on building it at the
worst moment. **Build M9 in parallel with the data pipeline (M2–M5)** so it is ready before M6's
final run.

### 2.2 Why no single automated metric is trusted

Automated evaluation of writing quality is unsolved; the best automated judges agree with human
preference only ~73–78% of the time. So the harness relies on a *constellation* of cheap proxies,
each capturing one facet, and treats the human-preference protocol as the ground truth they are
calibrated against. **Calibration rule:** if automated metrics improve but human preference does
not, the metrics are measuring the wrong thing and must be revised. Automated metrics gate
iteration speed; human judgment gates trust.

### 2.3 Why tiered thresholds, not binary pass/fail

Binary pass/fail says *whether* you failed, not *how close* you are — useless for iteration. Each
metric reports fail / warn / pass, so a change can be seen to move a metric from fail to warn even
before it reaches pass. That gradient makes the harness a development tool, not just a final gate.

### 2.4 The metric constellation and what each defends

| Metric | Captures | Defends against | Target |
|---|---|---|---|
| `slop_score` | banned-pattern density / 1k words | lexical slop | lower |
| `slop_delta` | reduction input→output | "doing nothing" | > 1.0 |
| `burstiness` | sentence-length variance | rhythm flattening | higher |
| `ner_preservation` | input entities surviving | content drift | ≥ 0.95 |
| `hallucination_rate` | output entities that are new | invented content | < 0.05 |
| `semantic_sim` in→out | content kept but edited | over/under-editing | 0.80–0.92 |
| `sem_vs_target` | closeness to human clean target | wrong target | ≥ 0.80 |

### 2.5 Why the adversarial tests exist

Proxies measure aggregate behavior; adversarial tests probe *specific failure modes* with
purpose-built inputs:
- **Over-edit** (clean prose in → expect near-no-change) validates M5's null pairs worked.
- **Voice-flattening** (strong narrators → expect profile preserved) probes M7 diversity / M8
  conditioning.
- **Tone-preservation** (tone-tagged → expect tone survives) catches altered-content edits.
- **Hallucination-introduction** (expect no new entities) guards the core preservation contract.

Together: proxies confirm the model does the right thing on average; adversarial tests confirm it
isn't doing specific wrong things on traps.

### 2.6 One definition, used in two places

`eval_metrics.py` is a pure-function library (text in, number out — no model/IO), so it is
independently unit-testable *and* importable as the in-loop `compute_metrics` during training. Same
definitions drive in-loop monitoring and final gating — no divergence between "training eval" and
"final eval." The metric **names are a frozen contract**: M6/M7/M8 completion tests read these exact
keys.

---

## 3. Inputs & Outputs Contract

**Inputs:** `data/test_raw.jsonl` (M5); any model directory under `models/`; adversarial fixtures.

**Outputs:**
- `scripts/eval_metrics.py` — metric functions (frozen names).
- `scripts/run_eval.py` — runner → `reports/<name>.json` (consumed by M6/M7/M8 tests).
- `scripts/run_adversarial.py` + fixtures `data/eval_clean_prose.jsonl`,
  `data/eval_strong_voice.jsonl`, `data/eval_tone_tagged.jsonl`.
- Documented human-preference protocol.

**Frozen report keys:** `slop_delta`, `ner_preservation`, `hallucination_rate`, `sem_vs_target`,
`burstiness_improved_rate` (plus informational `slop_in`, `slop_out`, `semantic_sim`).

---

## 4. Common Challenges & Solutions

**Challenge 1 — Automated metrics improve but humans aren't convinced.**
*Why:* proxies miss real quality dimensions. *Detect:* human win-rate flat while slop_delta rises.
*Solve:* expand the blacklist; add structural-slop checks; trust the human signal and recalibrate.

**Challenge 2 — `ner_preservation` is noisy.**
*Why:* spaCy mislabels invented/fantasy names. *Detect:* score varies on clearly-faithful outputs.
*Solve:* supplement NER with a capitalized-token heuristic; manually audit a sample.

**Challenge 3 — Over-edit score low on clean prose.**
*Why:* too few null pairs in M5. *Detect:* `run_adversarial` over-edit similarity low. *Solve:* raise
M5 null fraction; retrain.

**Challenge 4 — Report keys mismatch the module tests.**
*Why:* metric names drifted between M9 and the consumers. *Detect:* M6/M7/M8 tests KeyError. *Solve:*
freeze the names listed in §3; never rename without updating all consumers.

**Challenge 5 — Evaluation is slow.**
*Why:* running the full test set every training step. *Detect:* training stalls at eval. *Solve:* use
a 50-pair subset for in-loop eval; run the full set only at checkpoints.

**Challenge 6 — Tone judge is inconsistent.**
*Why:* single LLM-judge call has high variance. *Detect:* tone verdicts flip on re-run. *Solve:*
average multiple judgments, or human-tag the tone fixture.

---

## 5. Step-by-Step Implementation Guide

**Step 1 — Build the metric library** (`eval_metrics.py`) with the frozen-name functions (§6).

**Step 2 — Unit-test the metrics** on hand-built known inputs before trusting them on models.
```bash
pytest tests/test_eval_metrics.py -v
```

**Step 3 — Build the runner** (`run_eval.py`): load any model dir, edit every `test_raw.jsonl`
pair, compute metrics, print the tiered table, write the JSON report.

**Step 4 — Self-test on the M6 prototype** (proves the runner end-to-end before the 7B run).
```bash
python scripts/run_eval.py models/sft_lora reports/sft_proto_eval.json
```

**Step 5 — Build adversarial fixtures + runner.** Curate
`data/eval_clean_prose.jsonl` (slop-free human excerpts),
`data/eval_strong_voice.jsonl` (idiosyncratic narrators),
`data/eval_tone_tagged.jsonl` (`{text, tone}`); implement `run_adversarial.py`.

**Step 6 — Document the human-preference protocol** (blind pairwise, 20–30 pairs, win-rate tracked).

**Step 7 — Wire `evaluate_pair` into training** as the in-loop `compute_metrics` (subset).

---

## 6. Reference Implementation

### `scripts/eval_metrics.py`

```python
# scripts/eval_metrics.py  — pure functions; FROZEN metric names.
import re, spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

SLOP_PATTERNS = [
    r"\btapestry\b", r"\bdelve\b", r"\btestament\b", r"\bvibrant\b", r"\bnavigate\b",
    r"\bnuanced\b", r"\bresonate\b", r"\bpivotal\b", r"\bmultifaceted\b", r"\bunderscore\b",
    r"\bembark\b", r"\brealm\b", r"\bshimmered\b", r"\bunsettlingly\b",
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin addition\b", r"\bit is worth noting\b",
    r"\bsomething shifted\b", r"\beverything changed\b", r"\bbut here's the thing\b",
    r"\bat the end of the day\b", r"it's not .+, it's",
]
LABELS = {"PERSON", "GPE", "LOC", "ORG", "FAC"}

def slop_score(text):
    t = text.lower(); hits = sum(1 for p in SLOP_PATTERNS if re.search(p, t))
    w = len(text.split()); return hits / (w / 1000) if w else 0.0

def burstiness(text):
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(s.split()) for s in sents]
    if len(lengths) < 2: return 0.0
    m = sum(lengths) / len(lengths)
    return (sum((l - m) ** 2 for l in lengths) / len(lengths)) ** 0.5

def _ents(t): return {e.text.lower() for e in nlp(t).ents if e.label_ in LABELS}

def ner_preservation(inp, out):
    ie = _ents(inp); return 1.0 if not ie else len(ie & _ents(out)) / len(ie)

def hallucination_rate(inp, out):
    ie, oe = _ents(inp), _ents(out); return len(oe - ie) / max(len(oe), 1)

def sem_sim(a, b):
    return float(util.cos_sim(embedder.encode(a, convert_to_tensor=True),
                              embedder.encode(b, convert_to_tensor=True)))

def evaluate_pair(sloppy_input, model_output, clean_target=None):
    r = {"slop_in": slop_score(sloppy_input), "slop_out": slop_score(model_output),
         "slop_delta": slop_score(sloppy_input) - slop_score(model_output),
         "burstiness_out": burstiness(model_output),
         "burstiness_improved": burstiness(model_output) >= burstiness(sloppy_input),
         "ner_preservation": ner_preservation(sloppy_input, model_output),
         "hallucination_rate": hallucination_rate(sloppy_input, model_output),
         "semantic_sim": sem_sim(sloppy_input, model_output)}
    if clean_target: r["sem_vs_target"] = sem_sim(model_output, clean_target)
    return r
```

### `scripts/run_eval.py`

```python
# scripts/run_eval.py
import json, sys, torch
from pathlib import Path
from unsloth import FastLanguageModel
from eval_metrics import evaluate_pair
from prompt_config import build_messages

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/ftpo_model"
REPORT = sys.argv[2] if len(sys.argv) > 2 else "reports/eval.json"

model, tok = FastLanguageModel.from_pretrained(MODEL_DIR, max_seq_length=8192,
                                               dtype=None, load_in_4bit=True)
FastLanguageModel.for_inference(model)

def edit(sloppy):
    # Identical prompt contract and template used to build the training data (M5).
    ids = tok.apply_chat_template(build_messages(sloppy), tokenize=True,
                                  add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=1024, temperature=0.3,
                             do_sample=True, repetition_penalty=1.1)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

results = [evaluate_pair(p["sloppy"], edit(p["sloppy"]), p.get("clean"))
           for p in (json.loads(l) for l in open("data/test_raw.jsonl"))]
# Average only true numerics (exclude bool fields like burstiness_improved).
numeric = [k for k, v in results[0].items()
           if isinstance(v, (int, float)) and not isinstance(v, bool)]
avg = {k: sum(r[k] for r in results) / len(results) for k in numeric}
avg["burstiness_improved_rate"] = sum(r["burstiness_improved"] for r in results) / len(results)

TIERS = {"slop_delta": (1.0, 2.0), "ner_preservation": (0.90, 0.95),
         "hallucination_rate": (0.10, 0.05), "sem_vs_target": (0.75, 0.80),
         "burstiness_improved_rate": (0.50, 0.70)}
def tier(m, v):
    fail, passv = TIERS[m]
    if m == "hallucination_rate":
        return "PASS" if v < passv else "WARN" if v < fail else "FAIL"
    return "PASS" if v >= passv else "WARN" if v >= fail else "FAIL"

print("\n── Evaluation:", MODEL_DIR, "──")
for k in TIERS:
    print(f"  {k:28s} {avg.get(k, float('nan')):7.3f}   [{tier(k, avg.get(k, 0))}]")
print("  (informational)")
for k in ("slop_in", "slop_out", "semantic_sim", "burstiness_out"):
    print(f"  {k:28s} {avg.get(k, float('nan')):7.3f}")
Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
json.dump(avg, open(REPORT, "w"), indent=2)
print(f"\nReport → {REPORT}")
```

### `scripts/run_adversarial.py` (sketch)

```python
# scripts/run_adversarial.py
import json
from eval_metrics import sem_sim, hallucination_rate

def over_edit_score(model_edit, f="data/eval_clean_prose.jsonl"):
    """On already-clean prose, the model should barely change anything (expect ~0.97+)."""
    sims = [sem_sim(json.loads(l)["text"], model_edit(json.loads(l)["text"])) for l in open(f)]
    return sum(sims) / len(sims)
```

### Human-preference protocol (documented)

```
1. Sample 20-30 pairs from data/test_raw.jsonl; run the model to produce outputs.
2. Build a blind sheet: show INPUT and OUTPUT with random A/B labels.
3. Reviewer marks which reads as more natural / less AI-like.
4. Record win-rate (output preferred / total); track across checkpoints.
5. Calibration: if slop_delta rises but win-rate doesn't, revise the proxies.
```

---

## 7. Configuration & Parameters

| Parameter | Value | Rationale | If you change it |
|---|---|---|---|
| inference temp | 0.3 | Faithful, deterministic editing | Higher → creative drift |
| `max_new_tokens` | 1024 | Full rewrite of a 700-word excerpt | Lower → truncated output |
| embedder | all-MiniLM-L6-v2 | Consistent with M5 | — |
| tiers | fail/warn/pass | Iteration gradient | binary → no gradient |
| human sample | 20–30 | Manageable blind review | Fewer → noisier signal |
| in-loop subset | ~50 pairs | Fast periodic eval | Full set → slow training |

---

## 8. Alternatives Considered & Rejected

| Alternative | Why considered | Why rejected |
|---|---|---|
| Single quality score (e.g. one LLM judge) | Simple | Unreliable; hides which facet failed; no iteration signal |
| Perplexity as quality proxy | Cheap | Rewards bland, predictable text — anti-correlated with good prose |
| Off-the-shelf AI-detector only | Turnkey | Brittle, gameable; doesn't measure content preservation |
| Build M9 after M6 | Less upfront work | M6 can't be validated; blocks at the worst time |
| Skip adversarial tests | Faster | Misses the specific failure modes the pipeline is designed to prevent |

---

## 9. KPIs / Test File

**Test file:** `tests/test_eval_metrics.py`

```python
# tests/test_eval_metrics.py
from scripts.eval_metrics import (slop_score, burstiness, ner_preservation,
                                  hallucination_rate, sem_sim)

SLOPPY = ("Furthermore, the vibrant tapestry of the city resonated with a profound and "
          "multifaceted sense of nuance. It is worth noting that everything changed.")
CLEAN = "Rain slicked the streets. A tram rattled past. She pulled her coat tighter."

def test_slop_higher_on_slop():
    assert slop_score(SLOPPY) > slop_score(CLEAN)

def test_slop_zero_on_clean():
    assert slop_score("The dog ran across the yard and barked once.") == 0.0

def test_burstiness_varied_vs_uniform():
    uniform = "I walk home. I eat food. I read books. I sleep now. I wake up."
    varied = ("I walk. Then, after a long winding detour through the old quarter where the "
              "lamps flicker, I finally arrive home. I sleep.")
    assert burstiness(varied) > burstiness(uniform)

def test_ner_full_and_partial():
    inp = "Alice met Bob in Paris."
    assert ner_preservation(inp, "Alice met Bob in Paris last week.") == 1.0
    assert ner_preservation(inp, "Alice walked alone.") < 1.0

def test_hallucination_detected():
    assert hallucination_rate("Alice walked home.", "Alice and Bob walked to London.") > 0.0

def test_sem_sim_bounds():
    assert sem_sim("a cat sat on a mat", "a cat sat on a mat") > 0.98
    assert sem_sim("quantum chromodynamics", "a recipe for banana bread") < 0.5
```

---

## 10. Definition of Done

- `pytest tests/test_eval_metrics.py` passes (metric library correct).
- `run_eval.py` runs end-to-end on the M6 prototype, prints the tiered table, writes a JSON report.
- Adversarial fixtures built; `run_adversarial.py` runs.
- Human-preference protocol documented and runnable.
- `evaluate_pair` wired into training as in-loop `compute_metrics`.
- **Reached "done" before the M6 final run completes.**
