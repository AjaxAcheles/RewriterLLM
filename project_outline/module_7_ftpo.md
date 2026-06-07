# Module 7 — Slop Profiling & FTPO Suppression (Phase B2)

| Field | Value |
|---|---|
| **Phase** | Training (stage 2 of 2 — surgical suppression) |
| **Depends on** | M6 (SFT model + baseline), M9 (eval) |
| **Blocks** | M8 (optional) |
| **Critical path** | Yes (through M7) |
| **Owner effort** | 1–2 days |
| **Runtime budget** | profile ~1–2 hrs · FTPO-pair gen ~2–4 hrs · FTPO train ~2–6 hrs |

---

## 1. Primary Objective

Measure the residual slop fingerprint of the SFT model, auto-generate FTPO preference pairs from
its own outputs, and run Final Token Preference Optimization to surgically suppress residual slop
**without** degrading editing behavior or collapsing lexical diversity. Deliverable:
`models/ftpo_model/` plus `reports/ftpo_eval.json` showing slop down and preservation/diversity held
versus the M6 baseline.

---

## 2. Core Concepts — Deep Dive

### 2.1 Why a second stage exists

M6's SFT installed the editing transformation but, by the nature of its loss, left residual slop —
it rewarded clean outputs without actively suppressing patterns baked into the base model's RLHF.
M7 installs that missing suppression. The stages are complementary: **SFT teaches *how to edit*
(FTPO cannot); FTPO *removes residual slop* (SFT cannot).** FTPO is a scalpel applied after the
surgery, operating on a model that already edits.

### 2.2 Why FTPO and not DPO — the decisive evidence

The obvious tool is preference optimization (DPO/ORPO), and the pipeline's (clean, sloppy) pairs fit
its shape. The plan rejects DPO on evidence: on this exact task DPO **degrades writing quality and
collapses lexical diversity** while achieving *weaker* slop suppression than the purpose-built
alternative. Diversity collapse — convergence to a narrower vocabulary and more uniform sentence
structure — is itself a form of slop. DPO would risk trading lexical slop for structural slop. FTPO
was designed to avoid exactly this.

### 2.3 How FTPO works, mechanistically

A model emits each token from a probability distribution over the vocabulary; slop tokens have
inflated probability. FTPO lowers the probability of specific slop tokens at the moments they would
appear, while pinning the rest of the vocabulary in place. It trains on minimal **final-token
preference pairs**: a *prompt* (generation up to where slop would start, e.g. "The evening sky was
a"), a *rejected* token ("tapestry"), and *chosen* alternatives ("blaze", "wash", "field").

Its loss has three terms:
1. **Margin-based preference loss** — push chosen above rejected, then *deactivate* once the margin
   is met (prevents overtraining a token that's already fixed).
2. **Target regularization** — tether chosen/rejected logits near reference values.
3. **Non-target regularization (strong)** — anchor the *rest* of the vocabulary to reference.

The decisive choice is operating in **logit space with MSE regularization** (not probability space
with KL): this localizes the update to the adjusted tokens and prevents the whole-distribution
shift behind DPO's collapse. Reported result: ~90% slop reduction while maintaining or improving
quality, diversity, and cross-domain benchmarks.

### 2.4 The data is self-supervised against the model's own tendencies

FTPO pairs come from the *SFT model's own outputs*. Profiling first measures which patterns the
post-SFT model overuses versus the human baseline (M2's Gutenberg corpus). Then the Antislop Sampler
generates from the SFT model and, wherever it would emit a banned pattern, backtracks and records the
rejected token plus the coherent alternatives min-p filtering offers — producing a preference pair
automatically. The model is trained to suppress exactly the patterns it personally overuses.

### 2.5 Why human review of the fingerprint is mandatory

The profiler ranks overrepresented patterns. Some are true slop ("tapestry", "shimmered"); some are
*genre-appropriate* (a horror corpus legitimately overuses "shadows") or proper nouns frequent in
the data. Left in the banlist, FTPO suppresses legitimate vocabulary, damaging range. A human must
prune the fingerprint before training. Required, not optional.

### 2.6 The central risk and its three guardrails

An over-aggressive FTPO pass can disturb the editing behavior M6 installed (large logit shifts
perturb weights beyond slop-relevant distributions). Guardrails: a *low learning rate* (5e-5, an
order below SFT), FTPO's *margin deactivation*, and a *mandatory regression check* against the M6
baseline. Success = slop metrics improve **and** preservation/diversity hold — improvement without
regression.

---

## 3. Inputs & Outputs Contract

**Inputs:** `models/sft_merged/` (M6), `data/raw_texts/` (M2 human baseline),
`reports/sft_baseline.json` (M6), auto-antislop (M1).

**Outputs:**
- `antislop_data/slop_fingerprint.json` — profiled overrepresentation.
- `antislop_data/fingerprint_review.csv` + `antislop_data/banlist.json` — human-pruned banlist.
- `antislop_data/ftpo_pairs.jsonl` — `{prompt, rejected, chosen[]}` records.
- `models/ftpo_model/` — the suppressed model.
- `reports/ftpo_eval.json` — compared field-by-field to the M6 baseline.

**Gate:** slop_delta ≥ baseline; ner_preservation ≥ baseline − 0.02; burstiness_improved_rate ≥
baseline − 0.05; sem_vs_target ≥ baseline − 0.02.

---

## 4. Common Challenges & Solutions

**Challenge 1 — NER preservation drops after FTPO.**
*Why:* logit shifts too large; editing behavior disturbed. *Detect:* `ftpo_eval.json` ner below
baseline. *Solve:* lower LR to 1e-5; reduce epochs to 1; verify margin deactivation is firing.

**Challenge 2 — Diversity/burstiness drops.**
*Why:* non-target anchor too weak → partial distribution collapse. *Detect:* burstiness_improved_rate
falls. *Solve:* raise `lambda-nontarget`; raise `margin`.

**Challenge 3 — Slop barely reduced.**
*Why:* banlist over-pruned, or ban-strength too low. *Detect:* slop_delta ≈ baseline. *Solve:*
restore valid entries; raise ban-strength toward 1.0; generate more FTPO pairs.

**Challenge 4 — FTPO suppresses legitimate words.**
*Why:* genre words / proper nouns left in the banlist. *Detect:* outputs avoid setting-appropriate
vocabulary. *Solve:* re-review the fingerprint; prune more carefully; document decisions in the CSV.

**Challenge 5 — Output becomes incoherent.**
*Why:* model damaged by excessive shifts. *Detect:* garbled or repetitive generations. *Solve:*
lower LR; fewer steps; inspect the margin-deactivation rate (should be high once converged).

**Challenge 6 — Profiling is slow.**
*Why:* large sample/model. *Detect:* long runtime. *Solve:* reduce `n_samples` to 300; run the
merged model on GPU.

---

## 5. Step-by-Step Implementation Guide

**Step 1 — Profile the SFT model.**
```bash
python scripts/07_profile_slop.py
# writes antislop_data/slop_fingerprint.json (ranked overrepresentation vs Gutenberg)
```

**Step 2 — Human-prune the fingerprint.** Open the fingerprint; for each top pattern decide KEEP
(true slop) or PRUNE (genre/proper-noun/legitimate). Record in
`antislop_data/fingerprint_review.csv`; write KEEP-only entries to `antislop_data/banlist.json`.

**Step 3 — Generate FTPO pairs + train.**
```bash
python scripts/08_train_ftpo.py
# 1) Antislop Sampler → antislop_data/ftpo_pairs.jsonl
# 2) FTPO trainer → models/ftpo_model/
```

**Step 4 — Evaluate against the baseline.**
```bash
python scripts/run_eval.py models/ftpo_model reports/ftpo_eval.json
```

**Step 5 — Regression triage.** Compare `ftpo_eval.json` to `sft_baseline.json`. If ner or
burstiness regressed, lower LR to 1e-5 (and/or epochs to 1) and re-run Step 3–4.

**Step 6 — Verify.**
```bash
pytest tests/test_ftpo.py -v
```

---

## 6. Reference Implementation

### `scripts/07_profile_slop.py`

```python
# scripts/07_profile_slop.py
"""Profile SFT model residual slop vs the human (Gutenberg) baseline (wraps auto-antislop)."""
import subprocess, sys
from pathlib import Path

def _entry(name):
    # auto-antislop's CLI layout can vary by version; fail loudly with the actual file list
    # instead of an opaque subprocess error if the expected entry point isn't present.
    p = Path("auto-antislop") / name
    if not p.exists():
        avail = sorted(q.name for q in Path("auto-antislop").glob("*.py"))
        raise SystemExit(f"Expected '{p}' not found. Align to the cloned repo's CLI; "
                         f"available top-level scripts: {avail}")
    return str(p)

def main(model_dir="models/sft_merged", output_dir="antislop_data",
         baseline_corpus="data/raw_texts", n_samples=500):
    subprocess.run([sys.executable, _entry("profile_model.py"),
                    "--model", model_dir, "--output-dir", output_dir,
                    "--baseline-corpus", baseline_corpus, "--n-samples", str(n_samples),
                    "--temperature", "0.9", "--max-tokens", "512"], check=True)
    print(f"Fingerprint at {output_dir}/slop_fingerprint.json — REVIEW AND PRUNE before FTPO")

if __name__ == "__main__":
    main()
```

**Human pruning record — `antislop_data/fingerprint_review.csv`:**
```csv
pattern,ratio_vs_human,decision,reason
tapestry,2310,KEEP,classic AI filler
shadows,180,PRUNE,genre-appropriate for horror corpus
Elara,40000,KEEP,AI name fixation
deck,95,PRUNE,nautical setting legitimate
```

### `scripts/08_train_ftpo.py`

```python
# scripts/08_train_ftpo.py
"""Generate FTPO preference pairs from the SFT model, then run FTPO (wraps auto-antislop)."""
import subprocess, sys
from pathlib import Path

def _entry(name):
    p = Path("auto-antislop") / name
    if not p.exists():
        avail = sorted(q.name for q in Path("auto-antislop").glob("*.py"))
        raise SystemExit(f"Expected '{p}' not found. Align to the cloned repo's CLI; "
                         f"available top-level scripts: {avail}")
    return str(p)

def main(sft_model="models/sft_merged", banlist="antislop_data/banlist.json",
         pairs_out="antislop_data/ftpo_pairs.jsonl", output_dir="models/ftpo_model",
         n_pairs=2000, epochs=1, lr=5e-5):
    subprocess.run([sys.executable, _entry("generate_ftpo_data.py"),
                    "--model", sft_model, "--banlist", banlist, "--output", pairs_out,
                    "--n-samples", str(n_pairs), "--ban-strength", "0.7",
                    "--temperature", "0.9"], check=True)
    subprocess.run([sys.executable, _entry("train_ftpo.py"),
                    "--model", sft_model, "--data", pairs_out, "--output", output_dir,
                    "--epochs", str(epochs), "--learning-rate", str(lr),
                    "--margin", "2.0", "--lambda-target", "0.1",
                    "--lambda-nontarget", "1.0"], check=True)
    print(f"FTPO complete → {output_dir}. Re-run run_eval.py and compare to SFT baseline.")

if __name__ == "__main__":
    main()
```

---

## 7. Configuration & Parameters

| Parameter | Value | Rationale | If you change it |
|---|---|---|---|
| profile samples | 500 | Reliable frequency ratios | Fewer → noisier fingerprint |
| baseline | Gutenberg (M2) | Human reference | — |
| FTPO pairs | ~2,000 | Cover the banlist | Fewer → incomplete suppression |
| ban-strength | 0.7 | Soft-ban; allow when no good alt | Higher → more aggressive |
| epochs | 1 | Usually enough | More → regression risk |
| LR | 5e-5 (→1e-5) | Protect SFT behavior | Higher → damages editing |
| margin | 2.0 | Deactivation threshold | Lower → overtraining |
| λ_target | 0.1 | Mild tether | — |
| λ_nontarget | 1.0 | Anchor vocab → no collapse | Lower → diversity risk |

---

## 8. Alternatives Considered & Rejected

| Alternative | Why considered | Why rejected |
|---|---|---|
| DPO/ORPO | Pairs already exist | Degrades quality, collapses diversity, weaker suppression |
| Antislop Sampler at inference (no FTPO) | No extra training | 69–96% throughput hit — unusable for deployment; it's a data tool |
| Just expand the SFT banlist of words | Simpler | SFT can't *suppress* probabilities; only FTPO operates in logit space |
| Skip human pruning | Faster | Suppresses genre-legitimate vocabulary; damages range |
| High LR for stronger effect | More slop removed | Disturbs M6 editing behavior; fails the regression gate |

---

## 9. KPIs / Test File

**Test file:** `tests/test_ftpo.py`

```python
# tests/test_ftpo.py
import json
from pathlib import Path
import pytest

FP, BAN = "antislop_data/slop_fingerprint.json", "antislop_data/banlist.json"
PAIRS, FTPO_DIR = "antislop_data/ftpo_pairs.jsonl", "models/ftpo_model"
SFT_BASE, FTPO_EVAL = "reports/sft_baseline.json", "reports/ftpo_eval.json"

def test_fingerprint_generated():
    assert Path(FP).exists(); assert len(json.load(open(FP))) > 0

def test_banlist_pruned():
    assert Path(BAN).exists(), "Pruned banlist missing (human review skipped)"

def test_pairs_structure():
    for p in [json.loads(l) for l in open(PAIRS)][:50]:
        assert "prompt" in p and "rejected" in p
        assert p.get("chosen") and len(p["chosen"]) >= 1

def test_model_saved():
    assert Path(FTPO_DIR).is_dir()

@pytest.mark.skipif(not (Path(SFT_BASE).exists() and Path(FTPO_EVAL).exists()),
                    reason="eval reports not present")
def test_improvement_without_regression():
    b, f = json.load(open(SFT_BASE)), json.load(open(FTPO_EVAL))
    assert f["slop_delta"] >= b["slop_delta"], "slop not reduced further"
    assert f["ner_preservation"] >= b["ner_preservation"] - 0.02, "NER regressed"
    assert f["burstiness_improved_rate"] >= b["burstiness_improved_rate"] - 0.05, "diversity regressed"
    assert f["sem_vs_target"] >= b["sem_vs_target"] - 0.02, "target fidelity regressed"
```

---

## 10. Definition of Done

- Fingerprint generated; human pruning recorded; `banlist.json` written.
- `ftpo_pairs.jsonl` has valid (prompt, rejected, chosen[]) structure.
- `models/ftpo_model/` saved and coherent.
- `reports/ftpo_eval.json` shows slop down with preservation/diversity held vs. the M6 baseline.
- `pytest tests/test_ftpo.py` passes.
