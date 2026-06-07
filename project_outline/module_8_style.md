# Module 8 — Style Conditioning (Phase B3, Optional)

| Field | Value |
|---|---|
| **Phase** | Training (optional extension) |
| **Depends on** | M7 (FTPO model), M9 (eval) |
| **Blocks** | Nothing — terminal optional module |
| **Critical path** | No (deferrable without blocking completion) |
| **Owner effort** | 1–2 days |
| **Runtime budget** | pair build ~minutes · style train ~3–8 hrs |

---

## 1. Primary Objective

Add an optional capability: given a 3–5-paragraph style reference, the model bends its clean output
toward that author's measurable stylistic profile — without losing slop removal or content
preservation. Deliverable: `models/style_lora/` plus evidence of stylometric convergence and an
unchanged core-editing regression check.

---

## 2. Core Concepts — Deep Dive

### 2.1 What style conditioning changes

After Modules 6–7 the editor converges every input toward a generic "clean prose" register. Style
conditioning makes that target register *controllable*: given a reference block demonstrating an
author's voice, the cleaned output bends toward that voice — sentence-length distribution,
punctuation habits, dialogue density, register, vocabulary. The contract is unchanged: content
preserved, slop removed; only the *aesthetic register* shifts.

### 2.2 Why train it rather than prompt it

Style mimicry is the task where prompting is weakest. Models approximate style in structured
formats (news, email) but consistently fail at the implicit choices defining a writer's voice from a
few in-context examples. Prompting ("write like this") yields superficial imitation. Training on
examples of *how a reference influences editing decisions* installs the behavior at the weight
level — the model has seen many "given this reference, the styled output looks like this" cases and
generalizes the conditioning.

### 2.3 Why a light second pass on top of M7

Modules 6–7 installed editing and suppression at significant effort; an aggressive style retrain
risks overwriting them. So style is a fresh low-rank adapter, low LR, few epochs — adding the
conditioning while minimally perturbing existing weights. Same restraint principle as M7's FTPO
pass.

### 2.4 Why conditioned and unconditioned pairs must not be mixed

Mixing inputs with and without a `<style_reference>` block in this pass confuses the model about
when a block is expected and how to weight it. Training *only* on conditioned pairs gives a clean
signal: "when a reference is present, condition on it." The base model already handles the
no-reference case from Modules 6–7; at inference the block's presence selects the behavior.

### 2.5 How style data is constructed (same-author references)

For each (sloppy, clean) pair, attach a style reference of 3–5 *other* excerpts by the **same source
author** as the clean target (using M2's `source` as the author proxy). The model learns the
reference governs register while the same-author clean target demonstrates the correct styled
output. Cross-author references can be added later to test transfer; same-author pairing is the
training default.

### 2.6 Measuring success: convergence + no regression

Two tests. **Convergence:** the styled output's stylometric feature vector (avg/var sentence length,
comma rate, dialogue proxy, adverb rate) moves *closer to the reference* than the non-conditioned
baseline output does — optionally confirmed by an authorship-attribution classifier above chance.
**No regression:** M9 confirms core NER/slop metrics did not degrade versus M7.

---

## 3. Inputs & Outputs Contract

**Inputs:** `models/ftpo_model/` (M7), `data/filtered_pairs.jsonl` (M5),
`data/raw_excerpts.jsonl` (M2, for same-author references), `reports/ftpo_eval.json`.

**Outputs:**
- `data/style_train.jsonl` — style-template records (reference block + excerpt → styled clean).
- `models/style_lora/` — style adapters.
- `reports/style_eval.json` — regression check vs. M7.

**Invariants:** every training record contains a `<style_reference>` block + assistant turn; pairs
are conditioned-only; styled output converges toward the reference; core metrics not degraded.

---

## 4. Common Challenges & Solutions

**Challenge 1 — Core editing degrades after the style pass.**
*Why:* style LR too high or too many epochs overwrote M6–M7 behavior. *Detect:* `style_eval.json`
ner/slop below M7. *Solve:* lower LR to 2e-5; 1 epoch; lower rank.

**Challenge 2 — No stylometric movement.**
*Why:* the model ignores the reference block. *Detect:* convergence test shows styled ≈ baseline
distance. *Solve:* confirm pairs are conditioned-only; raise adapter rank; verify the block is
inside the user turn at train and inference.

**Challenge 3 — Content drift in styled output.**
*Why:* the pass loosened preservation while chasing style. *Detect:* NER/hallucination worsen.
*Solve:* re-emphasize preservation in the system prompt; tighten the regression gate.

**Challenge 4 — Context OOM.**
*Why:* reference block + excerpt + output exceeds budget. *Detect:* OOM at 8192. *Solve:* reduce
`k_ref`; trim reference excerpt length; lower `max_seq_length`.

**Challenge 5 — Style "bleeds" even without a reference at inference.**
*Why:* conditioned + unconditioned pairs were mixed. *Detect:* no-reference inputs come out stylized.
*Solve:* retrain the style pass on conditioned pairs ONLY.

**Challenge 6 — Same-author references too few for some authors.**
*Why:* a source has < k_ref+1 excerpts. *Detect:* builder skips many pairs. *Solve:* lower `k_ref`;
group thin authors; accept fewer style pairs (this is an optional capability).

---

## 5. Step-by-Step Implementation Guide

**Step 1 — Build style pairs.**
```bash
python scripts/build_style_pairs.py
# writes data/style_train.jsonl; each record has a <style_reference> block (same-author)
```

**Step 2 — Train the style adapter** on top of the M7 model.
```bash
python scripts/09_train_style.py 2>&1 | tee reports/style_train.log
```

**Step 3 — Regression check.**
```bash
python scripts/run_eval.py models/style_lora reports/style_eval.json
# compare to reports/ftpo_eval.json: core NER and slop must not regress
```

**Step 4 — Convergence check.** Run the model on a few held excerpts with a known-author reference;
compute style vectors of (reference, baseline output, styled output) and confirm the styled output
is closer to the reference.

**Step 5 — Verify.**
```bash
pytest tests/test_style.py -v
```

---

## 6. Reference Implementation

### `scripts/build_style_pairs.py`

```python
# scripts/build_style_pairs.py
"""Style-conditioned pairs; reference = 3-5 same-author excerpts. → data/style_train.jsonl"""
import json, random
from collections import defaultdict
from transformers import AutoTokenizer
from prompt_config import CANONICAL_TOKENIZER, build_messages, render

_tok = AutoTokenizer.from_pretrained(CANONICAL_TOKENIZER)

def fmt(style_ref, sloppy, clean):
    # Same frozen contract as M5/M6, in style mode; rendered with the model's own template.
    return {"text": render(_tok, build_messages(sloppy, clean, style_reference=style_ref))}

def main(filtered="data/filtered_pairs.jsonl", excerpts="data/raw_excerpts.jsonl",
         out="data/style_train.jsonl", seed=42, k_ref=4):
    random.seed(seed)
    by_author, id_src = defaultdict(list), {}
    for l in open(excerpts):
        e = json.loads(l); by_author[e["source"]].append(e["text"]); id_src[e["id"]] = e["source"]
    written = 0
    with open(out, "w") as f:
        for l in open(filtered):
            p = json.loads(l); src = id_src.get(p["id"].rsplit("_", 1)[0])
            if not src or len(by_author[src]) < k_ref + 1:
                continue
            refs = random.sample([t for t in by_author[src] if t != p["clean"]],
                                 min(k_ref, len(by_author[src]) - 1))
            f.write(json.dumps(fmt("\n\n".join(refs), p["sloppy"], p["clean"])) + "\n")
            written += 1
    print(f"Wrote {written} style-conditioned pairs to {out}")

if __name__ == "__main__":
    main()
```

### `scripts/09_train_style.py`

```python
# scripts/09_train_style.py
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_DIR, OUTPUT_DIR, MAX_SEQ = "models/ftpo_model", "models/style_lora", 8192

model, tokenizer = FastLanguageModel.from_pretrained(MODEL_DIR, max_seq_length=MAX_SEQ,
                                                     dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)
ds = load_dataset("json", data_files={"train": "data/style_train.jsonl"})
SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds["train"],
           dataset_text_field="text", max_seq_length=MAX_SEQ,
           args=SFTConfig(per_device_train_batch_size=1, gradient_accumulation_steps=8,
                          num_train_epochs=2, learning_rate=5e-5, warmup_steps=20,
                          lr_scheduler_type="cosine",
                          fp16=not torch.cuda.is_bf16_supported(),
                          bf16=torch.cuda.is_bf16_supported(), optim="adamw_8bit",
                          logging_steps=25, output_dir="checkpoints/style",
                          report_to="none")).train()
model.save_pretrained(OUTPUT_DIR); tokenizer.save_pretrained(OUTPUT_DIR)
print("Style pass complete → models/style_lora/")
```

### `scripts/check_style_convergence.py`

```python
# scripts/check_style_convergence.py
import re, numpy as np

def style_vector(text):
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(s.split()) for s in sents] or [0]
    words = text.split() or [""]
    return np.array([np.mean(lengths), np.std(lengths),
                     text.count(",") / max(len(sents), 1),
                     text.count('"') / max(len(words), 1),
                     sum(w.endswith("ly") for w in words) / len(words)], dtype=float)

def closer_to_reference(reference, baseline_out, styled_out):
    rv, bv, sv = map(style_vector, (reference, baseline_out, styled_out))
    return np.linalg.norm(sv - rv) < np.linalg.norm(bv - rv), \
           np.linalg.norm(bv - rv), np.linalg.norm(sv - rv)
```

---

## 7. Configuration & Parameters

| Parameter | Value | Rationale | If you change it |
|---|---|---|---|
| base | `models/ftpo_model` | Layer style on finished editor | — |
| rank | 16 | Small targeted change | Higher → more capacity, overwrite risk |
| epochs | 2 | Light pass | More → core regression |
| LR | 5e-5 | Don't overwrite editing/suppression | Higher → regression |
| `max_seq_length` | 8192 | Reference + excerpt + output | Lower → truncation |
| k references | 3–5 | Enough voice signal | More → context bloat |

---

## 8. Alternatives Considered & Rejected

| Alternative | Why considered | Why rejected |
|---|---|---|
| Prompt-only style control | No training | Superficial; fails on implicit voice features |
| Retrain editor from scratch with style | Unified model | Overwrites M6–M7 effort; high regression risk |
| Mix conditioned + unconditioned pairs | One pass covers both | Confuses the model about when to condition; style bleed |
| Cross-author references in training | Tests transfer | Weakens the reference↔target signal; keep for eval only |
| Full-rank style adapter | More capacity | Unnecessary; raises overwrite/OOM risk |

---

## 9. KPIs / Test File

**Test file:** `tests/test_style.py`

```python
# tests/test_style.py
import json
from pathlib import Path
import pytest

STYLE_DATA, STYLE_DIR = "data/style_train.jsonl", "models/style_lora"
FTPO_EVAL, STYLE_EVAL = "reports/ftpo_eval.json", "reports/style_eval.json"

def test_pairs_have_reference_block():
    from scripts.prompt_config import STYLE_SYSTEM_PROMPT, RESPONSE_TEMPLATE
    for p in [json.loads(l) for l in open(STYLE_DATA)][:50]:
        assert "<style_reference>" in p["text"]
        assert STYLE_SYSTEM_PROMPT in p["text"]      # frozen style-mode contract embedded
        assert RESPONSE_TEMPLATE in p["text"]

def test_model_saved():
    assert Path(STYLE_DIR).is_dir()

def test_convergence_function():
    from scripts.check_style_convergence import closer_to_reference
    reference = "Short. Punchy. Terse lines. He ran. She stopped. Done."
    baseline = ("The protagonist moved with a profound and multifaceted sense of vibrant "
                "determination through the bustling environment.")
    styled = "He moved fast. The crowd parted. He did not stop."
    ok, d_base, d_styled = closer_to_reference(reference, baseline, styled)
    assert ok, f"Styled not closer ({d_styled} vs {d_base})"

@pytest.mark.skipif(not (Path(FTPO_EVAL).exists() and Path(STYLE_EVAL).exists()),
                    reason="eval reports not present")
def test_no_core_regression():
    f, s = json.load(open(FTPO_EVAL)), json.load(open(STYLE_EVAL))
    assert s["ner_preservation"] >= f["ner_preservation"] - 0.03
    assert s["slop_delta"] >= f["slop_delta"] - 0.5
```

---

## 10. Definition of Done

- `data/style_train.jsonl` built from same-author references; every record conditioned.
- `models/style_lora/` saved and generates on a reference-conditioned prompt.
- Convergence demonstrated (styled closer to reference than baseline).
- Core NER/slop not degraded vs. M7.
- `pytest tests/test_style.py` passes.
- *(Optional — deferring does not block project completion.)*
