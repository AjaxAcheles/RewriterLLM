# Module 6 — SFT Training (Phase B1)

| Field | Value |
|---|---|
| **Phase** | Training (stage 1 of 2 — the core capability) |
| **Depends on** | M5 (dataset), M9 (eval harness must exist) |
| **Blocks** | M7 |
| **Critical path** | Yes |
| **Owner effort** | 1 day prototype + 1–2 days final run |
| **Runtime budget** | 4B proto: a few hrs · 7B final: ~10–24 hrs on 12 GB (data-size dependent) |

---

## 1. Primary Objective

Train the content-preserving editing transformation into the student model using Unsloth + QLoRA:
prototype on a 4B model to validate the pipeline cheaply, then train the final 7B model, and
record a clean evaluation baseline that becomes the contract Module 7 must beat. Deliverables:
`models/sft_lora/` (adapters), `models/sft_merged/` (merged 16-bit), `reports/sft_baseline.json`.

---

## 2. Core Concepts — Deep Dive

### 2.1 What SFT teaches — and the limit that defines the handoff to M7

SFT shows the model thousands of (sloppy→clean) examples and optimizes it to produce the clean
output. Over the dataset it learns the *editing transformation*: which surface patterns to change
(slop vocabulary, hollow transitions, tell-don't-show, flat rhythm) and which deep structures to
leave untouched (entities, events, causality, tone). This is the central capability of the project.

But SFT's loss only *rewards matching the target*; it does not *penalize* slop patterns the base
model already knows. It raises the probability of clean outputs without actively lowering the
probability of sloppy ones, so residual slop persists. That is not a bug to fix here — it is the
reason Module 7 exists. Module 6 therefore ends not by declaring victory but by **recording a
baseline** that leaves a measurable, improvable amount of residual slop for FTPO to attack.

### 2.2 Why QLoRA + Unsloth is the only way to fit 7B in 12 GB

Full fine-tuning of 7B needs 80–100 GB. QLoRA freezes the base in 4-bit NF4 (~4 GB for 7B) and
trains only small float LoRA adapters (~0.1–1% of params), landing near 10–12 GB. Unsloth then
rewrites the memory-heavy ops with custom Triton kernels, cutting activation memory enough to make
long context feasible — the difference between ~932 tokens (stock) and tens of thousands (Unsloth)
on a 12 GB card. Since the task needs the full sloppy excerpt *and* its full clean rewrite in
context at once, long context is mandatory, so Unsloth is mandatory. The computations are
numerically identical to standard QLoRA — Unsloth changes memory layout and kernel efficiency, not
results.

### 2.3 Why CPU/RAM offload is NOT used here (the contrast with M4)

M4 used llama.cpp hybrid offload because *inference* tolerates latency. *Training* does not, and
the offload path is blocked anyway: DeepSpeed ZeRO-3 (full CPU parameter offload) is incompatible
with bitsandbytes 4-bit quantization, which QLoRA requires — you cannot shard params to CPU *and*
keep them in 4-bit NF4. Also, LoRA trains so few parameters that offloading optimizer state saves
almost nothing while adding large PCIe latency. So Module 6 is GPU-only and the 32 GB RAM sits
idle — correct and expected.

### 2.4 The 4B→7B ladder as risk reduction

The 7B run is slow and expensive on this hardware; running it first means discovering pipeline bugs
(masking, format, OOM at a given context) after hours of wasted compute. The 4B prototype trains
fast, surfaces those bugs cheaply, and validates the whole path from dataset to evaluation. Only
once the prototype's eval is sane do we commit to 7B.

### 2.5 The signature silent failure: loss on the full sequence

If loss is computed over the entire sequence (system + user + assistant) rather than only the
assistant response, the model is rewarded for reproducing its *input*. It then learns to "edit" by
returning the sloppy text unchanged. Training loss looks fine; the failure only shows as near-zero
slop-delta in eval. Mitigation: response-only loss masking, implemented here with TRL's
`DataCollatorForCompletionOnlyLM` keyed on the frozen assistant marker (`RESPONSE_TEMPLATE` from
`prompt_config`), with sequence packing disabled so the mask aligns; plus a test that asserts the
model does not copy its input.

### 2.6 Hyperparameter intuition

`r`/`lora_alpha` set adapter capacity (higher for the harder 7B nuance). `grad-accum` simulates a
larger batch than VRAM allows (effective batch = per-device × accum). `adamw_8bit` halves
optimizer-state memory. A cosine schedule with warmup stabilizes the early, high-variance steps.
LR is lower for 7B (1e-4 vs 2e-4) because larger models destabilize more easily.

---

## 3. Inputs & Outputs Contract

**Inputs:** `data/train.jsonl`, `data/val.jsonl` (M5); Module 9's `eval_metrics.py` + `run_eval.py`.

**Outputs:**
- `models/sft_lora/` — PEFT adapters.
- `models/sft_merged/` — merged 16-bit model (required by M7 profiling/inference).
- `checkpoints/sft/` — intermediate checkpoints.
- `reports/sft_baseline.json` — recorded metrics; the **input contract for M7**.

**Baseline gate (asserted via M9 on the test set):** `slop_delta > 1.0`,
`ner_preservation ≥ 0.90`, `sem_vs_target ≥ 0.78`.

---

## 4. Common Challenges & Solutions

**Challenge 1 — The model copies its input verbatim.**
*Why:* loss applied to the full sequence, not just the response. *Detect:* `test_model_does_not_copy_input`
fails; slop_delta ≈ 0. *Solve:* ensure the `DataCollatorForCompletionOnlyLM` is wired in with
`packing=False`. If it raises *"Could not find response key"*, the marker did not tokenize
identically in-context (a known TRL gotcha with leading whitespace/newlines); pass token IDs instead
of the string: `DataCollatorForCompletionOnlyLM(tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False), tokenizer=tokenizer)`.

**Challenge 2 — CUDA OOM on the 7B final run.**
*Why:* 8192 context × activations exceeds 12 GB. *Detect:* OOM at first step or mid-epoch. *Solve:*
lower `max_seq_length` (e.g. 6144); keep batch 1; ensure `use_gradient_checkpointing="unsloth"`;
reduce grad-accum if needed.

**Challenge 3 — Validation loss rises after epoch 1.**
*Why:* overfitting a small dataset. *Detect:* val loss U-turns while train loss falls. *Solve:*
reduce to 2 epochs; raise dropout; add data; early-stop on val.

**Challenge 4 — slop_delta near zero despite training.**
*Why:* either the copy bug (Challenge 1) or weak training contrast (M5 `max_sem` too high).
*Detect:* eval shows little slop reduction. *Solve:* verify masking first; then check curation's
upper similarity bound.

**Challenge 5 — Training is far slower than expected.**
*Why:* Unsloth gradient checkpointing not enabled, or a non-Unsloth path. *Detect:* low it/s; high
memory. *Solve:* set `use_gradient_checkpointing="unsloth"`; load via `FastLanguageModel`.

**Challenge 6 — Merged-model save OOMs.**
*Why:* merging needs extra transient memory. *Detect:* OOM at `save_pretrained_merged`. *Solve:*
save adapters first; merge in a separate fresh process.

---

## 5. Step-by-Step Implementation Guide

**Step 1 — Prototype config.** In `06_train_sft.py` set `MODEL=4B`, `RANK=16`, `MAX_SEQ=4096`.

**Step 2 — Run the prototype.**
```bash
python scripts/06_train_sft.py 2>&1 | tee reports/sft_proto.log
# watch: train loss decreasing; periodic eval loss; no OOM
```

**Step 3 — Validate the prototype.**
```bash
python scripts/run_eval.py models/sft_lora reports/sft_proto_eval.json
```
Fix any masking/format/OOM bugs surfaced before spending on 7B.

**Step 4 — Final config.** Switch to `MODEL=7B`, `RANK=32`, `MAX_SEQ=8192`, `LR=1e-4`,
`grad-accum=8`.

**Step 5 — Run the final training**, then save adapters + merged model (the script does both).

**Step 6 — Record the baseline.**
```bash
python scripts/run_eval.py models/sft_lora reports/sft_baseline.json
cat reports/sft_baseline.json    # confirm gate: slop_delta>1, ner>=0.90, sem_vs_target>=0.78
```

**Step 7 — Verify.**
```bash
pytest tests/test_sft_training.py -v
```

---

## 6. Reference Implementation — `scripts/06_train_sft.py`

```python
# scripts/06_train_sft.py
"""SFT with Unsloth + QLoRA. Proto: 4B/r16/4096 ; Final: 7B/r32/8192."""
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from prompt_config import RESPONSE_TEMPLATE

MODEL = "unsloth/Qwen3-4B-bnb-4bit"     # final: "unsloth/Qwen3-7B-bnb-4bit"
RANK, MAX_SEQ = 16, 4096                  # final: 32, 8192
OUTPUT_DIR = "models/sft_lora"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ, dtype=None, load_in_4bit=True)

model = FastLanguageModel.get_peft_model(
    model, r=RANK, lora_alpha=RANK, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)

ds = load_dataset("json", data_files={"train": "data/train.jsonl",
                                       "validation": "data/val.jsonl"})
is_4b = "4B" in MODEL

# Loss only on the assistant response — never on the system+user prompt — so the model learns to
# EDIT rather than copy its input (see 2.5). Keyed on the frozen response marker; needs packing off.
collator = DataCollatorForCompletionOnlyLM(response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    data_collator=collator,
    dataset_text_field="text", max_seq_length=MAX_SEQ, dataset_num_proc=2,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4 if is_4b else 8,
        num_train_epochs=3,
        learning_rate=2e-4 if is_4b else 1e-4,
        warmup_steps=50, lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit", weight_decay=0.01, packing=False,
        logging_steps=25, eval_steps=100, save_steps=200, save_total_limit=3,
        output_dir="checkpoints/sft", report_to="none"))

trainer.train()
model.save_pretrained(OUTPUT_DIR); tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained_merged("models/sft_merged", tokenizer, save_method="merged_16bit")
print("SFT complete. Run scripts/run_eval.py to record the baseline.")
```

---

## 7. Configuration & Parameters

| Parameter | Proto (4B) | Final (7B) | Rationale | If you change it |
|---|---|---|---|---|
| base | Qwen3-4B | Qwen3-7B | fast iter → final capability | smaller → weaker prose |
| `r` / `alpha` | 16 | 32 | capacity for 7B nuance | higher → more VRAM, overfit risk |
| `max_seq_length` | 4096 | 8192 | room for long excerpt+output | lower → truncation; higher → OOM |
| grad-accum | 4 | 8 | effective batch w/o VRAM | higher → slower steps |
| LR | 2e-4 | 1e-4 | larger model stability | higher → divergence |
| epochs | 3 | 3 | narrow-task convergence | watch val loss |
| optim | adamw_8bit | adamw_8bit | half optimizer memory | — |
| grad ckpt | "unsloth" | "unsloth" | activation memory | omit → OOM/slow |
| loss masking | completion-only | completion-only | learn to edit, not copy input | full-seq → model copies input |

---

## 8. Alternatives Considered & Rejected

| Alternative | Why considered | Why rejected |
|---|---|---|
| Full fine-tuning | Max capacity | 80–100 GB VRAM — impossible on 12 GB |
| Stock QLoRA (no Unsloth) | Standard | ~932-token context on 12 GB — too short for excerpt+output |
| DeepSpeed ZeRO-3 CPU offload | Use the 32 GB RAM | Incompatible with bitsandbytes 4-bit; LoRA offload saves little |
| Skip prototype, train 7B directly | Save a step | Pipeline bugs cost hours of 7B compute to discover |
| DPO instead of SFT | Pairs already exist | DPO can't teach the transformation; degrades diversity (see M7) |

---

## 9. KPIs / Test File

**Test file:** `tests/test_sft_training.py`

```python
# tests/test_sft_training.py
import json
from pathlib import Path
import pytest

SFT_DIR, MERGED, BASELINE = "models/sft_lora", "models/sft_merged", "reports/sft_baseline.json"

def test_adapters_saved():
    assert Path(SFT_DIR).is_dir()
    assert any("adapter" in p.name for p in Path(SFT_DIR).iterdir())

def test_merged_saved():
    assert Path(MERGED).is_dir(), "Merged model needed by Module 7"

def test_model_does_not_copy_input():
    from unsloth import FastLanguageModel
    from scripts.prompt_config import build_messages
    model, tok = FastLanguageModel.from_pretrained(SFT_DIR, max_seq_length=4096,
                                                   dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    sloppy = ("Furthermore, the vibrant tapestry of the bustling city resonated with a "
              "profound sense of nuance, and she felt deeply sad.")
    ids = tok.apply_chat_template(build_messages(sloppy), tokenize=True,
                                  add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt").to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=200, temperature=0.3,
                         do_sample=True, repetition_penalty=1.1)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    assert text.strip() and text.strip() != sloppy.strip(), "Model copied input — check masking"

@pytest.mark.skipif(not Path(BASELINE).exists(), reason="baseline not recorded")
def test_baseline_gate():
    m = json.load(open(BASELINE))
    assert m["slop_delta"] > 1.0
    assert m["ner_preservation"] >= 0.90
    assert m["sem_vs_target"] >= 0.78
```

---

## 10. Definition of Done

- Prototype trained + evaluated; bugs resolved.
- 7B adapters + merged model saved.
- `reports/sft_baseline.json` clears the gate (slop_delta>1.0, ner≥0.90, sem_vs_target≥0.78).
- Model demonstrably does not copy input.
- `pytest tests/test_sft_training.py` passes.
