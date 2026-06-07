"""
06_train_sft — supervised fine-tuning of the editor model with Unsloth + QLoRA.

Pipeline position: M6 — Training, phase B1 (core capability)
Depends on: data/train.jsonl, data/val.jsonl (M5), scripts/eval_metrics.py + run_eval.py (M9)
Produces:   models/sft_lora/, models/sft_merged/, reports/sft_baseline.json

Trains the sloppy→clean editing transformation into the model. Two-stage process:
  1. Prototype run on 4B model (fast, surfaces pipeline bugs cheaply)
  2. Final run on 7B model (~10-24 hrs on 12 GB)

Only move to the 7B run after the 4B prototype's eval results look sane.

WHY QLOQA + UNSLOTH IS MANDATORY
Full fine-tuning of 7B needs 80-100 GB VRAM. QLoRA freezes the base in 4-bit NF4 (~4 GB)
and trains only small float LoRA adapters (~0.1-1% of params), landing near 10-12 GB.
Unsloth then rewrites memory-heavy ops with custom Triton kernels so long context (~8K) is
feasible — without Unsloth, stock QLoRA is limited to ~932 tokens on 12 GB, too short for
an excerpt + its full rewrite in a single context window.

CRITICAL: COMPLETION-ONLY LOSS MASKING
If loss is computed over the full sequence (system prompt + user turn + assistant response),
the model is rewarded for reproducing its own INPUT. It learns to "edit" by returning the
sloppy text unchanged. Training loss looks normal; the failure only shows as slop_delta ≈ 0
in evaluation. This is the most common silent failure in this kind of task.

Fix: DataCollatorForCompletionOnlyLM keyed on RESPONSE_TEMPLATE with packing=False.
     packing=False is required so the loss mask aligns correctly (packing concatenates
     multiple examples, shifting the response boundary).

TRL gotcha: if the collator raises "Could not find response key in token IDs", the response
marker did not tokenize identically in context. Pass the token IDs directly instead:
  DataCollatorForCompletionOnlyLM(
      tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False),
      tokenizer=tokenizer
  )

CONFIGURATION (toggle between proto and final at the top of main())
  Proto (4B):  MODEL="unsloth/Qwen3-4B-bnb-4bit",  RANK=16, MAX_SEQ=4096, LR=2e-4, ACCUM=4
  Final (7B):  MODEL="unsloth/Qwen3-7B-bnb-4bit",  RANK=32, MAX_SEQ=8192, LR=1e-4, ACCUM=8

Usage:
    python scripts/06_train_sft.py
    # then evaluate:
    python scripts/run_eval.py models/sft_lora reports/sft_baseline.json

Key implementation notes:
  - use_gradient_checkpointing="unsloth" is mandatory on 12 GB. Without it you will OOM
    on the 7B run at the very first step or shortly after.
  - Save adapters first (models/sft_lora/), then merge to 16-bit (models/sft_merged/).
    Merging requires transient extra memory; if it OOMs, run the merge in a separate fresh
    process after training finishes.
  - The merged model is required by M7's profiling step. Do not skip it.
  - Gate before declaring M6 done (assert via run_eval.py output):
      slop_delta > 1.0, ner_preservation >= 0.90, sem_vs_target >= 0.78
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from scripts.prompt_config import RESPONSE_TEMPLATE

# ---------------------------------------------------------------------------
# Toggle between prototype (4B) and final (7B) runs here.
# ---------------------------------------------------------------------------
MODEL = "unsloth/Qwen3-4B-bnb-4bit"    # final: "unsloth/Qwen3-7B-bnb-4bit"
RANK = 16                                # final: 32
MAX_SEQ = 4096                           # final: 8192
OUTPUT_DIR = "models/sft_lora"


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
