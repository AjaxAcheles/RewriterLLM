"""
run_eval — full test-set evaluation runner for any model checkpoint.

Pipeline position: M9 — Evaluation harness (cross-cutting, must exist before M6 final run)
Depends on: data/test_raw.jsonl (M5), scripts/eval_metrics.py (M9), any model dir
Produces:   reports/<name>.json (read by M6/M7/M8 completion tests)

Loads a model, runs inference on every pair in data/test_raw.jsonl, computes the full
metric constellation from eval_metrics.evaluate_pair(), prints a tiered pass/warn/fail
table, and writes the averages as a JSON report.

The report JSON keys are a FROZEN CONTRACT (same as the keys in evaluate_pair's return dict).
M6/M7/M8 completion tests read these keys by name. Changing them here breaks those tests.

TIERED THRESHOLDS (fail / warn / pass)
  slop_delta:              fail < 1.0  / warn < 2.0  / pass >= 2.0
  ner_preservation:        fail < 0.90 / warn < 0.95 / pass >= 0.95
  hallucination_rate:      pass < 0.05 / warn < 0.10 / fail >= 0.10
  sem_vs_target:           fail < 0.75 / warn < 0.80 / pass >= 0.80
  burstiness_improved_rate:fail < 0.50 / warn < 0.70 / pass >= 0.70

Tiered rather than binary: a change can be seen to move a metric from fail to warn before
it reaches pass, providing an iteration gradient rather than just a gate.

IN-LOOP USE
During training (M6/M7/M8), import evaluate_pair directly from eval_metrics and run on
a ~50-pair subset for speed. Full test-set eval only at checkpoints and final gates.

Usage:
    python scripts/run_eval.py <model_dir> <report_path>
    python scripts/run_eval.py models/sft_lora reports/sft_baseline.json
    python scripts/run_eval.py models/ftpo_model reports/ftpo_eval.json

Key implementation notes:
  - Load the model with FastLanguageModel.from_pretrained() (Unsloth), not AutoModelForCausalLM.
    The Unsloth path is required for inference efficiency and correct 4-bit handling.
  - Inference temperature: 0.3 — low enough to be deterministic/faithful, high enough not
    to be greedy. repetition_penalty=1.1 guards against degenerate repetition loops.
  - max_new_tokens=1024 covers a full 700-word rewrite. Do not set lower or outputs will be
    truncated and metrics (especially sem_vs_target) will be artificially degraded.
  - The prompt is built via prompt_config.build_messages(sloppy) + tokenizer's apply_chat_template
    with add_generation_prompt=True, enable_thinking=False. This must match training exactly.
  - burstiness_improved_rate is computed as mean(r["burstiness_improved"] for r in results)
    after the per-pair bools are collected — it is not a field in individual evaluate_pair results.
"""

import json
import sys
import torch
from pathlib import Path
from scripts.eval_metrics import evaluate_pair
from scripts.prompt_config import build_messages

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/ftpo_model"
REPORT = sys.argv[2] if len(sys.argv) > 2 else "reports/eval.json"

TIERS = {
    # metric: (fail_threshold, pass_threshold) — higher is better unless noted
    "slop_delta":              (1.0, 2.0),
    "ner_preservation":        (0.90, 0.95),
    "hallucination_rate":      (0.10, 0.05),   # lower is better
    "sem_vs_target":           (0.75, 0.80),
    "burstiness_improved_rate":(0.50, 0.70),
}


def tier(metric, value):
    """Return "PASS", "WARN", or "FAIL" for a given metric value."""
    fail_t, pass_t = TIERS[metric]
    if metric == "hallucination_rate":
        return "PASS" if value < pass_t else "WARN" if value < fail_t else "FAIL"
    return "PASS" if value >= pass_t else "WARN" if value >= fail_t else "FAIL"


def edit(model, tokenizer, sloppy):
    """Run the model on a single sloppy excerpt and return the decoded output string.
    TODO: implement inference call.
    """
    raise NotImplementedError


def main():
    """Load model, evaluate test set, print table, write report."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
