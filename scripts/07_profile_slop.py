"""
07_profile_slop — profile the SFT model's residual slop fingerprint vs. a human baseline.

Pipeline position: M7 — Training, phase B2 (step 1 of 2)
Depends on: models/sft_merged/ (M6), Gutenberg corpus sample
Produces:   reports/slop_profile.json, data/slop_banlist_draft.txt

The SFT model removes the bulk of the slop it was trained on but may still exhibit a
residual fingerprint — patterns that appear more often in its outputs than in human-written
prose of the same genre. This script measures that fingerprint so M7 can surgically suppress
it via FTPO (08_train_ftpo.py).

METHOD
  1. Sample N excerpts from a held-out Gutenberg corpus (not the M2 training corpus).
  2. Run the SFT model on the same sloppy inputs as the test set.
  3. For each pattern in eval_metrics.SLOP_PATTERNS (plus a broader candidate set), compute
     frequency in (a) model outputs and (b) Gutenberg baseline.
  4. Rank patterns by (model_freq - baseline_freq). A large positive delta = residual slop.
  5. Write the ranked list to slop_banlist_draft.txt for human review.

MANDATORY HUMAN STEP
The draft banlist must be manually pruned before running 08_train_ftpo.py. Some patterns
that appear frequently in model outputs are genre-legitimate (e.g. "shimmered" is expected
in fantasy; "realm" appears in historical fiction). Training FTPO to suppress a
genre-legitimate word will degrade prose quality in those genres. The human reviewer removes
these false positives before the final banlist is committed to eval_metrics.SLOP_PATTERNS.

Output schema for slop_profile.json:
  {
    "pattern": str,
    "model_freq_per_1k": float,
    "baseline_freq_per_1k": float,
    "delta": float
  }
  (list, sorted by delta descending)

Usage:
    python scripts/07_profile_slop.py [--model models/sft_merged]
                                      [--gutenberg-dir data/gutenberg_sample/]
                                      [--output reports/slop_profile.json]

Key implementation notes:
  - The Gutenberg baseline should be genre-matched to the M2 training corpus (fiction prose).
    A baseline drawn from technical or academic text will inflate deltas for formal vocabulary.
  - Use the same pattern list as eval_metrics.SLOP_PATTERNS plus a broader candidate set
    (e.g. the top-500 most common English words that are suspiciously over-represented in
    LLM output). The module_7_ftpo.md document has a suggested extended candidate list.
  - Frequency should be computed at the same normalisation as slop_score() — hits per 1,000
    words — so results are comparable to the eval metric values.
"""

import json
from pathlib import Path


def compute_pattern_frequencies(texts, patterns):
    """Return a dict mapping each pattern to its hits-per-1k-words rate across all texts."""
    raise NotImplementedError


def main(model_dir="models/sft_merged", output_path="reports/slop_profile.json"):
    """Run profiling and write the ranked banlist draft."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
