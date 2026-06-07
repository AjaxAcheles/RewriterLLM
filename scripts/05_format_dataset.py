"""
05_format_dataset — apply chat template and split filtered+null pairs into train/val/test.

Pipeline position: M5 — Data pipeline, stage 4 of 4 (step 3 of 3)
Depends on: data/filtered_pairs.jsonl, data/null_pairs.jsonl, scripts/prompt_config.py
Produces:   data/train.jsonl, data/val.jsonl, data/test.jsonl, data/test_raw.jsonl

Each pair is rendered into the chat template by calling prompt_config.render() with the
model's own tokenizer. This guarantees that the formatted training text is byte-identical
to what the model sees at inference time — the same code path, the same tokenizer template,
the same enable_thinking=False flag.

SPLIT BY BASE EXCERPT ID (non-negotiable)
M4 generated multiple sloppy versions per clean excerpt (one per teacher). All versions
share a base excerpt ID. If different versions of the same excerpt land in different splits,
the test set has effectively seen the training content — scores are inflated by memorisation.
Splitting by base ID makes leakage impossible by construction: all versions of a given
excerpt go to the same split.

Base ID extraction:
  null pairs:   strip the "null_" prefix → underlying excerpt ID
  normal pairs: strip the last "_teacher" suffix → base excerpt ID

The split is 80/10/10 over shuffled base IDs (seed=42 for reproducibility).

Output schemas:
  train.jsonl, val.jsonl, test.jsonl — {"text": "<fully rendered chat string>"}
  test_raw.jsonl — raw pair dicts (id, clean, sloppy, teacher) — used by M9 run_eval.py

WHY test_raw.jsonl EXISTS
The formatted test.jsonl wraps the pair in the chat template (useful for inspection and
training-symmetry checks). M9 needs the *raw* sloppy text to feed the model fresh and the
*raw* clean target to score against. Saving both avoids reformatting in the eval harness
and eliminates any risk of a template mismatch at eval time.

Usage:
    python scripts/05_format_dataset.py [--filtered data/filtered_pairs.jsonl]
                                        [--nulls data/null_pairs.jsonl]
                                        [--seed 42] [--val-frac 0.10] [--test-frac 0.10]

Key implementation notes:
  - Load the tokenizer ONCE (from prompt_config.CANONICAL_TOKENIZER) at module level or in
    main(). Reloading per pair is prohibitively slow.
  - The tokenizer download (~600 MB) will block on first run. If you're offline or want to
    use a local path, patch CANONICAL_TOKENIZER in prompt_config.py.
  - Print "train=N val=N test=N (by M base excerpts)" so the M5 KPI checks are visible.
"""

import json
import random
from transformers import AutoTokenizer
try:
    from scripts.prompt_config import CANONICAL_TOKENIZER, build_messages, render
except ImportError:
    from prompt_config import CANONICAL_TOKENIZER, build_messages, render


def base_id(pair_id):
    """Extract the base excerpt ID from a pair ID.

    null_abc123 → abc123
    abc123_llama3 → abc123
    """
    if pair_id.startswith("null_"):
        return pair_id[5:]
    return pair_id.rsplit("_", 1)[0]


def format_pair(tokenizer, sloppy, clean):
    """Render a sloppy/clean pair into a single chat-template string for training."""
    return {"text": render(tokenizer, build_messages(sloppy, clean))}


def main(filtered="data/filtered_pairs.jsonl", nulls="data/null_pairs.jsonl",
         seed=42, val_frac=0.10, test_frac=0.10):
    """Merge, format, split by base ID, and write all four output files."""
    tokenizer = AutoTokenizer.from_pretrained(CANONICAL_TOKENIZER)

    pairs = (
        [json.loads(l) for l in open(filtered)] +
        [json.loads(l) for l in open(nulls)]
    )

    # Group pairs by base excerpt ID — splitting by base prevents leakage
    by_base: dict = {}
    for p in pairs:
        bid = base_id(p["id"])
        by_base.setdefault(bid, []).append(p)

    bases = list(by_base.keys())
    random.seed(seed)
    random.shuffle(bases)

    n = len(bases)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    test_bases = set(bases[:n_test])
    val_bases  = set(bases[n_test:n_test + n_val])

    train, val, test = [], [], []
    for b, grp in by_base.items():
        if b in test_bases:
            test.extend(grp)
        elif b in val_bases:
            val.extend(grp)
        else:
            train.extend(grp)

    def write_formatted(records, path):
        with open(path, "w") as f:
            for p in records:
                f.write(json.dumps(format_pair(tokenizer, p["sloppy"], p["clean"])) + "\n")

    write_formatted(train, "data/train.jsonl")
    write_formatted(val,   "data/val.jsonl")
    write_formatted(test,  "data/test.jsonl")

    with open("data/test_raw.jsonl", "w") as f:
        for p in test:
            f.write(json.dumps(p) + "\n")

    print(f"train={len(train)} val={len(val)} test={len(test)} (by {n} base excerpts)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered",  default="data/filtered_pairs.jsonl")
    parser.add_argument("--nulls",     default="data/null_pairs.jsonl")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--val-frac",  type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    args = parser.parse_args()
    main(args.filtered, args.nulls, args.seed, args.val_frac, args.test_frac)
