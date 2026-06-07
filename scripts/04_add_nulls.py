"""
04_add_nulls — inject null-edit pairs (clean input, identical clean output) into the dataset.

Pipeline position: M5 — Data pipeline, stage 4 of 4 (step 2 of 3)
Depends on: data/filtered_pairs.jsonl (03_filter_pairs.py), data/raw_excerpts.jsonl (M2)
Produces:   data/null_pairs.jsonl

A model trained only on (sloppy→clean) pairs learns an implicit rule: always rewrite
heavily. On already-clean prose it then over-edits, damaging good writing. Null-edit pairs
break that rule by teaching "clean input → unchanged output." At 10-15% of the total set,
they instil restraint without diluting the slop-removal signal.

Their effect is directly measurable: M9's over-edit adversarial test feeds clean prose to
the model and checks that the semantic similarity between input and output is >= 0.97. If
the over-edit test fails after training, increase the null fraction here and retrain.

NULL ID FORMAT
IDs follow the format: null_{excerpt_id}
The "null_" prefix is how 05_format_dataset.py's base_id() function identifies these records
and extracts the underlying excerpt ID for leakage-safe splitting. Do not change the prefix.

Output schema (one JSON object per line):
  {"id": "null_{excerpt_id}", "clean": str, "sloppy": str, "teacher": "null"}
  where clean == sloppy (byte-identical).

Usage:
    python scripts/04_add_nulls.py [--filtered data/filtered_pairs.jsonl]
                                   [--excerpts data/raw_excerpts.jsonl]
                                   [--output data/null_pairs.jsonl]
                                   [--fraction 0.12] [--seed 42]

Key implementation notes:
  - Sample from raw_excerpts.jsonl, not from filtered_pairs. Nulls should be clean human
    prose that the model has NOT already seen as the "clean" side of a sloppy pair in
    training — otherwise the model could memorise rather than learn the rule.
  - Actually, the above concern is minor for this dataset size, but sampling broadly from
    the excerpt pool is still better practice than re-using filtered-pair clean sides.
  - The output count should be int(n_filtered * fraction), capped at the excerpt pool size.
    At fraction=0.12 and ~2,000 filtered pairs, expect ~240 null pairs.
"""

import json
import random
from pathlib import Path


def main(filtered="data/filtered_pairs.jsonl", excerpts="data/raw_excerpts.jsonl",
         out="data/null_pairs.jsonl", fraction=0.12, seed=42):
    """Sample clean excerpts and write them as null-edit pairs."""
    n_filtered = sum(1 for _ in open(filtered))
    pool = [json.loads(l) for l in open(excerpts)]
    random.seed(seed)
    n_nulls = min(int(n_filtered * fraction), len(pool))
    sample = random.sample(pool, n_nulls)

    with open(out, "w") as f:
        for e in sample:
            f.write(json.dumps({
                "id":      f"null_{e['id']}",
                "clean":   e["text"],
                "sloppy":  e["text"],  # byte-identical — no edit needed
                "teacher": "null",
            }) + "\n")

    print(f"Wrote {len(sample)} null pairs ({fraction:.0%} of {n_filtered} filtered pairs)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered",  default="data/filtered_pairs.jsonl")
    parser.add_argument("--excerpts",  default="data/raw_excerpts.jsonl")
    parser.add_argument("--output",    default="data/null_pairs.jsonl")
    parser.add_argument("--fraction",  type=float, default=0.12)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    main(args.filtered, args.excerpts, args.output, args.fraction, args.seed)
