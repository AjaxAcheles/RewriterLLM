"""
sample_adversarial_fixtures.py — bootstrap the three adversarial test fixture files.

run_adversarial.py needs three fixture files that cannot be generated automatically:
  data/eval_clean_prose.jsonl   — slop-free human prose (over-edit + hallucination tests)
  data/eval_strong_voice.jsonl  — distinctive narrator voice (voice-flattening test)
  data/eval_tone_tagged.jsonl   — prose with a tone label (tone-preservation test)

This script handles the first file automatically (samples clean sides from the test set)
and produces candidate pools for the other two that require a human to curate.

Usage:
    python scripts/sample_adversarial_fixtures.py [--n 30] [--seed 42]

After running:
  - data/eval_clean_prose.jsonl  is READY — no further action needed.
  - data/eval_strong_voice.jsonl contains CANDIDATES. Open it, keep only excerpts
    with a strong/idiosyncratic narrator voice, delete the rest, and remove the
    "_source" and "_todo" keys from every kept record.
  - data/eval_tone_tagged.jsonl  contains CANDIDATES. Open it, fill in the "tone"
    field for each excerpt (e.g. "darkly comic", "melancholy", "tense", "sardonic",
    "lyrical"), delete excerpts you can't confidently label, and remove "_source"
    and "_todo" keys.

The script will not overwrite existing fixture files unless --overwrite is passed.
"""

import argparse
import json
import random
from pathlib import Path


def main(n=30, seed=42, overwrite=False):
    random.seed(seed)

    test_raw = Path("data/test_raw.jsonl")
    if not test_raw.exists():
        raise SystemExit(
            "data/test_raw.jsonl not found. Run scripts/05_format_dataset.py first."
        )

    pairs = [json.loads(l) for l in open(test_raw)]
    if len(pairs) < n * 3:
        print(f"Warning: only {len(pairs)} test pairs available; fixture files will be smaller than {n} items.")

    # Oversample so each bucket has enough candidates after curation
    sample = random.sample(pairs, min(n * 3, len(pairs)))
    buckets = [
        sample[:n],           # clean prose — used as-is
        sample[n:n * 2],      # voice candidates — needs human curation
        sample[n * 2:],       # tone candidates — needs human annotation
    ]

    # ── eval_clean_prose.jsonl ────────────────────────────────────────────────
    out_clean = Path("data/eval_clean_prose.jsonl")
    if out_clean.exists() and not overwrite:
        print(f"Skipping {out_clean} (already exists — pass --overwrite to replace)")
    else:
        with open(out_clean, "w") as f:
            for p in buckets[0]:
                f.write(json.dumps({"text": p["clean"]}) + "\n")
        print(f"Wrote {len(buckets[0])} items to {out_clean}  [READY]")

    # ── eval_strong_voice.jsonl ───────────────────────────────────────────────
    out_voice = Path("data/eval_strong_voice.jsonl")
    if out_voice.exists() and not overwrite:
        print(f"Skipping {out_voice} (already exists — pass --overwrite to replace)")
    else:
        with open(out_voice, "w") as f:
            for p in buckets[1]:
                f.write(json.dumps({
                    "text":    p["clean"],
                    "_source": p["id"],
                    "_todo":   "Keep only if the narrator voice is strong/idiosyncratic. Delete this record and '_source' key before use.",
                }) + "\n")
        print(f"Wrote {len(buckets[1])} CANDIDATES to {out_voice}")
        print("  → Review: keep distinctive-voice excerpts, delete the rest.")
        print("    Remove '_source' and '_todo' keys from every kept record.")

    # ── eval_tone_tagged.jsonl ────────────────────────────────────────────────
    out_tone = Path("data/eval_tone_tagged.jsonl")
    if out_tone.exists() and not overwrite:
        print(f"Skipping {out_tone} (already exists — pass --overwrite to replace)")
    else:
        with open(out_tone, "w") as f:
            for p in buckets[2]:
                f.write(json.dumps({
                    "text":    p["clean"],
                    "tone":    "FILL_IN",
                    "_source": p["id"],
                    "_todo":   "Set 'tone' to one of: darkly comic | melancholy | tense | sardonic | lyrical | wry | ominous. Delete records you cannot confidently label. Remove '_source' and '_todo' keys.",
                }) + "\n")
        print(f"Wrote {len(buckets[2])} CANDIDATES to {out_tone}")
        print("  → Review: set the 'tone' field, delete unlabellable records.")
        print("    Remove '_source' and '_todo' keys from every kept record.")
        print("    WARNING: run_adversarial.py uses a naive string-match judge by default.")
        print("    Tone labels should be single words or short phrases that may literally")
        print("    appear in the edited text (e.g. 'melancholy' not 'sad but hopeful').")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap adversarial test fixtures from the test set."
    )
    parser.add_argument("--n",         type=int,  default=30, help="Target items per fixture")
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    main(args.n, args.seed, args.overwrite)
