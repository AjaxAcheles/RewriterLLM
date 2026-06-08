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

import re
import json
import random
import subprocess
import sys
from pathlib import Path

# Extended candidate slop patterns beyond eval_metrics.SLOP_PATTERNS
EXTENDED_PATTERNS = [
    r"\btapestry\b", r"\bdelve\b", r"\btestament\b", r"\bvibrant\b",
    r"\bnavigate\b", r"\bnuanced\b", r"\bresonate\b", r"\bpivotal\b",
    r"\bmultifaceted\b", r"\bunderscore\b", r"\bembark\b", r"\brealm\b",
    r"\bshimmered\b", r"\bunsettlingly\b", r"\bfurthermore\b", r"\bmoreover\b",
    r"\bin addition\b", r"\bit is worth noting\b", r"\bsomething shifted\b",
    r"\beverything changed\b", r"\bbut here's the thing\b",
    r"\bat the end of the day\b", r"\bprofound\b", r"\bharbor\b",
    r"\bjourney\b", r"\btransform\b", r"\bcrucial\b", r"\bfoster\b",
    r"\bsustainable\b", r"\bsynergy\b", r"\bleverag\b", r"\bparadigm\b",
    r"\bseamless\b", r"\bintricate\b", r"\bcomplex\b",
]


def compute_pattern_frequencies(texts, patterns):
    """Return a dict mapping each pattern to its hits-per-1k-words rate across all texts."""
    total_words = sum(len(t.split()) for t in texts)
    if total_words == 0:
        return {p: 0.0 for p in patterns}

    counts = {p: 0 for p in patterns}
    for text in texts:
        t_lower = text.lower()
        for p in patterns:
            counts[p] += len(re.findall(p, t_lower))

    # Normalize to hits per 1000 words
    scale = 1000.0 / total_words
    return {p: counts[p] * scale for p in patterns}


def main(model_dir="models/sft_merged", output_path="reports/slop_profile.json",
         n_samples=500):
    """Run profiling and write the ranked banlist draft."""
    import torch
    from unsloth import FastLanguageModel
    try:
        from scripts.prompt_config import build_messages
    except ImportError:
        from prompt_config import build_messages

    # Load human baseline (Gutenberg corpus)
    baseline_texts = []
    gutenberg_dir = Path("data/raw_texts")
    txt_files = sorted(gutenberg_dir.glob("*.txt"))
    random.seed(42)
    sample_files = random.sample(txt_files, min(n_samples, len(txt_files)))
    for f in sample_files:
        try:
            baseline_texts.append(f.read_text(errors="ignore")[:3000])
        except Exception:
            pass

    if not baseline_texts:
        raise SystemExit("No Gutenberg baseline texts found in data/raw_texts/")

    # Load the SFT model and generate outputs
    model, tok = FastLanguageModel.from_pretrained(
        model_dir, max_seq_length=8192, dtype=None, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)

    test_pairs_path = Path("data/test_raw.jsonl")
    if not test_pairs_path.exists():
        raise SystemExit("data/test_raw.jsonl not found — run M5 pipeline first")

    test_pairs = [json.loads(l) for l in open(test_pairs_path)]
    sample_pairs = random.sample(test_pairs, min(n_samples, len(test_pairs)))

    model_outputs = []
    for p in sample_pairs:
        ids = tok.apply_chat_template(
            build_messages(p["sloppy"]), tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=1024,
                                 temperature=0.9, do_sample=True)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        model_outputs.append(text)

    print(f"Generated {len(model_outputs)} model outputs for profiling")

    # Compute frequencies
    model_freqs    = compute_pattern_frequencies(model_outputs, EXTENDED_PATTERNS)
    baseline_freqs = compute_pattern_frequencies(baseline_texts, EXTENDED_PATTERNS)

    # Build ranked profile
    profile = []
    for pat in EXTENDED_PATTERNS:
        mf = model_freqs.get(pat, 0.0)
        bf = baseline_freqs.get(pat, 0.0)
        profile.append({
            "pattern":              pat,
            "model_freq_per_1k":    round(mf, 4),
            "baseline_freq_per_1k": round(bf, 4),
            "delta":                round(mf - bf, 4),
        })

    profile.sort(key=lambda x: x["delta"], reverse=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(profile, open(output_path, "w"), indent=2)

    # Write draft banlist
    banlist_path = "data/slop_banlist_draft.txt"
    with open(banlist_path, "w") as f:
        for entry in profile[:50]:
            if entry["delta"] > 0.5:
                # Strip regex word boundaries for the banlist
                word = re.sub(r'\\b', '', entry["pattern"])
                f.write(word + "\n")

    print(f"Profile written to {output_path}")
    print(f"Draft banlist written to {banlist_path}")
    print()
    print("NEXT STEP — REQUIRED BEFORE RUNNING 08_train_ftpo.py:")
    print("  1. Open data/slop_banlist_draft.txt")
    print("  2. Remove any genre-legitimate words (e.g. 'shimmered' in fantasy, 'realm' in historical fiction)")
    print("  3. Save the pruned file as: data/slop_banlist_final.txt")
    print("     cp data/slop_banlist_draft.txt data/slop_banlist_final.txt  # then edit")
    print("  4. Run: python scripts/08_train_ftpo.py")
    print("\nTop 20 residual patterns:")
    for e in profile[:20]:
        print(f"  {e['pattern']:40s} delta={e['delta']:+.3f} "
              f"(model={e['model_freq_per_1k']:.3f}, human={e['baseline_freq_per_1k']:.3f})")


if __name__ == "__main__":
    main()
