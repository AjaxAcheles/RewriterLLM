"""check_style_convergence.py — verify that style conditioning moves outputs toward the reference.

Computes a simple stylometric feature vector (sentence stats, punctuation rates) and checks
that the styled model output is closer to the reference than the baseline (unconditioned) output.

Usage:
    python scripts/check_style_convergence.py --reference "<text>" --baseline "<text>" --styled "<text>"

Or import and call closer_to_reference() directly.
"""
import re
import numpy as np


def style_vector(text):
    """Compute a 5-element stylometric feature vector for the given text."""
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(s.split()) for s in sents] or [0]
    words = text.split() or [""]
    return np.array([
        np.mean(lengths),                                              # avg sentence length
        np.std(lengths),                                               # sentence length variance
        text.count(",") / max(len(sents), 1),                          # comma rate per sentence
        text.count('"') / max(len(words), 1),                          # dialogue density
        sum(w.endswith("ly") for w in words) / len(words),             # adverb rate
    ], dtype=float)


def closer_to_reference(reference, baseline_out, styled_out):
    """Check whether styled_out is closer to reference than baseline_out.

    Returns (is_closer: bool, dist_baseline: float, dist_styled: float)
    """
    rv = style_vector(reference)
    bv = style_vector(baseline_out)
    sv = style_vector(styled_out)
    dist_baseline = float(np.linalg.norm(bv - rv))
    dist_styled   = float(np.linalg.norm(sv - rv))
    return dist_styled < dist_baseline, dist_baseline, dist_styled


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference")
    parser.add_argument("--baseline")
    parser.add_argument("--styled")
    args = parser.parse_args()
    if args.reference and args.baseline and args.styled:
        ok, d_base, d_styled = closer_to_reference(args.reference, args.baseline, args.styled)
        print(f"Closer to reference: {ok}")
        print(f"  Baseline distance: {d_base:.4f}")
        print(f"  Styled distance:   {d_styled:.4f}")
