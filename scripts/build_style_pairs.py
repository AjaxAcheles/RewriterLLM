"""build_style_pairs.py — build style-conditioned training pairs for M8.

For each filtered pair, attaches a style reference of 3-5 same-author excerpts
from data/raw_excerpts.jsonl. Writes data/style_train.jsonl.

Usage:
    python scripts/build_style_pairs.py [--filtered data/filtered_pairs.jsonl]
                                        [--excerpts data/raw_excerpts.jsonl]
                                        [--output data/style_train.jsonl]
                                        [--k-ref 4]
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer

try:
    from scripts.prompt_config import CANONICAL_TOKENIZER, build_messages, render
except ImportError:
    from prompt_config import CANONICAL_TOKENIZER, build_messages, render

_tok = None  # lazy-loaded


def _get_tokenizer():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(CANONICAL_TOKENIZER)
    return _tok


def fmt(style_ref, sloppy, clean):
    tok = _get_tokenizer()
    return {"text": render(tok, build_messages(sloppy, clean, style_reference=style_ref))}


def main(filtered="data/filtered_pairs.jsonl", excerpts="data/raw_excerpts.jsonl",
         out="data/style_train.jsonl", seed=42, k_ref=4):
    random.seed(seed)

    by_author = defaultdict(list)
    id_to_src = {}
    id_to_text = {}

    for line in open(excerpts):
        e = json.loads(line)
        by_author[e["source"]].append(e["text"])
        id_to_src[e["id"]] = e["source"]
        id_to_text[e["id"]] = e["text"]

    written = 0
    skipped = 0
    with open(out, "w") as f:
        for line in open(filtered):
            p = json.loads(line)
            base = p["id"].rsplit("_", 1)[0]
            src = id_to_src.get(base)
            if not src:
                skipped += 1
                continue
            author_texts = by_author[src]
            if len(author_texts) < k_ref + 1:
                skipped += 1
                continue
            # Exclude the clean target itself from the reference pool
            pool = [t for t in author_texts if t != p["clean"]]
            refs = random.sample(pool, min(k_ref, len(pool)))
            style_ref = "\n\n".join(refs)
            f.write(json.dumps(fmt(style_ref, p["sloppy"], p["clean"])) + "\n")
            written += 1

    print(f"Wrote {written} style-conditioned pairs to {out} ({skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered", default="data/filtered_pairs.jsonl")
    parser.add_argument("--excerpts", default="data/raw_excerpts.jsonl")
    parser.add_argument("--output",   default="data/style_train.jsonl")
    parser.add_argument("--k-ref",    type=int, default=4)
    args = parser.parse_args()
    main(args.filtered, args.excerpts, args.output, k_ref=args.k_ref)
