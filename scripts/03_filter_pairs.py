"""
03_filter_pairs — filter raw sloppy/clean pairs by NER overlap and semantic similarity.

Pipeline position: M5 — Data pipeline, stage 4 of 4 (step 1 of 3)
Depends on: data/raw_pairs.jsonl (M4)
Produces:   data/filtered_pairs.jsonl

This is the quality gate between noisy generation and expensive training. Applies three
filters, each catching a distinct class of bad pair:

  1. NER overlap >= 0.90
     Extracts named entities (PERSON/GPE/LOC/ORG/FAC) from both sides; requires the sloppy
     version to retain >= 90% of the clean version's entities. Primary defence against
     content drift — a dropped character or invented location is a hard failure signal.

  2. Semantic similarity >= 0.80 (lower bound)
     Catches meaning drift: different events, different emphasis, different resolution.

  3. Semantic similarity <= 0.97 (upper bound)
     The non-obvious filter: too-high similarity means the teacher barely changed the text,
     so there is little slop to remove. A near-identical pair teaches the model nothing (or,
     worse, that "do nothing" is the right response even on sloppy input). If the keep rate
     exceeds ~90%, tighten this bound to 0.94.

The script prints keep/drop counts and the keep rate. Act on the rate before proceeding:
  < 40%: widespread M4 drift — fix the teacher prompt in 02_generate_pairs.py
  > 90%: degenerate pairs slipping through — tighten max_sem

Output schema: raw_pairs.jsonl fields plus {"ner_overlap": float, "sem_sim": float}.

Usage:
    python scripts/03_filter_pairs.py [--input data/raw_pairs.jsonl]
                                      [--output data/filtered_pairs.jsonl]
                                      [--min-ner 0.90] [--min-sem 0.80] [--max-sem 0.97]

Key implementation notes:
  - Batch-encode ALL texts at once with embedder.encode([list]) — single-pair encoding is
    10-50x slower. Read all pairs into memory, encode the full list, then filter.
  - NER can mislabel fantasy/invented proper nouns. If you see valid pairs being dropped,
    inspect the rejected pairs manually and consider lowering min_ner to 0.85 or adding a
    capitalised-token heuristic for names spaCy misses.
  - The embedder used here MUST match the one in eval_metrics.py (all-MiniLM-L6-v2) so that
    similarity thresholds are on the same scale. Changing the embedder requires re-tuning
    the bounds and re-running M5.
"""

import json
import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

_NER_LABELS = {"PERSON", "GPE", "LOC", "ORG", "FAC", "NORP"}


def ner_overlap(clean, sloppy):
    """Fraction of clean-side entities that appear in sloppy side. Returns 1.0 if no entities."""
    raise NotImplementedError


def main(raw="data/raw_pairs.jsonl", out="data/filtered_pairs.jsonl",
         min_ner=0.90, min_sem=0.80, max_sem=0.97):
    """Filter raw_pairs.jsonl and write kept pairs with overlap/sim scores appended."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
