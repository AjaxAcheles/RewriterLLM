"""
eval_metrics — pure-function metric library for the AI-slop editor model.

THIS FILE'S EXPORTED NAMES ARE A FROZEN CONTRACT. The keys in the dict returned by
`evaluate_pair` are read verbatim by the completion tests in M6, M7, and M8:
  reports/sft_baseline.json  (M6 gate: slop_delta > 1.0, ner_preservation >= 0.90, sem_vs_target >= 0.78)
  reports/ftpo_eval.json     (M7 gate: slop down without regression vs. SFT baseline)
  reports/style_eval.json    (M8 gate: no core regression)

Do NOT rename any function or dict key. If you need a new metric, add it; never remove or rename.

WHY PURE FUNCTIONS
------------------
No model loading, no file I/O, no state. Text in, number out. This makes every metric
independently unit-testable with hand-crafted known inputs (see tests/test_eval_metrics.py)
*before* any model exists. It also means this module is safe to import inside the training
loop as `compute_metrics` — the same definitions drive in-loop monitoring and final gating,
so there is no divergence between "what we checked during training" and "what we gate on
at the end."

THE METRIC CONSTELLATION
-------------------------
No single automated metric for writing quality is trusted (automated judges agree with human
preference only ~73-78% of the time). The constellation below captures different facets:

  slop_score         — banned-pattern density per 1k words; measures lexical slop
  slop_delta         — reduction from input to output; defends against "doing nothing"
  burstiness         — sentence-length variance; defends against rhythm flattening
  ner_preservation   — fraction of input named entities surviving in output; defends against content drift
  hallucination_rate — fraction of output entities that are NEW (not in input); defends against invention
  semantic_sim       — cosine similarity input→output; catches over- and under-editing
  sem_vs_target      — closeness to the human clean target; the only metric with ground truth

The human-preference protocol (documented in module_9_evaluation.md §6) is the ground truth
these proxies are calibrated against. If automated metrics improve but human win-rate doesn't,
the metrics are measuring the wrong thing — revise them, don't trust them.

EXTENDING SLOP_PATTERNS
------------------------
The initial list covers well-known LLM verbal tics. After M7 profiles the SFT model's residual
slop fingerprint against a Gutenberg baseline, additional patterns will be added here. The human
review step in M7 must prune any genre-legitimate words from the banlist before adding them.
"""

import re
import spacy
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Lazy-loaded models (module-level singletons — load once, reuse)
# ---------------------------------------------------------------------------

# spaCy small model — adequate for NER on fiction; faster than md/lg.
# If fantasy/invented names score poorly, consider augmenting with a capitalized-token heuristic
# rather than switching to a larger model (the gain is marginal for this task).
nlp = spacy.load("en_core_web_sm")

# MiniLM is fast and adequate for paragraph-level semantic similarity. The same model is used in
# M5 (filtering) so scores are consistent across the pipeline. Upgrading the embedder here
# requires re-running M5 filtering to keep thresholds meaningful.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Slop pattern list — FROZEN after M5 dataset creation; extend only via M7 profiling
# ---------------------------------------------------------------------------

# Each entry is a regex pattern (case-insensitive search applied to lowercased text).
# Sources: well-known LLM verbal tics, hollow transition phrases, cliché narrative beats.
# M7 will append model-specific residual patterns after profiling the SFT output.
SLOP_PATTERNS = [
    # Overused individual words
    r"\btapestry\b",
    r"\bdelve\b",
    r"\btestament\b",
    r"\bvibrant\b",
    r"\bnavigate\b",
    r"\bnuanced\b",
    r"\bresonate\b",
    r"\bpivotal\b",
    r"\bmultifaceted\b",
    r"\bunderscore\b",
    r"\bembark\b",
    r"\brealm\b",
    r"\bshimmered\b",
    r"\bunsettlingly\b",
    # Hollow transition phrases
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bin addition\b",
    r"\bit is worth noting\b",
    # Cliché narrative beats
    r"\bsomething shifted\b",
    r"\beverything changed\b",
    r"\bbut here's the thing\b",
    r"\bat the end of the day\b",
    r"it's not .+, it's",
]

# NER label set used for both ner_preservation and hallucination_rate.
# NORP (nationalities, religious groups) is excluded here — it fires too readily on adjectives
# like "British" or "Catholic" which are not entities in the content-preservation sense.
_NER_LABELS = {"PERSON", "GPE", "LOC", "ORG", "FAC"}


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def slop_score(text):
    """Density of banned slop patterns, normalised to hits per 1,000 words.

    Returns 0.0 on empty input. Higher = more sloppy.
    """
    t = text.lower()
    hits = sum(1 for p in SLOP_PATTERNS if re.search(p, t))
    words = len(text.split())
    return hits / (words / 1000) if words else 0.0


def burstiness(text):
    """Standard deviation of sentence lengths (in words).

    High burstiness means varied rhythm — a hallmark of good prose. Flat rhythm (uniform
    sentence length) is a strong marker of AI-generated text. Defends against the model
    ironing out authorial sentence variety while "cleaning" the excerpt.

    Returns 0.0 if fewer than 2 sentences are detected.
    """
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return variance ** 0.5


def _ents(text):
    """Extract the set of lowercased named entity strings from text."""
    return {e.text.lower() for e in nlp(text).ents if e.label_ in _NER_LABELS}


def ner_preservation(inp, out):
    """Fraction of named entities in `inp` that also appear in `out`.

    Returns 1.0 if `inp` has no recognised entities (nothing to lose).
    The primary defence against content drift — a dropped character or invented location
    is a hard signal the pair is teaching rewriting, not editing.
    """
    inp_ents = _ents(inp)
    if not inp_ents:
        return 1.0
    return len(inp_ents & _ents(out)) / len(inp_ents)


def hallucination_rate(inp, out):
    """Fraction of output entities that do NOT appear in the input.

    New entities in the output are invented content — the editor must not introduce
    characters, places, or organisations that weren't in the original excerpt.
    Returns 0.0 if the output has no recognised entities.
    """
    inp_ents = _ents(inp)
    out_ents = _ents(out)
    return len(out_ents - inp_ents) / max(len(out_ents), 1)


def sem_sim(a, b):
    """Cosine similarity between the sentence-transformer embeddings of `a` and `b`.

    Used for two purposes:
      1. semantic_sim (input→output): should be 0.80–0.92 — edited but not rewritten.
      2. sem_vs_target (output→clean target): the only metric with ground truth; >= 0.80.
    """
    return float(
        util.cos_sim(
            embedder.encode(a, convert_to_tensor=True),
            embedder.encode(b, convert_to_tensor=True),
        )
    )


# ---------------------------------------------------------------------------
# Composite evaluation function — the frozen API consumed by run_eval.py
# ---------------------------------------------------------------------------

def evaluate_pair(sloppy_input, model_output, clean_target=None):
    """Compute the full metric constellation for one (input, output[, target]) triple.

    Args:
        sloppy_input:  The AI-sloppy text fed to the model at inference time.
        model_output:  The model's edited output.
        clean_target:  The human-written ground-truth clean version (optional).
                       When provided, `sem_vs_target` is added to the result dict.
                       Required for the M6/M7/M8 gate checks.

    Returns:
        Dict with the following keys (FROZEN — do not rename):

        Gated keys (read by M6/M7/M8 completion tests):
          slop_delta              — slop_in minus slop_out; > 1.0 to pass M6 gate
          ner_preservation        — fraction of input entities surviving; >= 0.90 (M6), >= 0.95 (M9 target)
          hallucination_rate      — fraction of new output entities; < 0.05 target
          sem_vs_target           — output cosine sim to human target; >= 0.78 (M6), >= 0.80 (M9 target)
          burstiness_improved_rate — aggregated in run_eval.py; True here = improved on this pair

        Informational keys (not gated, but printed in the eval table):
          slop_in                 — raw slop score of the input
          slop_out                — raw slop score of the output
          semantic_sim            — input→output cosine similarity
          burstiness_out          — absolute burstiness of the output
          burstiness_improved     — bool: output burstiness >= input burstiness
    """
    slop_in = slop_score(sloppy_input)
    slop_out = slop_score(model_output)

    result = {
        # Informational
        "slop_in": slop_in,
        "slop_out": slop_out,
        "burstiness_out": burstiness(model_output),
        "burstiness_improved": burstiness(model_output) >= burstiness(sloppy_input),
        "semantic_sim": sem_sim(sloppy_input, model_output),
        # Gated
        "slop_delta": slop_in - slop_out,
        "ner_preservation": ner_preservation(sloppy_input, model_output),
        "hallucination_rate": hallucination_rate(sloppy_input, model_output),
    }
    if clean_target is not None:
        result["sem_vs_target"] = sem_sim(model_output, clean_target)
    return result
