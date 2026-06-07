# tests/test_eval_metrics.py — unit tests for the pure-function metric library (M9).
#
# These tests use hand-crafted known inputs so results are predictable. They can and SHOULD
# be run before any model or data exists — the metric library is pure Python + spaCy +
# sentence-transformers, with no GPU or large model required.
#
# This is the FIRST test file that can go green after the conda environment is set up.
#
# pytest tests/test_eval_metrics.py -v
import pytest
pytest.importorskip("spacy", reason="spacy not installed — run in the editor conda env")
pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed — run in the editor conda env")

try:
    from scripts.eval_metrics import (
        slop_score,
        burstiness,
        ner_preservation,
        hallucination_rate,
        sem_sim,
    )
except ImportError:
    from eval_metrics import (
        slop_score,
        burstiness,
        ner_preservation,
        hallucination_rate,
        sem_sim,
    )

# ---------------------------------------------------------------------------
# Representative test texts
# ---------------------------------------------------------------------------

SLOPPY = (
    "Furthermore, the vibrant tapestry of the city resonated with a profound and "
    "multifaceted sense of nuance. It is worth noting that everything changed."
)

CLEAN = "Rain slicked the streets. A tram rattled past. She pulled her coat tighter."


# ---------------------------------------------------------------------------
# slop_score
# ---------------------------------------------------------------------------

def test_slop_higher_on_slop():
    """Sloppy text should score higher than clean prose."""
    assert slop_score(SLOPPY) > slop_score(CLEAN)


def test_slop_zero_on_clean():
    """Ordinary clean prose with no banned patterns should score 0."""
    assert slop_score("The dog ran across the yard and barked once.") == 0.0


def test_slop_empty_string():
    """Empty input should not raise and should return 0.0."""
    assert slop_score("") == 0.0


# ---------------------------------------------------------------------------
# burstiness
# ---------------------------------------------------------------------------

def test_burstiness_varied_vs_uniform():
    """A text with varied sentence lengths should have higher burstiness than uniform."""
    uniform = "I walk home. I eat food. I read books. I sleep now. I wake up."
    varied = (
        "I walk. Then, after a long winding detour through the old quarter where the "
        "lamps flicker, I finally arrive home. I sleep."
    )
    assert burstiness(varied) > burstiness(uniform)


def test_burstiness_single_sentence():
    """A single sentence can't have variance — should return 0.0 without raising."""
    assert burstiness("Just one sentence here.") == 0.0


# ---------------------------------------------------------------------------
# ner_preservation
# ---------------------------------------------------------------------------

def test_ner_full_preservation():
    """All entities present in output — expect 1.0."""
    inp = "Alice met Bob in Paris."
    assert ner_preservation(inp, "Alice met Bob in Paris last week.") == 1.0


def test_ner_partial_preservation():
    """One of two entities dropped — expect < 1.0."""
    inp = "Alice met Bob in Paris."
    assert ner_preservation(inp, "Alice walked alone.") < 1.0


def test_ner_no_entities_in_input():
    """Input with no recognised entities should return 1.0 (nothing to preserve)."""
    assert ner_preservation("The dog barked.", "A dog barked.") == 1.0


# ---------------------------------------------------------------------------
# hallucination_rate
# ---------------------------------------------------------------------------

def test_hallucination_detected():
    """New entities in output not present in input should be detected."""
    assert hallucination_rate("Alice walked home.", "Alice and Bob walked to London.") > 0.0


def test_no_hallucination():
    """Output entities are a subset of input entities — expect 0.0."""
    assert hallucination_rate("Alice met Bob in Paris.", "Alice met Bob.") == 0.0


# ---------------------------------------------------------------------------
# sem_sim
# ---------------------------------------------------------------------------

def test_sem_sim_identical():
    """Identical strings should have cosine similarity near 1.0."""
    assert sem_sim("a cat sat on a mat", "a cat sat on a mat") > 0.98


def test_sem_sim_unrelated():
    """Completely unrelated texts should have low cosine similarity."""
    assert sem_sim("quantum chromodynamics", "a recipe for banana bread") < 0.5
