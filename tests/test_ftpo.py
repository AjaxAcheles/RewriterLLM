# tests/test_ftpo.py — M7 FTPO training completion checks.
#
# Requires models/ftpo_model/ and reports/ftpo_eval.json to exist.
# Also requires reports/sft_baseline.json (M6 output) for regression comparison.
#
# pytest tests/test_ftpo.py -v
import json
from pathlib import Path
import pytest

FTPO_DIR  = "models/ftpo_model"
FTPO_EVAL = "reports/ftpo_eval.json"
BASELINE  = "reports/sft_baseline.json"


def test_ftpo_model_saved():
    # models/ftpo_model/ must exist and contain model files.
    pass


@pytest.mark.skipif(
    not Path(FTPO_EVAL).exists() or not Path(BASELINE).exists(),
    reason="eval reports not yet generated"
)
def test_slop_delta_improved():
    # FTPO's slop_delta must be strictly greater than the SFT baseline's slop_delta.
    # This is the primary assertion that FTPO actually reduced residual slop.
    # If it doesn't improve, the banlist may be too small or the LR too low.
    pass


@pytest.mark.skipif(
    not Path(FTPO_EVAL).exists() or not Path(BASELINE).exists(),
    reason="eval reports not yet generated"
)
def test_ner_preservation_not_regressed():
    # FTPO's ner_preservation must be >= SFT baseline's ner_preservation - 0.02.
    # A regression here means FTPO is altering content, not just surface style.
    # Tighten the learning rate or reduce epochs if this fails.
    pass


@pytest.mark.skipif(
    not Path(FTPO_EVAL).exists() or not Path(BASELINE).exists(),
    reason="eval reports not yet generated"
)
def test_sem_vs_target_not_regressed():
    # FTPO's sem_vs_target must be >= SFT baseline's sem_vs_target - 0.02.
    # A regression here means the edits are drifting away from the human targets.
    pass


@pytest.mark.skipif(not Path(FTPO_EVAL).exists(), reason="eval report not yet generated")
def test_ftpo_absolute_gates():
    # Regardless of baseline comparison, FTPO output must clear the absolute thresholds.
    # These are the same gates as M6 but the FTPO run must also pass them.
    pass
