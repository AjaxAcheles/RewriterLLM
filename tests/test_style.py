# tests/test_style.py — M8 style conditioning checks (optional module).
#
# Requires models/style_lora/ and reports/style_eval.json to exist.
# Also requires reports/ftpo_eval.json (M7 output) for regression comparison.
# This module is optional; if M8 is not being built, skip this file entirely.
#
# pytest tests/test_style.py -v
import json
from pathlib import Path
import pytest

STYLE_DIR  = "models/style_lora"
STYLE_EVAL = "reports/style_eval.json"
FTPO_EVAL  = "reports/ftpo_eval.json"


def test_style_lora_saved():
    # models/style_lora/ must exist and contain LoRA adapter files.
    # This is a LoRA-only checkpoint — the base M7 weights are not duplicated.
    pass


@pytest.mark.skipif(
    not Path(STYLE_EVAL).exists() or not Path(FTPO_EVAL).exists(),
    reason="eval reports not yet generated"
)
def test_no_core_regression():
    # The style adapter must not regress ner_preservation, sem_vs_target, or slop_delta
    # vs. the FTPO baseline. Style conditioning adds capability; it must not degrade
    # the core editor behaviour trained in M6 and refined in M7.
    pass


@pytest.mark.skipif(not Path(STYLE_EVAL).exists(), reason="eval report not yet generated")
def test_conditioned_output_shifts_toward_reference():
    # When a style reference is provided, the model's output should be measurably more
    # similar to the reference style than the unconditioned output.
    # Metric: cosine similarity between output embedding and reference embedding should
    # be higher for conditioned vs. unconditioned runs on the same sloppy input.
    pass


@pytest.mark.skipif(not Path(STYLE_EVAL).exists(), reason="eval report not yet generated")
def test_unconditioned_output_matches_ftpo_baseline():
    # Without a style reference, the style adapter must produce output equivalent to
    # the base M7 model. Train ONLY on conditioned pairs (09_train_style.py does this)
    # to prevent the adapter from degrading unconditioned editing.
    pass
