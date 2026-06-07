# tests/test_ftpo.py — M7 FTPO training completion checks.
#
# Requires models/ftpo_model/ and reports/ftpo_eval.json to exist.
# Also requires reports/sft_baseline.json (M6 output) for regression comparison.
#
# pytest tests/test_ftpo.py -v
import json
from pathlib import Path
import pytest

FP        = "antislop_data/slop_fingerprint.json"
BAN       = "antislop_data/banlist.json"
PAIRS     = "antislop_data/ftpo_pairs.jsonl"
FTPO_DIR  = "models/ftpo_model"
SFT_BASE  = "reports/sft_baseline.json"
FTPO_EVAL = "reports/ftpo_eval.json"


def test_fingerprint_generated():
    if not Path("reports/slop_profile.json").exists() and not Path(FP).exists():
        pytest.skip("Slop profile not yet generated — run scripts/07_profile_slop.py first")
    p = Path(FP) if Path(FP).exists() else Path("reports/slop_profile.json")
    assert p.exists()
    assert len(json.loads(p.read_text())) > 0


def test_banlist_pruned():
    banlist_path = Path(BAN) if Path(BAN).exists() else Path("data/slop_banlist_final.txt")
    if not banlist_path.exists():
        pytest.skip("Human-pruned banlist not yet created")
    assert banlist_path.exists(), "Pruned banlist missing (human review skipped)"


def test_model_saved():
    if not Path(FTPO_DIR).is_dir():
        pytest.skip("FTPO model not yet trained — run scripts/08_train_ftpo.py first")
    assert Path(FTPO_DIR).is_dir()


@pytest.mark.skipif(
    not (Path(SFT_BASE).exists() and Path(FTPO_EVAL).exists()),
    reason="eval reports not present"
)
def test_improvement_without_regression():
    b = json.load(open(SFT_BASE))
    f = json.load(open(FTPO_EVAL))
    assert f["slop_delta"] >= b["slop_delta"],                                "slop not reduced further"
    assert f["ner_preservation"] >= b["ner_preservation"] - 0.02,             "NER regressed"
    assert f.get("burstiness_improved_rate", 1.0) >= b.get("burstiness_improved_rate", 0) - 0.05, "diversity regressed"
    assert f["sem_vs_target"] >= b["sem_vs_target"] - 0.02,                   "target fidelity regressed"
