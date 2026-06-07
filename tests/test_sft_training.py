# tests/test_sft_training.py — M6 SFT training completion checks.
#
# Requires the M6 training run to have finished (models/sft_lora/ and models/sft_merged/ exist,
# reports/sft_baseline.json has been written by run_eval.py).
#
# The no-copy test loads the model and makes a live inference call — requires GPU.
#
# pytest tests/test_sft_training.py -v
import json
from pathlib import Path
import pytest

SFT_DIR  = "models/sft_lora"
MERGED   = "models/sft_merged"
BASELINE = "reports/sft_baseline.json"


def test_adapters_saved():
    """LoRA adapter files must exist in models/sft_lora/."""
    assert Path(SFT_DIR).is_dir(), f"Adapter directory not found: {SFT_DIR}"
    has_adapter = any("adapter" in p.name for p in Path(SFT_DIR).iterdir())
    assert has_adapter, f"No adapter files found in {SFT_DIR}"


def test_merged_model_saved():
    """Merged 16-bit model must exist — M7 profiling loads models/sft_merged/."""
    assert Path(MERGED).is_dir(), (
        f"Merged model not found at {MERGED}. "
        "Run: model.save_pretrained_merged('models/sft_merged', tokenizer, save_method='merged_16bit')"
    )


def test_model_does_not_copy_input():
    """The model must edit the sloppy input, not return it unchanged.

    This is the primary check for the completion-only loss masking in M6. If this fails,
    the DataCollatorForCompletionOnlyLM was not wired correctly and the model learned to copy.
    """
    from unsloth import FastLanguageModel
    from scripts.prompt_config import build_messages

    model, tok = FastLanguageModel.from_pretrained(
        SFT_DIR, max_seq_length=4096, dtype=None, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)

    sloppy = (
        "Furthermore, the vibrant tapestry of the bustling city resonated with a "
        "profound sense of nuance, and she felt deeply sad."
    )
    ids = tok.apply_chat_template(
        build_messages(sloppy),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda")

    import torch
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, max_new_tokens=200, temperature=0.3,
            do_sample=True, repetition_penalty=1.1,
        )
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    assert text.strip(), "Model produced empty output"
    assert text.strip() != sloppy.strip(), (
        "Model copied its input verbatim — completion-only loss masking is broken. "
        "Check DataCollatorForCompletionOnlyLM is wired in with packing=False."
    )


@pytest.mark.skipif(not Path(BASELINE).exists(), reason="baseline not yet recorded")
def test_baseline_gate():
    """SFT baseline must clear the M6 → M7 gate before FTPO starts."""
    m = json.load(open(BASELINE))
    assert m["slop_delta"] > 1.0,          f"slop_delta={m['slop_delta']:.3f} must be > 1.0"
    assert m["ner_preservation"] >= 0.90,  f"ner_preservation={m['ner_preservation']:.3f} must be >= 0.90"
    assert m["sem_vs_target"] >= 0.78,     f"sem_vs_target={m['sem_vs_target']:.3f} must be >= 0.78"
