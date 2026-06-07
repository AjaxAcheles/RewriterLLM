# tests/test_environment.py — M1 environment verification.
#
# Run this on the TARGET 12 GB machine after following module_1_environment.md §5.
# Every test here maps to a fragile dependency class from M1 §2. They are designed to fail
# fast (at setup time) rather than fail mysteriously deep in Module 6.
#
# pytest tests/test_environment.py -v
#
# All tests must be green before starting any other module.
import importlib
import shutil
from pathlib import Path
import pytest


def test_cuda_available():
    """GPU is present and visible to PyTorch. If this fails, the PyTorch wheel is CPU-only."""
    import torch
    assert torch.cuda.is_available(), "CUDA not available — check PyTorch CUDA build"
    print("GPU:", torch.cuda.get_device_name(0))


def test_bitsandbytes_4bit_kernel_runs():
    """Live 4-bit op so a bnb/CUDA mismatch fails HERE, not 10 hours into Module 6."""
    import torch
    import bitsandbytes as bnb
    layer = bnb.nn.Linear4bit(64, 64, bias=False).cuda()
    out = layer(torch.randn(2, 64, device="cuda", dtype=torch.float16))
    assert out.shape == (2, 64)


def test_bf16_detection_runs():
    """Confirm bf16 detection doesn't crash (used to choose fp16 vs bf16 in training)."""
    import torch
    _ = torch.cuda.is_bf16_supported()


@pytest.mark.parametrize("mod", [
    "torch", "transformers", "datasets", "trl", "peft", "accelerate",
    "bitsandbytes", "unsloth", "spacy", "sentence_transformers",
])
def test_critical_imports(mod):
    """Every training/data stack package must import without error."""
    importlib.import_module(mod)


def test_spacy_model_present():
    """spaCy en_core_web_sm must be downloaded (used by eval_metrics.py and 03_filter_pairs.py)."""
    import spacy
    spacy.load("en_core_web_sm")


def test_llamacpp_binary_compiled():
    """llama-cli must be compiled with -DGGML_CUDA=ON for M4 generation performance."""
    assert Path("llama.cpp/build/bin/llama-cli").exists(), (
        "llama.cpp not built — run: cd llama.cpp && cmake -B build -DGGML_CUDA=ON "
        "&& cmake --build build --config Release -j$(nproc)"
    )


def test_antislop_repo_present():
    """auto-antislop repo must be cloned (used by M7 profiling)."""
    assert Path("auto-antislop").is_dir(), "auto-antislop repo missing"


def test_all_stub_files_exist():
    """All script stubs must exist so imports resolve throughout the pipeline from day one."""
    expected = [
        "scripts/01_segment.py",
        "scripts/extract_outlines.py",
        "scripts/02_generate_pairs.py",
        "scripts/03_filter_pairs.py",
        "scripts/04_add_nulls.py",
        "scripts/05_format_dataset.py",
        "scripts/06_train_sft.py",
        "scripts/07_profile_slop.py",
        "scripts/08_train_ftpo.py",
        "scripts/09_train_style.py",
        "scripts/eval_metrics.py",
        "scripts/run_eval.py",
        "scripts/run_adversarial.py",
        "scripts/prompt_config.py",
    ]
    missing = [f for f in expected if not Path(f).exists()]
    assert not missing, f"Missing stub files: {missing}"


def test_directory_tree_exists():
    """All required directories must exist before any data is generated."""
    for d in ["data/raw_texts", "models/teachers", "antislop_data",
              "checkpoints", "scripts", "tests", "reports"]:
        assert Path(d).is_dir(), f"Missing directory: {d}"


def test_disk_headroom():
    """At least 50 GB free disk space. Checkpoints corrupt when disk fills mid-write."""
    free_gb = shutil.disk_usage(".").free / (1024 ** 3)
    assert free_gb >= 50, (
        f"Only {free_gb:.1f} GB free; need >= 50 GB "
        "(checkpoints + GGUFs + datasets + model weights)"
    )
