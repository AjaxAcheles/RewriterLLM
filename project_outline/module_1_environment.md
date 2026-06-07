# Module 1 — Environment & Infrastructure

| Field | Value |
|---|---|
| **Phase** | Foundation (pre-pipeline) |
| **Depends on** | Nothing |
| **Blocks** | Every other module |
| **Critical path** | Yes — first node |
| **Owner effort** | 0.5–1 day |
| **Runtime budget** | conda env ~3 min · pip stack ~8–15 min · llama.cpp CUDA build ~5–10 min · downloads vary |

---

## 1. Primary Objective

Produce a reproducible, fully-verified machine environment in which every downstream module
runs without dependency, version, or hardware surprises, and establish the canonical project
file tree and stub files so that all later modules read from and write to agreed-upon locations.
The deliverable is a machine state a verification test certifies as correct, plus a tree where
every file that *will* exist already exists as a stub, plus a lock file that reproduces the state
on a second machine.

---

## 2. Core Concepts — Deep Dive

### 2.1 The four fragile dependency classes

This module exists because the stack mixes four kinds of dependency that fail in *non-local*
ways — the error surfaces far from its cause:

1. **GPU-native Python libraries** (PyTorch, bitsandbytes, Unsloth) compiled against a specific
   CUDA version. A mismatch does not error at install; it errors on the first GPU kernel call,
   which is during *training* (Module 6). Symptoms: silent CPU fallback, or
   `CUDA error: no kernel image is available for execution on the device`.
2. **A compiled C++ binary** (llama.cpp). Built without CUDA it still runs — on CPU — making
   Module 4 generation 10–20× slower with no error.
3. **A third-party research repo** (auto-antislop) that pins loosely and can silently downgrade a
   training-stack package.
4. **Disk/memory headroom.** Checkpoints corrupt when disk fills mid-write.

The unifying lesson: **shift every failure left.** A verification test that exercises each
fragile point at setup converts a Module-6 mystery into a Module-1 assertion.

### 2.2 The CUDA ↔ PyTorch ↔ bitsandbytes compatibility chain

```
NVIDIA driver  ── supports up to ──►  CUDA runtime X.Y
PyTorch wheel  ── built for ────────►  CUDA X.Y   (the cuXYZ in the pip index URL)
bitsandbytes   ── ships kernels for ►  CUDA X.Y   (auto-detected from torch at import)
```

The relationship is asymmetric: a *newer* driver runs an *older* runtime, but a wheel built for
cu124 will not find kernels under a cu118 runtime. **Safe recipe:** read the driver's max CUDA
from `nvidia-smi`, install the highest PyTorch CUDA wheel at or below it, let bitsandbytes
auto-detect from torch. The verification test then runs a live 4-bit linear layer so a bnb
mismatch fails at setup rather than in Module 6.

### 2.3 Why QLoRA's components shape the environment

The whole project hinges on fitting a 7B model into 12 GB. That is only possible because
bitsandbytes provides 4-bit NF4 quantization and Unsloth provides memory-efficient kernels +
gradient checkpointing. These two libraries are therefore not optional conveniences — they are
load-bearing, and their correct installation is the point of this module. If either is broken or
falls back to a generic path, Module 6 will OOM.

### 2.4 Why a single conda environment

conda pins the *interpreter version* (3.10) independent of system Python — eliminating the most
common "works on my machine" divergence. One environment named `editor` holds training, data, and
antislop deps so a single resolver reconciles their versions; cross-concern scripts
(`run_eval.py` imports both torch and spaCy) then cannot hit an inter-env version skew.

### 2.5 Why stub every file now

Import errors and path-not-found errors are the top cause of "the run broke deep in and I don't
know why." Stubbing all scripts makes imports resolve from day one, makes total scope visible,
and lets each module fill its file without touching the tree. `scripts/__init__.py` additionally
makes `from scripts.x import y` work inside the test suite.

---

## 3. Inputs & Outputs Contract

**Inputs:** none (root module).

**Outputs other modules rely on:**

| Path | Consumed by |
|---|---|
| conda env `editor` (Python 3.10) | all |
| `llama.cpp/build/bin/llama-cli` (CUDA) | M4 |
| `auto-antislop/` (installed) | M7 |
| tree: `data/ models/ antislop_data/ checkpoints/ reports/ scripts/ tests/` | all |
| 14 script stubs + `scripts/__init__.py` + `tests/test_environment.py` | each owning module |
| `scripts/prompt_config.py` (frozen prompt contract) | M5, M6, M8, M9 |
| `requirements.lock.txt` · `reports/env_manifest.txt` | reproduction / diagnostics |

---

## 4. Common Challenges & Solutions

**Challenge 1 — PyTorch installs the CPU build.**
*Why:* pip resolves a default wheel without the CUDA index. *Detect:*
`python -c "import torch;print(torch.version.cuda)"` prints `None`. *Solve:* reinstall with the
explicit `--index-url https://download.pytorch.org/whl/cuXYZ` matching your driver.

**Challenge 2 — bitsandbytes imports but kernels fail at training.**
*Why:* bnb shipped kernels for a different CUDA than torch was built for. *Detect:* the live
4-bit kernel test (§10) errors. *Solve:* align torch's CUDA build, then
`pip install --force-reinstall bitsandbytes`.

**Challenge 3 — llama.cpp runs but only on CPU.**
*Why:* compiled without CUDA enabled. *Detect:* `llama-cli` startup log shows
`offloaded 0/N layers to GPU`, and generation is glacial. *Solve:* clean rebuild with CUDA on:
`cd llama.cpp && rm -rf build && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc)`.

**Challenge 4 — auto-antislop downgrades the training stack.**
*Why:* its `requirements.txt` pins an older transformers/peft. *Detect:* after its install,
`import unsloth` raises, or `pip show transformers` shows an unexpected downgrade. *Solve:*
install auto-antislop *last*, then force-reinstall the training stack; record exact versions in
the lock file.

**Challenge 5 — tests can't import `scripts.*`.**
*Why:* missing `scripts/__init__.py` or pytest run from the wrong directory. *Detect:*
`ModuleNotFoundError: No module named 'scripts'`. *Solve:* ensure `scripts/__init__.py` exists;
run `pytest` from the repo root.

**Challenge 6 — disk fills during a later run, corrupting a checkpoint.**
*Why:* under-provisioned disk. *Detect:* `test_disk_headroom` warns; later `OSError: No space
left on device`. *Solve:* provision ≥ 50 GB (100 recommended); prune old checkpoints with
`save_total_limit`.

---

## 5. Step-by-Step Implementation Guide

> Run from the project root. Each step lists the command and what you should see before moving on.

**Step 0 — Inspect the GPU.**
```bash
nvidia-smi
```
Confirm the card and note the CUDA version (top-right). Pick the PyTorch wheel index ≤ that.

**Step 1 — Create and activate the environment.**
```bash
conda create -n editor python=3.10 -y && conda activate editor
python --version          # expect: Python 3.10.x
```

**Step 2 — Install PyTorch (CUDA).**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect: a version, a CUDA number (not None), and True
```

**Step 3 — Install the training stack (one resolver pass).**
```bash
pip install unsloth transformers datasets trl peft accelerate bitsandbytes
```

**Step 4 — Install the data stack and the spaCy model.**
```bash
pip install spacy sentence-transformers
python -m spacy download en_core_web_sm
python -c "import spacy; spacy.load('en_core_web_sm'); print('spacy ok')"
```

**Step 5 — Build llama.cpp with CUDA.** Current llama.cpp builds via CMake; the binary lands in
`build/bin/`.
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j"$(nproc)" && cd ..
test -f llama.cpp/build/bin/llama-cli && echo "binary ok"
```

**Step 6 — Install auto-antislop, then re-verify.**
```bash
git clone https://github.com/sam-paech/auto-antislop
cd auto-antislop && pip install -r requirements.txt && cd ..
python -c "import unsloth, transformers, trl, peft, bitsandbytes; print('stack intact')"
```
If that import fails, force-reinstall the training stack (Challenge 4).

**Step 7 — Create the tree and stubs.** Run `scripts/00_setup.sh` (it is idempotent and does
steps 1–7), or create the tree manually per §3. Confirm:
```bash
ls scripts/ && ls data/ models/ antislop_data/ checkpoints/ reports/
```

**Step 8 — Capture a manifest and freeze.**
```bash
{ nvidia-smi; echo; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"; echo; pip freeze; } > reports/env_manifest.txt
pip freeze > requirements.lock.txt
```

**Step 9 — Verify.**
```bash
pytest tests/test_environment.py -v
```
Iterate installs until every test is green **on the target machine**.

---

## 6. Reference Implementation — `scripts/00_setup.sh`

```bash
#!/usr/bin/env bash
# scripts/00_setup.sh — idempotent environment + tree setup. Safe to re-run.
set -euo pipefail
log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }

ENV_NAME="editor"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"  # override via env

if ! conda env list | grep -q "^${ENV_NAME}\b"; then
  log "creating conda env ${ENV_NAME} (python 3.10)"; conda create -n "${ENV_NAME}" python=3.10 -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${ENV_NAME}"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "installing PyTorch from ${CUDA_INDEX}"; pip install torch torchvision --index-url "${CUDA_INDEX}"
fi

log "installing training stack"
pip install unsloth transformers datasets trl peft accelerate bitsandbytes

log "installing data stack + spaCy model"
pip install spacy sentence-transformers && python -m spacy download en_core_web_sm

if [ ! -f "llama.cpp/build/bin/llama-cli" ]; then
  log "building llama.cpp (CUDA)"; [ -d llama.cpp ] || git clone https://github.com/ggerganov/llama.cpp
  ( cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j"$(nproc)" )
fi

if [ ! -d "auto-antislop" ]; then
  log "installing auto-antislop"; git clone https://github.com/sam-paech/auto-antislop
  ( cd auto-antislop && pip install -r requirements.txt )
fi
log "re-verifying training stack"
python -c "import unsloth, transformers, trl, peft, bitsandbytes" \
  || { log "stack broken by antislop — reinstalling"; pip install --force-reinstall unsloth transformers trl peft bitsandbytes; }

log "creating tree + stubs"
mkdir -p data/raw_texts models/teachers antislop_data checkpoints scripts tests reports
STUBS=( scripts/01_segment.py scripts/02_generate_pairs.py scripts/03_filter_pairs.py
  scripts/04_add_nulls.py scripts/05_format_dataset.py scripts/06_train_sft.py
  scripts/07_profile_slop.py scripts/08_train_ftpo.py scripts/09_train_style.py
  scripts/eval_metrics.py scripts/run_eval.py scripts/extract_outlines.py
  scripts/run_adversarial.py scripts/prompt_config.py )
for f in "${STUBS[@]}"; do [ -f "$f" ] || printf '"""STUB — implemented in its module."""\n' > "$f"; done
[ -f scripts/__init__.py ] || touch scripts/__init__.py

pip freeze > requirements.lock.txt
log "done. Run: pytest tests/test_environment.py"
```

Reproduction on a second machine: `conda create -n editor python=3.10 -y && conda activate
editor && pip install -r requirements.lock.txt`, then rebuild llama.cpp and re-clone
auto-antislop.

---

## 7. Configuration & Parameters

| Parameter | Value | Rationale | If you change it |
|---|---|---|---|
| Python | 3.10 | Most reliable wheels for Unsloth + bnb | 3.11/3.12 may lack a wheel — verify first |
| CUDA wheel | match `nvidia-smi` | Mismatch → CPU fallback / kernel error | Pick highest ≤ driver ceiling |
| Env tool | conda | Pins interpreter version; one resolver | venv won't manage the Python version |
| llama.cpp build | `cmake -DGGML_CUDA=ON` | CPU-only build is 10–20× slower | Rebuild clean if CUDA was off |
| Min disk | 50 GB (100 rec.) | Weights + GGUFs + data + checkpoints | < 50 GB risks checkpoint corruption |

---

## 8. Alternatives Considered & Rejected

| Alternative | Why considered | Why rejected |
|---|---|---|
| Docker | Clean isolation | Single-box GPU workflow doesn't need it; adds passthrough complexity |
| `venv` | Lighter | Doesn't manage Python *version*; the 3.10 pin is part of the contract |
| Per-concern envs | Smaller footprint | Cross-concern imports reproduce the version skew this module prevents |
| `llama-cpp-python` wheel | No compile | CUDA wheels lag and are fragile; local build is more reliable for hybrid offload |
| Skip stubs | Less clutter | Reintroduces deep import-error failures |

---

## 9. KPIs / Test File

**Test file:** `tests/test_environment.py`

```python
# tests/test_environment.py
import importlib, shutil
from pathlib import Path
import pytest


def test_cuda_available():
    import torch
    assert torch.cuda.is_available(), "CUDA not available — check PyTorch CUDA build"
    print("GPU:", torch.cuda.get_device_name(0))


def test_bitsandbytes_4bit_kernel_runs():
    """Live 4-bit op so a bnb/CUDA mismatch fails HERE, not in Module 6."""
    import torch, bitsandbytes as bnb
    layer = bnb.nn.Linear4bit(64, 64, bias=False).cuda()
    out = layer(torch.randn(2, 64, device="cuda", dtype=torch.float16))
    assert out.shape == (2, 64)


def test_bf16_detection_runs():
    import torch
    _ = torch.cuda.is_bf16_supported()


@pytest.mark.parametrize("mod", [
    "torch", "transformers", "datasets", "trl", "peft", "accelerate",
    "bitsandbytes", "unsloth", "spacy", "sentence_transformers"])
def test_critical_imports(mod):
    importlib.import_module(mod)


def test_spacy_model_present():
    import spacy; spacy.load("en_core_web_sm")


def test_llamacpp_binary_compiled():
    assert Path("llama.cpp/build/bin/llama-cli").exists(), "llama.cpp not built with -DGGML_CUDA=ON"


def test_antislop_repo_present():
    assert Path("auto-antislop").is_dir(), "auto-antislop repo missing"


def test_all_stub_files_exist():
    expected = ["scripts/01_segment.py", "scripts/02_generate_pairs.py",
        "scripts/03_filter_pairs.py", "scripts/04_add_nulls.py",
        "scripts/05_format_dataset.py", "scripts/06_train_sft.py",
        "scripts/07_profile_slop.py", "scripts/08_train_ftpo.py",
        "scripts/09_train_style.py", "scripts/extract_outlines.py",
        "scripts/eval_metrics.py", "scripts/run_eval.py", "scripts/run_adversarial.py",
        "scripts/prompt_config.py"]
    missing = [f for f in expected if not Path(f).exists()]
    assert not missing, f"Missing stub files: {missing}"


def test_directory_tree_exists():
    for d in ["data/raw_texts", "models/teachers", "antislop_data",
              "checkpoints", "scripts", "tests", "reports"]:
        assert Path(d).is_dir(), f"Missing directory: {d}"


def test_disk_headroom():
    free_gb = shutil.disk_usage(".").free / (1024 ** 3)
    assert free_gb >= 50, f"Only {free_gb:.1f} GB free; need >= 50 GB"
```

---

## 10. Definition of Done

- `pytest tests/test_environment.py` passes on the **target 12 GB machine** (incl. the live
  4-bit kernel test).
- `requirements.lock.txt` and `reports/env_manifest.txt` committed.
- All 13 script stubs, `scripts/__init__.py`, and the full tree exist.
- Reproduction from the lock file verified or documented.
