# RewriterLLM — AI Slop Editor

Fine-tunes a Qwen3-7B model (via QLoRA + Unsloth) to strip AI-generated writing patterns from prose while preserving every character, event, and plot beat.

---

## What it does

The model takes sloppy, AI-flavored prose and rewrites it to sound like a human wrote it. It removes patterns like overused words ("tapestry", "delve", "resonate"), empty transitions ("Furthermore, Moreover"), parallel three-part structures, and unearned emotional pivots — without changing the story.

Training uses **inverse injection**: clean human text → teacher LLM adds slop → `(sloppy, clean)` training pair. The model learns to reverse that transformation.

---

## Hardware requirements

- GPU with **≥ 12 GB VRAM** (tested on RTX 3060 12 GB)
- CUDA 12.1+
- ~60 GB disk (models, corpus, checkpoints)

---

## Quick start

### 1. Set up the environment

```bash
bash scripts/00_setup.sh
conda activate editor
pytest tests/test_environment.py -v
```

The setup script creates the `editor` conda environment, installs all dependencies, and builds llama.cpp with CUDA support.

> **CUDA wheel note:** After `conda activate editor`, run:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Replace `cu121` with the highest CUxx at or below your driver's max CUDA version (visible in `nvidia-smi`).
> **CUDA compilation note:** If you are running a modern Linux distribution with a recent `glibc`, CUDA toolkit compilations (like `llama.cpp`) may fail due to a missing `noexcept(true)` specifier in `math_functions.h`. 
> 
> The `00_setup.sh` script will automatically detect this and attempt to safely patch the CUDA header using `sed`. You may be prompted for your `sudo` password during this step. A backup of your original header is automatically saved as `math_functions.h.bak`.

### 2. Download teacher models

Place Q4_K_M GGUF files in `models/teachers/`:

```
models/teachers/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
models/teachers/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 3. Run the data pipeline

```bash
# Fetch source corpus from HuggingFace (sedthh/gutenberg_english)
python scripts/fetch_gutenberg.py

# Segment into 300-700 word excerpts
python scripts/01_segment.py

# Extract structured outlines (characters, events, setting)
python scripts/extract_outlines.py
# *** HUMAN GATE: review data/outlines.json — reconstruction accuracy >= 90% required ***

# Generate sloppy rewrites via teacher LLMs
python scripts/02_generate_pairs.py

# Filter pairs by NER overlap and semantic similarity
python scripts/03_filter_pairs.py

# Add null-edit pairs (clean input → unchanged output, ~12% of dataset)
python scripts/04_add_nulls.py

# Apply chat template and split train/val/test
python scripts/05_format_dataset.py
```

### 4. Train

```bash
# Phase B1 — SFT on (sloppy → clean) pairs
python scripts/06_train_sft.py

# Evaluate SFT baseline (must clear: slop_delta > 1.0, ner_preservation >= 0.90)
python scripts/run_eval.py models/sft_lora reports/sft_baseline.json

# Phase B2 — Slop profiling + FTPO suppression
python scripts/07_profile_slop.py
# *** HUMAN GATE: prune antislop_data/fingerprint_review.csv before running FTPO ***
python scripts/08_train_ftpo.py

# Phase B3 (optional) — Style conditioning adapter
python scripts/09_train_style.py
```

### 5. Evaluate

```bash
# Full test-set evaluation
python scripts/run_eval.py models/ftpo_model reports/ftpo_eval.json

# Adversarial tests (requires hand-curated fixtures in data/)
python scripts/run_adversarial.py models/ftpo_model
```

---

## Module map

```
M1 Environment ──┬─> M2 Corpus ──> M3 Outline ──> M4 Generation ──> M5 Curation ──┐
                 │                                                                  │
                 └─> M9 Evaluation Harness ──────────────────────────────────────── ┤
                                                                                    ▼
                                                              M6 SFT ──> M7 FTPO ──> M8 Style
```

| Module | Script(s) | Description |
|--------|-----------|-------------|
| M1 | `00_setup.sh` | Environment setup, dependency install, llama.cpp build |
| M2 | `fetch_gutenberg.py`, `01_segment.py` | Corpus download and segmentation |
| M3 | `extract_outlines.py` | LLM-based structured outline extraction |
| M4 | `02_generate_pairs.py` | Teacher LLM sloppy pair generation |
| M5 | `03_filter_pairs.py`, `04_add_nulls.py`, `05_format_dataset.py` | Curation and dataset split |
| M6 | `06_train_sft.py` | SFT training with Unsloth + QLoRA |
| M7 | `07_profile_slop.py`, `08_train_ftpo.py` | Slop profiling and FTPO suppression |
| M8 | `09_train_style.py`, `build_style_pairs.py` | Style conditioning adapter |
| M9 | `eval_metrics.py`, `run_eval.py`, `run_adversarial.py` | Evaluation harness |

Full module documentation is in [`project_outline/`](project_outline/).

---

## Key design decisions

- **Completion-only loss masking** (`DataCollatorForCompletionOnlyLM` with `packing=False`) — the single most important training detail. Full-sequence loss causes the model to copy its input; masking trains actual editing.
- **`enable_thinking=False`** — prevents Qwen3 from injecting `<think>…</think>` scaffolding at both training and inference.
- **Split by base excerpt ID** — all sloppy versions of the same excerpt go to the same split, making leakage impossible by construction.
- **Null-edit pairs at 12%** — prevents the model from over-editing already-clean prose.
- **Frozen prompt contract** — `scripts/prompt_config.py` is the single source of truth for system prompts, imported identically by training and inference.

---

## Running tests

```bash
# Unit tests (no GPU or data required)
pytest tests/test_eval_metrics.py tests/test_segmentation.py -v

# Full test suite (skips tests whose data/models aren't present)
pytest tests/ -v
```

---

## Phase gates

Do not start a downstream stage until its upstream KPIs are green:

| Gate | Metric |
|------|--------|
| M3 → M4 | Manual outline reconstruction ≥ 90% |
| M5 → M6 | Zero base-ID leakage; ≥ 1,800 filtered pairs |
| M6 → M7 | `slop_delta > 1.0`, `ner_preservation ≥ 0.90`, `sem_vs_target ≥ 0.78` |
| M7 → M8 | FTPO shows slop reduction with preservation/diversity held vs. SFT baseline |

---

## Adversarial test fixtures

`run_adversarial.py` requires three hand-curated fixtures (20-30 items each):

```
data/eval_clean_prose.jsonl    # {"text": str}  — slop-free human excerpts
data/eval_strong_voice.jsonl   # {"text": str}  — idiosyncratic narrator voice
data/eval_tone_tagged.jsonl    # {"text": str, "tone": str}
```

The script skips tests whose fixture files are absent.

---

## License

See [LICENSE](LICENSE) (if present) or the original Qwen3 and Gutenberg corpus terms.
