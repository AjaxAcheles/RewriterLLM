#!/usr/bin/env bash
# scripts/00_setup.sh — idempotent environment + tree setup. Safe to re-run.
set -euo pipefail

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }

ENV_NAME="editor"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

if ! conda env list | grep -q "^${ENV_NAME}\b"; then
  log "creating conda env ${ENV_NAME} (python 3.10)"
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "installing PyTorch from ${CUDA_INDEX}"
  pip install torch torchvision --index-url "${CUDA_INDEX}"
fi

log "installing training stack"
pip install unsloth transformers datasets trl peft accelerate bitsandbytes

log "installing data stack + spaCy model"
pip install spacy sentence-transformers
python -m spacy download en_core_web_sm

if [ ! -f "llama.cpp/build/bin/llama-cli" ]; then
  log "building llama.cpp (CUDA)"
  [ -d llama.cpp ] || git clone https://github.com/ggerganov/llama.cpp
  ( cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j"$(nproc)" )
fi

if [ ! -d "auto-antislop" ]; then
  log "installing auto-antislop"
  git clone https://github.com/sam-paech/auto-antislop
  ( cd auto-antislop && pip install -r requirements.txt )
fi

log "re-verifying training stack"
python -c "import unsloth, transformers, trl, peft, bitsandbytes" \
  || { log "stack broken by antislop — reinstalling"
       pip install --force-reinstall unsloth transformers trl peft bitsandbytes; }

log "creating tree + stubs"
mkdir -p data/raw_texts models/teachers antislop_data checkpoints scripts tests reports

STUBS=(
  scripts/01_segment.py scripts/02_generate_pairs.py scripts/03_filter_pairs.py
  scripts/04_add_nulls.py scripts/05_format_dataset.py scripts/06_train_sft.py
  scripts/07_profile_slop.py scripts/08_train_ftpo.py scripts/09_train_style.py
  scripts/eval_metrics.py scripts/run_eval.py scripts/extract_outlines.py
  scripts/run_adversarial.py scripts/prompt_config.py
)
for f in "${STUBS[@]}"; do
  [ -f "$f" ] || printf '"""STUB — implemented in its module."""\n' > "$f"
done
[ -f scripts/__init__.py ] || touch scripts/__init__.py

pip freeze > requirements.lock.txt
log "done. Run: pytest tests/test_environment.py"
