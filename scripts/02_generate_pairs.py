"""
02_generate_pairs — generate sloppy rewrites of clean excerpts using teacher LLMs.

Pipeline position: M4 — Data pipeline, stage 3 of 4
Depends on: data/raw_excerpts.jsonl (M2), data/outlines.json (M3)
Produces:   data/raw_pairs.jsonl

For each clean excerpt, calls >= 2 teacher LLMs to produce a "sloppy" rewrite — one that
injects the kinds of AI-generated writing patterns the editor model will learn to remove.
The teacher is given the excerpt's structured outline (from M3) as a constraint, so it can
change the prose style but cannot change characters, events, or causality.

WHY MULTIPLE TEACHERS
Using a single teacher family would teach the student to remove that family's specific tics.
We need at least two families (e.g. Llama + Mistral) so the student learns general
slop removal — patterns that appear across model families — rather than becoming a
one-teacher de-slopifier that leaves other families' output dirty.

PAIR ID FORMAT
Pair IDs follow the format: {base_excerpt_id}_{teacher_name}
The underscore separator is relied on by 05_format_dataset.py's base_id() function.
Do not use underscores inside teacher_name if you want the split logic to remain simple.

LLAMA.CPP HYBRID OFFLOAD
Teachers run via llama.cpp with hybrid CPU/GPU offload (-ngl flag controls how many layers
go to GPU). Inference is latency-tolerant (unlike training), so offloading partial layers
to CPU is acceptable here. The llama-cli binary must be compiled with -DGGML_CUDA=ON
(verified by test_environment.py). Expect ~2-10 tokens/s depending on offload ratio.

CHECKPOINTING
The script reads raw_pairs.jsonl on startup, builds a set of already-generated pair IDs,
and skips those. This makes the run resumable after a crash or interruption. A pair is
written to the file immediately after generation (not batched) so a crash loses at most
one in-flight generation.

Output schema (one JSON object per line):
  {"id": str, "clean": str, "sloppy": str, "teacher": str}
  where id = "{base_excerpt_id}_{teacher_name}"

Target: >= 4,500 raw pairs (>= 2,250 base excerpts × >= 2 teachers). M5 filters to ~2,000.

Usage:
    python scripts/02_generate_pairs.py [--excerpts data/raw_excerpts.jsonl]
                                        [--outlines data/outlines.json]
                                        [--output data/raw_pairs.jsonl]
                                        [--teachers llama3,mistral]

Key implementation notes:
  - The teacher prompt must include the outline's events list explicitly — just saying
    "preserve the plot" is not enough. Enumerate the beats and instruct the teacher to
    hit all of them.
  - Generate with temperature 0.9-1.0 to get varied slop. Low temperature produces
    near-copies that pass the semantic similarity upper bound filter in M5 (too high sim →
    dropped as "no slop to remove"). If keep rate in M5 is >90%, slop injection is too weak.
  - GGUF files for teacher models go in models/teachers/. Use Q4_K_M quantisation for the
    quality/speed tradeoff at this context length.
  - The outline constraint is not 100% effective — some teachers still drift. M5's NER
    filter catches this. Do not relax the M5 thresholds to compensate for teacher drift;
    fix the prompt here instead.
"""

import json
import subprocess
from pathlib import Path


def load_existing_ids(output_path):
    """Return set of pair IDs already written to output_path."""
    p = Path(output_path)
    if not p.exists():
        return set()
    return {json.loads(line)["id"] for line in p.open()}


def call_teacher(llama_cli, model_path, prompt, n_gpu_layers=20, max_tokens=1024):
    """Invoke llama-cli and return the generated text.

    llama_cli: path to the compiled llama-cli binary (e.g. "llama.cpp/build/bin/llama-cli")
    TODO: implement subprocess call with appropriate flags.
    """
    raise NotImplementedError


def build_slop_prompt(excerpt, outline):
    """Build the teacher prompt that produces a sloppy rewrite.

    The outline's events must be enumerated explicitly so the teacher hits every beat.
    TODO: implement prompt template.
    """
    raise NotImplementedError


def main(excerpts_path="data/raw_excerpts.jsonl", outlines_path="data/outlines.json",
         output_path="data/raw_pairs.jsonl", teachers=None):
    """Generate sloppy pairs for all excerpts × all teachers not yet in the checkpoint."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
