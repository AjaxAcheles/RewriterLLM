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

import argparse
import json
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Teacher configuration — add/remove entries here, one per model family.
# Adjust n_gpu_layers to the max that loads without OOM on your GPU.
# ---------------------------------------------------------------------------
TEACHERS = {
    "llama3": {
        "path": "models/teachers/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "n_gpu_layers": 28,
        "slop_block": (
            "- Overused words: tapestry, delve, testament, vibrant, navigate, resonate\n"
            "- Empty transitions: Furthermore, Moreover, It is worth noting that\n"
            "- Tell emotions directly rather than showing them"
        ),
    },
    "mistral": {
        "path": "models/teachers/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "n_gpu_layers": 30,
        "slop_block": (
            "- Parallel three-part structures and snappy triads\n"
            "- Unearned pivots: 'Something shifted', 'Everything changed'\n"
            "- Overused words: shimmered, nuanced, embark, realm, underscore"
        ),
    },
}

SLOP_PROMPT_TEMPLATE = """You are generating a deliberately AI-style draft of a story segment.

REQUIRED story elements — you MUST include ALL of these exactly:
{outline}

Write the segment using these AI writing patterns:
{slop_block}

STRICT RULES:
- Keep ALL character names, locations, and events IDENTICAL to the outline
- Do NOT add any character, location, or event not listed above
- Do NOT change the order of events
- Length: approximately {length} words

Draft:"""

LLAMA_CLI = "llama.cpp/build/bin/llama-cli"


def load_existing_ids(output_path):
    """Return set of pair IDs already written to output_path."""
    p = Path(output_path)
    if not p.exists():
        return set()
    return {json.loads(line)["id"] for line in p.open()}


def _outline_to_text(outline_entry):
    """Convert an outline dict (or raw string) to a structured text block."""
    if isinstance(outline_entry, str):
        return outline_entry
    outline = outline_entry.get("outline", outline_entry)
    chars = outline.get("characters", [])
    char_str = "\n".join(
        f"  - {c['name']}: {c['role']}" if isinstance(c, dict) else f"  - {c}"
        for c in chars
    ) or "  (none named)"
    events = outline.get("events", [])
    events_str = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(events))
    setting = outline.get("setting", "")
    pov = outline.get("pov_state", "")
    return (
        f"CHARACTERS:\n{char_str}\n"
        f"SETTING: {setting}\n"
        f"EVENTS (in order):\n{events_str}\n"
        f"POV STATE: {pov}"
    )


def build_slop_prompt(excerpt, outline, slop_block):
    """Build the full teacher prompt that produces a sloppy rewrite.

    The outline's events are enumerated explicitly so the teacher hits every beat.
    `excerpt` is used to derive the target length so output length matches the original.
    """
    return SLOP_PROMPT_TEMPLATE.format(
        outline=_outline_to_text(outline),
        slop_block=slop_block,
        length=len(excerpt.split()),
    )


def call_teacher(llama_cli, model_path, prompt, n_gpu_layers=20, max_tokens=1024):
    """Invoke llama-cli and return the generated text, or None on failure."""
    cmd = [
        llama_cli, "--model", model_path,
        "--n-gpu-layers", str(n_gpu_layers),
        "--ctx-size", "4096",
        "--n-predict", str(max_tokens),
        "--temp", "0.85",
        "--repeat-penalty", "1.1",
        "--no-display-prompt",
        "--prompt", prompt,
    ]
    for attempt in range(2):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=480)
            out = r.stdout.strip()
            if len(out.split()) >= 100:
                return out
        except subprocess.TimeoutExpired:
            print(f"  timeout (attempt {attempt + 1})")
    return None


def main(excerpts_path="data/raw_excerpts.jsonl", outlines_path="data/outlines.json",
         output_path="data/raw_pairs.jsonl", teachers=None):
    """Generate sloppy pairs for all excerpts × all teachers not yet in the checkpoint."""
    if not Path(LLAMA_CLI).exists():
        raise SystemExit(
            f"llama-cli not found at {LLAMA_CLI}. "
            "Build llama.cpp with: cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)"
        )

    active_teachers = {
        k: v for k, v in TEACHERS.items()
        if (teachers is None or k in teachers) and Path(v["path"]).exists()
    }
    if not active_teachers:
        raise SystemExit(
            "No teacher GGUF files found. Download to models/teachers/:\n"
            "  wget -P models/teachers/ https://huggingface.co/bartowski/"
            "Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        )

    outlines = json.load(open(outlines_path))
    written = load_existing_ids(output_path)
    total_new = 0

    with open(output_path, "a") as out_f:
        for line in open(excerpts_path):
            item = json.loads(line)
            iid = item["id"]
            if iid not in outlines:
                continue
            for name, cfg in active_teachers.items():
                pair_id = f"{iid}_{name}"
                if pair_id in written:
                    continue
                prompt = build_slop_prompt(item["text"], outlines[iid], cfg["slop_block"])
                sloppy = call_teacher(
                    LLAMA_CLI, cfg["path"], prompt,
                    n_gpu_layers=cfg["n_gpu_layers"],
                    max_tokens=min(len(item["text"].split()) + 200, 1000),
                )
                if sloppy:
                    out_f.write(json.dumps({
                        "id": pair_id, "clean": item["text"],
                        "sloppy": sloppy, "teacher": name,
                    }) + "\n")
                    out_f.flush()
                    written.add(pair_id)
                    total_new += 1
                    if total_new % 100 == 0:
                        print(f"  Generated {total_new} new pairs...")

    print(f"Generation complete. New pairs this run: {total_new}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excerpts", default="data/raw_excerpts.jsonl")
    parser.add_argument("--outlines", default="data/outlines.json")
    parser.add_argument("--output",   default="data/raw_pairs.jsonl")
    parser.add_argument("--teachers", nargs="+")
    args = parser.parse_args()
    main(args.excerpts, args.outlines, args.output, args.teachers)
