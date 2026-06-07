"""
extract_outlines — extract structured 4-section outlines from clean excerpts via LLM.

Pipeline position: M3 — Data pipeline, stage 2 of 4
Depends on: data/raw_excerpts.jsonl (M2)
Produces:   data/outlines.json

For each excerpt, calls a teacher LLM and asks it to extract a structured outline with
four sections: characters (list of names + role), setting (location + time of day/season),
events (ordered list of plot beats), and pov_state (whose head we're in, emotional state).
The outline is the content-preservation mechanism for M4: the sloppy-pair teacher is
constrained by the outline, so it can change the prose style but not the story content.

A drifting outline (one that omits events or invents characters) breaks the content-
preservation invariant for every pair generated from it. This is why a MANUAL GATE is
required before M4 starts: sample 50 outlines, reconstruct the prose from each outline
alone, and require >= 90% reconstructability. If the score is below 90%, fix the extraction
prompt before continuing.

Processing is done in batches with checkpointing. The script is resumable: on restart it
reads the existing outlines.json (if any), identifies which excerpt IDs are already done,
and skips them. The checkpoint is written after every batch.

Output schema (single JSON object, dict keyed by excerpt ID):
  {
    "<excerpt_id>": {
      "id": str,
      "outline": {
        "characters": [{"name": str, "role": str}, ...],
        "setting": str,
        "events": [str, ...],
        "pov_state": str
      }
    },
    ...
  }

Usage:
    python scripts/extract_outlines.py [--excerpts data/raw_excerpts.jsonl]
                                       [--output data/outlines.json]
                                       [--batch-size 50]

Key implementation notes:
  - Use a structured output / JSON mode call if the teacher supports it; otherwise parse
    freeform and retry on parse failure (the outline schema is simple enough to extract
    with a fallback regex for the common failure modes).
  - The extraction prompt must explicitly instruct the model NOT to add events that are
    implied but not stated — the goal is a lossless structural summary, not an expansion.
  - Checkpoint after each batch by writing the full dict back to outlines.json. A partial
    run that crashes loses at most one batch.
  - The manual reconstruction gate (see module_3_outline.md §5 Step 3) is a human step
    and is NOT automated here. The script just produces the outlines; the human does the
    50-sample spot-check before running 02_generate_pairs.py.
"""

import json
import re
import subprocess
from pathlib import Path

EXTRACTION_PROMPT = """You are a structural analyst. Read this prose excerpt and extract a structured outline.

EXCERPT:
{text}

Extract the outline as a JSON object with EXACTLY these four keys:
{{
  "characters": [{{"name": "...", "role": "..."}}],
  "setting": "location and time description",
  "events": ["event 1 [because ... / leading to ...]", "event 2", ...],
  "pov_state": "narrator/POV name: knows [X], feels [Y as fact, not prose]"
}}

Rules:
- characters: list every named character with their role in this scene
- setting: one sentence, location + time context
- events: ordered list, each with causal link where possible
- pov_state: fact form only — "grief over X" not "sorrow washed over her"
- Do NOT add events that are implied but not stated
- Output ONLY valid JSON, nothing else

JSON:"""


def load_checkpoint(output_path):
    """Return the existing outlines dict (keyed by ID) or {} if none exists."""
    p = Path(output_path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def extract_outline(excerpt_text, teacher_fn):
    """Call teacher_fn with the extraction prompt and return the parsed outline dict.

    teacher_fn: callable(prompt: str) -> str
    Returns: {"characters": [...], "setting": str, "events": [...], "pov_state": str}
    """
    prompt = EXTRACTION_PROMPT.format(text=excerpt_text)
    raw = teacher_fn(prompt)

    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        raise ValueError(f"No JSON found in teacher response: {raw[:200]}")

    try:
        outline = json.loads(json_match.group())
    except json.JSONDecodeError:
        fixed = json_match.group().replace("'", '"')
        outline = json.loads(fixed)

    required = {"characters", "setting", "events", "pov_state"}
    missing = required - set(outline.keys())
    if missing:
        raise ValueError(f"Outline missing keys: {missing}")

    return outline


def main(excerpts_path="data/raw_excerpts.jsonl", output_path="data/outlines.json",
         batch_size=50):
    """Extract outlines for all excerpts not yet in the checkpoint."""
    outlines = load_checkpoint(output_path)
    done_ids = set(outlines.keys())

    excerpts = [json.loads(l) for l in open(excerpts_path)]
    todo = [e for e in excerpts if e["id"] not in done_ids]
    print(f"Already done: {len(done_ids)} | Remaining: {len(todo)}")

    llama_cli = "llama.cpp/build/bin/llama-cli"
    teacher_model = next(iter(sorted(Path("models/teachers").glob("*.gguf"))), None)
    if not teacher_model:
        raise SystemExit("No teacher GGUF found in models/teachers/. Download one first.")

    def teacher_fn(prompt):
        cmd = [llama_cli, "--model", str(teacher_model),
               "--n-gpu-layers", "28", "--ctx-size", "4096",
               "--n-predict", "512", "--temp", "0.1",
               "--no-display-prompt", "--prompt", prompt]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return r.stdout.strip()
        except subprocess.TimeoutExpired:
            return ""

    failed = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        for e in batch:
            try:
                outline = extract_outline(e["text"], teacher_fn)
                outlines[e["id"]] = {"id": e["id"], "outline": outline}
            except Exception as ex:
                print(f"  FAILED {e['id']}: {ex}")
                failed += 1
        Path(output_path).write_text(json.dumps(outlines, indent=2))
        print(f"  Checkpointed {min(i + batch_size, len(todo))}/{len(todo)}")

    print(f"Done. {len(outlines)} outlines written; {failed} failed.")


if __name__ == "__main__":
    main()
