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
from pathlib import Path


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
    TODO: implement prompt construction, LLM call, and JSON parsing with retry.
    """
    raise NotImplementedError


def main(excerpts_path="data/raw_excerpts.jsonl", output_path="data/outlines.json",
         batch_size=50):
    """Extract outlines for all excerpts not yet in the checkpoint."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
