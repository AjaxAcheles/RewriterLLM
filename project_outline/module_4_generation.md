# Module 4 — Synthetic Sloppy-Pair Generation

| Field | Value |
|---|---|
| **Phase** | Data pipeline (stage 3 of 4) |
| **Depends on** | M1, M3 |
| **Blocks** | M5 |
| **Critical path** | Yes |
| **Owner effort** | 1 day setup + multi-day unattended run |
| **Runtime budget** | ~2–3 tok/s hybrid offload → minutes per excerpt × teachers × thousands = days |

---

## 1. Primary Objective

Generate the *sloppy* (input) side of every training pair by feeding each outline to two or more
open-source teacher LLMs instructed to write AI-slop-laden prose covering exactly the outline's
events. Produce a resumable, diverse corpus of raw (sloppy, clean) pairs in
`data/raw_pairs.jsonl`.

---

## 2. Core Concepts — Deep Dive

### 2.1 Inverse injection, and why direction doesn't matter

The intuitive approach — collect AI text, pay humans to clean it — fails on economics and
consistency. The project exploits a directional asymmetry: **adding slop to clean text is far
easier and more reliable than removing it.** We already have the clean side (M2); we manufacture
the dirty side. The pair is valid regardless of how it was built: at training and at deployment
the model runs sloppy→clean. That we *constructed* it clean→sloppy is invisible to the trained
model.

### 2.2 Slop is model-specific — the empirical basis for multiple teachers

Slop fingerprints cluster within a model family and differ between families. The Antislop research
found individual patterns appearing on the order of 1,000× more often than in human text, and
specific tokens (e.g. certain character names) tens of thousands of times more often — with the
overused set differing by model. A single-teacher dataset teaches the editor to remove *one*
family's signature and may leave others untouched. Since deployment input could come from any
model, the training data must contain slop from multiple families. Rotating 2–3 open-source
families embeds that diversity and teaches the general concept of slop rather than one signature.

### 2.3 Why teachers must be open-source

Distilling a closed model (GPT-4, Claude, Gemini) into a deployed student likely violates the
provider's terms, which generally forbid using outputs to train competing models. Open-source
instruct models (Llama, Mistral, Qwen) carry licenses permitting this. Hard constraint.

### 2.4 Hybrid CPU/GPU offload — where 32 GB RAM finally earns its keep

This is the *only* place offload helps. Batch teacher inference is latency-tolerant: a draft
taking 5 s or 5 min doesn't matter for an offline job. That tolerance lets llama.cpp split a
larger teacher across GPU and system RAM via `--n-gpu-layers`, running 13–30B models that would
never fit in 12 GB of VRAM. (Contrast Module 6: training is *not* latency-tolerant and the offload
path is blocked by the ZeRO-3 × bitsandbytes incompatibility.)

### 2.5 The teacher prompt does three jobs at once

1. **Force content coverage** — every outline event must appear (prevents drift).
2. **Force slop injection** — name the patterns to use (guarantees measurable slop to remove).
3. **Forbid content addition** — explicitly bar new characters/locations/events (counters the
   teacher's urge to embellish).

All three are essential; dropping any one degrades pair quality in a way Module 5 then has to
filter out, wasting generation.

### 2.6 Resumability as a first-class requirement

At a few tok/s across multiple teachers and thousands of excerpts, a full run spans days. A crash
at hour 40 must not restart from zero. The loop checkpoints completed base-excerpt IDs to disk and
skips them on restart; output is appended and flushed per pair. Because an excerpt is only
checkpointed once *all* its teachers finish, a crash partway through an excerpt would otherwise
re-emit the teachers it already wrote — so on startup the loop also reads the pair IDs already on
disk and resumes at pair granularity, never duplicating a pair. Retrofitting this after a multi-day
crash is the worst possible time — so it is engineered in from the start.

---

## 3. Inputs & Outputs Contract

**Inputs:** `data/raw_excerpts.jsonl` (M2), `data/outlines.json` (M3),
`models/teachers/*.gguf` (≥ 2 families).

**Outputs:**
- `data/raw_pairs.jsonl` — one object per (excerpt × teacher):
  ```json
  {"id": "a1b2c3d4e5f6_llama3", "clean": "<M2 text>", "sloppy": "<teacher output>", "teacher": "llama3"}
  ```
- `data/generated_ids.txt` — checkpoint of completed *base* excerpt IDs.

**Invariants:** `sloppy` ≥ 100 words; ≥ 2 distinct `teacher` values; every base ID references a
real M2 excerpt; re-running skips already-written pairs. **Sizing:** generate from enough base
excerpts that the raw count is roughly **twice** Module 5's filtered target — so at the low end of
the acceptable keep band (~40%) at least ~1,800 pairs still survive curation. Concretely: ≥ 2,250
base excerpts × ≥ 2 teachers → ≥ 4,500 raw pairs.

---

## 4. Common Challenges & Solutions

**Challenge 1 — Generation is unbearably slow.**
*Why:* too few GPU layers, or teacher too large. *Detect:* `llama-cli` log shows few offloaded
layers; tok/s < 1. *Solve:* raise `--n-gpu-layers` to the OOM boundary; use a smaller teacher for
part of the run; lower `--n-predict`.

**Challenge 2 — OOM on model load.**
*Why:* `--n-gpu-layers` too high for 12 GB. *Detect:* CUDA OOM at startup. *Solve:* reduce layers
by 4 and retry until it loads; record the working value per teacher.

**Challenge 3 — Sloppy side omits outline events.**
*Why:* prompt too weak, or the M3 outline was terse. *Detect:* spot-audit shows missing
characters/events; M5 NER filter later drops many pairs. *Solve:* strengthen "MUST include ALL";
verify M3 outline quality; lower temperature slightly so the model follows instructions.

**Challenge 4 — Sloppy side invents new content.**
*Why:* the no-addition rule was ignored at high temperature. *Detect:* new named entities appear.
*Solve:* emphasize STRICT RULES; lower `--temp` to ~0.8; shorten max tokens to reduce rambling.

**Challenge 5 — Output is barely sloppy (near-identical to clean).**
*Why:* slop block too soft; temperature too low. *Detect:* M5 upper similarity bound drops these.
*Solve:* name more patterns; raise temperature a touch; per-teacher slop emphasis.

**Challenge 6 — Run restarts from zero after a crash.**
*Why:* checkpoint not flushed. *Detect:* re-run regenerates done excerpts. *Solve:* confirm
`ck_f.flush()` after each excerpt and append-mode opens.

**Challenge 7 — Disk fills mid-run.**
*Why:* output larger than estimated. *Detect:* free space shrinking; `No space left`. *Solve:*
monitor disk; rotate/compress `raw_pairs.jsonl`; provision more space.

---

## 5. Step-by-Step Implementation Guide

**Step 1 — Download teachers (≥ 2 families).**
```bash
wget -P models/teachers/ https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
wget -P models/teachers/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

**Step 2 — Tune `--n-gpu-layers` per teacher.** Start at 32, decrease by 4 until the model loads
without OOM; record each working value into `TEACHERS` in the script.
```bash
./llama.cpp/build/bin/llama-cli -m models/teachers/<file>.gguf --n-gpu-layers 32 -p "test" -n 16
# if OOM, retry 28, 24, ... ; confirm the startup log shows layers offloaded to GPU
```

**Step 3 — Smoke test on 10 excerpts.** Temporarily cap the loop to 10 excerpts; run; hand-inspect
that outputs are sloppy and cover the outline events.

**Step 4 — Launch the full unattended run.**
```bash
nohup python scripts/02_generate_pairs.py > reports/generation.log 2>&1 &
tail -f reports/generation.log         # watch progress; safe to detach
```

**Step 5 — Monitor.** Periodically check disk and the log for repeated timeouts/short-output
rejects (signals a misbehaving teacher or prompt).

**Step 6 — Spot-audit ~30 finished pairs**, then verify.
```bash
pytest tests/test_generation.py -v
```

---

## 6. Reference Implementation — `scripts/02_generate_pairs.py`

```python
# scripts/02_generate_pairs.py
"""Generate sloppy versions of clean excerpts using multiple open-source teachers.
Resumable via data/generated_ids.txt.
Inputs: data/raw_excerpts.jsonl, data/outlines.json  →  Output: data/raw_pairs.jsonl"""
import json, subprocess
from pathlib import Path

PROMPT = """You are generating a deliberately AI-style draft of a story segment.

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

TEACHERS = {
    "llama3": {"path": "models/teachers/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
               "n_gpu_layers": 28,
               "slop_block": ("- Overused words: tapestry, delve, testament, vibrant, navigate, resonate\n"
                              "- Empty transitions: Furthermore, Moreover, It is worth noting that\n"
                              "- Tell emotions directly rather than showing them")},
    "mistral": {"path": "models/teachers/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                "n_gpu_layers": 30,
                "slop_block": ("- Parallel three-part structures and snappy triads\n"
                               "- Unearned pivots: 'Something shifted', 'Everything changed'\n"
                               "- Overused words: shimmered, nuanced, embark, realm, underscore")},
}


def call_teacher(prompt, model_path, n_gpu_layers, max_tokens=700, retries=2):
    cmd = ["./llama.cpp/build/bin/llama-cli", "--model", model_path,
           "--n-gpu-layers", str(n_gpu_layers), "--ctx-size", "4096",
           "--n-predict", str(max_tokens), "--temp", "0.85",
           "--repeat-penalty", "1.1", "--no-display-prompt", "--prompt", prompt]
    for attempt in range(retries):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=480)
            out = r.stdout.strip()
            if len(out.split()) >= 100:
                return out
        except subprocess.TimeoutExpired:
            print(f"  timeout (attempt {attempt + 1})")
    return None


def load_checkpoint(path):
    return set(Path(path).read_text().splitlines()) if Path(path).exists() else set()


def load_written_pair_ids(path):
    """Pairs already on disk, so a crash partway through an excerpt's teachers never duplicates."""
    if not Path(path).exists():
        return set()
    return {json.loads(l)["id"] for l in open(path) if l.strip()}


def main(excerpts="data/raw_excerpts.jsonl", outlines_file="data/outlines.json",
         output="data/raw_pairs.jsonl", checkpoint="data/generated_ids.txt"):
    done = load_checkpoint(checkpoint)              # fully-completed excerpts
    written = load_written_pair_ids(output)         # individual pairs already emitted
    outlines = json.load(open(outlines_file))
    with open(output, "a") as out_f, open(checkpoint, "a") as ck_f:
        for line in open(excerpts):
            item = json.loads(line); iid = item["id"]
            if iid in done or iid not in outlines:
                continue
            length = len(item["text"].split())
            for name, cfg in TEACHERS.items():
                pair_id = f"{iid}_{name}"
                if pair_id in written:              # resume at pair granularity, not excerpt
                    continue
                prompt = PROMPT.format(outline=outlines[iid],
                                       slop_block=cfg["slop_block"], length=length)
                sloppy = call_teacher(prompt, cfg["path"], cfg["n_gpu_layers"],
                                      max_tokens=min(length + 200, 1000))
                if sloppy:
                    out_f.write(json.dumps({"id": pair_id, "clean": item["text"],
                                            "sloppy": sloppy, "teacher": name}) + "\n")
                    out_f.flush(); written.add(pair_id)
            ck_f.write(iid + "\n"); ck_f.flush()


if __name__ == "__main__":
    main()
```

---

## 7. Configuration & Parameters

| Parameter | Value | Rationale | If you change it |
|---|---|---|---|
| teachers | ≥ 2 families | Diverse fingerprints → generalization | 1 → editor blind to other models' slop |
| quant | Q4_K_M | Best size/quality for hybrid inference | Smaller → quality drop; larger → slower |
| `--n-gpu-layers` | tuned per model | Max GPU layers w/o OOM | Higher → faster but OOM risk |
| `--temp` | 0.85 | Varied slop, still coherent | Higher → more invention; lower → less slop |
| min words | 100 | Reject truncated/failed gens | — |
| timeout | 480 s | Headroom for slow offload | Too low → false rejects |

---

## 8. Alternatives Considered & Rejected

| Alternative | Why considered | Why rejected |
|---|---|---|
| Closed-model teachers | Higher-quality slop | TOS violation for training a deployed student |
| Single teacher | Simpler, faster | Editor learns one fingerprint; poor generalization |
| GPU-only teachers (≤7B in VRAM) | Faster | Forgoes larger, more capable teachers RAM offload enables |
| vLLM/TGI server | High throughput | Server process conflicts with one-machine offline simplicity; offload is the point |
| Hand-written slop rules (templating) | Deterministic | Produces unnatural, low-diversity slop the editor overfits to |

---

## 9. KPIs / Test File

**Test file:** `tests/test_generation.py`

```python
# tests/test_generation.py
import json
from pathlib import Path
import pytest

RAW, EXCERPTS, CKPT = "data/raw_pairs.jsonl", "data/raw_excerpts.jsonl", "data/generated_ids.txt"

@pytest.fixture(scope="module")
def pairs():
    assert Path(RAW).exists(), "Run scripts/02_generate_pairs.py first"
    return [json.loads(l) for l in open(RAW)]

def test_min_count(pairs):
    assert len(pairs) >= 4500, f"Only {len(pairs)}; want >= 4500 so >= 1,800 survive M5 curation"

def test_fields(pairs):
    for p in pairs: assert set(p) >= {"id", "clean", "sloppy", "teacher"}

def test_sloppy_substantial(pairs):
    for p in pairs: assert len(p["sloppy"].split()) >= 100

def test_teacher_diversity(pairs):
    assert len({p["teacher"] for p in pairs}) >= 2

def test_real_base_ids(pairs):
    valid = {json.loads(l)["id"] for l in open(EXCERPTS)}
    for p in pairs:
        assert p["id"].rsplit("_", 1)[0] in valid

def test_checkpoint_nonempty_if_present():
    if Path(CKPT).exists():
        assert Path(CKPT).read_text().strip()
```

---

## 10. Definition of Done

- ≥ 4,500 raw pairs (≈ 2,250 base excerpts × ≥ 2 teacher families); every sloppy side ≥ 100 words.
- Resumability verified (re-run skips done IDs).
- ~30-pair manual audit confirms slop present + events covered.
- `pytest tests/test_generation.py` passes.
