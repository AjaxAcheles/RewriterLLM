# AI-Slop Editor Model — Module Index

This directory contains the nine work-package modules decomposed from Master Plan v2. Each file
is self-contained: objective, deep rationale, how-it-works, granular action steps, fleshed-out
code, configuration tables with rationale, required resources, a runnable `test_*.py` KPI file,
a definition of done, and a failure-mode table.

## Read order & dependency map

```
M1 Environment ──┬─> M2 Corpus ──> M3 Outline ──> M4 Generation ──> M5 Curation ──┐
                 │                                                                 │
                 └─> M9 Evaluation Harness (build in parallel) <───────────────────┤
                                      │                                            │
                                      ▼                                            ▼
                              (defines success for) ────────> M6 SFT ──> M7 FTPO ──> M8 Style
```

**Critical path:** M1 → M2 → M3 → M4 → M5 → M6 → M7 (→ M8 optional).
**Parallel track:** M9 must be built during the M2–M5 window — M6 cannot be declared complete
without it.

## Files

| # | File | Topic | Critical path | Optional |
|---|---|---|---|---|
| M1 | `module_1_environment.md` | Environment & infrastructure | yes | no |
| M2 | `module_2_corpus.md` | Source corpus & segmentation | yes | no |
| M3 | `module_3_outline.md` | Outline extraction & validation | yes | no |
| M4 | `module_4_generation.md` | Synthetic sloppy-pair generation | yes | no |
| M5 | `module_5_curation.md` | Pair filtering & dataset curation | yes | no |
| M6 | `module_6_sft.md` | SFT training (Phase B1) | yes | no |
| M7 | `module_7_ftpo.md` | Slop profiling & FTPO (Phase B2) | yes | no |
| M8 | `module_8_style.md` | Style conditioning (Phase B3) | no | yes |
| M9 | `module_9_evaluation.md` | Evaluation harness | yes (early) | no |

## Phase gates (do not start downstream until upstream KPIs are green)

- **M3 → M4:** manual outline-reconstruction gate ≥ 90%.
- **M5 → M6:** zero base-ID leakage; ≥ 1,800 filtered pairs (M4 generates ~2× this so the floor
  holds at the low end of the keep band).
- **M6 → M7:** recorded SFT baseline clears `slop_delta>1.0`, `ner_preservation≥0.90`,
  `sem_vs_target≥0.78`.
- **M7 → M8:** FTPO shows slop down with preservation/diversity held vs. SFT baseline.

## Cross-module contracts

- **Join key:** excerpt `id` (M2) threads through outlines (M3), pairs (M4), splits (M5).
- **Report keys:** the metric names in M9's `eval_metrics.py` are read verbatim by the
  completion tests in M6/M7/M8 — freeze them.
- **Prompt contract:** the system prompt and chat template live once in `scripts/prompt_config.py`
  and are imported by M5 (formatting), M6/M8 (training), and M9 (inference) — rendered through the
  model's own tokenizer so training text and inference input are byte-identical.
