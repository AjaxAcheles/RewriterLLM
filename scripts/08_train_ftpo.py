"""
08_train_ftpo — Final Token Preference Optimization to suppress residual slop.

Pipeline position: M7 — Training, phase B2 (step 2 of 2)
Depends on: models/sft_merged/ (M6), reports/slop_profile.json + human-pruned banlist (M7 step 1)
Produces:   models/ftpo_model/, reports/ftpo_eval.json

FTPO surgically lowers the model's probability of generating tokens in the human-pruned
slop banlist without disturbing the core editing behaviour trained in M6. It is the second
and final training phase for the base editor (M8 style conditioning is optional on top).

WHY NOT DPO
Preference optimisation in its standard form (DPO) shifts the entire output distribution
toward preferred outputs and away from rejected ones. For writing-quality tasks this causes
diversity collapse: the model learns to avoid disfavoured words broadly, flattening prose
variety. FTPO instead applies logit-space MSE regularisation — it penalises high logit
values specifically for the listed tokens, leaving the rest of the distribution intact.
The margin deactivation mechanism (do not penalise tokens already below the target logit
level) means FTPO does not over-suppress words that are already infrequent.

HYPERPARAMETERS
  LR: 5e-5  — intentionally much lower than M6's 1e-4. Higher LR overwrites the editing
              transformation trained in M6 rather than just suppressing individual tokens.
  epochs: 1-2  — a single pass over a small number of FTPO pairs is sufficient; more epochs
                 risks degrading M6 behaviour.

GATE (must pass before declaring M7 done — assert via run_eval.py):
  slop_delta must improve vs. reports/sft_baseline.json
  ner_preservation must NOT regress vs. sft_baseline.json
  sem_vs_target must NOT regress vs. sft_baseline.json

Usage:
    python scripts/08_train_ftpo.py [--base models/sft_merged]
                                    [--banlist data/slop_banlist_final.txt]
                                    [--output models/ftpo_model]
    # then evaluate:
    python scripts/run_eval.py models/ftpo_model reports/ftpo_eval.json

Key implementation notes:
  - Load the merged SFT model (models/sft_merged), not the LoRA adapters. FTPO adds its own
    LoRA layer on top of the merged weights.
  - The banlist file (data/slop_banlist_final.txt) is the human-pruned version of
    slop_banlist_draft.txt from 07_profile_slop.py. Never run FTPO on the raw draft — the
    genre-legitimate words will degrade genre-appropriate prose.
  - Tokenise the banlist to get the actual token IDs. Some slop phrases tokenise to multiple
    tokens; FTPO's penalty should target the first (or most diagnostic) token of each phrase.
  - Margin deactivation: if a token's logit is already below a threshold (e.g. -5.0), skip
    the penalty for that token in that forward pass — it is already suppressed.
  - Validate that slop_delta improves AND ner/sem do not regress. If ner or sem regress,
    lower the learning rate and re-run before accepting the checkpoint.
"""

from pathlib import Path


def load_banlist_token_ids(banlist_path, tokenizer):
    """Convert the banlist text file into a set of token IDs.

    Each line in the banlist is a word or short phrase. Tokenise each and take the first
    (or most discriminative) token ID.
    TODO: implement.
    """
    raise NotImplementedError


def main(base_model="models/sft_merged", banlist="data/slop_banlist_final.txt",
         output_dir="models/ftpo_model"):
    """Run FTPO on the SFT-merged model and save the result."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
