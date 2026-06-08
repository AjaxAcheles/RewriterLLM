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
import torch


def load_banlist_token_ids(banlist_path, tokenizer):
    """Convert the banlist text file into a set of token IDs.

    Each line in the banlist is a word or short phrase. Tokenise each and take the first
    (or most discriminative) token ID.
    """
    from pathlib import Path as P
    lines = P(banlist_path).read_text().strip().splitlines()
    token_ids = set()
    for word in lines:
        word = word.strip()
        if not word:
            continue
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if ids:
            token_ids.add(ids[0])
    return token_ids


def main(base_model="models/sft_merged", banlist="data/slop_banlist_final.txt",
         output_dir="models/ftpo_model"):
    """Run FTPO on the SFT-merged model and save the result."""
    import torch
    import json
    from pathlib import Path as P

    # Use auto-antislop if available
    antislop_dir = P("auto-antislop")
    if antislop_dir.exists():
        import subprocess, sys

        def _entry(name):
            p = antislop_dir / name
            if not p.exists():
                avail = sorted(q.name for q in antislop_dir.glob("*.py"))
                raise SystemExit(f"Expected '{p}'. Available scripts: {avail}")
            return str(p)

        pairs_out = "antislop_data/ftpo_pairs.jsonl"

        subprocess.run([sys.executable, _entry("generate_ftpo_data.py"),
                        "--model", base_model, "--banlist", banlist,
                        "--output", pairs_out, "--n-samples", "2000",
                        "--ban-strength", "0.7", "--temperature", "0.9"],
                       check=True)

        subprocess.run([sys.executable, _entry("train_ftpo.py"),
                        "--model", base_model, "--data", pairs_out,
                        "--output", output_dir, "--epochs", "1",
                        "--learning-rate", "5e-5", "--margin", "2.0",
                        "--lambda-target", "0.1", "--lambda-nontarget", "1.0"],
                       check=True)

        print(f"FTPO complete → {output_dir}")
        print("Run: python scripts/run_eval.py models/ftpo_model reports/ftpo_eval.json")
        return

    # Fallback: logit-suppression FTPO using pure PyTorch (no auto-antislop)
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
    try:
        from scripts.prompt_config import RESPONSE_TEMPLATE
    except ImportError:
        from prompt_config import RESPONSE_TEMPLATE

    print("auto-antislop not found. Running logit-suppression FTPO fallback.")

    if not P(banlist).exists():
        raise SystemExit(f"Banlist not found: {banlist}. Run 07_profile_slop.py and review first.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model, max_seq_length=8192, dtype=None, load_in_4bit=True
    )
    ban_ids = load_banlist_token_ids(banlist, tokenizer)
    print(f"Loaded {len(ban_ids)} banned token IDs from {banlist}")

    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    ds = load_dataset("json", data_files={"train": "data/train.jsonl"})
    response_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_ids, tokenizer=tokenizer)

    class FTPOTrainer(SFTTrainer):
        """SFTTrainer subclass that adds per-token logit-space MSE penalties for banned tokens.

        For each forward pass, after computing the standard completion-only CE loss, this
        trainer adds a penalty proportional to how far each banned token's logit exceeds
        `margin`. Tokens already below `margin` are unaffected (margin deactivation).
        """
        def __init__(self, ban_ids, margin=-5.0, penalty_weight=0.1, **kwargs):
            super().__init__(**kwargs)
            self._ban_ids = sorted(ban_ids)
            self.margin = margin
            self.penalty_weight = penalty_weight
            self._ban_tensor = None

        def _ban_tensor_on(self, device):
            if self._ban_tensor is None or self._ban_tensor.device != device:
                self._ban_tensor = torch.tensor(
                    self._ban_ids, dtype=torch.long, device=device
                )
            return self._ban_tensor

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss  # completion-only CE loss from masked labels

            # Logit-space MSE penalty: push banned token logits below `margin`
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            ban_t = self._ban_tensor_on(logits.device)
            ban_logits = logits[:, :, ban_t]  # (batch, seq_len, n_banned)
            excess = torch.clamp(ban_logits - self.margin, min=0.0)
            penalty = self.penalty_weight * (excess ** 2).mean()
            loss = loss + penalty

            return (loss, outputs) if return_outputs else loss

    trainer = FTPOTrainer(
        ban_ids=ban_ids, margin=-5.0, penalty_weight=0.1,
        model=model, tokenizer=tokenizer,
        train_dataset=ds["train"], data_collator=collator,
        dataset_text_field="text", max_seq_length=8192,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            learning_rate=5e-5,
            warmup_steps=20,
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            output_dir="checkpoints/ftpo",
            report_to="none",
        ),
    )
    trainer.train()

    P(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"FTPO (simplified) complete → {output_dir}")


if __name__ == "__main__":
    main()
