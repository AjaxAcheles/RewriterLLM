"""
prompt_config — frozen prompt contract for the AI-slop editor model.

THIS FILE IS A FROZEN CONTRACT. Every module that touches the model imports from here:
  M5  (05_format_dataset.py)  — renders training pairs into the chat template
  M6  (06_train_sft.py)       — provides RESPONSE_TEMPLATE for completion-only loss masking
  M8  (09_train_style.py)     — uses STYLE_SYSTEM_PROMPT and STYLE_USER_TEMPLATE
  M9  (run_eval.py)           — builds inference prompts; must match training exactly

WHY THIS FILE EXISTS
--------------------
The system prompt encodes the preservation rules in natural language. If that text changes
between training (M5/M6) and inference (M9), the model is asked at test time to behave under
a contract it was never trained on. Similarly, the chat template rendered here must be
byte-identical to the template rendered at inference — a single extra newline or a missing
special token is a silent but real distribution shift.

The solution is: define the contract once, import it everywhere, render it through the
*model's own tokenizer* (via `apply_chat_template`) rather than hand-writing ChatML markup.
That way training text and inference input are produced by literally the same code path.

CHANGING THIS FILE
------------------
Do not rename, reorder, or reword anything after M5 has produced training data. If you must
change the prompt, you must also regenerate all training data (M5) and retrain (M6+).
The one safe extension: M8 appends to EDITOR_SYSTEM_PROMPT via STYLE_SYSTEM_PROMPT —
a strict superset that doesn't alter the base editor's behavior.
"""

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

EDITOR_SYSTEM_PROMPT = (
    "You are a prose editor. Your job is to remove AI slop patterns from the excerpt. "
    "Preserve every character, event, location, causal link, and plot beat exactly. "
    "Do not add new content. Do not improve the plot. Only improve the prose."
)

# M8 only — extends the base contract without replacing it. If you change EDITOR_SYSTEM_PROMPT
# you must audit whether STYLE_SYSTEM_PROMPT still reads correctly as a whole sentence.
STYLE_SYSTEM_PROMPT = EDITOR_SYSTEM_PROMPT + (
    " Adapt the prose toward the style shown in the style reference."
)

# ---------------------------------------------------------------------------
# User turn templates
# ---------------------------------------------------------------------------

USER_TEMPLATE = "Edit this excerpt:\n\n{sloppy}"

# The style reference block uses XML-style tags so the model can cleanly distinguish reference
# material from the excerpt to be edited. Used by M8; must match STYLE_SYSTEM_PROMPT's intent.
STYLE_USER_TEMPLATE = (
    "<style_reference>\n{reference}\n</style_reference>\n\n" + USER_TEMPLATE
)

# ---------------------------------------------------------------------------
# Tokenizer / template constants
# ---------------------------------------------------------------------------

# The 4B proto and 7B final both use Qwen3's ChatML template, so either tokenizer produces
# identical structure. The 7B canonical is used everywhere so a single `render()` call works
# across both training runs.
CANONICAL_TOKENIZER = "unsloth/Qwen3-7B-bnb-4bit"

# The string that opens the assistant turn in Qwen3's ChatML format.
# Used by M6's DataCollatorForCompletionOnlyLM as the response boundary — loss is computed
# ONLY on tokens AFTER this marker. If loss covers the full sequence (system + user prompt)
# the model learns to copy its input verbatim. That is the single most common silent failure
# in this kind of SFT task.
#
# TRL gotcha: if DataCollatorForCompletionOnlyLM raises "Could not find response key in
# token IDs", it means the marker didn't tokenize identically in-context (often a whitespace
# difference). Fix: pass `tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)`
# as the response_template argument instead of the string.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def build_messages(sloppy, clean=None, style_reference=None):
    """Build a chat message list for the editor task.

    Args:
        sloppy: The AI-generated (sloppy) prose to be edited.
        clean:  The human-written target. If provided, appended as the assistant turn,
                producing a complete training example. Omit for inference.
        style_reference: 3-5 paragraph style excerpt (M8 only). Switches to STYLE_SYSTEM_PROMPT
                         and STYLE_USER_TEMPLATE when provided.

    Returns:
        List of {"role": ..., "content": ...} dicts ready for `apply_chat_template`.
    """
    system = STYLE_SYSTEM_PROMPT if style_reference else EDITOR_SYSTEM_PROMPT
    user = (
        STYLE_USER_TEMPLATE.format(reference=style_reference, sloppy=sloppy)
        if style_reference
        else USER_TEMPLATE.format(sloppy=sloppy)
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if clean is not None:
        msgs.append({"role": "assistant", "content": clean})
    return msgs


def render(tokenizer, msgs, add_generation_prompt=False):
    """Apply the model's own chat template to a message list.

    Always passes `enable_thinking=False` because Qwen3, when given reasoning capability,
    injects <think>...</think> scaffolding that the editing task neither produces nor expects.
    Training without that flag on a base that was fine-tuned to use it causes a mismatch
    between the model's natural outputs and the training targets.

    Args:
        tokenizer: A loaded Qwen3 tokenizer (or any tokenizer whose template is ChatML-compatible).
        msgs: Output of `build_messages(...)`.
        add_generation_prompt: True at inference (appends the opening assistant marker so the
                               model continues from there); False for training (the full
                               assistant turn is already in `msgs`).

    Returns:
        A single string containing the fully-rendered chat text.
    """
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
