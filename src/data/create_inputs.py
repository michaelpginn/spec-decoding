import logging
from typing import Literal

from torch import device

logger = logging.getLogger(__name__)
_logged_template_mode = False


Task = Literal["translation"]


def create_prompt(task: Task, language: str, input: str):
    if task == "translation":
        # modified to avoid triggering chain-of-thought reasoning
        # in Qwen3 models even when enable_thinking=True is set elsewhere.
        return f"Translate the following English text to {language}. Output only the translation, nothing else.\n\n{input}"
    else:
        raise NotImplementedError()


def create_inputs(
    message: str,
    tokenizer,
    device: device | None = None,
    debug=False,
):
    """Create chat messages for translation task."""
    messages = [{"role": "user", "content": message}]
    # enable_thinking=False: disables Qwen3's chain-of-thought <think>...</think> mode.
    # Falls back gracefully on non-Qwen3 tokenizers that don't support this kwarg.
    global _logged_template_mode
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if not _logged_template_mode:
            logger.info("Chat template applied successfully with enable_thinking=False (Qwen3 non-thinking mode).")
            _logged_template_mode = True
    except TypeError:
        # Tokenizer does not support enable_thinking (e.g. Qwen2.5, older models).
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not _logged_template_mode:
            logger.info("Tokenizer does not support enable_thinking kwarg; falling back to standard apply_chat_template.")
            _logged_template_mode = True
    if debug:
        print(f"\n[DEBUG] Prompt:\n{prompt}\n{'=' * 60}")
    inputs = tokenizer(prompt, return_tensors="pt")
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs
