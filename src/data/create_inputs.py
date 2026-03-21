from typing import Literal

from torch import device

Task = Literal["translation"]


def create_prompt(task: Task, language: str, input: str):
    if task == "translation":
        return f"You are translating from English to {language}. Output only the raw translation, no labels, explanations, notes, or alternatives. Please translate the following: {input}"
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
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if debug:
        print(f"\n[DEBUG] Prompt:\n{prompt}\n{'=' * 60}")
    inputs = tokenizer(prompt, return_tensors="pt")
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs
