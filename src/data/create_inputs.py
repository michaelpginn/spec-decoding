from typing import Literal

from torch import device


Task = Literal["translation", "story_gen"]


def create_prompt(task: Task, language: str, input: str):
    if task == "translation":
        return f"Translate the following English text to {language}. Output only the translation, nothing else.\n\n{input}"
    elif task == "story_gen":
        return f"Write a short story in {language} about a(n) {input}. Output only the story, nothing else."
    else:
        raise NotImplementedError(f"Unknown task: {task!r}")


def create_inputs(
    message: str,
    tokenizer,
    device: device | None = None,
    debug=False,
):
    """Tokenize a prompt string into model inputs."""
    messages = [{"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if debug:
        print(f"\n[DEBUG] Prompt:\n{prompt}\n{'=' * 60}")
    inputs = tokenizer(prompt, return_tensors="pt")
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs
