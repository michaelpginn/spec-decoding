import itertools

import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def create_prompt(language: str, adj_n: bool = False, num_prompts: int = 10):
    """Generate a dict of story prompts. Keys are 1-indexed integers."""
    prompts: dict = {}
    if not adj_n:
        nouns = list(wn.all_synsets('n'))
        sample_nouns = [synset.name().split('.')[0] for synset in nouns[:num_prompts]]
        for i in range(num_prompts):
            prompts[i + 1] = f"Write a short story in {language} about a(n) {sample_nouns[i]}. Output only the story, nothing else."
    else:
        adjs = [s.name().split('.')[0] for s in itertools.islice(wn.all_synsets('a'), num_prompts)]
        nouns = [s.name().split('.')[0] for s in itertools.islice(wn.all_synsets('n'), num_prompts)]
        wombos = [f"{a} {n}" for a, n in zip(adjs, nouns)]
        for i in range(num_prompts):
            prompts[i + 1] = f"Write a short story in {language} about a(n) {wombos[i]}. Output only the story, nothing else."
    return prompts


def create_inputs_story(message: str, tokenizer, device=None, debug: bool = False):
    """Tokenize a story prompt with Qwen3 non-thinking mode.

    Uses enable_thinking=False so Qwen3 skips the <think>...</think> preamble
    and goes straight to the story. This keeps generated token counts clean and
    prevents reasoning traces from inflating speculative decoding metrics.
    Falls back silently on older models (Qwen2.5, etc.) that don't support the kwarg.
    """
    messages = [{"role": "user", "content": message}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # Older tokenizers (pre-Qwen3) don't support enable_thinking.
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if debug:
        print(f"\n[DEBUG] Story prompt:\n{prompt}\n{'=' * 60}")
    inputs = tokenizer(prompt, return_tensors="pt")
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs
