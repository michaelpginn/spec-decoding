import time
from typing import Literal, cast

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config.config import ExperimentConfig
from src.spec_decode import speculative_decode_greedy


def generate_output(
    inputs: dict,
    model,
    tokenizer: PreTrainedTokenizer,
    draft_model: PreTrainedModel | None,
    draft_tokenizer: PreTrainedTokenizer | None,
    config: ExperimentConfig,
) -> tuple[str, dict]:
    """Generates an output, optionally using speculative decoding."""
    same_tokenizer = (draft_tokenizer is None) or (tokenizer.vocab_size == draft_tokenizer.vocab_size)
    is_cuda = inputs["input_ids"].device.type == "cuda"
    prompt_len = inputs["input_ids"].shape[1]

    # Use our custom spec dec implementation
    if config.draft_model_type != 'none' and not config.use_hf_assisted:
        output_ids, metrics = speculative_decode_greedy(
            target_model=model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            input_ids=inputs['input_ids'],
            max_new_tokens=config.max_new_tokens,
            gamma=config.gamma, # type:ignore
            device=inputs["input_ids"].device,
            track_iterations=config.track_iterations
        )
        decoded = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        decoded = cast(str, decoded).strip()
        return decoded, metrics

    # Otherwise, use HF decoding
    with torch.no_grad():
        # Measure prefill time (one forward pass to fill KV cache)
        if is_cuda:
            torch.cuda.synchronize()
        prefill_start = time.time()
        model(inputs["input_ids"], use_cache=True)
        if is_cuda:
            torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start
        
        if config.draft_model_type != 'none':
            generate_kwargs = {
                "assistant_model": draft_model,
                "num_assistant_tokens": config.gamma,
                "num_assistant_tokens_schedule": config.hf_schedule,
            }
            if not same_tokenizer:
                generate_kwargs['tokenizer'] = tokenizer
                generate_kwargs['assistant_tokenizer'] = draft_tokenizer
        else:
            generate_kwargs = {}

        # Generate (this re-does prefill internally)
        if is_cuda:
            torch.cuda.synchronize()
        gen_start = time.time()
        out = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
        )
        if is_cuda:
            torch.cuda.synchronize()
        total_time = time.time() - gen_start

    decode_time = total_time - prefill_time

    # Decode only the new tokens (after the prompt)
    generated_token_count = out.shape[1] - prompt_len
    decoded = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    decoded = cast(str, decoded).strip()
    return decoded, {
        "generated_tokens": generated_token_count,
        "time": decode_time,
        "toks_per_sec": generated_token_count / decode_time if decode_time > 0 else 0,
    }
