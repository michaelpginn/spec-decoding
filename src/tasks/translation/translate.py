from src.decoding.spec_decode import speculative_decode_greedy
import time
import torch

def create_translation_messages(source: str, target_lang: str) -> list:
    """Create chat messages for translation task."""
    return [
        {
            "role": "system",
            "content": f"You are a translator. Translate English to {target_lang}. Output ONLY the translation, nothing else. No explanations, no notes, no alternatives."
        },
        {
            "role": "user",
            "content": source
        }
    ]

def translate_target(model, tokenizer, source: str, target_lang: str, max_new_tokens: int = 512, device=None, debug: bool = False):
    """Translate using the model's chat template.
    
    Returns:
        tuple of (translation, generated_token_count, decode_time)
            decode_time: Wall-clock generation time excluding prefill (seconds)
    """
    messages = create_translation_messages(source, target_lang)
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    if debug:
        print(f"\n[DEBUG] Prompt:\n{prompt}\n{'='*60}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    is_cuda = inputs["input_ids"].device.type == "cuda"

    # Measure prefill time (one forward pass to fill KV cache)
    with torch.no_grad():
        if is_cuda:
            torch.cuda.synchronize()
        prefill_start = time.time()
        model(inputs["input_ids"], use_cache=True)
        if is_cuda:
            torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start
    
    # Generate (this re-does prefill internally)
    with torch.no_grad():
        if is_cuda:
            torch.cuda.synchronize()
        gen_start = time.time()
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        if is_cuda:
            torch.cuda.synchronize()
        total_time = time.time() - gen_start
    
    decode_time = total_time - prefill_time
    
    # Decode only the new tokens (after the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    generated_token_count = out.shape[1] - prompt_len
    decoded = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return decoded.strip(), generated_token_count, decode_time


def speculative_decode_translate(
    target_model,
    draft_model,
    tokenizer,
    source: str,
    target_lang: str,
    max_new_tokens: int = 256,
    gamma: int = 5,
    device=None,
    debug: bool = False,
    track_iterations: bool = True,
):
    """
    Wrapper for speculative_decode_greedy for translation tasks.
    Same tokenizer, greedy decoding only.
    
    Args:
        target_model: The large target model
        draft_model: The smaller draft model (MUST share same tokenizer)
        tokenizer: Shared tokenizer
        source: Source text to translate
        target_lang: Target language name (e.g., "Nepali")
        max_new_tokens: Maximum new tokens to generate
        gamma: Number of draft tokens per iteration
        device: Device to run on
        debug: Print debug info
    
    Returns:
        translation: Translated text
        metrics: Dict with acceptance_rate, time, draft_tokens, matched_tokens, etc.
    
    Raises:
        NotImplementedError: If tokenizers are different or sampling is requested
    """
    if device is None:
        device = next(target_model.parameters()).device
    
    messages = create_translation_messages(source, target_lang)
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    if debug:
        print(f"\n[DEBUG] Prompt:\n{prompt}\n{'='*60}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    
    # Run speculative decoding
    output_ids, metrics = speculative_decode_greedy(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        device=device,
        track_iterations=track_iterations
    )
    
    # Decode translation (only new tokens)
    translation = tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True
    ).strip()
    
    return translation, metrics


def assisted_decode_hf(
    target_model,
    target_tokenizer,
    draft_model,
    draft_tokenizer,
    source: str,
    target_lang: str,
    max_new_tokens: int = 512,
    device=None,
    return_metrics: bool = True,
    num_assistant_tokens: int = 5,
    num_assistant_tokens_schedule: str = "heuristic",
):
    """
    Use HuggingFace's optimized assisted generation (speculative decoding).
    
    Args:
        num_assistant_tokens: Number of tokens draft model generates before verification.
            Default is 5. Higher values = more speculative, may be faster if draft is good.
        num_assistant_tokens_schedule: "heuristic" (dynamic adjustment based on acceptance) 
            or "constant" (fixed number). Default is "heuristic".
    """
    if device is None:
        device = next(target_model.parameters()).device

    messages = create_translation_messages(source, target_lang)

    prompt = target_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = target_tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]
    is_cuda = input_ids.device.type == "cuda"

    # Measure prefill time (one forward pass to fill KV cache)
    with torch.no_grad():
        if is_cuda:
            torch.cuda.synchronize()
        prefill_start = time.time()
        target_model(input_ids, use_cache=True)
        if is_cuda:
            torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start

    if is_cuda:
        torch.cuda.synchronize()
    start_time = time.time()
    
    # Check if tokenizers are the same object (only skip tokenizer params in this case)
    same_tokenizer = target_tokenizer is draft_tokenizer
    
    generate_kwargs = {
        "attention_mask": attention_mask,
        "assistant_model": draft_model,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": target_tokenizer.eos_token_id,
        "num_assistant_tokens": num_assistant_tokens,
        "num_assistant_tokens_schedule": num_assistant_tokens_schedule,
    }
    
    # handle both same and different tokenizer cases
    if same_tokenizer:
        with torch.no_grad():
            outputs = target_model.generate(input_ids, **generate_kwargs)
    else:
        # Different tokenizer objects - try without first, then with if needed
        try:
            with torch.no_grad():
                outputs = target_model.generate(input_ids, **generate_kwargs)
        except ValueError as e:
            if "different tokenizers" in str(e).lower():
                # Need to pass tokenizers for universal assisted decoding
                generate_kwargs["tokenizer"] = target_tokenizer
                generate_kwargs["assistant_tokenizer"] = draft_tokenizer
                with torch.no_grad():
                    outputs = target_model.generate(input_ids, **generate_kwargs)
            elif "not required" in str(e).lower():
                # Tokenizers are same, already handled above but just in case
                with torch.no_grad():
                    outputs = target_model.generate(input_ids, **generate_kwargs)
            else:
                raise
    
    if is_cuda:
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    decode_time = total_time - prefill_time
    generated_tokens = outputs.shape[1] - prompt_len

    translation = target_tokenizer.decode(
        outputs[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    if return_metrics:
        metrics = {
            "time": decode_time,
            "generated_tokens": generated_tokens,
            "decode_tps": generated_tokens / decode_time if decode_time > 0 else 0,
        }
        return translation, metrics
    else:
        return translation