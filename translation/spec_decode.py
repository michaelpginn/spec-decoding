"""
speculative decoding for translation

Contains:
- speculative_decode_greedy: Custom implementation (same tokenizer, greedy decoding)
- assisted_decode: HuggingFace's assisted generation wrapper
"""
import torch
import time

# CUSTOM IMPLEMENTATION
def speculative_decode_greedy(
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    gamma: int = 5,
    eos_token_id: int = None,
    device=None,
):
    """
    Custom speculative decoding implementation.
    Simplest case: same tokenizer, greedy decoding.
    
    Args:
        target_model: The large target model
        draft_model: The smaller draft model
        tokenizer: Shared tokenizer (must be same for both models)
        input_ids: Input token IDs [1, seq_len]
        max_new_tokens: Maximum new tokens to generate
        gamma: Number of draft tokens to generate per iteration
        eos_token_id: End of sequence token ID
        device: Device to run on
    
    Returns:
        output_ids: Generated token IDs
        metrics: Dict with acceptance_rate, total_time, etc.
    """
    if device is None:
        device = next(target_model.parameters()).device
    
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    # Ensure input is on device
    input_ids = input_ids.to(device)
    current_ids = input_ids.clone()
    
    # Metrics tracking
    total_draft_tokens = 0
    total_matched_tokens = 0  # Draft tokens that matched target (for acceptance rate)
    start_time = time.time()
    
    generated_count = 0
    
    with torch.no_grad():
        while generated_count < max_new_tokens:
            # Step 1: Draft model generates gamma tokens greedily 
            draft_ids = current_ids.clone()
            draft_tokens = []
            
            for _ in range(gamma):
                draft_outputs = draft_model(draft_ids)
                draft_logits = draft_outputs.logits[:, -1, :]  # [1, vocab_size]
                next_token = torch.argmax(draft_logits, dim=-1, keepdim=True)  # [1, 1]
                draft_tokens.append(next_token.item())
                draft_ids = torch.cat([draft_ids, next_token], dim=-1)
                
                # Stop drafting if EOS
                if next_token.item() == eos_token_id:
                    break
            
            num_drafted = len(draft_tokens)
            if num_drafted == 0:
                break
                
            total_draft_tokens += num_drafted
            
            #  Step 2: Target model verifies all draft tokens in one forward pass 
            # Feed the sequence with all draft tokens to target model
            target_outputs = target_model(draft_ids)
            target_logits = target_outputs.logits  # [1, seq_len, vocab_size]
            
            # Get target model's predictions for each position where we have a draft
            # Position i in target_logits predicts token at position i+1
            # So to verify draft_tokens[j], we look at target_logits at position (original_len + j - 1)
            original_len = current_ids.shape[1]
            
            #  Step 3: Accept tokens until first mismatch 
            num_accepted = 0
            num_matched = 0  # Count only exact matches for acceptance rate
            for j, draft_token in enumerate(draft_tokens):
                # Target's prediction for this position
                target_pred_logits = target_logits[:, original_len + j - 1, :]
                target_token = torch.argmax(target_pred_logits, dim=-1).item()
                
                if target_token == draft_token:
                    num_accepted += 1
                    num_matched += 1  # Draft was correct
                else:
                    # Mismatch: use target's token instead and stop
                    num_accepted += 1  # We still add a token (target's correction)
                    draft_tokens[j] = target_token  # Replace with target's choice
                    # num_matched stays the same (this was NOT a match)
                    break
            
            total_matched_tokens += num_matched
            
            #  Step 4: Update sequence with accepted tokens 
            accepted_tokens = draft_tokens[:num_accepted]
            if accepted_tokens:
                accepted_tensor = torch.tensor([accepted_tokens], device=device)
                current_ids = torch.cat([current_ids, accepted_tensor], dim=-1)
                generated_count += num_accepted
            
            # Check for EOS
            if accepted_tokens and accepted_tokens[-1] == eos_token_id:
                break
    
    total_time = time.time() - start_time
    
    # Calculate acceptance rate (matched draft tokens / total draft tokens)
    acceptance_rate = total_matched_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    
    metrics = {
        "total_time": total_time,
        "generated_tokens": generated_count,
        "total_draft_tokens": total_draft_tokens,
        "total_matched_tokens": total_matched_tokens,
        "acceptance_rate": acceptance_rate,
    }
    
    return current_ids, metrics


def speculative_decode_translate(
    target_model,
    draft_model,
    tokenizer,
    source: str,
    target_lang: str,
    max_length: int = 256,
    gamma: int = 5,
    device=None,
    debug: bool = False,
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
        max_length: Maximum new tokens
        gamma: Number of draft tokens per iteration
        device: Device to run on
        debug: Print debug info
    
    Returns:
        translation: Translated text
        metrics: Dict with acceptance_rate, total_time, etc.
    
    Raises:
        NotImplementedError: If tokenizers are different or sampling is requested
    """
    if device is None:
        device = next(target_model.parameters()).device
    
    # Build prompt
    messages = [
        {
            "role": "system",
            "content": f"You are a professional translator. Translate the following text from English to {target_lang}. Maintain the original style and tone. Only output the translation."
        },
        {
            "role": "user",
            "content": f"Translate this:\n\n{source}"
        }
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    if debug:
        print(f"\n[DEBUG] Prompt:\n{prompt}\n{'='*60}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    
    # Run speculative decoding
    output_ids, metrics = speculative_decode_greedy(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=max_length,
        gamma=gamma,
        device=device,
    )
    
    # Decode translation (only new tokens)
    translation = tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True
    ).strip()
    
    return translation, metrics


def speculative_decode_with_sampling():
    """Speculative decoding with nucleus/temperature sampling."""
    raise NotImplementedError(
        "Sampling-based speculative decoding not implemented yet. "
        "Use speculative_decode_greedy for greedy decoding."
    )


def speculative_decode_different_tokenizers():
    """Speculative decoding with different tokenizers (universal assisted decoding)."""
    raise NotImplementedError(
        "Different tokenizer speculative decoding not implemented yet. "
        "Use HuggingFace's assisted_decode for this case."
    )


# HUGGINGFACE ASSISTED GENERATION WRAPPER
def assisted_decode_hf(
    target_model,
    target_tokenizer,
    draft_model,
    draft_tokenizer,
    source: str,
    target_lang: str,
    max_length: int = 512,
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

    messages = [
        {
            "role": "system",
            "content": f"You are a professional translator. Translate the following text from English to {target_lang}. Maintain the original style and tone. Only output the translation."
        },
        {
            "role": "user",
            "content": f"Translate this:\n\n{source}"
        }
    ]

    prompt = target_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    start_time = time.time()
    
    # Check if tokenizers are the same object (only skip tokenizer params in this case)
    same_tokenizer = target_tokenizer is draft_tokenizer
    
    generate_kwargs = {
        "attention_mask": attention_mask,
        "assistant_model": draft_model,
        "max_new_tokens": max_length,
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
    
    total_time = time.time() - start_time
    generated_tokens = outputs.shape[1] - prompt_len

    translation = target_tokenizer.decode(
        outputs[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    if return_metrics:
        metrics = {
            "total_time": total_time,
            "generated_tokens": generated_tokens,
            "decode_tps": generated_tokens / total_time if total_time > 0 else 0,
        }
        return translation, metrics
    else:
        return translation
