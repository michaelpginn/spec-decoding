"""
speculative decoding for translation

Contains:
- speculative_decode_greedy: Custom implementation (same tokenizer, greedy decoding)
- assisted_decode: HuggingFace's assisted generation wrapper
"""
import torch
import time

def crop_past_key_values(past_key_values, new_length):
    """
    slice the KV cache of the sequence length.
    """
    if past_key_values is None:
        return None

    new_past = []
    for layer_past in past_key_values:
        key_state, value_state = layer_past
        k_cropped = key_state[:, :, :new_length, :]
        v_cropped = value_state[:, :, :new_length, :]
        new_past.append((k_cropped, v_cropped))

    return tuple(new_past)

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
    Speculative Decoding with KV Caching.
    Key features:
    
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

    # targte model prefill
    target_outputs = target_model(input_ids, use_cache=True)
    target_past_key_values = target_outputs.past_key_values
    target_next_logit = target_outputs.logits[:, -1, :]

    # draft model prefill
    draft_outputs = draft_model(input_ids, use_cache=True)
    draft_past_key_values = draft_outputs.past_key_values
    draft_first_token = torch.argmax(draft_outputs.logits[:, -1, :], dim=-1, keepdim=True)
    
    # Metrics tracking
    total_draft_tokens = 0
    total_matched_tokens = 0  
    generated_count = 0

    start_time = time.time()
    
    with torch.no_grad():
        while generated_count < max_new_tokens:
            
            all_draft_tokens = [draft_first_token.item()]
            current_draft_input = draft_first_token

            if draft_first_token.item() == eos_token_id:
                pass
            else:
                for _ in range(gamma - 1):
                    draft_out = draft_model(
                        input_ids=current_draft_input,
                        past_key_values=draft_past_key_values,
                        use_cache=True,
                    )
                    draft_past_key_values = draft_out.past_key_values
                    new_token_id = torch.argmax(draft_out.logits[:, -1, :], dim=-1, keepdim=True)
                    all_draft_tokens.append(new_token_id.item())
                    current_draft_input = new_token_id
                    
                    if new_token_id.item() == eos_token_id:
                        break
            
            num_drafted = len(all_draft_tokens)
            if num_drafted == 0:
                break
            
            total_draft_tokens += num_drafted
            
            #  Step 2: Target model verifies
            draft_token_tensor = torch.tensor([all_draft_tokens], device=device)

            target_out = target_model(
                input_ids=draft_token_tensor,
                past_key_values=target_past_key_values,
                use_cache=True,
            )

            if num_drafted > 1:
                verification_logits = torch.cat([
                    target_next_logit.unsqueeze(1),     
                    target_out.logits[:, :-1, :]         
                ], dim=1)
            else:
                verification_logits = target_next_logit.unsqueeze(1)

            #  Step 3: Accept tokens until first mismatch 
            accepted_tokens = []
            all_match = True

            for i in range(num_drafted):
                draft_id = all_draft_tokens[i]
                target_id = torch.argmax(verification_logits[:, i, :]).item()
                
                if draft_id == target_id:
                    accepted_tokens.append(draft_id)
                else:
                    # Mismatch: accept target's correction and stop
                    accepted_tokens.append(target_id)
                    all_match = False
                    break
            
            num_accepted = len(accepted_tokens)
            num_matched = num_accepted - 1 if not all_match else num_accepted
            total_matched_tokens += num_matched

            # cache management
            if all_match:
                bonus_logits = target_out.logits[:, -1, :]
                bonus_token = torch.argmax(bonus_logits, dim=-1).item()
                accepted_tokens.append(bonus_token)

                # update sequence
                accepted_tensor = torch.tensor([accepted_tokens], device=device)
                current_ids = torch.cat([current_ids, accepted_tensor], dim=-1)
                generated_count += len(accepted_tokens)

                # target cache
                target_past_key_values = target_out.past_key_values
                bonus_token_tensor = torch.tensor([[bonus_token]], device=device)
                bonus_out = target_model(
                    input_ids=bonus_token_tensor,
                    past_key_values=target_past_key_values,
                    use_cache=True,
                )
                target_past_key_values = bonus_out.past_key_values
                target_next_logit = bonus_out.logits[:, -1, :]

                draft_bonus_out = draft_model(
                    input_ids=bonus_token_tensor,
                    past_key_values=draft_past_key_values,
                    use_cache=True,
                )
                draft_past_key_values = draft_bonus_out.past_key_values
                draft_first_token = torch.argmax(draft_bonus_out.logits[:, -1, :], dim=-1, keepdim=True)

            else:
                accepted_tensor = torch.tensor([accepted_tokens], device=device)
                current_ids = torch.cat([current_ids, accepted_tensor], dim=-1)
                generated_count += len(accepted_tokens)

                valid_cache_len = current_ids.shape[1] - 1

                target_past_key_values = crop_past_key_values(
                    target_out.past_key_values,
                    valid_cache_len
                )
                correction_token = accepted_tokens[-1]
                correction_tensor = torch.tensor([[correction_token]], device=device)
                correction_out = target_model(
                    input_ids=correction_tensor,
                    past_key_values=target_past_key_values,
                    use_cache=True,
                )
                target_past_key_values = correction_out.past_key_values
                target_next_logit = correction_out.logits[:, -1, :]
                
                # Crop draft cache and process correction
                draft_past_key_values = crop_past_key_values(
                    draft_past_key_values,
                    valid_cache_len
                )
                draft_correction_out = draft_model(
                    input_ids=correction_tensor,
                    past_key_values=draft_past_key_values,
                    use_cache=True,
                )
                draft_past_key_values = draft_correction_out.past_key_values
                draft_first_token = torch.argmax(draft_correction_out.logits[:, -1, :], dim=-1, keepdim=True)
            
            if accepted_tokens[-1] == eos_token_id:
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
