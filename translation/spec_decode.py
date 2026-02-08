"""
Speculative decoding for translation.

Contains:
- speculative_decode_greedy: Custom implementation 
- speculative_decode_translate: Translation wrapper for speculative_decode_greedy
- assisted_decode_hf: HuggingFace's assisted generation wrapper
"""
import time

import torch

from translation.models import create_translation_messages


def get_stop_token_ids(tokenizer, eos_token_id=None):
    """
    Get all stop token IDs for chat models.
    Supports: Qwen, Llama, Mistral, Gemma, and others.
    """
    stop_ids = set()
    
    if eos_token_id is not None:
        stop_ids.add(eos_token_id)
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    
    stop_tokens = [
        "<|im_end|>",        # Qwen
        "<|endoftext|>",     # Qwen, GPT
        "<|eot_id|>",        # Llama 3
        "<|end_of_text|>",   # Llama 3
        "</s>",              # Mistral, Llama 2
        "<end_of_turn>",     # Gemma
        "<eos>",             # Gemma
        "[/INST]",           # Mistral
    ]
    
    for token in stop_tokens:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids and len(ids) == 1:
                stop_ids.add(ids[0])
        except Exception:
            pass
    
    return stop_ids


def crop_kv_cache(past_key_values, new_length):
    """
    Crop KV cache to a specific sequence length.
    Handles both DynamicCache objects and tuple format.
    """
    if past_key_values is None:
        return None
    
    if hasattr(past_key_values, 'crop'):
        past_key_values.crop(new_length)
        return past_key_values
    else:
        new_past = []
        for layer_past in past_key_values:
            key_state, value_state = layer_past
            k_cropped = key_state[:, :, :new_length, :]
            v_cropped = value_state[:, :, :new_length, :]
            new_past.append((k_cropped, v_cropped))
        return tuple(new_past)


def speculative_decode_greedy(
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    gamma: int = 5,
    eos_token_id: int | None = None,
    device=None,
    track_iterations: bool = False
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
    
    stop_token_ids = torch.tensor(list(get_stop_token_ids(tokenizer, eos_token_id)), device=device)
    input_ids = input_ids.to(device)

    # B,S+max_new
    generated_tokens = torch.concat(
        [
            input_ids,
            torch.zeros(input_ids.size(0), max_new_tokens, device=device, dtype=torch.int64),
        ], dim=-1
    )
    cur_gen_idx = input_ids.size(1)

    with torch.no_grad():
        # Preload kv cache for prompts
        target_out = target_model(input_ids, use_cache=True)
        target_kv_cache = target_out.past_key_values
        draft_kv_cache = draft_model(input_ids, use_cache=True).past_key_values

        # Add the first new token
        last_target_token = target_out.logits[:,-1,:].argmax(dim=-1)
        generated_tokens[:, cur_gen_idx] = last_target_token
        cur_gen_idx += 1
        
        # Metrics
        total_draft_tokens = 0
        total_matched_tokens = 0  
        iteration_history = []
        start_time = time.time()
    
        while cur_gen_idx < generated_tokens.size(-1):
            # Step 1: Draft tokens
            # B * gamma (unless gamma > remaining tokens)
            new_draft_tokens = torch.zeros(
                (
                    input_ids.size(0),
                    min(gamma, generated_tokens.size(-1) - cur_gen_idx),
                ),
                device=input_ids.device,
                dtype=torch.int64
            )
            draft_input_ids = generated_tokens[:, cur_gen_idx-1:cur_gen_idx]
            for idx in range(new_draft_tokens.size(-1)):
                draft_out = draft_model(
                    input_ids=draft_input_ids,
                    past_key_values=draft_kv_cache,
                    use_cache=True,
                )
                draft_kv_cache = draft_out.past_key_values
                next_draft_token = draft_out.logits[:, -1, :].argmax(dim=-1)
                new_draft_tokens[:, idx] = next_draft_token
                draft_input_ids = next_draft_token.unsqueeze(-1)
                if torch.isin(next_draft_token, stop_token_ids).any():
                    # Trim draft tokens tensor since it's shorter than usual
                    new_draft_tokens = new_draft_tokens[:,:idx+1]
                    break
            
            #  Step 2: Target model verifies
            target_input_ids = torch.concat([
                generated_tokens[:, cur_gen_idx-1:cur_gen_idx],
                new_draft_tokens
            ], dim=-1)
            target_out = target_model(
                input_ids=target_input_ids,
                past_key_values=target_kv_cache,
                use_cache=True,
            )
            # Find the first collision, if any
            target_preds_for_draft = target_out.logits.argmax(dim=-1)
            collisions = (new_draft_tokens != target_preds_for_draft[:,:-1])
            total_draft_tokens += collisions.size(-1)
            if collisions.any():
                first_collision_idx = collisions.int().argmax(dim=-1).min().item()
                tokens_to_add = torch.concat(
                    [
                        new_draft_tokens[:, :first_collision_idx],
                        target_preds_for_draft[:, first_collision_idx].unsqueeze(-1),
                    ],
                    dim=-1,
                )
                total_matched_tokens += first_collision_idx
            else:
                if torch.isin(new_draft_tokens[:,-1], stop_token_ids).any():
                    # If we've reached <eos>, don't add bonus token
                    tokens_to_add = new_draft_tokens

                else:
                    # If no collision, add all draft tokens plus the bonus token (if room)
                    if cur_gen_idx + new_draft_tokens.size(-1) < generated_tokens.size(-1):
                        tokens_to_add = torch.concat(
                            [new_draft_tokens, target_preds_for_draft[:, -1].unsqueeze(-1)],
                            dim=-1,
                        )
                    else:
                        tokens_to_add = new_draft_tokens

                total_matched_tokens += new_draft_tokens.size(-1)
            # Actually add the new tokens and update idxs
            new_gen_idx = cur_gen_idx + tokens_to_add.size(-1)
            generated_tokens[:, cur_gen_idx : new_gen_idx] = tokens_to_add
            cur_gen_idx = new_gen_idx
            
            # Update kv caches
            # Either cache should not include the last generated tok (either correction or bonus token)
            draft_kv_cache = crop_kv_cache(draft_kv_cache, new_gen_idx - 1)
            target_kv_cache = crop_kv_cache(target_kv_cache, new_gen_idx - 1)
            
            if track_iterations:
                # FIXME: If we ever do batching this is wrong
                draft_text = tokenizer.convert_ids_to_tokens(new_draft_tokens[0])
                last_token = tokenizer.convert_ids_to_tokens(tokens_to_add[0][-1:])
                if not collisions.any():
                    result = f"ALL ACCEPTED ({len(new_draft_tokens[0])}) + BONUS '{last_token}'"
                else:
                    first_collision_idx = collisions.int().argmax(dim=-1)
                    rejected = tokenizer.convert_ids_to_tokens(new_draft_tokens[0][first_collision_idx])
                    result = f"ACCEPTED {first_collision_idx}, REJECTED '{rejected}' -> TARGET '{last_token}'"
                iteration_history.append({
                    "iter": len(iteration_history),
                    "drafted": draft_text,
                    "result": result,
                })

            if torch.isin(generated_tokens[:,cur_gen_idx-1], stop_token_ids).any():
                # Get rid of extra 0s
                generated_tokens = generated_tokens[:,:cur_gen_idx]
                break
    
    total_time = time.time() - start_time
    
    # Calculate acceptance rate (matched draft tokens / total draft tokens)
    acceptance_rate = total_matched_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    total_generated_tokens = torch.count_nonzero(generated_tokens)
    
    metrics = {
        "total_time": total_time,
        "generated_tokens": total_generated_tokens,
        "total_draft_tokens": total_draft_tokens,
        "total_matched_tokens": total_matched_tokens,
        "acceptance_rate": acceptance_rate,
    }

    if track_iterations:
        metrics["iteration_history"] = iteration_history
    
    return generated_tokens, metrics


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
    
    messages = create_translation_messages(source, target_lang)
    
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
        track_iterations=track_iterations
    )
    
    # Decode translation (only new tokens)
    translation = tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=False
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

    messages = create_translation_messages(source, target_lang)

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
