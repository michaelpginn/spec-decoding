"""
greedy speculative decoding for translation
"""
import torch
import torch.nn.functional as F
import time


def crop_past_key_values(past_key_values, keep_length: int):
    """
    Crops the KV cache to a specific length.
    Handles both DynamicCache (newer) and tuple (legacy) formats.
    """
    if past_key_values is None:
        return None
    
    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        num_layers = len(past_key_values.key_cache)
        for i in range(num_layers):
            # Get current cache tensors
            k = past_key_values.key_cache[i]
            v = past_key_values.value_cache[i]
            
            # Crop to keep_length
            if k.shape[2] > keep_length:
                past_key_values.key_cache[i] = k[:, :, :keep_length, :]
                past_key_values.value_cache[i] = v[:, :, :keep_length, :]
        
        return past_key_values  
    
    if isinstance(past_key_values, tuple):
        new_past = []
        for layer_past in past_key_values:
            if isinstance(layer_past, tuple) and len(layer_past) == 2:
                k, v = layer_past
                if k.shape[2] > keep_length:
                    k = k[:, :, :keep_length, :]
                    v = v[:, :, :keep_length, :]
                new_past.append((k, v))
            else:
                new_past.append(layer_past)
        return tuple(new_past)
    
    return past_key_values


def compute_kl_divergence(draft_logits: torch.Tensor, target_logits: torch.Tensor) -> float:
    """
    Compute KL divergence between draft and target distributions.
    KL(target || draft) = sum(target * log(target / draft))
    """
    min_vocab = min(draft_logits.shape[-1], target_logits.shape[-1])
    draft_logits = draft_logits[..., :min_vocab]
    target_logits = target_logits[..., :min_vocab]

    draft_probs = F.softmax(draft_logits, dim=-1)
    target_probs = F.softmax(target_logits, dim=-1)
    
    eps = 1e-10
    draft_probs = draft_probs + eps
    target_probs = target_probs + eps
    
    # KL divergence
    kl = (target_probs * (target_probs.log() - draft_probs.log())).sum(dim=-1)
    return kl.mean().item()


def speculative_decode(
    target_model,
    target_tokenizer,
    draft_model,
    draft_tokenizer,  # same as target tokenizer for same family models
    source: str,
    target_lang: str,
    draft_k: int = 4,
    max_length: int = 512,
    device=None,
    return_metrics: bool = True,
):
    """
    Greedy speculative decoding with proper KV cache management.
    
    Returns: 
        translation: str
        metrics: dict (if return_metrics=True)
    """
    if device is None:
        device = next(target_model.parameters()).device

    # prompt
    messages = [
        {
            "role": "system",
            "content": f"You are a professional translator. Translate the following text from English to {target_lang.upper()}. Maintain the original style and tone. Only output the translation."
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

    # Tokenize
    inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    # prefill
    prefill_start = time.time()
    
    with torch.no_grad():
        # target model prefill
        target_outputs = target_model(input_ids, use_cache=True)
        target_past_kv = target_outputs.past_key_values
        
        # draft model prefill 
        draft_outputs = draft_model(input_ids, use_cache=True)
        draft_past_kv = draft_outputs.past_key_values
        
        # Generate first token from target
        next_token_logits = target_outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    # Initialize generated sequence
    generated_ids = torch.cat([input_ids, next_token_id], dim=1)
    
    # Update both caches with the first token
    with torch.no_grad():
        # Update target cache
        target_out = target_model(next_token_id, past_key_values=target_past_kv, use_cache=True)
        target_past_kv = target_out.past_key_values
        
        # Update draft cache
        draft_out = draft_model(next_token_id, past_key_values=draft_past_kv, use_cache=True)
        draft_past_kv = draft_out.past_key_values
    
    prefill_time = time.time() - prefill_start

    # metrics 
    accepted_tokens_per_step = []
    total_draft_tokens = 0
    total_accepted_tokens = 0
    kl_divergences = []
    num_steps = 0
    decode_start = time.time()

    # decoding
    while generated_ids.shape[1] < max_length:
        num_steps += 1
        
        # Draft model proposes K tokens (only pass LAST token, use cache)
        draft_tokens = []
        draft_logits_list = []
        
        for _ in range(draft_k):
            with torch.no_grad():
                # Only pass the last token (use cache for history)
                last_token = generated_ids[:, -1:] if len(draft_tokens) == 0 else draft_tokens[-1]
                
                draft_out = draft_model(last_token, past_key_values=draft_past_kv, use_cache=True)
                draft_past_kv = draft_out.past_key_values
                
                draft_logits = draft_out.logits[:, -1, :]
                draft_logits_list.append(draft_logits)
                
                draft_token = torch.argmax(draft_logits, dim=-1, keepdim=True)
                draft_tokens.append(draft_token)
        
        total_draft_tokens += draft_k
        draft_tokens_tensor = torch.cat(draft_tokens, dim=1)

        # Target model verifies K tokens in ONE forward pass
        with torch.no_grad():
            target_out = target_model(draft_tokens_tensor, past_key_values=target_past_kv, use_cache=True)
            target_logits = target_out.logits 
        
        # kl divergence
        draft_logits_stacked = torch.stack(draft_logits_list, dim=1) 
        kl_div = compute_kl_divergence(draft_logits_stacked, target_logits)
        kl_divergences.append(kl_div)

        # accept/reject
        accepted_count = 0
        correction_token = None
        
        for i in range(draft_k):
            draft_token_id = draft_tokens[i].item()
            target_token_id = torch.argmax(target_logits[:, i, :], dim=-1).item()
            
            if draft_token_id == target_token_id:
                accepted_count += 1
                total_accepted_tokens += 1
            else:
                # Reject: target's prediction is the correction
                correction_token = torch.tensor([[target_token_id]], device=device)
                break
        
        accepted_tokens_per_step.append(accepted_count)
        
        # Add accepted tokens to generated sequence
        if accepted_count > 0:
            accepted_segment = draft_tokens_tensor[:, :accepted_count]
            generated_ids = torch.cat([generated_ids, accepted_segment], dim=1)
        
        # Add correction token or sample next token
        if accepted_count < draft_k:
            # Rejected: add correction token
            generated_ids = torch.cat([generated_ids, correction_token], dim=1)
        else:
            # All accepted: sample one more token from target
            last_target_logits = target_logits[:, -1, :]
            next_token = torch.argmax(last_target_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            correction_token = next_token  # For syncing draft

        current_seq_len = generated_ids.shape[1]
        
        target_past_kv = crop_past_key_values(target_out.past_key_values, current_seq_len - 1)
        draft_past_kv = crop_past_key_values(draft_past_kv, current_seq_len - 1)
        
        # Sync BOTH models with the last token
        sync_token = correction_token if correction_token is not None else next_token
        
        with torch.no_grad():
            draft_sync_out = draft_model(sync_token, past_key_values=draft_past_kv, use_cache=True)
            draft_past_kv = draft_sync_out.past_key_values
            
            target_sync_out = target_model(sync_token, past_key_values=target_past_kv, use_cache=True)
            target_past_kv = target_sync_out.past_key_values

        # Check EOS
        if generated_ids[0, -1].item() == target_tokenizer.eos_token_id:
            break

    decode_time = time.time() - decode_start
    total_time = prefill_time + decode_time

    translation = target_tokenizer.decode(
        generated_ids[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    if return_metrics:
        generated_tokens = generated_ids.shape[1] - prompt_len
        metrics = {
            "total_time": total_time,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "num_steps": num_steps,
            "accepted_tokens_per_step": accepted_tokens_per_step,
            "total_draft_tokens": total_draft_tokens,
            "total_accepted_tokens": total_accepted_tokens,
            "generated_tokens": generated_tokens,
            "kl_divergences": kl_divergences,
            # Computed metrics
            "acceptance_rate": total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0,
            "mean_accepted_tokens": sum(accepted_tokens_per_step) / len(accepted_tokens_per_step) if accepted_tokens_per_step else 0,
            "block_efficiency": (sum(accepted_tokens_per_step) / len(accepted_tokens_per_step) / draft_k) if accepted_tokens_per_step else 0,
            "avg_kl_divergence": sum(kl_divergences) / len(kl_divergences) if kl_divergences else 0,
            "decode_tps": generated_tokens / decode_time if decode_time > 0 else 0,
        }
        return translation, metrics
    else:
        return translation