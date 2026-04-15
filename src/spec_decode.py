"""
Speculative decoding implementation.

Contains:
- speculative_decode_greedy: Custom greedy speculative decoding with KV caching
- get_stop_token_ids: Stop token detection for various chat models
- crop_kv_cache: KV cache management utility
"""

import time
from typing import Literal

import torch


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
        "<|im_end|>",  # Qwen
        "<|endoftext|>",  # Qwen, GPT
        "<|eot_id|>",  # Llama 3
        "<|end_of_text|>",  # Llama 3
        "</s>",  # Mistral, Llama 2
        "<end_of_turn>",  # Gemma
        "<eos>",  # Gemma
        "[/INST]",  # Mistral
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

    if hasattr(past_key_values, "crop"):
        past_key_values.crop(new_length)
        return past_key_values
    else:
        new_past = []
        for layer_past in past_key_values:
            # NGramModel-style cache: a single tensor per layer
            if isinstance(layer_past, torch.Tensor):
                # Crop along the sequence-length dimension (assumed last)
                new_past.append(layer_past[..., :new_length])
            # Standard (key, value) pair cache from HF
            elif len(layer_past) == 2:
                key_state, value_state = layer_past
                k_cropped = key_state[..., :new_length, :]
                v_cropped = value_state[..., :new_length, :]
                new_past.append((k_cropped, v_cropped))
        return tuple(new_past)


def get_kv_cache_length(past_key_values) -> int:
    """Helper to get the current sequence length of a KV cache."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
        if len(past_key_values[0][0].shape) == 4:
            # HF cache
            return past_key_values[0][0].size(2)
        else:
            return past_key_values[0][0].size(-1)
    return 0


def speculative_decode(
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor,
    mode: Literal["greedy", "sample"],
    max_new_tokens: int = 256,
    gamma: int = 5,
    eos_token_id: int | None = None,
    device=None,
    track_iterations: bool = False,
):
    """
    Speculative Decoding with KV Caching.
    Key features:

    Args:
        target_model: The large target model
        draft_model: The smaller draft model
        tokenizer: Shared tokenizer (must be same for both models)
        input_ids: Input token IDs [1, seq_len]
        mode: 'greedy' | 'sample'
        max_new_tokens: Maximum new tokens to generate
        gamma: Number of draft tokens to generate per iteration
        eos_token_id: End of sequence token ID
        device: Device to run on

    Returns:
        output_ids: Generated token IDs
        metrics: Dict with acceptance_rate, time, draft_tokens, matched_tokens, etc.
    """
    bs = input_ids.size(0)
    assert bs == 1, "Speculative decoding only supports batch_size=1"

    if device is None:
        device = next(target_model.parameters()).device

    def select_index(logits: torch.Tensor):
        return sample(logits, mode)

    stop_token_ids = torch.tensor(
        list(get_stop_token_ids(tokenizer, eos_token_id)), device=device
    )
    input_ids = input_ids.to(device)

    # This is okay because if we've gotten this far, we know the actual tokenizers are the same length.
    # Just be aware that logits may have a slightly shorter dimension
    d_vocab = max(draft_model.config.vocab_size, target_model.config.vocab_size)

    # B,S+max_new
    generated_tokens = torch.concat(
        [
            input_ids,
            torch.zeros(
                input_ids.size(0), max_new_tokens, device=device, dtype=torch.int64
            ),
        ],
        dim=-1,
    )
    cur_gen_idx = input_ids.size(1)

    is_cuda = device.type == "cuda"
    with torch.no_grad():
        # Preload kv cache for prompts
        target_out = target_model(input_ids, use_cache=True)
        target_kv_cache = target_out.past_key_values
        draft_kv_cache = draft_model(input_ids, use_cache=True).past_key_values

        # Add the first new token
        last_target_token = select_index(
            torch.log_softmax(target_out.logits[:, -1, :], dim=-1)
        )
        generated_tokens[:, cur_gen_idx] = last_target_token
        cur_gen_idx += 1

        # Metrics
        total_draft_tokens = 0
        total_matched_tokens = 0
        num_iterations = 0
        iteration_history = []
        if is_cuda:
            torch.cuda.synchronize()
        start_time = time.time()

        while cur_gen_idx < generated_tokens.size(-1):
            num_iterations += 1
            # Step 1: Draft tokens
            # B * gamma (unless gamma > remaining tokens)
            max_draft_tokens = min(gamma, generated_tokens.size(-1) - cur_gen_idx)
            new_draft_tokens = torch.zeros(
                (bs, max_draft_tokens),
                device=input_ids.device,
                dtype=torch.int64,
            )
            new_draft_token_logprobs = torch.full(
                (bs, max_draft_tokens, d_vocab),
                fill_value=float("-inf"),
                device=input_ids.device,
            )

            # Determine how many tokens the draft model is missing from its cache
            cache_len = get_kv_cache_length(draft_kv_cache)
            expected_len = cur_gen_idx - 1
            if cache_len < expected_len:
                draft_input_ids = generated_tokens[:, cache_len:cur_gen_idx]
            else:
                draft_input_ids = generated_tokens[:, cur_gen_idx - 1 : cur_gen_idx]

            for idx in range(new_draft_tokens.size(-1)):
                draft_out = draft_model(
                    input_ids=draft_input_ids,
                    past_key_values=draft_kv_cache,
                    use_cache=True,
                )
                draft_kv_cache = draft_out.past_key_values
                draft_out_logprobs = torch.log_softmax(
                    draft_out.logits[:, -1, :], dim=-1
                )
                next_draft_token = select_index(draft_out_logprobs)  # (bs,)
                new_draft_tokens[:, idx] = next_draft_token
                new_draft_token_logprobs[:, idx, : draft_out_logprobs.shape[-1]] = (
                    draft_out_logprobs
                )
                draft_input_ids = next_draft_token.unsqueeze(-1)
                if torch.isin(next_draft_token, stop_token_ids).any():
                    # Trim draft tokens tensor since it's shorter than usual
                    new_draft_tokens = new_draft_tokens[:, : idx + 1]
                    new_draft_token_logprobs = new_draft_token_logprobs[:, : idx + 1, :]
                    break

            #  Step 2: Target model verifies
            target_input_ids = torch.concat(
                [generated_tokens[:, cur_gen_idx - 1 : cur_gen_idx], new_draft_tokens],
                dim=-1,
            )
            target_out = target_model(
                input_ids=target_input_ids,
                past_key_values=target_kv_cache,
                use_cache=True,
            )
            # Find the first collision, if any
            target_out_logprobs = torch.log_softmax(
                target_out.logits, dim=-1
            )  # (bs,seq,d_vocab)
            target_out_chosen_logprobs = target_out_logprobs[:,:-1,:].gather(
                -1, new_draft_tokens.unsqueeze(-1)
            ).squeeze(-1)  # (bs,seq)
            draft_out_chosen_logprobs = new_draft_token_logprobs.gather(
                -1, new_draft_tokens.unsqueeze(-1)
            ).squeeze(-1)  # (bs,seq)

            # First, check if p_draft(t) <= p_target(t)
            lower_draft_prob = draft_out_chosen_logprobs <= target_out_chosen_logprobs

            # If this fails, we still accept a token with p = p_target(t) / p_draft(t)
            random_accept = target_out_chosen_logprobs - draft_out_chosen_logprobs
            random_accept = (
                torch.log(torch.rand_like(target_out_chosen_logprobs)) < random_accept
            )

            # Figure out the first rejected token
            rejected = ~(lower_draft_prob | random_accept)
            total_draft_tokens += rejected.size(-1)
            if rejected.any():
                first_collision_idx = rejected.int().argmax(dim=-1).item()
                assert isinstance(first_collision_idx, int)
                total_matched_tokens += first_collision_idx

                # Resample token from p_target(x) - p_draft(x)
                resample_dist = (
                    torch.exp(target_out_logprobs[:, first_collision_idx])
                    - torch.exp(new_draft_token_logprobs[:, first_collision_idx])
                ).clamp(min=0)
                resample_dist = resample_dist / resample_dist.sum(dim=-1, keepdim=True)
                resample_dist = torch.log(resample_dist)
                resampled_token = select_index(resample_dist)

                tokens_to_add = torch.concat(
                    [
                        new_draft_tokens[:, :first_collision_idx],
                        resampled_token.unsqueeze(-1),
                    ],
                    dim=-1,
                )
            else:
                if torch.isin(new_draft_tokens[:, -1], stop_token_ids).any():
                    # If we've reached <eos>, don't add bonus token
                    tokens_to_add = new_draft_tokens
                else:
                    # If no collision, add all draft tokens plus the bonus token (if room)
                    if cur_gen_idx + new_draft_tokens.size(-1) < generated_tokens.size(
                        -1
                    ):
                        bonus_token = select_index(target_out_logprobs[:, -1])
                        tokens_to_add = torch.concat(
                            [
                                new_draft_tokens,
                                bonus_token.unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                    else:
                        tokens_to_add = new_draft_tokens

                total_matched_tokens += new_draft_tokens.size(-1)

            # Actually add the new tokens and update idxs
            new_gen_idx = cur_gen_idx + tokens_to_add.size(-1)
            generated_tokens[:, cur_gen_idx:new_gen_idx] = tokens_to_add
            cur_gen_idx = new_gen_idx

            # Update kv caches
            # Either cache should not include the last generated tok (either correction or bonus token)
            target_kv_cache = crop_kv_cache(target_kv_cache, new_gen_idx - 1)
            draft_kv_cache = crop_kv_cache(draft_kv_cache, new_gen_idx - 1)

            if track_iterations:
                # FIXME: If we ever do batching this is wrong
                draft_ids = new_draft_tokens[0].tolist()
                draft_text = [tokenizer.decode([tid]) for tid in draft_ids]
                last_token_str = tokenizer.decode([int(tokens_to_add[0, -1].item())])
                if not rejected.any():
                    result = (
                        f"ALL ACCEPTED ({len(draft_ids)}) + BONUS '{last_token_str}'"
                    )
                else:
                    first_collision_idx = rejected.int().argmax(dim=-1).item()
                    assert isinstance(first_collision_idx, int)
                    rejected_str = tokenizer.decode(
                        [new_draft_tokens[0][first_collision_idx].item()]
                    )
                    result = f"ACCEPTED {first_collision_idx}, REJECTED '{rejected_str}' -> TARGET '{last_token_str}'"
                iteration_history.append(
                    {
                        "iter": len(iteration_history),
                        "drafted": draft_text,
                        "result": result,
                    }
                )

            if torch.isin(generated_tokens[:, cur_gen_idx - 1], stop_token_ids).any():
                # Get rid of extra 0s
                generated_tokens = generated_tokens[:, :cur_gen_idx]
                break

    if is_cuda:
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    # Calculate acceptance rate (matched draft tokens / total draft tokens)
    acceptance_rate = (
        total_matched_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    )
    total_generated_tokens = cur_gen_idx - input_ids.size(1)

    metrics = {
        "time": total_time,
        "generated_tokens": total_generated_tokens,
        "draft_tokens": total_draft_tokens,
        "matched_tokens": total_matched_tokens,
        "acceptance_rate": acceptance_rate,
        "num_iterations": num_iterations,
        "toks_per_sec": total_generated_tokens / total_time if total_time > 0 else 0,
    }

    if track_iterations:
        metrics["iteration_history"] = iteration_history

    return generated_tokens, metrics



def sample(logprobs: torch.Tensor, mode: Literal["greedy", "sample"]):
    # TODO: Add top-k and top-p
    
    if mode == "greedy":
        return logprobs.argmax(dim=-1)
    else:
        return torch.distributions.Categorical(logits=logprobs).sample()


def speculative_decode_different_tokenizers():
    """Speculative decoding with different tokenizers (universal assisted decoding)."""
    raise NotImplementedError(
        "Different tokenizer speculative decoding not implemented yet. "
        "Use HuggingFace's assisted_decode for this case."
    )
