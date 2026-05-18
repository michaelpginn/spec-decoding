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
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.1,
    repetition_penalty_window: int = 16,
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
        top_k: If > 0, only sample from the top k tokens
        top_p: If > 0 and < 1, keep the smallest set of tokens whose cumulative prob >= p
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

    def apply_filters(logprobs: torch.Tensor) -> torch.Tensor:
        return filter_logprobs(logprobs, top_k=top_k, top_p=top_p)

    def select_index(logprobs: torch.Tensor):
        return sample(logprobs, mode)

    def penalize_logits(
        logits: torch.Tensor,
        confirmed_len: int,
        draft_so_far: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply windowed repetition penalty to raw logits (single position).

        Builds the penalty context as:
            last `repetition_penalty_window` tokens of
            generated_tokens[:, :confirmed_len] ++ draft_so_far (if any)

        This is called with raw logits (before log_softmax) so the
        positive/negative sign distinction in the penalty formula is meaningful.

        Args:
            logits:       Raw model logits, shape [bs, d_vocab].
            confirmed_len: Number of confirmed tokens in generated_tokens
                          (i.e. cur_gen_idx at the time of the call).
            draft_so_far: Draft tokens generated in the current iteration
                          so far, shape [bs, n_draft]. None or empty = no drafts yet.
        Returns:
            Penalized logits, same shape as input.
        """
        if repetition_penalty == 1.0:
            return logits
        # Build context: confirmed portion of generated_tokens + any draft tokens
        ctx = generated_tokens[:, :confirmed_len]
        if draft_so_far is not None and draft_so_far.size(-1) > 0:
            ctx = torch.cat([ctx, draft_so_far], dim=-1)
        # Slide to the last `repetition_penalty_window` tokens
        ctx = ctx[:, -repetition_penalty_window:]
        return apply_repetition_penalty(logits, ctx, repetition_penalty)

    def penalize_logits_batched(
        logits: torch.Tensor,
        confirmed_len: int,
        draft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply windowed repetition penalty to all verification positions at once.

        Instead of calling penalize_logits in a Python loop for each position
        j in [0, n_draft], this builds a (n_positions, vocab_size) counts
        matrix incrementally and applies the penalty in one vectorized pass.

        Position j's context = confirmed_tokens[-W:] ++ draft_tokens[:j].
        Position n_draft (bonus) uses all draft_tokens.

        Args:
            logits:        Raw target logits, shape [bs, n_draft+1, d_vocab].
            confirmed_len: Number of confirmed tokens in generated_tokens.
            draft_tokens:  Draft tokens from this iteration, shape [bs, n_draft].
        Returns:
            Penalized logits, same shape as input.
        """
        if repetition_penalty == 1.0:
            return logits
        return apply_repetition_penalty_batched(
            logits, generated_tokens, confirmed_len, draft_tokens,
            repetition_penalty, repetition_penalty_window,
        )

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

    # Track average time for draft and verifier forward pass for speedup factor
    draft_start,draft_end,verifier_start, verifier_end   = None, None, None, None
    draft_times_acc = (0., 0)
    verifier_times_acc = (0., 0)
    if device.type == 'cuda':
        draft_start = torch.cuda.Event(enable_timing=True)
        draft_end = torch.cuda.Event(enable_timing=True)
        verifier_start = torch.cuda.Event(enable_timing=True)
        verifier_end = torch.cuda.Event(enable_timing=True)

    def get_time():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.time()
        
    with torch.no_grad():
        # Preload kv cache for prompts
        target_out = target_model(input_ids, use_cache=True)
        target_kv_cache = target_out.past_key_values
        draft_kv_cache = draft_model(input_ids, use_cache=True).past_key_values

        # Add the first new token.
        # Penalty context: the last W tokens of the prompt (no generated tokens yet).
        first_logits = penalize_logits(target_out.logits[:, -1, :], confirmed_len=cur_gen_idx)
        last_target_token = select_index(
            apply_filters(torch.log_softmax(first_logits, dim=-1))
        )
        generated_tokens[:, cur_gen_idx] = last_target_token
        cur_gen_idx += 1

        # Metrics
        total_draft_tokens = 0
        total_matched_tokens = 0
        num_iterations = 0
        iteration_history = []
        start_time = get_time()

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

            _ = draft_start and draft_start.record()
            for idx in range(new_draft_tokens.size(-1)):
                draft_out = draft_model(
                    input_ids=draft_input_ids,
                    past_key_values=draft_kv_cache,
                    use_cache=True,
                )
                draft_kv_cache = draft_out.past_key_values
                # Penalty context: confirmed tokens + draft tokens produced so far
                # (new_draft_tokens[:, :idx] holds the idx tokens drafted this iteration).
                draft_out_logprobs = apply_filters(
                    torch.log_softmax(
                        penalize_logits(
                            draft_out.logits[:, -1, :],
                            confirmed_len=cur_gen_idx,
                            draft_so_far=new_draft_tokens[:, :idx],
                        ),
                        dim=-1,
                    )
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
            _ = draft_end and draft_end.record()

            #  Step 2: Target model verifies
            target_input_ids = torch.concat(
                [generated_tokens[:, cur_gen_idx - 1 : cur_gen_idx], new_draft_tokens],
                dim=-1,
            )
            _ = verifier_start and verifier_start.record()
            target_out = target_model(
                input_ids=target_input_ids,
                past_key_values=target_kv_cache,
                use_cache=True,
            )
            _ = verifier_end and verifier_end.record()
            
            # Record times (CUDA only)
            if draft_start and draft_end and verifier_start and verifier_end:
                torch.cuda.synchronize()
                draft_times_acc = (draft_times_acc[0] + draft_start.elapsed_time(draft_end), draft_times_acc[1] + new_draft_tokens.size(-1))
                verifier_times_acc = (verifier_times_acc[0] + verifier_start.elapsed_time(verifier_end), verifier_times_acc[1] + 1)
            
            # Find the first collision, if any.
            #
            # Apply repetition penalty per-position BEFORE log_softmax so that
            # the target distribution matches what the draft model saw at each step.
            # Position j verifies draft token j, so its penalty context is:
            #   confirmed tokens + new_draft_tokens[:, :j]  (same as draft step j).
            # Position `n_draft` (the bonus token) uses all draft tokens as context.
            #
            # We must use the raw logits here, not the post-softmax values, because
            # apply_repetition_penalty relies on logit sign to decide divide vs multiply.
            n_draft = new_draft_tokens.size(-1)  # may be < gamma if EOS was hit early
            target_raw_logits = target_out.logits  # (bs, n_draft+1, d_vocab)
            penalized_target_logits = penalize_logits_batched(
                target_raw_logits,
                confirmed_len=cur_gen_idx,
                draft_tokens=new_draft_tokens,
            )
            target_out_logprobs = apply_filters(
                torch.log_softmax(penalized_target_logits, dim=-1)
            )  # (bs, n_draft+1, d_vocab)
            target_out_chosen_logprobs = (
                target_out_logprobs[:, :-1, :]
                .gather(-1, new_draft_tokens.unsqueeze(-1))
                .squeeze(-1)
            )  # (bs, n_draft)
            draft_out_chosen_logprobs = new_draft_token_logprobs.gather(
                -1, new_draft_tokens.unsqueeze(-1)
            ).squeeze(-1)  # (bs, n_draft)

            # First, check if p_draft(t) <= p_target(t)
            lower_draft_prob = draft_out_chosen_logprobs <= target_out_chosen_logprobs

            # If this fails, we still accept a token with p = p_target(t) / p_draft(t)
            random_accept = target_out_chosen_logprobs - draft_out_chosen_logprobs
            random_accept = (
                torch.log(torch.rand_like(target_out_chosen_logprobs)) < random_accept
            )

            # Figure out the first rejected token
            rejected = ~(lower_draft_prob | random_accept)
            if rejected.any():
                first_collision_idx = rejected.int().argmax(dim=-1).item()
                assert isinstance(first_collision_idx, int)
                total_matched_tokens += first_collision_idx
                total_draft_tokens += first_collision_idx + 1

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
                total_matched_tokens += new_draft_tokens.size(-1)
                total_draft_tokens += new_draft_tokens.size(-1)
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

    total_time = get_time() - start_time

    # Acceptance rate (matched draft tokens / total verified draft tokens)
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
    
    # Forward pass times for speedup factor
    if draft_times_acc[1] > 0 and verifier_times_acc[1] > 0:
        average_draft_time = draft_times_acc[0] / draft_times_acc[1]
        average_verifier_time = verifier_times_acc[0] / verifier_times_acc[1]
        metrics["average_draft_time"] = average_draft_time / 1000 # seconds
        metrics["average_verifier_time"] = average_verifier_time / 1000

    if track_iterations:
        metrics["iteration_history"] = iteration_history

    return generated_tokens, metrics


def filter_logprobs(
    logprobs: torch.Tensor, top_k: int = 0, top_p: float = 0.0
) -> torch.Tensor:
    """Apply top-k and/or top-p filtering, then renormalize to valid log-probs."""
    filtered = logprobs
    if top_k > 0:
        filtered = apply_top_k(filtered, k=top_k)
    if 0.0 < top_p < 1.0:
        filtered = apply_top_p(filtered, p=top_p)
    if top_k > 0 or 0.0 < top_p < 1.0:
        filtered = torch.log_softmax(filtered, dim=-1)
    return filtered


def sample(logprobs: torch.Tensor, mode: Literal["greedy", "sample"]):
    """Sample a token index from (already filtered) log-probs."""
    if mode == "greedy":
        return logprobs.argmax(dim=-1)
    return torch.distributions.Categorical(logits=logprobs).sample()


def speculative_decode_different_tokenizers():
    """Speculative decoding with different tokenizers (universal assisted decoding)."""
    raise NotImplementedError(
        "Different tokenizer speculative decoding not implemented yet. "
        "Use HuggingFace's assisted_decode for this case."
    )


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Filters logits to only keep the top k values."""
    if k < 1:
        raise ValueError(f"top_k must be >= 1, got {k}")

    if k >= logits.size(-1):
        return logits

    top_values, _ = torch.topk(logits, k, dim=-1)
    kth_value = top_values[..., -1, None]
    indices_to_remove = logits < kth_value
    logits_filtered = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits_filtered


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Filters logits to keep the smallest set of top tokens whose cumulative prob >= p."""
    if p < 0.0 or p > 1.0:
        raise ValueError(f"top_p must be between 0.0 and 1.0, got {p}")

    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p

    # to keep the borderline token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    logits_filtered = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits_filtered


def apply_repetition_penalty(
    logits: torch.Tensor,
    context_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply multiplicative repetition penalty to raw logits (single position).

    Only tokens that appear **more than once** in `context_ids` are penalized.
    A token's first occurrence in the window is free — the penalty only fires
    when the model tries to emit a token it has already repeated within the
    current window.

    For each such repeated token ID:
        - If logit > 0  ->  logit /= penalty   (shrinks toward zero -> less probable)
        - If logit < 0  ->  logit *= penalty   (pushed further negative -> even less probable)

    The penalty must be applied to *raw logits* (before log_softmax) so that the
    positive/negative sign distinction in the formula is meaningful.

    Args:
        logits:      Raw model logits, shape [bs, d_vocab].
        context_ids: Token IDs in the penalty window, shape [bs, ctx_len].
        penalty:     Penalty factor (> 1.0). 1.0 = no-op.

    Returns:
        Penalized logits, same shape and dtype as input.
    """
    if penalty == 1.0 or context_ids.size(-1) == 0:
        return logits

    logits = logits.clone()
    vocab_size = logits.size(-1)
    for b in range(logits.size(0)):
        ctx = context_ids[b]
        # Defensive guard: keep only valid token IDs in [0, vocab_size).
        # Current callers already ensure clean slices, so this is a safety net.
        ctx = ctx[(ctx >= 0) & (ctx < vocab_size)]
        if ctx.numel() == 0:
            continue

        # Count how many times each token ID appears in the window.
        counts = torch.bincount(ctx, minlength=vocab_size)  # shape [vocab_size]

        # Only penalize tokens that appear MORE THAN ONCE (genuine repetition).
        repeated_ids = counts.nonzero(as_tuple=False).squeeze(-1)
        repeated_ids = repeated_ids[counts[repeated_ids] > 1]

        if repeated_ids.numel() == 0:
            continue

        score = logits[b, repeated_ids]
        logits[b, repeated_ids] = torch.where(
            score > 0, score / penalty, score * penalty
        )
    return logits


def apply_repetition_penalty_batched(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    confirmed_len: int,
    draft_tokens: torch.Tensor,
    penalty: float,
    window: int,
) -> torch.Tensor:
    """Vectorized repetition penalty for all verification positions at once.

    Instead of calling apply_repetition_penalty in a Python loop for each
    verification position j in [0, n_draft], this function:
      1. Computes bincount once for the shared confirmed-token window.
      2. Incrementally accumulates draft token counts to build a
         (n_positions, vocab_size) counts matrix in one forward pass.
      3. Applies the penalty to all positions simultaneously.

    Position j's context = confirmed_tokens[-W:] ++ draft_tokens[:j].
    Position n_draft (bonus) uses confirmed_tokens + all draft_tokens.

    Args:
        logits:           Raw target logits, shape [bs, n_draft+1, d_vocab].
        generated_tokens: Full generated token buffer.
        confirmed_len:    Number of confirmed tokens in generated_tokens.
        draft_tokens:     Draft tokens from this iteration, shape [bs, n_draft].
        penalty:          Penalty factor (> 1.0).
        window:           Repetition penalty window size.

    Returns:
        Penalized logits, same shape as input.
    """
    if penalty == 1.0:
        return logits

    bs, n_positions, vocab_size = logits.shape
    n_draft = draft_tokens.size(-1)  # n_positions = n_draft + 1
    logits = logits.clone()

    for b in range(bs):
        # Step 1: Base counts from confirmed tokens
        # This is the shared prefix for ALL verification positions.
        confirmed = generated_tokens[b, :confirmed_len]
        # The full window at position n_draft (bonus) is:
        #   confirmed[-max(0, W - n_draft):] ++ draft_tokens[b, :n_draft]
        # At position 0, no draft tokens → context is just confirmed[-W:].
        # We compute the base counts over the confirmed portion that is
        # within the window even at the widest position (the bonus token).
        confirmed_in_window = confirmed[-window:] if window > 0 else confirmed
        base_counts = torch.bincount(confirmed_in_window, minlength=vocab_size)

        # Step 2: Build incremental counts matrix
        # all_counts[j] = counts for position j's full context window.
        # Position 0: only confirmed tokens (windowed).
        # Position j: confirmed tokens + draft_tokens[:j] (windowed).
        #
        # As j increases, we add one more draft token. But also, if the window
        # is smaller than confirmed_len + j, older confirmed tokens slide out.
        all_counts = base_counts.unsqueeze(0).expand(n_positions, -1).clone()  # (n_pos, V)

        for j in range(1, n_positions):
            # Add draft token j-1 to position j's context
            dt = draft_tokens[b, j - 1].item()
            if 0 <= dt < vocab_size:
                all_counts[j, dt] += 1
                # Also propagate to all subsequent positions
                # (positions j+1..n_draft all include this draft token)
                if j + 1 < n_positions:
                    all_counts[j + 1:, dt] += 1

            # If adding draft tokens pushes us past the window, remove the
            # oldest confirmed token that just slid out of the window.
            tokens_in_ctx = confirmed_in_window.size(0) + j
            if tokens_in_ctx > window and window > 0:
                # Absolute index of the token sliding out (0-indexed relative to confirmed sequence start)
                slide_out_abs_idx = confirmed_len - window + j - 1
                
                if slide_out_abs_idx < confirmed_len:
                    # Token sliding out was originally from the confirmed prefix
                    if slide_out_abs_idx >= 0:
                        old_token = generated_tokens[b, slide_out_abs_idx].item()
                        if 0 <= old_token < vocab_size:
                            all_counts[j:, old_token] = torch.clamp(
                                all_counts[j:, old_token] - 1, min=0
                            )
                else:
                    # Token sliding out was from the draft tokens we added earlier this iteration
                    draft_idx = slide_out_abs_idx - confirmed_len
                    if 0 <= draft_idx < n_draft:
                        old_token = draft_tokens[b, draft_idx].item()
                        if 0 <= old_token < vocab_size:
                            all_counts[j:, old_token] = torch.clamp(
                                all_counts[j:, old_token] - 1, min=0
                            )

        # Step 3: Find repeated tokens and apply penalty
        # A token is "repeated" if its count >= 2 at that position.
        repeated_mask = all_counts > 1  # (n_positions, vocab_size)

        if not repeated_mask.any():
            continue

        # Apply penalty: positive logits /= penalty, negative logits *= penalty
        pos_logits = logits[b]  # (n_positions, vocab_size)
        logits[b] = torch.where(
            repeated_mask & (pos_logits > 0),
            pos_logits / penalty,
            torch.where(
                repeated_mask & (pos_logits <= 0),
                pos_logits * penalty,
                pos_logits,
            ),
        )

    return logits
