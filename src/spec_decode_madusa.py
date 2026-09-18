"""
Speculative decoding implementation.

Contains:
- speculative_decode_greedy: Custom greedy speculative decoding with KV caching
- get_stop_token_ids: Stop token detection for various chat models
- crop_kv_cache: KV cache management utility
"""

import time
from typing import Literal, cast

import torch
import torch.nn as nn

from src.models.madusa import madusa


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


def crop_kv_cache(past_key_values, new_length, best_path, max_accept_len):
    """
    Crop KV cache to a specific sequence length.
    Handles both DynamicCache objects and tuple format.
    """
    if past_key_values is None:
        return None

    accepted_tree_idx = []
    for i in range(max_accept_len):
        node_idx = best_path[i]
        accepted_tree_idx.append(past_key_values+node_idx)

    keep_idx = torch.concat(range(new_length), accepted_tree_idx)
    if hasattr(past_key_values, "select_indices") or hasattr(past_key_values, "select_idx"):
        return past_key_values.select_index(keep_idx)
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

def default_tree(preset_type:str):
    if preset_type.lower() not in ["lightweight", "greedy_linear", "standard"]:
        raise ValueError("Must be one of the following options: ['lightweight', 'greedy_linear', 'standard']")
    if preset_type.lower() == "lightweight":
        """
        16-node tree focused on high-probability top-1/top-2 branches
        Reduces significantly matrix multiplication sizes and tree-attention mask generation overhead during the verification step.
        Ideal for memory-constrained devices or edge deployment.
        """
        return [
            [0], [0, 0], [0, 0, 0], [0, 0, 0, 0],
            [1], [0, 1], [1, 0], [0, 0, 1],
            [2], [0, 2], [2, 0], [0, 1, 0],
            [3], [0, 0, 2], [1, 1], [0, 0, 0, 1]
        ]
    elif preset_type.lower() == "greedy_linear":
        """
        Minimal single-path execution without branching
        Simplest memory footprint; eliminates complex 2D branching logic and minimizes KV
        cache slicing operations.
        """
        return [
            [0],
            [0, 0],
            [0, 0, 0],
            [0, 0, 0, 0]
        ]
    elif preset_type.lower() == "standard":
        """
        Standard 64-Node Tree
        Maximizes the expected token acceptance per iteration rate; explores a diverse range
        of branches across up to 4 Medusa heads.
        """
        return[
            [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3],
            [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0],
            [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0],
            [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1],
            [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0],
            [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0],
            [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6],
            [0, 7, 0]
        ]

def build_tree(
    logits,
    tree_choice,
    cur_gen_idx,
    past_kv_len,
    top_k,
    top_p,
    mode
):
    num_heads = logits.size(dim=1)
    top_token_per_head = []
    for i in range(num_heads):
        max_rank = max(tree_choice[i])
        head_logits = logits[0,i,:]
        filter_logits = filter_logprobs(
            nn.LogSoftmax(head_logits, dim=1),
            top_k=top_k,
            top_p=top_p
        )
        top_ids = torch.topk(logits[0, i], k=max_rank + 1).indices
        top_token_per_head.append(top_ids)

    nodes = []
    node_dict = {}
    paths = []

    for path in tree_choice:
        node_path = []
        for depth in range(len(path)):
            rank = path[depth]
            token_id = top_token_per_head[depth][rank]
            prefix = (depth, rank, token_id)

            if prefix not in node_dict:
                new_node_idx = len(nodes)
                node_dict[prefix] = new_node_idx

                parent_prefix = prefix[:-1]
                parent_idx = node_dict[parent_prefix] if len(parent_prefix)>0 else None

                nodes.append(
                    {
                        "node_idx": new_node_idx,
                        "token_id": token_id,
                        "depth": depth,
                        "parent_idx": parent_idx
                    }
                )
                node_path.append(node_dict[parent_prefix])
    size = len(nodes)

    draft_tree_tokens = torch.zeros(1, size)
    pos_idx = torch.zeros(1, size)

    for idx in range(size):
        draft_tree_tokens[0,idx] = nodes[idx]["token_id"]
        pos_idx[0, idx] = cur_gen_idx + nodes[idx]["depth"]

    attn_mask = torch.full(
        size=(size, past_kv_len + size),
        fill_value=-float('inf')
    )
    attn_mask[:, :, :, :past_kv_len] = 0.0

    for i in range(size):
        cur_node = nodes[i]
        while cur_node is not None:
            attn_mask[0, 0, i, past_kv_len+cur_node["node_idx"]] = 0.0
            cur_node = nodes[cur_node["parent_idx"]] if cur_node["parent_idx"] is not None else None

    return {
        "tokens": draft_tree_tokens,
        "attention": attn_mask,
        "pos_idx": pos_idx,
        "paths": paths,
        "nodes": nodes
    }

def speculative_decode(
    target_model,
    tokenizer,
    input_ids,
    mode:Literal["greedy", "sample"],
    madusa:nn.Module,
    max_new_tokens=128,
    tree_choices:str|list[list]="Standard",
    top_k=0,
    top_p=0.0,
    repetition_penalty=1.1,
    repetition_penalty_window=16,
    eos_token_id=None,
    device=None,
    track_iterations=False
):
    """
    Medusa Speculative Decoding with KV Caching.
    Key features:

    Args:
        target_model: The large target model
        medusa_heads: The amount of heads for trees
        tokenizer: tokenizer from taraget model
        input_ids: Input token IDs [1, seq_len]
        input_ids: Input token IDs [1, seq_len]
        mode: 'greedy' | 'sample'
        max_new_tokens: Maximum new tokens to generate
        tree_choices: What approach of tree standard used
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

    if isinstance(tree_choices, str):
        tree_choices = default_tree(tree_choices)

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

    stop_token_ids = torch.tensor(
        list(get_stop_token_ids(tokenizer, eos_token_id)), device=device
    )
    input_ids = input_ids.to(device)

    # This is okay because if we've gotten this far, we know the actual tokenizers are the same length.
    # Just be aware that logits may have a slightly shorter dimension
    d_vocab = max(madusa.vocab_size, target_model.config.vocab_size)

    # B,S+max_new
    generated_tokens = torch.concat(
        [
            input_ids,
            torch.zeros(
                bs, max_new_tokens, device=device, dtype=torch.int64
            ),
        ],
        dim=-1,
    )
    prompt_len = input_ids.size(-1)
    cur_gen_idx = input_ids.size(-1)

    # Track average time for draft and verifier forward pass for speedup factor
    # Each accumulator: (sum_of_times, sum_of_squared_times, count)
    draft_start,draft_end,verifier_start, verifier_end = None, None, None, None
    draft_times_acc = (0., 0., 0)
    verifier_times_acc = (0., 0., 0)
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
        target_out = target_model(input_ids, use_cache=True, output_hidden_states=True)
        target_kv_cache = target_out.past_key_values
        last_hidden = target_out.hidden_states[-1][:,-1:, :] #shape [bs, 1, hidden_dim]

        # Add the first new token.
        # Penalty context: the last W tokens of the prompt (no generated tokens yet).
        first_logits = penalize_logits(target_out.logits[:, -1, :], confirmed_len=cur_gen_idx)
        first_target_token = select_index(
            apply_filters(torch.log_softmax(first_logits, dim=-1))
        )
        generated_tokens[:, cur_gen_idx] = first_target_token
        cur_gen_idx += 1

        prev_target_logits = first_logits

        # Metrics
        total_draft_tokens = 0
        total_matched_tokens = 0
        # Per-position acceptance for the octiles (eg 16, 32, ..., 128 if we use max_tokens=128)
        # These are offsets after the prompt length, not absolute indices
        octile_offsets = [i * (max_new_tokens // 8) + 1 for i in range(8)]
        per_position_accept_count = [0] * 8
        per_position_draft_count = [0] * 8
        num_iterations = 0
        iteration_history = []
        start_time = get_time()

        while cur_gen_idx < generated_tokens.size(-1):
            num_iterations += 1
            past_kv_len = get_kv_cache_length(target_kv_cache)
            medusa_logits = madusa(last_hidden)

            # Step 1: parallel draft candidate tree generation via the medusa heads
            tree_data = build_tree(
                logits=medusa_logits,
                cur_gen_idx=cur_gen_idx,
                tree_choice=tree_choices,
                past_kv_len=past_kv_len,
                top_k=top_k,
                top_p=top_p,
                mode=mode
            )
            draft_tree_tokens = tree_data["tokens"]
            tree_atten_mask = tree_data["attention"]
            tree_pos_id = tree_data["pos_idx"]
            tree_paths = tree_data["paths"]
            tree_nodes = tree_data["nodes"]

            total_draft_tokens += draft_tree_tokens.size(-1)

            # Step 2: Target Model Parallel Verification Pass over Candidate Tree
            target_out = target_model(
                input_ids=draft_tree_tokens,
                past_kv_values=target_kv_cache,
                attention_mask=tree_atten_mask,
                position_ids=tree_pos_id,
                use_cache=True,
                output_hidden_states=True
            )
            verify_logits = target_out.logits

            # step 3: evaluate candidate of tree paths to find the longest valid branch
            best_path = None
            best_accepted_token = []
            best_bonus_token = None
            best_bonus_logits = None
            max_accept_len = -1

            for path in tree_paths:
                accepted_in_path = []
                bonus_token = None
                bonus_logits = None
                path_matched = True

                for depth in range(len(path)):
                    node_i = path[depth]
                    draft_token = draft_tree_tokens[:, node_i]

                    if depth == 0:
                        raw_pred_logits = prev_target_logits
                    else:
                        parent_node_idx = tree_nodes[node_i]["parent_idx"]
                        raw_pred_logits = verify_logits[:, parent_node_idx, :]
                    node_raw_logits = penalize_logits(
                        raw_pred_logits,
                        confirmed_len=cur_gen_idx+depth
                    )
                    target_dist = apply_filters(nn.LogSoftmax(node_raw_logits, dim=-1))
                    verified_token = select_index(target_dist)

                    if draft_token == verified_token:
                        accepted_in_path.append(draft_token)
                    else:
                        bonus_token = verified_token
                        bonus_logits = node_raw_logits
                        path_matched = False
                        break

                if path_matched and bonus_token is None:
                    last_node_idx = path[-1]
                    bonnus_raw_logits = penalize_logits(
                        verify_logits[:, last_node_idx, :],
                        confirmed_len=cur_gen_idx+len(path),
                        draft_so_far=generated_tokens
                    )
                    bonus_token = select_index(
                        apply_filters(
                            nn.LogSoftmax(bonnus_raw_logits, dim=-1)
                        )
                    )
                    bonus_logits = bonnus_raw_logits

                if len(accepted_in_path) > max_accept_len:
                    max_accept_len = len(accepted_in_path)
                    best_accepted_token = accepted_in_path
                    best_bonus_token = bonus_token
                    best_bonus_logits = bonus_logits
                    best_path = path
            # step 4: updating octile acceptance counters
            gen_offset = cur_gen_idx - prompt_len
            draft_depth = len(best_path) if best_path is not None else 0

            for i in range(7):
                checkpoint = octile_offsets[i]
                if gen_offset <= checkpoint and checkpoint < (gen_offset + draft_depth):
                    per_position_draft_count[i] = per_position_draft_count[i]+1
                    rel_depth = checkpoint - gen_offset
                    if rel_depth == checkpoint - gen_offset:
                        per_position_accept_count[i] = per_position_accept_count[i]+1

            # step 5: commit accepted tokens and update seq len
            tokens_to_add = torch.concat((best_accepted_token, best_bonus_token))
            new_gen_idx = cur_gen_idx + tokens_to_add.size(dim=-1)
            generated_tokens[:, cur_gen_idx:new_gen_idx] = tokens_to_add

            total_matched_tokens += len(best_accepted_token)

            # step 6: prune any unused tree kv cache and extract hidden state for next medusa pass
            target_kv_cache = crop_kv_cache(
                target_out.past_key_values,
                new_gen_idx-1,
                best_path,
                max_accept_len
            )
            last_hidden = (target_out.hidden_States[-1])[:, best_path[max_accept_len-1]:best_path[max_accept_len+1, :]] if max_accept_len > 0
            prev_target_logits = best_bonus_logits

            cur_gen_idx = best_bonus_logits

            if generated_tokens[:, cur_gen_idx] in stop_token_ids:
                generated_tokens = generated_tokens[:, :cur_gen_idx]
                break

    total_time = get_time()- start_time
    acceptance_rate = total_matched_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0

    octile_position_acceptance = [
        acc / draf if draf > 0 else None for acc, draf in zip(per_position_accept_count, per_position_draft_count)
    ]

    metrics = {
        "time": total_time,
        "generated_tokens": cur_gen_idx-prompt_len,
        "draft_tokens": total_draft_tokens,
        "matched_tokens": total_matched_tokens,
        "acceptance_rate": acceptance_rate,
        "num_iterations"
        "octile_position_acceptance": octile_position_acceptance,
        "octile_positions": octile_offsets,
        "num_iterations": num_iterations,
        "toks_per_sec": (cur_gen_idx-prompt_len) / total_time if total_time > 0 else 0,
    }

    # Forward pass times for speedup factor
    if draft_times_acc[2] > 0 and verifier_times_acc[2] > 0:
        v_sum, v_sum_sq, v_n = verifier_times_acc
        average_verifier_time = v_sum / v_n        # ms
        metrics["average_verifier_time"] = average_verifier_time / 1000
        # Variance of individual forward pass times (population variance, in ms^2)
        raw_verifier_variance = v_sum_sq / v_n - average_verifier_time**2
        metrics["verifier_time_variance"] = max(raw_verifier_variance, 0.0) / 1e6  # s^2
        metrics["verifier_time_count"] = v_n

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
    """Apply multiplicative repetition penalty to raw logits (single position)."""
    if penalty == 1.0 or context_ids.size(-1) == 0:
        return logits

    for b in range(logits.size(0)):
        # window_counts = torch.nn.functional.one_hot(context_ids[b]).sum(dim=-2)
        max_vocab_index = torch.max(context_ids[b]).item() + 1
        max_vocab_index = cast(int, max_vocab_index)
        window_counts = torch.zeros(
            max_vocab_index,
            dtype=torch.long,
            device=context_ids[b].device
        )
        window_counts.scatter_add_(
            dim=0,
            index=context_ids[b],
            src=torch.ones_like(context_ids[b])
        )

        per_token_penalty = penalty ** window_counts
        logits[b,:max_vocab_index] = torch.where(
            logits[b,:max_vocab_index] > 0,
            logits[b,:max_vocab_index] / per_token_penalty,
            logits[b,:max_vocab_index] * per_token_penalty,
        )
    return logits


def apply_repetition_penalty_batched(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    confirmed_len: int,
    penalty: float,
    window: int,
) -> torch.Tensor:
    """Vectorized repetition penalty for all verification positions at once.

    Position j's context = generated_tokens[j-window:j]
    """
    if penalty == 1.0:
        return logits

    bs, seq_len = generated_tokens.shape
    device = generated_tokens.device

    for b in range(bs):
        # Only positions before the current pos
        mask = ~torch.triu(torch.ones(seq_len + 1, seq_len, dtype=torch.bool, device=device))

        # Only positions after the start of the window
        window_start = torch.clamp(
            torch.arange(
                seq_len + 1,
                device=device
            ) - window,
            min=0
        ).unsqueeze(-1)
        start_mask = torch.arange(seq_len, device=device) >= window_start
        mask *= start_mask

        # Replace masked positions with an unused index (hack to avoid using 0)
        unused_idx = torch.max(generated_tokens).item() + 1
        unused_idx = cast(int, unused_idx)
        window_tokens = generated_tokens[b].expand(seq_len + 1, seq_len).masked_fill(~mask, unused_idx)

        # Old way, OOM:
        # window_counts = torch.nn.functional.one_hot(window_tokens).sum(dim=1)[...,:-1] # cut off the unused one

        window_counts = torch.zeros(window_tokens.size(0), unused_idx + 1, dtype=torch.long, device=window_tokens.device)
        window_counts.scatter_add_(dim=1, index=window_tokens, src=torch.ones_like(window_tokens))
        window_counts  = window_counts[...,:-1] # cut off the unused vocab item

        per_token_penalty = penalty ** window_counts
        per_token_penalty = per_token_penalty[confirmed_len:]

        logits[b,:,:unused_idx] = torch.where(
            logits[b,:,:unused_idx] > 0,
            logits[b,:,:unused_idx] / per_token_penalty,
            logits[b,:,:unused_idx] * per_token_penalty,
        )

    return logits
