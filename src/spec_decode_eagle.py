import time
from typing import Literal, cast

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

def sliding_window(tokens, window):
    if window <= 0 or len(tokens) <= window:
        return tokens
    return tokens[-window:]

def any_in_set(tokens, target_set):
    for token in tokens:
        if token in target_set:
            return True
    return False


def extract_token_ids(tree_nodes):
    return torch.tensor([node.token_id for node in tree_nodes])

def extract_all_leaf_paths(tree_nodes):
    leaf_paths = []
    for node in tree_nodes:
        if len(node.children) == 0:
            path = []
            curr_node = node
            while curr_node is not None:
                path.append(curr_node)
                curr_node = curr_node.parent
            leaf_paths.append(path)
    return leaf_paths

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

def crop_align_eagle_kv_cache(kv_cache, accepted_path_nodes, base_seq_len):
    if kv_cache is None:
        return None

    accepted_idxs = [node.index for node in accepted_path_nodes]
    updated_kv_cache = []

    for layer_kv in kv_cache:
        if isinstance(layer_kv, torch.Tensor):
            base_state = layer_kv[..., :base_seq_len]
            path_state = layer_kv[..., base_seq_len+accepted_idxs]
            new_state = torch.concat(
                base_state,
                path_state,
                dim=-1
            )
        elif len(layer_kv) == 2:
            key_state, value_states = layer_kv

            base_keys = key_state[..., :base_seq_len, :]
            base_values = value_states[..., :base_seq_len, :]
            path_keys = key_state[..., base_seq_len+accepted_idxs, :]
            path_values = value_states[..., base_seq_len+accepted_idxs, :]

            new_keys = torch.concat(
                base_keys,
                path_keys,
                dim=-2
            )
            new_values = torch.concat(
                base_values,
                path_values,
                dim=-2
            )

            updated_kv_cache.append(
                (new_keys, new_values)
            )
    return tuple(updated_kv_cache)

def get_node_depth(node):
    depth = 0
    curr = node
    while curr.parent is not None:
        depth += 1
        curr = curr.parent
    return depth

def get_ancestor_tokens(node):
    tokens = []
    curr_node = node
    while curr_node is not None:
        tokens.append(curr_node.token_id)
        curr_node = curr_node.parent
    return tokens

def build_eagle_tree_attn(tree_nodes, base_seq_len):
    num_nodes = len(tree_nodes)
    attn_mask = [[float('-inf')] * num_nodes for _ in range(num_nodes)]
    pos_ids = [None] * num_nodes

    for i in range(num_nodes):
        node_i = tree_nodes[i]
        try:
            depth = node_i.depth
        except Exception:
            depth = get_node_depth(node=node_i)

        pos_ids[i] = base_seq_len + depth

        cur_ancestor = node_i
        while cur_ancestor is not None:
            attn_mask[i][cur_ancestor.index] = 0.0
            cur_ancestor = cur_ancestor.parent

    return attn_mask, pos_ids

class Node:
    def __init__(self, index, parent, token_id, hidden_state, depth, score):
        self.index = index
        self.parent = parent
        self.token_id = token_id
        self.hidden_state = hidden_state
        self.depth = depth
        self.score = score
        self.children = []

        if parent is not None:
            parent.children.append(self)

def create_node(index, parent, token_id, hidden_state, depth, score):
    return Node(index, parent, token_id, hidden_state, depth, score)

def eagle_draft_tree_expansion(
    eagle_module,
    root_hidden,
    root_token_id,
    tree_choices,
    top_k,
    top_p,
    repetition_penalty,
    repetition_penalty_window,
    generated_tokens,
    cur_gen_idx,
    mode
):
    tree_nodes = []

    root_node = create_node(
        index=0,
        parent=None,
        token_id=root_token_id,
        hidden_state=root_hidden,
        depth=0,
        score=1.0
    )

    tree_nodes.append(root_node)
    active_parents = [root_node]

    for depth in range(tree_choices):
        branch_factor = tree_choices[depth]
        next_parents = []

        for parent in active_parents:
            token_emb = eagle_module.get_embedding(parent.token_id)
            fused_feat = torch.concat(token_emb, parent.hidden_state, dim=-1)
            next_hidden = eagle_module.decoder_layer(fused_feat)
            draft_raw_logits = eagle_module.lm_head(next_hidden)

            path_tokens = get_ancestor_tokens(parent)
            full_context = torch.concat(
                generated_tokens[0, :cur_gen_idx],
                path_tokens
            )
            penalty_context = sliding_window(
                full_context,
                window=repetition_penalty_window
            )

            penalized_logits = apply_repetition_penalty(
                draft_raw_logits,
                penalty_context,
                repetition_penalty
            )
            draft_logorbs = filter_logprobs(
                torch.log_softmax(penalized_logits),
                top_k=top_k,
                top_p=top_p
            )

            top_scores, top_token_ids = top_k_sampling(
                draft_logorbs,
                k=branch_factor
            )

            for k_idx in range(branch_factor):
                child_node = create_node(
                    index=len(tree_nodes),
                    parent=parent,
                    token_id=top_token_ids[k_idx],
                    hidden_state=next_hidden,
                    depth=depth,
                    score=parent.score * torch.exp(top_scores[k_idx])
                )
                tree_nodes.append(child_node)
                next_parents.append(child_node)
        active_parents = next_parents
    return tree_nodes

def spec_decode_eagle(
    target_model,
    eagle_module,
    tokenizer,
    input_ids: torch.Tensor,
    mode: Literal["greedy", "sample"],
    max_new_tokens: int = 128,
    tree_choices:list[int]=[1,4,2,2],
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.1,
    repetition_penalty_window: int = 16,
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
        ctx = generated_tokens[0, max(
            0, cur_gen_idx-repetition_penalty_window
        ):cur_gen_idx]
        if draft_so_far is not None and draft_so_far.size(-1) > 0:
            ctx = torch.cat([ctx, draft_so_far], dim=-1)
        # Slide to the last `repetition_penalty_window` tokens
        ctx = ctx[:, -repetition_penalty_window:]
        return apply_repetition_penalty(logits, ctx, repetition_penalty)

    stop_token_ids = torch.tensor(
        list(get_stop_token_ids(tokenizer, eos_token_id)), device=device
    )
    input_ids = input_ids.to(device)

    generated_tokens = torch.concat(
        [
            input_ids,
            torch.zeros(
                input_ids.size(0), max_new_tokens, device=device, dtype=torch.int64
            ),
        ],
        dim=-1,
    )

    prompt_len = input_ids.sie(dim=-1)
    cur_gen_idx = prompt_len

    total_draft_tokens = 0
    total_matched_tokens = 0
    num_iterations = 0

    max_depth = len(tree_choices)
    per_position_draft_count = [0] * max_depth
    per_position_accept_count = [0] * max_depth
    octile_offsets = list(range(1, max_depth + 1))

    draft_times_acc = [0.0, 0.0, 0]     # [sum_ms, sum_sq_ms, count]
    verifier_times_acc = [0.0, 0.0, 0]  # [sum_ms, sum_sq_ms, count]
    iteration_history = []

    def get_time():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.time()

    with torch.no_grad():
        start_time = get_time()

        # Preload kv cache for prompts
        target_out = target_model(input_ids, use_cache=True)
        target_kv_cache = target_out.past_key_values

        last_hidden = target_out.last_hidden_state[:, -1, :]
        first_logits = penalize_logits(target_out.logits[:, -1, :], confirmed_len=cur_gen_idx)
        first_logprobs = filter_logprobs(
            torch.log_softmax(first_logits),
            top_k,
            top_p
        )
        first_token = sample(first_logprobs, mode)

        generated_tokens[:, cur_gen_idx] = first_token
        cur_gen_idx += 1

        while cur_gen_idx < generated_tokens.size(-1):
            num_iterations += 1
            base_seq_len = cur_gen_idx
            curr_token = generated_tokens[:, cur_gen_idx]

            t_d0 = get_time()
            tree_nodes = eagle_draft_tree_expansion(
                eagle_module=eagle_module,
                root_hidden=last_hidden,
                root_token_id=curr_token,
                tree_choices=tree_choices,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_penalty_window=repetition_penalty_window,
                generated_tokens=generated_tokens,
                cur_gen_idx=cur_gen_idx,
                mode=mode
            )
            t_d1 = get_time()

            d_ms = (t_d1 - t_d0) * 1000.0
            draft_times_acc[0] += d_ms
            draft_times_acc[1] += d_ms ** 2
            draft_times_acc[2] += 1

            for node in tree_nodes[1:]:
                if 1 <= node.depth <= max_depth:
                    per_position_draft_count[node.depth - 1] += 1

            flattened_tokens = extract_token_ids(tree_nodes=tree_nodes)
            tree_attn_mask, tree_pos_ids = build_eagle_tree_attn(
                tree_nodes=tree_nodes,
                base_seq_len=base_seq_len
            )

            target_out = target_model(
                input_ids=flattened_tokens,
                attention_mask=tree_attn_mask,
                pos_ids=tree_pos_ids,
                past_key_values=target_kv_cache,
                use_cache=True,
                output_hidden_states=True
            )

            target_raw_logits = target_out.logits
            verifier_hiddens = target_out.last_hidden_state

            penalize_target_logits = apply_repetition_penalty_batched(
                logits=target_raw_logits,
                generated_tokens=torch.concat(
                    generated_tokens[:, :cur_gen_idx],
                    flattened_tokens
                ),
                confirmed_len=cur_gen_idx,
                penalty=repetition_penalty,
                window=repetition_penalty_window
            )
            target_logprobs = filter_logprobs(
                torch.log_softmax(
                    penalize_target_logits
                ),
                top_k=top_k,
                top_p=top_p
            )

            candidate_paths = extract_all_leaf_paths(tree_nodes=tree_nodes)
            best_path = []
            best_accept_count = -1
            bonus_token = None
            new_last_hidden = None

            for path in candidate_paths:
                accepted_in_paths = []
                for step in range(len(path)-1):
                    parent_node = path[step]
                    child_node = path[step+1]

                    pred_token = sample(
                        target_logprobs[:, parent_node.index,:],
                        mode
                    )

                    if pred_token == child_node.token_id:
                        accepted_in_paths.append(child_node)
                    else:
                        break
                if len(accepted_in_paths) > best_accept_count:
                    best_accept_count = len(accepted_in_paths)
                    best_path = accepted_in_paths

                    last_accepted_node = accepted_in_paths[-1] if len(accepted_in_paths) else path[0]
                    bonus_token = sample(
                        target_logprobs[:, last_accepted_node.index, :],
                        mode
                    )
                    new_last_hidden = verifier_hiddens[:, last_accepted_node.index, :]
            # Record accepted position statistics
            for node in best_path:
                if 1 <= node.depth <= max_depth:
                    per_position_accept_count[node.depth - 1] += 1

            accepted_tokens = [
                node.token_id for node in best_path
            ]
            tokens_to_add = torch.concat(
                accepted_tokens,
                [bonus_token]
            )

            new_gen_idx = cur_gen_idx + len(tokens_to_add)
            generated_tokens[:, cur_gen_idx:new_gen_idx] = tokens_to_add
            cur_gen_idx = new_gen_idx

            total_matched_tokens = total_matched_tokens + best_accept_count
            total_draft_tokens = total_draft_tokens + len(tree_nodes) - 1

            target_kv_cache = crop_align_eagle_kv_cache(
                kv_cache=target_out.past_key_values,
                accepted_path_nodes=best_path,
                base_seq_len=base_seq_len
            )
            last_hidden = new_last_hidden

            if tokens_to_add in stop_token_ids:
                generated_tokens = generated_tokens[:, :cur_gen_idx]
                break

    total_time = get_time() - start_time
    acceptance_rate = total_matched_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    total_generated_tokens = cur_gen_idx - prompt_len

    octile_position_acceptance = [
        acc / draf if draf > 0 else None
        for acc, draf in zip(per_position_accept_count, per_position_draft_count)
    ]

    metrics = {
        "time": total_time,
        "generated_tokens": total_generated_tokens,
        "draft_tokens": total_draft_tokens,
        "matched_tokens": total_matched_tokens,
        "acceptance_rate": acceptance_rate,
        "octile_position_acceptance": octile_position_acceptance,
        "octile_positions": octile_offsets,
        "num_iterations": num_iterations,
        "toks_per_sec": total_generated_tokens / total_time if total_time > 0 else 0,
    }

    if draft_times_acc[2] > 0 and verifier_times_acc[2] > 0:
        d_sum, d_sum_sq, d_n = draft_times_acc
        v_sum, v_sum_sq, v_n = verifier_times_acc

        avg_draft = d_sum / d_n
        avg_verifier = v_sum / v_n

        metrics["average_draft_time"] = avg_draft / 1000.0
        metrics["average_verifier_time"] = avg_verifier / 1000.0

        raw_draft_var = (d_sum_sq / d_n) - (avg_draft ** 2)
        raw_verifier_var = (v_sum_sq / v_n) - (avg_verifier ** 2)

        metrics["draft_time_variance"] = max(raw_draft_var, 0.0) / 1e6
        metrics["verifier_time_variance"] = max(raw_verifier_var, 0.0) / 1e6
        metrics["draft_time_count"] = d_n
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

def top_k_sampling(logprods, k):
    sorted_logprobs, sorted_idxs = torch.sort(logprods)

    top_scores = sorted_logprobs[:k]
    top_token_ids = sorted_idxs[:k]
    return top_scores, top_token_ids

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
        window_counts = torch.zeros(max_vocab_index, dtype=torch.long, device=context_ids[b].device)
        window_counts.scatter_add_(dim=0, index=context_ids[b], src=torch.ones_like(context_ids[b]))

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
        window_start = torch.clamp(torch.arange(seq_len + 1, device=device) - window, 0).unsqueeze(-1)
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
