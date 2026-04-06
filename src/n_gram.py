import logging
from collections import defaultdict
from typing import cast

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


class NGramModel:
    def __init__(self, n: int, tokenizer: PreTrainedTokenizer):
        """
        takes tokenizer and checks if it is a huggingface and takes the tokenizer from it
        n is the number of gram.
        """
        self.n = n
        self.tokenizer = tokenizer

        # gram_freq[order][(context_tuple)][next_token] = count
        self.gram_freq: dict[int, dict[tuple[int, ...], dict[int, int]]] = {
            order: defaultdict(lambda: defaultdict(int)) for order in range(1, n + 1)
        }
        # conditional_probs[order][(context_tuple)][next_token] = probability
        self.conditional_probs: dict[int, dict[tuple[int, ...], dict[int, float]]] = {}

    def train(self, train: Dataset):
        """Learn n-gram counts for all orders 1..n, then compute conditional probabilities."""
        for sentence in train["text"]:
            token_ids: list[int] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(sentence)
            )  # type:ignore
            for order in range(1, self.n + 1):
                ctx_len = order - 1
                for idx in range(len(token_ids) - order + 1):
                    context = tuple(token_ids[idx : idx + ctx_len])
                    target = token_ids[idx + ctx_len]
                    self.gram_freq[order][context][target] += 1

        total_unique = sum(
            len(targets) for order_freqs in self.gram_freq.values()
            for targets in order_freqs.values()
        )
        for order in range(1, self.n + 1):
            self.conditional_probs[order] = {}
            for context_key, token_freqs in self.gram_freq[order].items():
                marginal_sum = sum(token_freqs.values())
                self.conditional_probs[order][context_key] = {
                    k: freq / marginal_sum for k, freq in token_freqs.items()
                }
        logger.info(
            f"Trained {self.n}-gram model with {total_unique} unique entries across all orders"
        )

    def predict(self, tokens: list[int] | str):
        """Predict next token with Katz-style backoff from order n down to 1.
        Returns a (vocab_size,) tensor of normalized probabilities."""
        if isinstance(tokens, str):
            tokens = cast(list[int], self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tokens)))

        vocab_size = len(self.tokenizer)
        probabilities = torch.zeros(vocab_size)
        mass_left = 1.0

        for order in range(self.n, 0, -1):
            ctx_len = order - 1
            if len(tokens) < ctx_len:
                continue
            context_key = tuple(tokens[-ctx_len:]) if ctx_len > 0 else ()
            cond = self.conditional_probs.get(order, {}).get(context_key)
            if cond:
                for token_id, prob in cond.items():
                    if probabilities[token_id] == 0:
                        probabilities[token_id] = prob * mass_left
                mass_left *= 0.4

        total = probabilities.sum()
        if total.item() == 0:
            return torch.full((vocab_size,), 1 / vocab_size)
        return probabilities / total

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[torch.Tensor] | None = None,
        use_cache: bool = True,
    ):
        """This is an adapter method that allows for duck typing in spec_decode.py.
            - Inputs and outputs should match the forward method of an AutoModelForCausalLM.
            - We do a sneaky trick where the "kv cache" is a (1, seq_length) tensor of token IDs
        """
        assert input_ids.shape[0] == 1 and len(input_ids.shape) == 2
        if past_key_values is not None:
            assert len(past_key_values) == 1 and (past_key_values[0].shape[0] == 1)
            full_seq = torch.concat([past_key_values[0], input_ids], dim=-1).to(input_ids.device)
        else:
            full_seq = input_ids
        logits = self.predict(full_seq[0].tolist()).to(input_ids.device)
        logits = logits.unsqueeze(0).unsqueeze(0) # (batch_size, seq_length, d_vocab)
        return CausalLMOutputWithPast(logits=logits, past_key_values=(full_seq,))  # type:ignore
