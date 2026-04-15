import logging
from collections import defaultdict
from types import SimpleNamespace
from typing import cast

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


class NGramModel:
    def __init__(self, n: int, tokenizer: PreTrainedTokenizer, vocab_size:int):
        """
        Args:
            n: Gram size
            tokenizer: Target model's tokenizer
            vocab_size: The target model's vocab size (which is often rounded up from the tokenizer vocab)
            
        """
        self.n = n
        self.gram_freq: dict[tuple[int, ...], dict[int, int]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )
        self.conditional_probs: dict[tuple[int, ...], dict[int, float]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(vocab_size=vocab_size)

    def train(self, train: Dataset):
        """Learn an n-gram model with gram frequencies"""
        for sentence in train["text"]:
            token_ids: list[int] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(sentence)
            )  # type:ignore
            for idx in range(len(token_ids) - self.n + 1):
                context = tuple(token_ids[idx : idx + self.n - 1])
                target = token_ids[idx + self.n - 1]
                self.gram_freq[context][target] += 1
        self.ngram_vocab_size = sum(len(c) for c in self.gram_freq.values())
        for context_key, token_freqs in self.gram_freq.items():
            marginal_sum = sum(token_freqs.values())
            self.conditional_probs[context_key] = {
                k: freq / marginal_sum for k, freq in token_freqs.items()
            }
        logger.info(
            f"N-gram model trained with {self.ngram_vocab_size} unique {self.n}-grams"
        )

    def predict(self, tokens: list[int] | str):
        """Predict the next token using the last (n-1)-gram. Returns a (vocab_size,) tensor of normalized probabilities."""
        if isinstance(tokens, str):
            tokens = cast(list[int], self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tokens)))

        if len(tokens) < self.n - 1:
            # Uniform distribution
            return torch.full((len(self.tokenizer),), 1 / len(self.tokenizer))

        probabilities = torch.zeros(len(self.tokenizer))
        context_key = tuple(tokens[-(self.n - 1) :])
        for token_id, prob in self.conditional_probs[context_key].items():
            probabilities[token_id] = prob

        if probabilities.sum().item() == 0:
            # Unseen gram
            return torch.full((len(self.tokenizer),), 1 / len(self.tokenizer))
        return probabilities

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
