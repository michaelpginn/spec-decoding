import logging
import math
from collections import defaultdict
from types import SimpleNamespace
from typing import cast

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


class NGramModel:
    def __init__(self, n: int, tokenizer: PreTrainedTokenizer, vocab_size: int):
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
        self.conditional_logprobs: dict[tuple[int, ...], dict[int, float]] = (
            defaultdict(lambda: defaultdict(lambda: 0))
        )
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self._logprob_buf: torch.Tensor | None = None
        self._last_modified: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._full_buf: torch.Tensor | None = None

    def _ensure_device(self, device: torch.device):
        if self._device == device:
            return
        self._device = device
        self._logprob_buf = torch.full((self.vocab_size,), float("-inf"), device=device)
        self._full_buf = torch.full((self.vocab_size,), 1 / len(self.tokenizer), device=device)
        self._last_modified = None

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
            self.conditional_logprobs[context_key] = {
                k: math.log(freq / marginal_sum) for k, freq in token_freqs.items()
            }
        # Precompute context cache for fast forward
        self._context_cache = {
            ctx: (torch.tensor(list(d.keys()), dtype=torch.long),
                torch.tensor(list(d.values()), dtype=torch.float32))
            for ctx, d in self.conditional_logprobs.items()
        }
        logger.info(
            f"N-gram model trained with {self.ngram_vocab_size} unique {self.n}-grams"
        )

    def predict(self, tokens: list[int] | str) -> torch.Tensor:
        """Predict the next token using the last (n-1)-gram. Returns a (vocab_size,) tensor of log probabilities."""
        if isinstance(tokens, str):
            tokens = cast(
                list[int],
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tokens)),
            )
        if self._logprob_buf is None:
            self._ensure_device(torch.device("cpu"))
        assert self._logprob_buf is not None and self._full_buf is not None

        device = self._device or torch.device("cpu")
        context_key = tuple(tokens[-(self.n - 1) :])
        if len(tokens) < self.n - 1 or context_key not in self.conditional_logprobs:
            return self._full_buf

        if self._last_modified is not None:
            self._logprob_buf.index_fill_(0, self._last_modified, float("-inf"))

        indices, values = self._context_cache[context_key]
        if indices.device != device:
            indices = indices.to(device)
            values = values.to(device)
            self._context_cache[context_key] = (indices, values)

        self._logprob_buf[indices] = values
        self._last_modified = indices
        return self._logprob_buf

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
            full_seq = torch.concat([past_key_values[0], input_ids], dim=-1).to(
                input_ids.device
            )
        else:
            full_seq = input_ids
        self._ensure_device(input_ids.device)
        logits = self.predict(full_seq[0, -(self.n - 1):].tolist())
        logits = logits.unsqueeze(0).unsqueeze(0)
        return CausalLMOutputWithPast(logits=logits, past_key_values=(full_seq,))  # type:ignore
