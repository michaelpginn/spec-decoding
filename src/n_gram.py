import logging
import math
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import NamedTuple, cast

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


class SparseRow(NamedTuple):
    """Filtered draft distribution q over a few candidate tokens (-inf elsewhere)."""
    ids: np.ndarray  # int64 token ids
    logprobs: np.ndarray  # float64 log q, normalized over `ids`


class FallbackRow(NamedTuple):
    """Uniform draft distribution q for an unseen context: log q = `logprob` for every
    token id below `end` that is not in `excluded`, -inf elsewhere."""
    end: int
    excluded: list[int]  # sorted
    logprob: float


class NGramModel:
    def __init__(
        self,
        n: int,
        tokenizer: PreTrainedTokenizer,
        vocab_size: int,
        sparse_drafting: bool = True,
    ):
        """
        Args:
            n: Gram size
            tokenizer: Target model's tokenizer
            vocab_size: The target model's vocab size (which is often rounded up from the tokenizer vocab)
            sparse_drafting: If True, spec_decode.py drafts with `draft_round`, which samples
                over the candidate tokens on the CPU. If False, it drafts through `__call__`
                like a neural model, over the full vocabulary. Both sample from the same q.
        """
        self.n = n
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.sparse_drafting = sparse_drafting
        self.config = SimpleNamespace(vocab_size=vocab_size)
        # context -> (next-token ids sorted ascending, their log-probs), on the CPU
        self._context_cache: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        # Same as _context_cache, moved to self._device on first use (dense path only)
        self._device_cache: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._logprob_buf: torch.Tensor | None = None
        self._last_modified: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._full_buf: torch.Tensor | None = None

    def _ensure_device(self, device: torch.device):
        if self._device == device:
            return
        self._device = device
        self._device_cache = {}
        self._logprob_buf = torch.full((self.vocab_size,), float("-inf"), device=device)
        self._full_buf = torch.full((self.vocab_size,), 1 / len(self.tokenizer), device=device)
        self._last_modified = None

    def train(self, train: Dataset, batch_size: int = 1000):
        """Learn an n-gram model with gram frequencies"""
        gram_freq: dict[tuple[int, ...], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        texts = train["text"]
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start : start + batch_size], add_special_tokens=False)
            for token_ids in batch["input_ids"]:
                for idx in range(len(token_ids) - self.n + 1):
                    context = tuple(token_ids[idx : idx + self.n - 1])
                    target = token_ids[idx + self.n - 1]
                    gram_freq[context][target] += 1
        self.ngram_vocab_size = sum(len(c) for c in gram_freq.values())
        # Keep only the per-context tensors; pop the counts as we go to keep peak memory low
        self._context_cache = {}
        while gram_freq:
            context_key, token_freqs = gram_freq.popitem()
            marginal_sum = sum(token_freqs.values())
            ids = sorted(token_freqs)
            self._context_cache[context_key] = (
                torch.tensor(ids, dtype=torch.long),
                torch.tensor([math.log(token_freqs[k] / marginal_sum) for k in ids], dtype=torch.float32),
            )
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
        if len(tokens) < self.n - 1 or context_key not in self._context_cache:
            # A copy, because spec_decode applies the repetition penalty in place, which
            # would otherwise build up in _full_buf across calls
            return self._full_buf.clone()

        if self._last_modified is not None:
            self._logprob_buf.index_fill_(0, self._last_modified, float("-inf"))

        if context_key not in self._device_cache:
            indices, values = self._context_cache[context_key]
            self._device_cache[context_key] = (indices.to(device), values.to(device))
        indices, values = self._device_cache[context_key]

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

    # ------------------------------------------------------------------
    # Sparse drafting: the same q as the dense path through spec_decode.py
    # (repetition penalty -> log_softmax -> top-k -> top-p -> log_softmax),
    # computed over the context's candidate tokens instead of the full vocab.
    # ------------------------------------------------------------------

    def sparse_distribution(
        self,
        history: list[int],
        penalty_window: list[int],
        repetition_penalty: float,
        top_k: int,
        top_p: float,
    ) -> SparseRow | FallbackRow:
        """Filtered draft distribution for the next token after `history`."""
        context_key = tuple(history[-(self.n - 1) :])
        if len(history) < self.n - 1 or context_key not in self._context_cache:
            return self._fallback_distribution(penalty_window, repetition_penalty, top_k, top_p)

        ids_t, logprobs_t = self._context_cache[context_key]
        ids = ids_t.numpy()
        logprobs = logprobs_t.numpy().astype(np.float64)

        # Repetition penalty, as in apply_repetition_penalty: n-gram log-probs are <= 0,
        # so a token seen c times in the window is multiplied by penalty**c.
        if repetition_penalty != 1.0 and penalty_window:
            for token, count in Counter(penalty_window).items():
                pos = np.searchsorted(ids, token)
                if pos < len(ids) and ids[pos] == token:
                    factor = repetition_penalty**count
                    if logprobs[pos] > 0:
                        logprobs[pos] = logprobs[pos] / factor
                    else:
                        logprobs[pos] = logprobs[pos] * factor

        logprobs = _log_softmax(logprobs)
        filtered = top_k > 0 or 0.0 < top_p < 1.0
        keep = np.ones(len(ids), dtype=bool)
        # Top-k, as in apply_top_k: drop everything below the k-th largest value. With fewer
        # than k candidates the k-th largest of the full vocab is -inf, so nothing is dropped.
        if top_k > 0 and top_k < self.vocab_size and len(ids) >= top_k:
            kth_value = np.partition(logprobs, -top_k)[-top_k]
            keep &= logprobs >= kth_value
        # Top-p, as in apply_top_p: keep the smallest prefix (by descending prob) whose
        # cumulative prob reaches p, including the token that crosses it.
        if 0.0 < top_p < 1.0:
            kept_idx = np.flatnonzero(keep)
            order = kept_idx[np.argsort(-logprobs[kept_idx], kind="stable")]
            cumulative = np.cumsum(np.exp(_log_softmax(logprobs[order])))
            remove = cumulative > top_p
            remove[1:] = remove[:-1].copy()
            remove[0] = False
            keep[order[remove]] = False
        ids, logprobs = ids[keep], logprobs[keep]
        if filtered:
            logprobs = _log_softmax(logprobs)
        return SparseRow(ids=ids, logprobs=logprobs)

    def _fallback_distribution(
        self,
        penalty_window: list[int],
        repetition_penalty: float,
        top_k: int,
        top_p: float,
    ) -> FallbackRow:
        """Unseen context. The dense path returns a constant vector over the vocab; the
        penalty lowers the window tokens, top-k then drops them (they fall below the
        k-th value, which is the constant), and top-p keeps the first ~p of the remaining
        equal-probability tokens in sort order (ascending id, since the sort is stable).
        This reproduces that: uniform over the lowest-id tokens left.

        One deliberate difference: in float32 the penalty's shift of the constant (~4e-7)
        is often rounded away by log_softmax, so the dense path only drops *some* window
        tokens. Here all of them are dropped, as intended. That changes the support by at
        most `repetition_penalty_window` tokens out of ~220k."""
        excluded: list[int] = []
        if repetition_penalty != 1.0 and (top_k > 0 or 0.0 < top_p < 1.0):
            excluded = sorted({t for t in penalty_window if t < self.vocab_size})
        n_available = self.vocab_size - len(excluded)
        n_keep = n_available
        if 0.0 < top_p < 1.0:
            # Token i (1-indexed) survives iff the cumulative prob before it, (i-1)/n, is <= p
            n_keep = min(n_available, math.floor(top_p * n_available) + 1)
        end = _nth_available_id(n_keep - 1, excluded) + 1
        return FallbackRow(end=end, excluded=[t for t in excluded if t < end], logprob=-math.log(n_keep))

    def draft_round(
        self,
        history: list[int],
        max_tokens: int,
        repetition_penalty: float,
        repetition_penalty_window: int,
        top_k: int,
        top_p: float,
        mode: str,
        stop_token_ids: set[int],
    ) -> tuple[list[int], list[SparseRow | FallbackRow]]:
        """Draft up to `max_tokens` tokens after `history` (prompt + confirmed tokens).

        Mirrors the neural draft loop in speculative_decode: the penalty window is the
        last `repetition_penalty_window` tokens of history + drafts so far, and drafting
        stops after a stop token. Returns the drafted tokens and each one's q.
        """
        drafts: list[int] = []
        rows: list[SparseRow | FallbackRow] = []
        for _ in range(max_tokens):
            sequence = history + drafts
            window = sequence[-repetition_penalty_window:] if repetition_penalty_window > 0 else []
            row = self.sparse_distribution(sequence, window, repetition_penalty, top_k, top_p)
            token = _select_from_row(row, mode)
            drafts.append(token)
            rows.append(row)
            if token in stop_token_ids:
                break
        return drafts, rows

    @staticmethod
    def fill_dense_rows(dense: torch.Tensor, rows: list[SparseRow | FallbackRow]):
        """Write each row's log q into `dense` (n_rows, d_vocab), which must be all -inf."""
        sparse = [(r, row) for r, row in enumerate(rows) if isinstance(row, SparseRow)]
        if sparse:
            row_idx = np.concatenate([np.full(len(row.ids), r) for r, row in sparse])
            tok_idx = np.concatenate([row.ids for _, row in sparse])
            values = np.concatenate([row.logprobs for _, row in sparse])
            dense[
                torch.from_numpy(row_idx).to(dense.device),
                torch.from_numpy(tok_idx).to(dense.device),
            ] = torch.from_numpy(values).to(dense.device, dense.dtype)
        for r, row in enumerate(rows):
            if isinstance(row, FallbackRow):
                dense[r, : row.end] = row.logprob
                if row.excluded:
                    dense[r, torch.tensor(row.excluded, device=dense.device)] = float("-inf")


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def _nth_available_id(n: int, excluded: list[int]) -> int:
    """The n-th (0-indexed) token id in ascending order that is not in sorted `excluded`."""
    token = n
    for e in excluded:
        if e <= token:
            token += 1
        else:
            break
    return token


def _select_from_row(row: SparseRow | FallbackRow, mode: str) -> int:
    """Greedy (lowest id among ties, like argmax) or sample, using torch's CPU RNG."""
    if isinstance(row, FallbackRow):
        n_keep = row.end - len(row.excluded)
        index = 0 if mode == "greedy" else int(torch.randint(n_keep, ()).item())
        return _nth_available_id(index, row.excluded)
    if mode == "greedy":
        return int(row.ids[np.argmax(row.logprobs)])
    cumulative = np.cumsum(np.exp(row.logprobs))
    u = torch.rand(()).item() * cumulative[-1]
    return int(row.ids[min(np.searchsorted(cumulative, u, side="right"), len(row.ids) - 1)])
