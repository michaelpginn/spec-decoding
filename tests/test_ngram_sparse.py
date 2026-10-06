"""Check that the n-gram's sparse drafting path samples from the same q as the dense path.

The dense path is what spec_decode.py did for every drafter before: the n-gram scatters its
log-probs into a full-vocab vector, then repetition penalty -> log_softmax -> top-k -> top-p
-> log_softmax. The sparse path (NGramModel.draft_round) does the same over the candidates.

    uv run python tests/test_ngram_sparse.py          # q equivalence, CPU, ~1 min
    uv run python tests/test_ngram_sparse.py --e2e    # + full decoding with Qwen3.5-0.8B (MPS if available)
"""

import argparse
import random

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.n_gram import NGramModel, SparseRow, _select_from_row
from src.spec_decode import apply_repetition_penalty, filter_logprobs

MODEL = "Qwen/Qwen3.5-0.8B"
VOCAB_SIZE = 248320


def dense_q(model: NGramModel, sequence, window, penalty, top_k, top_p) -> torch.Tensor:
    """log q exactly as the neural draft loop in speculative_decode computes it."""
    logits = model.predict(sequence).clone().unsqueeze(0)
    if penalty != 1.0:
        logits = apply_repetition_penalty(logits, torch.tensor([window]), penalty)
    return filter_logprobs(torch.log_softmax(logits, dim=-1), top_k=top_k, top_p=top_p)[0]


def sparse_q(model: NGramModel, sequence, window, penalty, top_k, top_p):
    row = model.sparse_distribution(sequence, window, penalty, top_k, top_p)
    dense = torch.full((1, VOCAB_SIZE), float("-inf"))
    NGramModel.fill_dense_rows(dense, [row])
    return dense[0], row


def _cumulative_before(model: NGramModel, sequence, window, penalty, top_k) -> dict[int, float]:
    """For each candidate (after penalty and top-k), the probability mass ranked above it:
    what top-p compares against p."""
    row = model.sparse_distribution(sequence, window, penalty, top_k, 1.0)
    assert isinstance(row, SparseRow)
    order = np.argsort(-row.logprobs, kind="stable")
    probs = np.exp(row.logprobs[order])
    before = np.cumsum(probs) - probs
    return dict(zip(row.ids[order].tolist(), before.tolist()))


def synthetic_model(tokenizer) -> tuple[NGramModel, list[int]]:
    """Zipf-distributed word salad, so contexts have hundreds of successors and many ties."""
    rng = random.Random(0)
    words = [w for w in "the of and to in is was for on that with as by at from his her it an be this which or are had not but were have one they you all their there been has more when will would who so no".split()]
    words += [f"w{i}" for i in range(400)]
    weights = [1 / (i + 1) for i in range(len(words))]
    texts = [" ".join(rng.choices(words, weights, k=rng.randint(5, 40))) for _ in range(4000)]
    model = NGramModel(n=2, tokenizer=tokenizer, vocab_size=VOCAB_SIZE)
    model.train(Dataset.from_dict({"text": texts}))
    corpus_ids = tokenizer(" ".join(texts[:200]), add_special_tokens=False)["input_ids"]
    return model, corpus_ids


def test_q_equivalence(model: NGramModel, corpus_ids: list[int]):
    rng = random.Random(1)
    settings = [(1.1, 50, 0.9), (1.1, 0, 0.9), (1.1, 5, 0.0), (1.0, 50, 0.9), (1.1, 50, 0.5), (1.0, 0, 0.0)]
    n_seen = n_fallback = n_exact_support = 0
    worst_tvd = 0.0
    for penalty, top_k, top_p in settings:
        for _ in range(300):
            start = rng.randrange(len(corpus_ids) - 20)
            sequence = corpus_ids[start : start + rng.randint(1, 20)]
            if rng.random() < 0.15:
                sequence = sequence + [rng.randrange(200000, 240000)]  # unseen context
            window = (sequence + [rng.choice(corpus_ids) for _ in range(16)])[-16:]
            d = dense_q(model, sequence, window, penalty, top_k, top_p)
            s, row = sparse_q(model, sequence, window, penalty, top_k, top_p)
            if isinstance(row, SparseRow):
                # The supports may differ only at the top-p cutoff: the dense path's float32
                # cumsum over the full vocab and the sparse path's float64 cumsum can round a
                # token whose cumulative prob sits right at p differently (CPU vs CUDA dense
                # would too). Every other token must match exactly.
                n_seen += 1
                differing = torch.nonzero(torch.isfinite(d) ^ torch.isfinite(s)).flatten().tolist()
                if differing:
                    assert 0.0 < top_p < 1.0, (penalty, top_k, top_p, sequence)
                    cum_before = _cumulative_before(model, sequence, window, penalty, top_k)
                    for t in differing:
                        assert abs(cum_before[t] - top_p) < 1e-3, (t, cum_before[t], sequence)
                else:
                    n_exact_support += 1
                    tvd = 0.5 * (d.exp() - s.exp()).abs().sum().item()
                    worst_tvd = max(worst_tvd, tvd)
                    assert tvd < 1e-4, (tvd, penalty, top_k, top_p, sequence)
                assert _select_from_row(row, "greedy") == int(d.argmax())
            else:
                # Unseen context: uniform over ~top_p of the vocab in both. The sparse path
                # drops every penalty-window token; float32 rounding makes the dense path drop
                # only some (see NGramModel._fallback_distribution), so the supports may differ
                # by up to the window size, plus rounding in the dense cumsum.
                n_fallback += 1
                d_finite, s_finite = torch.isfinite(d), torch.isfinite(s)
                assert (s_finite & ~d_finite).sum() <= 16 + 10, int((s_finite & ~d_finite).sum())
                assert (d_finite & ~s_finite).sum() <= 16 + 10, int((d_finite & ~s_finite).sum())
                assert abs(d[d_finite].exp().mean().item() - s[s_finite].exp().mean().item()) < 1e-9
    print(
        f"q equivalence OK: {n_seen} seen-context cases ({n_exact_support} identical up to "
        f"TVD {worst_tvd:.1e}, the rest differ only at the top-p cutoff) and "
        f"{n_fallback} unseen-context cases"
    )


def test_sampling_frequencies(model: NGramModel, corpus_ids: list[int]):
    """_select_from_row draws tokens with frequency q."""
    torch.manual_seed(0)
    sequence = corpus_ids[:1]
    row = model.sparse_distribution(sequence, [], 1.1, 50, 0.9)
    assert isinstance(row, SparseRow)
    n = 50000
    counts: dict[int, int] = {}
    for _ in range(n):
        t = _select_from_row(row, "sample")
        counts[t] = counts.get(t, 0) + 1
    q = dict(zip(row.ids.tolist(), np.exp(row.logprobs).tolist()))
    assert set(counts) <= set(q)
    worst = max(abs(counts.get(t, 0) / n - p) for t, p in q.items())
    assert worst < 0.01, worst
    print(f"sampling OK: {len(q)} candidates, max |freq - q| = {worst:.4f}")


def test_end_to_end(tokenizer):
    """Full speculative decoding, greedy: dense and sparse drafting must give identical output."""
    from src.data.create_inputs import create_inputs, create_prompt
    from src.data.dataset import assemble_dataset
    from src.spec_decode import speculative_decode
    from src.utils import load_model

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    target, _ = load_model(MODEL, device=device.type)
    data = assemble_dataset("que", "mono", tokenizer, 20000)
    models = {}
    for sparse in (False, True):
        models[sparse] = NGramModel(n=2, tokenizer=tokenizer, vocab_size=target.config.vocab_size, sparse_drafting=sparse)
        models[sparse].train(data["train"])

    sources = ["The child is playing in the garden.", "We will travel to the city tomorrow.", "My mother cooks potatoes every day."]
    for mode in ("greedy", "sample"):
        alphas = {False: [], True: []}
        for i, source in enumerate(sources):
            inputs = create_inputs(create_prompt("translation", "Quechua", source), tokenizer, device)
            outputs = {}
            for sparse in (False, True):
                torch.manual_seed(i)
                out, metrics = speculative_decode(
                    target, models[sparse], tokenizer, inputs["input_ids"], mode=mode,
                    max_new_tokens=48, gamma=3, top_k=50, top_p=0.9,
                    repetition_penalty=1.1, repetition_penalty_window=16,
                )
                outputs[sparse] = out
                alphas[sparse].append(metrics["acceptance_rate"])
            if mode == "greedy":
                assert torch.equal(outputs[False], outputs[True]), f"greedy outputs differ for {source!r}"
        print(f"end-to-end {mode}: alpha dense={np.mean(alphas[False]):.3f} sparse={np.mean(alphas[True]):.3f}"
              + (" (identical outputs)" if mode == "greedy" else " (different RNG draws; should be close)"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", action="store_true")
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ngram, ids = synthetic_model(tok)
    test_q_equivalence(ngram, ids)
    test_sampling_frequencies(ngram, ids)
    if args.e2e:
        test_end_to_end(tok)
