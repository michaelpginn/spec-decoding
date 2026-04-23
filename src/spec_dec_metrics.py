"""
evaluation and metrics
"""

import logging
from pathlib import Path
from statistics import median

import wandb

from src.config.config import ExperimentConfig

logger = logging.getLogger(__name__)


def summarize_metrics(
    all_metrics: list[dict],
    gamma: int,
    used_spec_dec: bool,
    verbose: bool = False,
):
    """Aggregates a list of per-sentence metrics."""
    times = [r["time"] for r in all_metrics]
    token_counts = [r["generated_tokens"] for r in all_metrics]
    per_sentence, summary = _compute_common_metrics(times, token_counts)
    if used_spec_dec:
        per_sentence_spec, summary_spec = _compute_spec_metrics(
            all_metrics, gamma, verbose
        )
        per_sentence = [
            {**per_sentence[i], **per_sentence_spec[i]}
            for i in range(len(per_sentence))
        ]
        summary.update(summary_spec)
    return per_sentence, summary


def _compute_common_metrics(
    times: list[float],
    token_counts: list[int],
) -> tuple[list[dict], dict]:
    """
    Base metrics shared by baseline and spec runs.

    Args:
        times: Per-sentence wall-clock times.
        token_counts: Per-sentence generated token counts.

    Returns:
        (per_sentence, summary) with unified key names usable by both run types.
    """
    if not times:
        return [], {}

    n = len(times)
    total_time = sum(times)
    total_tokens = sum(token_counts)

    time_per_token_list = [
        t / tc if tc > 0 else 0.0 for t, tc in zip(times, token_counts)
    ]

    per_sentence = [
        {
            "sentence/time": t,
            "sentence/time_per_token": tpt,
            "sentence/generated_tokens": tc,
            "sentence_idx": i,
        }
        for i, (t, tpt, tc) in enumerate(zip(times, time_per_token_list, token_counts))
    ]

    summary = {
        "avg_time_per_sentence": total_time / n,
        "median_time_per_sentence": median(times),
        "avg_time_per_token": sum(time_per_token_list) / n,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
    }

    return per_sentence, summary


def _compute_spec_metrics(
    spec_results: list[dict],
    gamma: int,
    verbose: bool,
) -> tuple[list[dict], dict]:
    """
    Compute per-sentence and summary metrics for a speculative decoding run.

    Args:
        spec_results: List of dicts with per-sentence keys:
            - time: Decode time for this sentence
            - generated_tokens: Tokens generated for this sentence
            - draft_tokens: Draft tokens proposed for this sentence
            - matched_tokens: Draft tokens that matched target for this sentence
            - acceptance_rate: Per-sentence acceptance rate
            - average_draft_time: Average time (s) of draft forward pass
            - average_verifier_time:  Average time (s) of verifier forward pass
        gamma: Number of draft tokens per iteration
        verbose: Print metrics to console

    Returns:
        (per_sentence, summary) with unified + spec-specific metrics.
    """
    if not spec_results:
        return [], {}

    per_sentence = [
        {
            "sentence/acceptance_rate": m["acceptance_rate"],
            "sentence/draft_tokens": m["draft_tokens"],
            "sentence/matched_tokens": m["matched_tokens"],
        }
        for m in spec_results
    ]

    total_generated = sum(r["generated_tokens"] for r in spec_results)
    total_draft = sum(r["draft_tokens"] for r in spec_results)
    total_matched = sum(r["matched_tokens"] for r in spec_results)

    total_iterations = sum(
        r.get("num_iterations", r["draft_tokens"] / gamma) for r in spec_results
    )
    mean_accepted = total_matched / total_iterations if total_iterations > 0 else 0

    summary = {}
    summary["draft_to_output_ratio"] = (
        total_draft / total_generated if total_generated > 0 else 0
    )
    summary["token_weighted_acceptance_rate"] = (
        total_matched / total_draft if total_draft > 0 else 0
    )
    summary["sentence_avg_acceptance_rate"] = sum(
        r["acceptance_rate"] for r in spec_results
    ) / len(spec_results)
    summary["mean_accepted_tokens"] = mean_accepted
    summary["block_efficiency"] = mean_accepted / gamma if gamma > 0 else 0
    
    # Compute speedup factor (CUDA only)
    summary["average_draft_time"] = sum(
        r.get("average_draft_time", 0) for r in spec_results
    ) / len(spec_results)
    summary["average_verifier_time"] = sum(
        r.get("average_verifier_time", 0) for r in spec_results
    ) / len(spec_results)
    if summary["average_verifier_time"] > 0 and summary["average_draft_time"] > 0:
        # Compute overall speedup factor
        drafter_cost_ratio = summary["average_draft_time"] / summary["average_verifier_time"]
        if summary["sentence_avg_acceptance_rate"] < 1:
            speedup_factor = (1 - summary["sentence_avg_acceptance_rate"] ** (gamma + 1)) / (
                (1 - summary["sentence_avg_acceptance_rate"]) * (gamma * drafter_cost_ratio + 1)
            )
        else:
            speedup_factor = float("inf")
        summary["speedup_factor"] = speedup_factor

    if verbose:
        print("\n=== Speculative Decoding Metrics ===")
        print(
            f"Acceptance Rate (token-weighted): {summary['token_weighted_acceptance_rate']:.2%}"
        )
        if "speedup_factor" in summary:
            print(
                f"Speedup Factor (sentence-weighted): {summary['speedup_factor']:.2f}x"
            )
        print(f"Mean Accepted Tokens (per iteration): {mean_accepted:.2f}")
        print(f"Block Efficiency: {summary['block_efficiency']:.2%}")
        print(f"Tokens/sec:  {summary['tokens_per_second']:.2f}")

    return per_sentence, summary


def log_token_flow(
    inputs: list[str], all_metrics: list[dict], config: ExperimentConfig
):
    # 7. Log token flow trace as wandb Table
    if any("iteration_history" in r for r in all_metrics):
        flow_table = wandb.Table(
            columns=[
                "sample_idx",
                "source_text",
                "iteration",
                "drafted_tokens",
                "num_drafted",
                "result",
            ]
        )
        for i, result in enumerate(all_metrics):
            for item in result.get("iteration_history", []):
                flow_table.add_data(
                    i,
                    inputs[i],
                    item["iter"],
                    " ".join(f"[{t}]" for t in item["drafted"]),
                    len(item["drafted"]),
                    item["result"],
                )
        wandb.log({"token_flow": flow_table})

    # 8. Save token flow trace locally
    output_dir = Path(f"./outputs/{config.language_code}")
    output_dir.mkdir(parents=True, exist_ok=True)

    iteration_file = output_dir / f"token_flow_{config.language_code}.txt"
    with open(iteration_file, "w", encoding="utf-8") as f:
        for i, result in enumerate(all_metrics):
            if "iteration_history" in result:
                f.write(f"\n{'=' * 60}\n")
                f.write(f'Sample {i + 1}: "{inputs[i]}"\n')
                f.write(f"{'=' * 60}\n")
                for item in result["iteration_history"]:
                    drafted_str = " ".join(f"[{t}]" for t in item["drafted"])
                    f.write(f"  Iter {item['iter']}: {drafted_str}\n")
                    f.write(f"           -> {item['result']}\n")
    logger.info(f"Token flow trace saved to {iteration_file}")
