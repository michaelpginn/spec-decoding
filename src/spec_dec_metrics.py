"""
evaluation and metrics
"""

import logging
import math
from pathlib import Path
from statistics import median, stdev

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

    # Per-sentence tokens/sec for std computation
    tps_per_sentence = [
        tc / t if t > 0 else 0.0 for t, tc in zip(times, token_counts)
    ]

    summary = {
        "avg_time_per_sentence": total_time / n,
        "median_time_per_sentence": median(times),
        "avg_time_per_token": sum(time_per_token_list) / n,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
        "sentence_avg_tokens_per_second": sum(tps_per_sentence) / n,
    }
    if n >= 2:
        summary["sentence_std_tokens_per_second"] = stdev(tps_per_sentence)

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

    total_iterations = sum(r["num_iterations"] for r in spec_results)
    mean_accepted = total_matched / total_iterations if total_iterations > 0 else 0

    # Per-sentence values for std computation
    per_sentence_acceptance_rates = [r["acceptance_rate"] for r in spec_results]
    per_sentence_accepted_tokens = [
        r["matched_tokens"] / r["num_iterations"] if r["num_iterations"] > 0 else 0.0
        for r in spec_results
    ]

    n = len(spec_results)
    summary = {}
    summary["draft_to_output_ratio"] = (
        total_draft / total_generated if total_generated > 0 else 0
    )
    summary["token_weighted_acceptance_rate"] = (
        total_matched / total_draft if total_draft > 0 else 0
    )
    summary["sentence_avg_acceptance_rate"] = sum(per_sentence_acceptance_rates) / n
    summary["mean_accepted_tokens"] = mean_accepted
    summary["block_efficiency"] = mean_accepted / gamma if gamma > 0 else 0

    # Standard deviations
    if n >= 2:
        summary["sentence_std_acceptance_rate"] = stdev(per_sentence_acceptance_rates)
        summary["sentence_std_mean_accepted_tokens"] = stdev(per_sentence_accepted_tokens)
    
    # Compute speedup factor and its std (CUDA only)
    # Pool forward-pass timing stats across all sentences
    total_d_sum = sum(r.get("average_draft_time", 0) * r.get("draft_time_count", 0) for r in spec_results)
    total_d_sum_sq = sum(r.get("draft_time_variance", 0) * r.get("draft_time_count", 0) + r.get("average_draft_time", 0)**2 * r.get("draft_time_count", 0) for r in spec_results)
    total_d_n = sum(r.get("draft_time_count", 0) for r in spec_results)

    total_v_sum = sum(r.get("average_verifier_time", 0) * r.get("verifier_time_count", 0) for r in spec_results)
    total_v_sum_sq = sum(r.get("verifier_time_variance", 0) * r.get("verifier_time_count", 0) + r.get("average_verifier_time", 0)**2 * r.get("verifier_time_count", 0) for r in spec_results)
    total_v_n = sum(r.get("verifier_time_count", 0) for r in spec_results)

    if total_d_n > 0 and total_v_n > 0:
        mu_d = total_d_sum / total_d_n
        mu_v = total_v_sum / total_v_n
        # Population variance of individual forward pass times
        var_d = max(total_d_sum_sq / total_d_n - mu_d**2, 0)
        var_v = max(total_v_sum_sq / total_v_n - mu_v**2, 0)

        summary["average_draft_time"] = mu_d
        summary["average_verifier_time"] = mu_v
        summary["draft_time_std"] = math.sqrt(var_d)
        summary["verifier_time_std"] = math.sqrt(var_v)
        summary["draft_time_count"] = total_d_n
        summary["verifier_time_count"] = total_v_n

        if mu_d > 0 and mu_v > 0:
            c = mu_d / mu_v  # drafter cost ratio
            alpha = summary["sentence_avg_acceptance_rate"]

            if alpha < 1:
                speedup = (1 - alpha ** (gamma + 1)) / (
                    (1 - alpha) * (gamma * c + 1)
                )
            else:
                speedup = float("inf")
            summary["speedup_factor"] = speedup

            # Variance via delta method (error propagation)
            # Var(mean_d) = var_d / n_d,  Var(mean_v) = var_v / n_v
            # Var(c) ≈ c^2 * [Var(mean_d)/mu_d^2 + Var(mean_v)/mu_v^2]
            var_c = c**2 * (var_d / (total_d_n * mu_d**2) + var_v / (total_v_n * mu_v**2))

            if alpha < 1 and speedup != float("inf"):
                # Var(alpha) from sentence-level std (already computed above)
                var_alpha = summary.get("sentence_std_acceptance_rate", 0)**2 / n if n >= 2 else 0

                # Sensitivity of speedup to changes in the cost ratio (c)
                df_dc = -(1 - alpha**(gamma+1)) * gamma / ((1 - alpha) * (gamma*c + 1)**2)

                # Sensitivity of speedup to changes in the acceptance rate (alpha)
                df_dalpha = (-(gamma+1) * alpha**gamma * (1-alpha) + (1 - alpha**(gamma+1))) / (
                    (1 - alpha)**2 * (gamma * c + 1)
                )

                var_speedup = df_dc**2 * var_c + df_dalpha**2 * var_alpha
                summary["speedup_factor_std"] = math.sqrt(max(var_speedup, 0))
    else:
        summary["average_draft_time"] = 0
        summary["average_verifier_time"] = 0

    if verbose:
        print("\n=== Speculative Decoding Metrics ===")
        print(
            f"Acceptance Rate (token-weighted): {summary['token_weighted_acceptance_rate']:.2%}"
        )
        if "speedup_factor" in summary:
            std_str = f" ± {summary['speedup_factor_std']:.2f}" if "speedup_factor_std" in summary else ""
            print(
                f"Speedup Factor (sentence-weighted): {summary['speedup_factor']:.2f}x{std_str}"
            )
        print(f"Mean Accepted Tokens (per iteration): {mean_accepted:.2f}")
        print(f"Block Efficiency: {summary['block_efficiency']:.2%}")
        if "tokens_per_second" in summary:
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
