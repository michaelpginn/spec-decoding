import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# BLEU, tps spec, speedup, acceptance rate, toks/second

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#E5E5E5'
plt.rcParams['axes.linewidth'] = 0.8

langs = ["amh","ber","chr","grn","haw","ibo","npi","oci","que","yor","zgh","zh"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 14
})

def fake_data(langs):
    acceptance_rate = (random.randint(1, 100) / 100) * 100

    len_langs = len(langs)
    gamma = 5
    c = 0.15
    base_tps = 40.0

    alpha = np.array([random.uniform(0, 0.999) for _ in range(len_langs)])

    speedup = (1 - alpha**(gamma + 1)) / ((1 - alpha) * (gamma * c + 1))

    tps_auto = np.random.normal(loc=base_tps, scale=1.5, size=len_langs)
    tps_spec = tps_auto * speedup

    bleu_auto = np.array([random.random() for _ in range(len_langs)])
    bleu_spec = bleu_auto + np.random.uniform(-0.001, 0.001, len_langs)

    data = np.random.randn(len_langs)
    data = (data - np.mean(data)) / np.std(data)
    # 3. Scale to target
    fake_data = (data * 5) + 7

    stdev = np.std(fake_data)

    np.random.seed(42)
    time = np.linspace(0, 10, 20)
    num_simulations = 10

    simulations = [np.sin(time) + np.random.normal(0, 0.2, 20) for _ in range(num_simulations)]

    mean_pass = np.mean(simulations, axis=0)
    std_pass = np.std(simulations, axis=0)

    # Dictionary construction
    return {
        "tps spec": tps_spec,
        "tps auto": tps_auto,
        "bleu auto": bleu_auto,
        "bleu spec": bleu_spec,
        "speedup": speedup,
        "acceptance rate": acceptance_rate,
        "standard deviation": stdev,
        "average forward pass": mean_pass,
        "average forward pass stdev": std_pass
    }

def placeholder_graphs():
    models = {
        "Baseline": fake_data(langs),
        "Task-Specific Distill": fake_data(langs),
        "General Domain Distill": fake_data(langs),
        "N-Gram": fake_data(langs),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    x_indexes = np.arange(len(langs))
    total_group_width = 0.8
    num_bars = len(models)
    width = total_group_width / num_bars

    colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442']

    for idx, (model_name, model_data) in enumerate(models.items()):
        offset = (idx - (num_bars - 1) / 2) * width

        ax.bar(
            x = x_indexes + offset,
            height = model_data["tps spec"],
            width=width,
            color=colors[idx],
            edgecolor='#333333',
            linewidth=0.4,
            label=model_name,
            yerr= model_data["standard deviation"],
            capsize=3,
        )

    for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor('#333333')

    # Mapping center coordinates back to textual language categories
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(langs, rotation=0)

    ax.set_ylabel("Tokens / Second (Spec)")

    # Place legend cleanly without taking up critical horizontal plot space
    ax.legend(
        frameon=False,
        fontsize=12,
        loc='center left',
        bbox_to_anchor=(1.02, 1)  # (x, y) coordinates starting right at the edge of the axis
    )

    plt.tight_layout()

    # Create destination output directory safely
    Path("../viz").mkdir(parents=True, exist_ok=True)
    plt.savefig("../viz/tps_spec.pdf", format="pdf", bbox_inches="tight")

    fig, ax2 = plt.subplots(figsize=(8,5))

    for idx, (model_name, model_data) in enumerate(models.items()):
        offset = (idx - (num_bars - 1) / 2) * width

        ax2.bar(
            x = x_indexes + offset,
            height = model_data["bleu spec"],
            width=width,
            color=colors[idx],
            edgecolor='#333333',
            linewidth=0.4,
            label=model_name,
            yerr= model_data["standard deviation"],
            capsize=3,
        )

    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#333333')

    # Mapping center coordinates back to textual language categories
    ax2.set_xticks(x_indexes)
    ax2.set_xticklabels(langs, rotation=0)

    ax2.set_ylabel("BLEU")

    # Place legend cleanly without taking up critical horizontal plot space
    ax2.legend(
        frameon=False,
        fontsize=12,
        loc='center left',
        bbox_to_anchor=(1.02, 1)  # (x, y) coordinates starting right at the edge of the axis
    )

    plt.tight_layout()

    # Create destination output directory safely
    Path("../viz").mkdir(parents=True, exist_ok=True)
    plt.savefig("../viz/bleu_spec.pdf", format="pdf", bbox_inches="tight")

    fig, ax3 = plt.subplots(figsize=(8,5))

    for idx, (model_name, model_data) in enumerate(models.items()):
        offset = (idx - (num_bars - 1) / 2) * width

        ax3.bar(
            x = x_indexes + offset,
            height = model_data["speedup"],
            width=width,
            color=colors[idx],
            edgecolor='#333333',
            linewidth=0.4,
            label=model_name,
            yerr= model_data["standard deviation"],
            capsize=3,
        )

    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#333333')

    # Mapping center coordinates back to textual language categories
    ax3.set_xticks(x_indexes)
    ax3.set_xticklabels(langs, rotation=0)

    ax3.set_ylabel("Speedup")

    # Place legend cleanly without taking up critical horizontal plot space
    ax3.legend(
        frameon=False,
        fontsize=12,
        loc='center left',
        bbox_to_anchor=(1.02, 1)  # (x, y) coordinates starting right at the edge of the axis
    )

    plt.tight_layout()

    # Create destination output directory safely
    Path("../viz").mkdir(parents=True, exist_ok=True)
    plt.savefig("../viz/speedup.pdf", format="pdf", bbox_inches="tight")

    fig, ax4 = plt.subplots(figsize=(8,5))

    for idx, (model_name, model_data) in enumerate(models.items()):
        offset = (idx - (num_bars - 1) / 2) * width

        ax4.bar(
            x = x_indexes + offset,
            height = model_data["acceptance rate"],
            width=width,
            color=colors[idx],
            edgecolor='#333333',
            linewidth=0.4,
            label=model_name,
            yerr= model_data["standard deviation"],
            capsize=3,
        )

    for spine in ax4.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#333333')

    # Mapping center coordinates back to textual language categories
    ax4.set_xticks(x_indexes)
    ax4.set_xticklabels(langs, rotation=0)

    ax4.set_ylabel("Acceptance Rate")

    # Place legend cleanly without taking up critical horizontal plot space
    ax4.legend(
        frameon=False,
        fontsize=12,
        loc='center left',
        bbox_to_anchor=(1.02, 1)  # (x, y) coordinates starting right at the edge of the axis
    )

    plt.tight_layout()

    # Create destination output directory safely
    Path("../viz").mkdir(parents=True, exist_ok=True)
    plt.savefig("../viz/acceptance_rate.pdf", format="pdf", bbox_inches="tight")

    fig, ax5 = plt.subplots(figsize=(8,5))

    for idx, (model_name, model_data) in enumerate(models.items()):
        offset = (idx - (num_bars - 1) / 2) * width

        ax5.bar(
            x = x_indexes + offset,
            height = model_data["average forward pass"],
            width=width,
            color=colors[idx],
            edgecolor='#333333',
            linewidth=0.4,
            label=model_name,
            yerr= model_data["average forward pass stdev"],
            capsize=3,
        )

    for spine in ax5.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#333333')

    # Mapping center coordinates back to textual language categories
    ax5.set_xticks(x_indexes)
    ax5.set_xticklabels(langs, rotation=0)

    ax5.set_ylabel("Average Forward Pass")

    # Place legend cleanly without taking up critical horizontal plot space
    ax5.legend(
        frameon=False,
        fontsize=12,
        loc='center left',
        bbox_to_anchor=(1.02, 1)  # (x, y) coordinates starting right at the edge of the axis
    )

    plt.tight_layout()

    # Create destination output directory safely
    Path("../viz").mkdir(parents=True, exist_ok=True)
    plt.savefig("../viz/placeholder_avg_forw_pass.pdf", format="pdf", bbox_inches="tight")

def graphs(
    data:dict[str, dict[str, int|float]] | None,
    y_label:str,
    ind_filename: dict[str,str],
    average_forward_pass: dict[str, int|float]
):
    '''
    ind_filename: {metric name: filename name, ...}
    data: {model: {metric: value, ...}, ...}
    average_forward_pass: {model: average forward pass value, ...}
    '''
    if data is None:
        raise ValueError("Need to provide data in the form of {Model: {'metric': value}}")

    for metric, title in ind_filename.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        x_indexes = np.arange(len(langs))
        total_group_width = 0.8
        num_bars = len(data)
        width = total_group_width / num_bars

        colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442']

        for idx, (model_name, model_data) in enumerate(data.items()):
            offset = (idx - (num_bars - 1) / 2) * width

            ax.bar(
                x = x_indexes + offset,
                height = model_data[metric],
                width=width,
                color=colors[idx],
                edgecolor='#333333',
                linewidth=0.4,
                label=model_name,
                yerr= model_data["standard deviation"],
                capsize=3,
            )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor('#333333')

        # Mapping center coordinates back to textual language categories
        ax.set_xticks(x_indexes)
        ax.set_xticklabels(langs, rotation=0)

        ax.set_ylabel(y_label)

        # Place legend cleanly without taking up critical horizontal plot space
        ax.legend(
            frameon=False,
            fontsize=12,
            loc='center left',
            bbox_to_anchor=(1.02, 1)  # (x, y) coordinates starting right at the edge of the axis
        )

        plt.tight_layout()

        # Create destination output directory safely
        Path("../viz").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"../viz/{title}.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    placeholder_graphs()
