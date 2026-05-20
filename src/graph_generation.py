import colorsys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "font.size": 10,
    "font.weight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "axes.axisbelow": True,
})

langs = ["amh","ber","chr","grn","haw","ibo","npi","oci","que","yor","zgh","zh"]

PALETTE = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7']

SETTINGS = ["Baseline", "Task-Specific Distill", "General Domain Distill", "N-Gram"]
FORWARD_PASS_MODELS = ["Qwen 0.8b", "Qwen 2b", "Qwen 4b", "Qwen 9b", "n-gram"]


def _shades(hex_color: str, n: int, light: float = 0.78, dark: float = 0.25) -> list[str]:
    h, _, s = colorsys.rgb_to_hls(*mcolors.to_rgb(hex_color))
    return [
        mcolors.to_hex(colorsys.hls_to_rgb(h, ll, s))
        for ll in np.linspace(light, dark, n)
    ]


def _placeholder_mean_std(center: float, spread: float, std_val: float) -> dict:
    n = len(langs)
    return {
        s: {
            "mean": np.random.normal(center, spread, n),
            "std": np.full(n, std_val),
        }
        for s in SETTINGS
    }


def fake_forward_pass(n_runs: int = 200) -> dict[str, np.ndarray]:
    centers = {"Qwen 0.8b": 12, "Qwen 2b": 22, "Qwen 4b": 38, "Qwen 9b": 65, "n-gram": 4}
    scales  = {"Qwen 0.8b": 1.5, "Qwen 2b": 2.5, "Qwen 4b": 4.0, "Qwen 9b": 6.0, "n-gram": 0.4}
    return {
        m: np.random.normal(loc=centers[m], scale=scales[m], size=n_runs)
        for m in FORWARD_PASS_MODELS
    }


def make_placeholder_data() -> dict:
    n = len(langs)
    return {
        "tps": _placeholder_mean_std(50, 8, 2.0),
        "speedup": _placeholder_mean_std(1.5, 0.3, 0.1),
        "acceptance rate": _placeholder_mean_std(70, 10, 5),
        "bleu": {
            "mean": np.random.uniform(0.1, 0.9, n),
            "std": np.full(n, 0.05),
        },
        "forward_pass": fake_forward_pass(),
    }


def _style_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor('black')


def _finalize(fig, filename: str):
    plt.tight_layout(pad=0.2)
    Path("viz").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"viz/{filename}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.show()
    plt.close(fig)


def _bar_plot(metric_data: dict, ylabel: str, filename: str):
    settings = list(metric_data.keys())

    rows = []
    for setting, vals in metric_data.items():
        mean = np.asarray(vals["mean"])
        std = np.asarray(vals["std"])
        for i, lang in enumerate(langs):
            rows.append({"lang": lang, "setting": setting, "mean": mean[i], "std": std[i]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 2))

    sns.barplot(
        data=df,
        x="lang",
        y="mean",
        hue="setting",
        hue_order=settings,
        order=langs,
        palette=PALETTE[:len(settings)],
        edgecolor='#333333',
        linewidth=0.4,
        errorbar=None,
        ax=ax,
    )

    for container, setting in zip(ax.containers, settings):
        s_df = df[df["setting"] == setting].set_index("lang").loc[langs]
        xs = [patch.get_x() + patch.get_width() / 2 for patch in container]
        ys = [patch.get_height() for patch in container]
        ax.errorbar(
            xs, ys,
            yerr=s_df["std"].to_numpy(),
            fmt='none',
            ecolor='#333333',
            capsize=3,
            linewidth=0.8,
        )

    _style_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)

    ax.legend(
        frameon=False,
        fontsize=10,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(settings),
        title=None,
        borderaxespad=0.1,
    )

    _finalize(fig, filename)


def _violin_plot(forward_pass: dict[str, np.ndarray], ylabel: str, filename: str):
    order = [m for m in FORWARD_PASS_MODELS if m in forward_pass]
    rows = [
        {"model": m, "value": v}
        for m in order
        for v in forward_pass[m]
    ]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(4, 3))

    sns.violinplot(
        data=df,
        x="model",
        y="value",
        order=order,
        hue="model",
        hue_order=order,
        palette=_shades(PALETTE[0], len(order)),
        legend=False,
        inner="quartile",
        linewidth=0.6,
        width=0.95,
        ax=ax,
    )

    _style_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelsize=10, rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    _finalize(fig, filename)


def _bleu_acceptance_line_plot(bleu: dict, acceptance_rate: dict, filename: str):
    bleu_mean = np.asarray(bleu["mean"])
    sort_idx = np.argsort(bleu_mean)
    sorted_bleu = bleu_mean[sort_idx]
    sorted_langs = [langs[i] for i in sort_idx]

    settings = list(acceptance_rate.keys())

    fig, ax = plt.subplots(figsize=(8, 4))

    for x in sorted_bleu:
        ax.axvline(x, linestyle='--', color='#BFBFBF', linewidth=0.6, alpha=0.7, zorder=0)

    for idx, setting in enumerate(settings):
        ar = np.asarray(acceptance_rate[setting]["mean"])[sort_idx]
        ax.plot(
            sorted_bleu, ar,
            marker='o',
            markersize=5,
            color=PALETTE[idx],
            label=setting,
            linewidth=1.5,
            linestyle='--',
        )

    _style_spines(ax)
    ax.set_xlabel("BLEU")
    ax.set_ylabel("Acceptance Rate")

    secax = ax.secondary_xaxis('top')
    secax.set_xticks(sorted_bleu)
    secax.set_xticklabels(sorted_langs, fontsize=10, fontweight='bold')
    secax.tick_params(axis='x', length=0, pad=2)

    ax.legend(
        frameon=False,
        fontsize=10,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(settings),
        title=None,
        borderaxespad=0.1,
    )

    _finalize(fig, filename)


def create_graphs(data: dict):
    _bar_plot(data["tps"],             "Tokens / Second (Spec)", "tps_spec")
    _bar_plot(data["speedup"],         "Speedup",                "speedup")
    _bar_plot(data["acceptance rate"], "Acceptance Rate",        "acceptance_rate")

    _bleu_acceptance_line_plot(data["bleu"], data["acceptance rate"], "bleu_vs_acceptance")

    _violin_plot(data["forward_pass"], "Average Forward Pass (ms)", "avg_forward_pass")


if __name__ == "__main__":
    USE_PLACEHOLDER = True

    if USE_PLACEHOLDER:
        data = make_placeholder_data()
    else:
        raise NotImplementedError("Real data loading not yet implemented")

    create_graphs(data)
