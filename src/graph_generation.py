import colorsys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import wandb
import re
import logging

logger = logging.getLogger(__name__)

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

SETTINGS = ["Baseline", "N-Gram", "Distilled (task)", "Distilled (general)"]
FORWARD_PASS_MODELS = ["N-Gram", "0.8B", "2B", "4B", "9B"]
KEY_TO_TITLE = {
    "sentence_avg_tokens_per_second": "Tokens/sec",
    "sentence_avg_acceptance_rate": "Acceptance Rate (α)",
    "speedup_factor": "Speedup Factor",
    "average_draft_time": "Forward Pass Time (ms)"
}

def _shades(hex_color: str, n: int, light: float = 0.78, dark: float = 0.25) -> list[str]:
    h, _, s = colorsys.rgb_to_hls(*mcolors.to_rgb(hex_color))
    return [
        mcolors.to_hex(colorsys.hls_to_rgb(h, ll, s))
        for ll in np.linspace(light, dark, n)
    ]


def load_real_data() -> pd.DataFrame:
    records = []
    logger.info("Loading runs")
    for run in tqdm(wandb.Api().runs(path="lecs-general/speculative decoding v2", lazy=False, filters={"state": "finished", "config.task": "translation"})):
        if run.config["draft_model"] is None:
            setting = "N-Gram"
            size = "N-Gram"
        else:
            size = re.match(r".*Qwen3.5-([\d\.]+B)", run.config["draft_model"]).group(1) # type:ignore
            if "general" in run.config["draft_model"]:
                setting = "Distilled (general)"
            elif "translation" in run.config["draft_model"]:
                setting = "Distilled (translation)"
            else:
                setting = "Baseline"
        records.append({
            "language": run.config["language_code"],
            "draft_model": run.config["draft_model"] or "ngram",
            "gamma": run.config["gamma"],
            "setting":setting,
            "model_size": size,
            "task": run.config["task"],
            **run.summary,
        })
    df = pd.DataFrame.from_records(records)
    del records
    df = df[df["sentence_avg_acceptance_rate"].notna()]
    best_gamma = df.groupby(["language", "draft_model", "task"])["sentence_avg_acceptance_rate"].idxmax()
    df = df.loc[best_gamma]
    return df


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


def _bar_plot(data: pd.DataFrame, y: str, y_std: str):
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.barplot(
        data=data,
        x="language",
        y=y,
        hue="setting",
        hue_order=SETTINGS,
        order=langs,
        palette=PALETTE[:len(SETTINGS)],
        edgecolor='#333333',
        linewidth=0.4,
        errorbar=None,
        ax=ax,
    )

    for container, setting in zip(ax.containers, SETTINGS):
        s_df = data[data["setting"] == setting]
        xs = [patch.get_x() + patch.get_width() / 2 for patch in container]
        ys = [patch.get_height() for patch in container]
        try:
            ax.errorbar(
                xs, ys,
                yerr=s_df[y_std].to_numpy(),
                fmt='none',
                ecolor='#333333',
                capsize=3,
                linewidth=0.8,
            )
        except:
            breakpoint()

    _style_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel(KEY_TO_TITLE[y])

    ax.legend(
        frameon=False,
        fontsize=10,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(SETTINGS),
        title=None,
        borderaxespad=0.1,
    )

    _finalize(fig, y)


def _violin_plot(data, x: str, y: str, y_std: str):
    order = [m for m in FORWARD_PASS_MODELS if m in set(data['model_size'])]
    fig, ax = plt.subplots(figsize=(4, 3))

    sns.violinplot(
        data=data,
        x=x,
        y=y,
        order=order,
        hue=x,
        hue_order=order,
        palette=_shades(PALETTE[0], len(order)),
        legend=False,
        inner="quartile",
        linewidth=0.6,
        width=0.95,
        cut=0,
        ax=ax,
    )
    ax.set_ylim(bottom=0)

    _style_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel(KEY_TO_TITLE[y])
    ax.tick_params(axis='x', labelsize=10, rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    _finalize(fig, y)


def _chrf_acceptance_plot(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    data = data.copy()
    baseline_chrf_by_lang = (
        data[data['setting'] == 'Baseline'].set_index('language')['chrf2']
    )
    data['chrf2'] = data['language'].map(baseline_chrf_by_lang)
    data = data.dropna(subset=['chrf2'])
    data = data.sort_values(by=["chrf2"])
    baseline_chrfs = data[data['setting'] == 'Baseline']

    for x in baseline_chrfs['chrf2']:
        ax.axvline(x, linestyle='--', color='#BFBFBF', linewidth=0.6, alpha=0.7, zorder=0)
    settings = [s for s in SETTINGS if s in set(data['setting'])]
    for idx, setting in enumerate(settings):
        setting_data = data[data['setting'] == setting]
        ax.plot(
            setting_data['chrf2'], setting_data['sentence_avg_acceptance_rate'],
            marker='o',
            markersize=5,
            color=PALETTE[idx],
            label=setting,
            linewidth=1.5,
            linestyle='--',
        )

    _style_spines(ax)
    ax.set_xlabel("chrF++")
    ax.set_ylabel("Acceptance Rate")

    fig.canvas.draw()
    base_y = 1.01
    row_spacing = 0.05
    inv = ax.transData.inverted()
    placed: list[tuple[float, float, int]] = []
    for _, lrow in baseline_chrfs.sort_values('chrf2').iterrows():
        txt = ax.text(
            lrow['chrf2'], base_y, lrow['language'],
            transform=ax.get_xaxis_transform(),
            fontsize=10, fontweight='bold', ha='center', va='bottom',
        )
        bbox = txt.get_window_extent()
        x_left = inv.transform((bbox.x0, 0))[0]
        x_right = inv.transform((bbox.x1, 0))[0]
        row = 0
        while any(r == row and not (x_right < pl or x_left > pr) for pl, pr, r in placed):
            row += 1
        txt.set_y(base_y + row * row_spacing)
        placed.append((x_left, x_right, row))
    max_row = max((r for _, _, r in placed), default=0)
    legend_y = base_y + (max_row + 1) * row_spacing + 0.02

    ax.legend(
        frameon=False,
        fontsize=10,
        loc='lower center',
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(settings),
        title=None,
        borderaxespad=0.1,
    )

    _finalize(fig, "chrf_acceptance")

def create_graphs(data: pd.DataFrame):
    # _bar_plot(data,             "Tokens / Second (Spec)", "tps_spec")
    _bar_plot(data, "sentence_avg_tokens_per_second", "sentence_std_tokens_per_second")
    _bar_plot(data, "speedup_factor", "speedup_factor_std")
    _bar_plot(data, "sentence_avg_acceptance_rate", "sentence_std_acceptance_rate")
    _chrf_acceptance_plot(data)

    forward_pass_data = data[data['setting'] == 'Baseline'][['model_size', "average_draft_time", "draft_time_std"]]
    ngram_pass_data = data[data['setting'] == 'N-Gram'][['model_size', "average_draft_time", "draft_time_std"]]
    _violin_plot(
        pd.concat([forward_pass_data, ngram_pass_data]),  # type:ignore
        "model_size", "average_draft_time", "draft_time_std")


if __name__ == "__main__":
    data = load_real_data()
    create_graphs(data)
