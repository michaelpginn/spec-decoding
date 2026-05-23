import colorsys
import logging
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import wandb

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
    "sentence_avg_tokens_per_second": "Tokens/s",
    "sentence_avg_acceptance_rate": "Acceptance Rate (α)",
    "speedup_factor": "Speedup Factor",
    "average_draft_time": "Forward Pass Time (s)"
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
    for run in tqdm(wandb.Api().runs(path="lecs-general/speculative decoding v2", lazy=False, filters={"state": "finished"})):
        if run.config["draft_model"] is None:
            setting = "N-Gram"
            size = "N-Gram"
        else:
            size = re.match(r".*Qwen3.5-([\d\.]+B)", run.config["draft_model"]).group(1) # type:ignore
            if "general" in run.config["draft_model"]:
                setting = "Distilled (general)"
            elif "translation" in run.config["draft_model"]:
                setting = "Distilled (task)"
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


def _bar_plot(data: pd.DataFrame, y: str, y_std: str, filename: str):
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
        ys = np.array([patch.get_height() for patch in container])
        stds = s_df[y_std].to_numpy()
        lower = np.minimum(stds, ys)
        try:
            ax.errorbar(
                xs, ys,
                yerr=[lower, stds],
                fmt='none',
                ecolor='#444444',
                capsize=2,
                linewidth=0.6,
                capthick=0.6,
            )
        except:
            breakpoint()

    ax.set_ylim(bottom=0)

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

    _finalize(fig, filename)


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


def load_distill_data() -> pd.DataFrame:
    records = []
    logger.info("Loading distillation runs")
    for run in tqdm(wandb.Api().runs(
        path="lecs-general/spec-dec-distill",
        lazy=False,
        filters={"state": "finished"},
    )):
        best_loss = run.summary.get("eval/best_loss")
        if best_loss is None:
            continue
        student = run.config.get("student_model", "")
        m = re.match(r".*Qwen3\.5-([\d\.]+B)", student)
        if not m:
            continue
        records.append({
            "language": run.config["language_code"],
            "model_size": m.group(1),
            "task": run.config.get("task", "translation"),
            "eval_ce_loss": best_loss,
        })
    return pd.DataFrame.from_records(records)


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
    translation_data = data[data['task'] == 'translation']
    _bar_plot(translation_data, "sentence_avg_tokens_per_second", "sentence_std_tokens_per_second", "translation_tps")
    _bar_plot(translation_data, "speedup_factor", "speedup_factor_std", "translation_speedup")
    _bar_plot(translation_data, "sentence_avg_acceptance_rate", "sentence_std_acceptance_rate", "translation_acceptance")
    _chrf_acceptance_plot(translation_data)

    story_data = data[data['task'] == 'story_gen']
    _bar_plot(story_data, "sentence_avg_tokens_per_second", "sentence_std_tokens_per_second", "story_tps")
    _bar_plot(story_data, "speedup_factor", "speedup_factor_std", "story_speedup")
    _bar_plot(story_data, "sentence_avg_acceptance_rate", "sentence_std_acceptance_rate", "story_acceptance")


    forward_pass_data = translation_data[translation_data["setting"] == "Baseline"][
        ["model_size", "average_draft_time", "draft_time_std"]
    ]
    verifier_data = translation_data.drop(columns=["average_draft_time", "draft_time_std"]).rename(
        columns={
            "average_verifier_time": "average_draft_time",
            "verifier_time_std": "draft_time_std",
        }
    )[translation_data["setting"] == "Baseline"][["model_size", "average_draft_time", "draft_time_std"]]
    verifier_data["model_size"] = "9B"
    ngram_pass_data = translation_data[translation_data["setting"] == "N-Gram"][
        ["model_size", "average_draft_time", "draft_time_std"]
    ]
    _violin_plot(
        pd.concat([forward_pass_data, verifier_data, ngram_pass_data]),  # type:ignore
        "model_size",
        "average_draft_time",
        "draft_time_std",
    )


def _pinsker_plot(kl_df: pd.DataFrame, spec_df: pd.DataFrame):
    kl_df = kl_df.rename(columns={"language_code": "language"})

    distilled_spec = spec_df[
        (spec_df["task"] == "translation") &
        (spec_df["setting"].isin(["Distilled (translation)", "Distilled (general)"]))
    ].copy()
    distilled_spec["distill_type"] = distilled_spec["setting"].map({
        "Distilled (translation)": "translation",
        "Distilled (general)": "general",
    })

    merged = distilled_spec.groupby(["language", "distill_type"])["sentence_avg_acceptance_rate"].mean().reset_index()
    merged = merged.merge(kl_df[["language", "kl_divergence"]], on="language", how="inner")

    if merged.empty:
        logger.warning("Pinsker plot: no data after merging distill and spec decode runs")
        return

    # Pinsker bound curve: acceptance >= 1 - sqrt(KL/2)
    kl_max = merged["kl_divergence"].max() + 0.3
    kl_range = np.linspace(0, kl_max, 300)
    pinsker_bound = np.maximum(0.0, 1.0 - np.sqrt(kl_range / 2))

    type_to_color  = {"translation": PALETTE[0], "general": PALETTE[1]}
    type_to_label  = {"translation": "Distilled (translation)", "general": "Distilled (general)"}
    type_to_marker = {"translation": "o", "general": "s"}

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        kl_range, pinsker_bound,
        color="black", linewidth=1.2, linestyle="--",
        label="Pinsker bound",
        zorder=1,
    )

    for dtype in ["translation", "general"]:
        subset = merged[merged["distill_type"] == dtype]
        if subset.empty:
            continue
        ax.scatter(
            subset["kl_divergence"],
            subset["sentence_avg_acceptance_rate"],
            color=type_to_color[dtype],
            label=type_to_label[dtype],
            marker=type_to_marker[dtype],
            s=40,
            zorder=3,
            edgecolors="black",
            linewidths=0.4,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["language"],
                (row["kl_divergence"], row["sentence_avg_acceptance_rate"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("KL Divergence (teacher ∥ student)")
    ax.set_ylabel("Acceptance Rate (α)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_spines(ax)
    _finalize(fig, "pinsker_bound")


if __name__ == "__main__":
    kl_data = pd.read_csv("viz/kl_results.csv")
    spec_data = load_real_data()
    _pinsker_plot(kl_data, spec_data)
    create_graphs(spec_data)
