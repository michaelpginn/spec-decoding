import colorsys
import logging
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
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
    "speedup_factor": "Speed-up Fact. (f)",
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


def _log_stats(data: pd.DataFrame, group: str, y: str, label: str):
    logger.info(f"Stats for {label} ({y}):")
    for group_value, sub in data.groupby(group):
        logger.info(
            f"  {group_value}: avg={sub[y].mean():.4f}, min={sub[y].min():.4f}, max={sub[y].max():.4f}"
        )


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


def _bar_plot(data: pd.DataFrame, y: str, y_std: str, filename: str, show_legend: bool = True):
    _log_stats(data, "setting", y, filename)
    fig, ax = plt.subplots(figsize=(8, 1.6 if show_legend else 1.3))
    bar_palette = ['#8C8C8C', *PALETTE[1:len(SETTINGS)]]
    sns.barplot(
        data=data,
        x="language",
        y=y,
        hue="setting",
        hue_order=SETTINGS,
        order=langs,
        palette=bar_palette,
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

    if y != "speedup_factor":
        ax.set_ylim(bottom=0)

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='y', which='minor', length=0)
    ax.grid(which='minor', axis='y', color='#E5E5E5', linewidth=0.5, alpha=0.7)

    _style_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel(KEY_TO_TITLE[y])

    if show_legend:
        ax.legend(
            frameon=False,
            fontsize=10,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(SETTINGS),
            title=None,
            borderaxespad=0.1,
        )
    else:
        ax.get_legend().remove()

    _finalize(fig, filename)


def _violin_plot(data, x: str, y: str, y_std: str):
    _log_stats(data, x, y, y)
    order = [m for m in FORWARD_PASS_MODELS if m in set(data['model_size'])]
    fig, ax = plt.subplots(figsize=(4, 1.5))

    sns.violinplot(
        data=data,
        x=y,
        y=x,
        order=order,
        hue=x,
        hue_order=order,
        palette=_shades(PALETTE[0], len(order)),
        legend=False,
        inner="quartile",
        linewidth=0.6,
        width=0.95,
        cut=0,
        orient="h",
        ax=ax,
    )
    ax.set_xlim(left=0)

    _style_spines(ax)
    ax.set_ylabel("")
    ax.set_xlabel(KEY_TO_TITLE[y])

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
    _log_stats(data, "setting", "sentence_avg_acceptance_rate", "chrf_acceptance")
    fig, ax = plt.subplots(figsize=(8, 2))
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

    r = baseline_chrfs['chrf2'].corr(baseline_chrfs['sentence_avg_acceptance_rate'])
    logger.info(f"Pearson r (chrF++ vs acceptance rate) for Baseline: r={r:.4f} (n={len(baseline_chrfs)})")

    ax.scatter(
        baseline_chrfs['chrf2'], baseline_chrfs['sentence_avg_acceptance_rate'],
        facecolors="none",
        edgecolors="#8C8C8C",
        s=40,
        linewidths=1.0,
        zorder=3,
    )
    slope, intercept = np.polyfit(baseline_chrfs['chrf2'], baseline_chrfs['sentence_avg_acceptance_rate'], 1)
    x_fit = np.array([baseline_chrfs['chrf2'].min(), baseline_chrfs['chrf2'].max()])
    ax.plot(x_fit, slope * x_fit + intercept, color="#8C8C8C", linewidth=1.5, linestyle='--')
    ax.text(
        0.98, 0.02, f"r = {r:.3f}",
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=10, fontweight='bold',
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
    _finalize(fig, "chrf_acceptance")

def _task_acceptance_scatter(data: pd.DataFrame):
    settings_to_show = ["Baseline", "Distilled (task)"]
    setting_to_color = {"Baseline": "#8C8C8C", "Distilled (task)": PALETTE[2]}

    pivoted = (
        data[data["setting"].isin(settings_to_show)]
        .pivot_table(
            index=["language", "setting"],
            columns="task",
            values="sentence_avg_acceptance_rate",
        )
        .dropna(subset=["translation", "story_gen"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(4, 3))

    x_lo, x_hi = pivoted["translation"].min(), pivoted["translation"].max()
    y_lo, y_hi = pivoted["story_gen"].min(), pivoted["story_gen"].max()
    x_span, y_span = x_hi - x_lo, y_hi - y_lo
    x_lims = (max(0, x_lo - 0.05 * x_span), min(1, x_hi + 0.05 * x_span))
    y_lims = (max(0, y_lo - 0.08 * y_span), min(1, y_hi + 0.08 * y_span))
    diag_lo = min(x_lims[0], y_lims[0])
    diag_hi = max(x_lims[1], y_lims[1])
    # ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="#BFBFBF", linewidth=0.8, linestyle="--", zorder=0)

    paired = pivoted.pivot(index="language", columns="setting", values=["translation", "story_gen"])
    for lang, row in paired.iterrows():
        if pd.isna(row[("translation", "Baseline")]) or pd.isna(row[("translation", "Distilled (task)")]):
            continue
        ax.annotate(
            "",
            xy=(row[("translation", "Distilled (task)")], row[("story_gen", "Distilled (task)")]),
            xytext=(row[("translation", "Baseline")], row[("story_gen", "Baseline")]),
            arrowprops=dict(arrowstyle="-", color="#999999", alpha=0.25, linewidth=0.6, shrinkA=5, shrinkB=5),
            zorder=1,
        )

    for setting in settings_to_show:
        sub = pivoted[pivoted["setting"] == setting]
        if setting == "Baseline":
            ax.scatter(
                sub["translation"], sub["story_gen"],
                facecolors="none",
                edgecolors=setting_to_color[setting],
                label=setting,
                s=40,
                zorder=3,
                linewidths=1.0,
            )
        else:
            ax.scatter(
                sub["translation"], sub["story_gen"],
                color=setting_to_color[setting],
                label=setting,
                s=40,
                zorder=3,
                edgecolors="black",
                linewidths=0.4,
            )
        for _, row in sub.iterrows():
            ax.annotate(
                row["language"],
                (row["translation"], row["story_gen"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_xlabel("Acceptance Rate (Translation)")
    ax.set_ylabel("Acceptance Rate (Story Generation)")
    ax.legend(
        frameon=False,
        fontsize=10,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(settings_to_show),
        borderaxespad=0.1,
    )
    _style_spines(ax)
    _finalize(fig, "task_acceptance_scatter")


def _size_scaling_plot(data: pd.DataFrame, y: str, filename: str):
    size_order = ["0.8B", "2B", "4B"]
    lang_order = ["amh", "ber", "grn"]
    sub = data[
        (data["task"] == "translation")
        & (data["setting"] == "Baseline")
        & (data["language"].isin(lang_order))
        & (data["model_size"].isin(size_order))
    ]
    pivot = sub.pivot_table(index="model_size", columns="language", values=y).reindex(size_order)

    fig, ax = plt.subplots(figsize=(4, 2.2))
    colors = _shades(PALETTE[0], len(lang_order))
    xs = np.arange(len(size_order))
    for lang, color in zip(lang_order, colors):
        ys = pivot[lang].to_numpy()
        ax.plot(xs, ys, color=color, linewidth=1.5, marker="o", markersize=4, markeredgecolor="black", markeredgewidth=0.4)
        last = np.where(~np.isnan(ys))[0]
        if len(last):
            i = last[-1]
            ax.annotate(
                lang,
                (xs[i], ys[i]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=color,
                va="center",
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(size_order)
    ax.set_xlim(-0.2, len(size_order) - 1 + 0.5)
    ax.set_xlabel("Draft Model Size")
    ax.set_ylabel(KEY_TO_TITLE[y])
    _style_spines(ax)
    _finalize(fig, filename)


def create_graphs(data: pd.DataFrame):
    # _bar_plot(data,             "Tokens / Second (Spec)", "tps_spec")
    translation_data = data[(data['task'] == 'translation') & (data['model_size'].isin(['0.8B', 'N-Gram']))]
    _bar_plot(translation_data, "sentence_avg_tokens_per_second", "sentence_std_tokens_per_second", "translation_tps")
    _bar_plot(translation_data, "speedup_factor", "speedup_factor_std", "translation_speedup", show_legend=False)
    _bar_plot(translation_data, "sentence_avg_acceptance_rate", "sentence_std_acceptance_rate", "translation_acceptance")
    _chrf_acceptance_plot(translation_data)

    story_data = data[(data['task'] == 'story_gen')  & (data['model_size'].isin(['0.8B', 'N-Gram']))]
    _bar_plot(story_data, "sentence_avg_tokens_per_second", "sentence_std_tokens_per_second", "story_tps")
    _bar_plot(story_data, "speedup_factor", "speedup_factor_std", "story_speedup")
    _bar_plot(story_data, "sentence_avg_acceptance_rate", "sentence_std_acceptance_rate", "story_acceptance")

    _task_acceptance_scatter(data[data['model_size'].isin(['0.8B', 'N-Gram'])])


    forward_pass_data = data[
        (data["task"] == "translation") & (data["setting"] == "Baseline")
    ][["model_size", "average_draft_time", "draft_time_std"]]
    verifier_data = (
        data[(data["task"] == "translation") & (data["setting"] == "Baseline") & (data["model_size"] == "0.8B")]
        .drop(columns=["average_draft_time", "draft_time_std"])
        .rename(
            columns={
                "average_verifier_time": "average_draft_time",
                "verifier_time_std": "draft_time_std",
            }
        )[translation_data["setting"] == "Baseline"][
            ["model_size", "average_draft_time", "draft_time_std"]
        ]
    )
    verifier_data["model_size"] = "9B"
    ngram_pass_data = data[(data["task"] == "translation") & (data["setting"] == "N-Gram")][
        ["model_size", "average_draft_time", "draft_time_std"]
    ]
    _violin_plot(
        pd.concat([forward_pass_data, verifier_data, ngram_pass_data]),  # type:ignore
        "model_size",
        "average_draft_time",
        "draft_time_std",
    )

    _size_scaling_plot(data, "sentence_avg_acceptance_rate", "size_scaling_acceptance")
    _size_scaling_plot(data, "speedup_factor", "size_scaling_speedup")


def _pinsker_plot(kl_df: pd.DataFrame, spec_df: pd.DataFrame):
    kl_df = kl_df.rename(columns={"language_code": "language"})

    distilled_spec = spec_df[
        (spec_df["task"] == "translation") &
        (spec_df["setting"] == "Distilled (task)") & (spec_df['model_size'].isin(['0.8B', 'N-Gram']))
    ].copy()
    distilled_spec["distill_type"] = distilled_spec["setting"].map({
        "Distilled (task)": "translation",
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

    type_to_color  = {"translation": PALETTE[2], "general": PALETTE[1]}
    type_to_label  = {"translation": "Distilled (translation)", "general": "Distilled (general)"}
    type_to_marker = {"translation": "o", "general": "s"}

    fig, ax = plt.subplots(figsize=(4, 3))

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

    ax.set_xlabel("KL Divergence (teacher || student)")
    ax.set_ylabel("Acceptance Rate (α)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_spines(ax)
    _finalize(fig, "pinsker_bound")


if __name__ == "__main__":
    kl_data = pd.read_csv("viz/kl_results.csv")
    spec_data = load_real_data()
    spec_data = spec_data[spec_data['language'] != 'zh']
    _pinsker_plot(kl_data, spec_data)
    create_graphs(spec_data)
