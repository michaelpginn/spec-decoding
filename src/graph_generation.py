import colorsys
import logging
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.transforms import Bbox
from tqdm import tqdm

import wandb

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
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

langs = ["amh","ber","chr","grn","haw","ibo","npi","oci","que","yor","zgh"]

PALETTE = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7']

SETTINGS = ["Baseline", "N-Gram", "Distilled (task)", "Distilled (general)"]

# Per model family: how to parse the draft-model size, the order of models to
# show along the forward-pass axis, the draft size used for the setting
# comparison plots, and the size label used for the verifier.
FAMILIES = {
    "qwen": {
        "size_pattern": r".*Qwen3.5-([\d\.]+B)",
        "forward_pass_models": ["N-Gram", "0.8B", "2B", "4B", "9B"],
        "baseline_size": "0.8B",
        "verifier_size": "9B",
    },
    "llama": {
        "size_pattern": r".*Llama-3\.2-([\d\.]+B)",
        "forward_pass_models": ["N-Gram", "1B", "3B"],
        "baseline_size": "1B",
        "verifier_size": "3B",
    },
}
KEY_TO_TITLE = {
    "sentence_avg_tokens_per_second": "Tokens/s",
    "sentence_avg_acceptance_rate": "Acceptance Rate (α)",
    "speedup_factor": "Speed-up Fact. (f)",
    "average_draft_time": "Forward Pass Time (s)"
}

# One colour per model family, shared by every plot that distinguishes them, so
# blue always means Qwen and green always means Llama. Per-family plots (violin,
# size scaling) build their shade ramp from the same colour.
FAMILY_STYLE = {
    "qwen": {"marker": "o", "color": PALETTE[0], "label": "Qwen"},
    "llama": {"marker": "s", "color": PALETTE[2], "label": "Llama"},
}


# Colour per setting on the plots that compare settings head to head (the dual
# acceptance scatters and the n-gram vs. distilled scatters).
SETTING_COLOR = {
    "Baseline": PALETTE[0],
    "Distilled (task)": PALETTE[1],
    "Distilled (general)": PALETTE[4],
    "N-Gram": "black",
}


def _family_color(family: str | None) -> str:
    return FAMILY_STYLE.get(family or "", {}).get("color", PALETTE[0])  # type:ignore[return-value]


def _shades(hex_color: str, n: int, light: float = 0.78, dark: float = 0.25) -> list[str]:
    h, _, s = colorsys.rgb_to_hls(*mcolors.to_rgb(hex_color))
    return [
        mcolors.to_hex(colorsys.hls_to_rgb(h, ll, s))
        for ll in np.linspace(light, dark, n)
    ]


def _detect_family(target_model: str) -> str:
    """Map a run's target model onto a model family in FAMILIES."""
    if "Llama" in target_model:
        return "llama"
    return "qwen"


def load_real_data() -> pd.DataFrame:
    records = []
    logger.info("Loading runs")
    for run in tqdm(wandb.Api().runs(path="lecs-general/speculative decoding v2", lazy=False, filters={"state": "finished"})):
        family = _detect_family(run.config["target_model"])
        draft_model = run.config["draft_model"]
        # N-Gram runs are flagged by draft_model_type. Qwen leaves draft_model
        # unset for these, but Llama keeps the base draft model name, so we can't
        # rely on draft_model being None to detect them.
        if run.config.get("draft_model_type") == "ngram" or draft_model is None:
            setting = "N-Gram"
            size = "N-Gram"
            draft_label = "ngram"
        else:
            size = re.match(FAMILIES[family]["size_pattern"], draft_model).group(1) # type:ignore
            if "general" in draft_model:
                setting = "Distilled (general)"
            elif "translation" in draft_model:
                setting = "Distilled (task)"
            else:
                setting = "Baseline"
            draft_label = draft_model
        records.append({
            "language": run.config["language_code"],
            "family": family,
            "draft_model": draft_label,
            "gamma": run.config["gamma"],
            "setting":setting,
            "model_size": size,
            "task": run.config["task"],
            **run.summary,
        })
    df = pd.DataFrame.from_records(records)
    del records
    df = df[df["sentence_avg_acceptance_rate"].notna()]
    # Include family because the "ngram" draft label is shared across families
    # and would otherwise collide.
    best_gamma = df.groupby(["family", "language", "draft_model", "task"])["sentence_avg_acceptance_rate"].idxmax()
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


def _finalize(fig, filename: str, family: str | None = None):
    plt.tight_layout(pad=0.2)
    out_dir = Path("viz") / family if family else Path("viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{filename}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.show()
    plt.close(fig)


def _bar_plot(data: pd.DataFrame, y: str, y_std: str, filename: str, family: str | None = None, show_legend: bool = True):
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

    _finalize(fig, filename, family)


def _violin_plot(data, x: str, y: str, y_std: str, family: str):
    _log_stats(data, x, y, y)
    order = [m for m in FAMILIES[family]["forward_pass_models"] if m in set(data['model_size'])]
    fig, ax = plt.subplots(figsize=(4, 1.5))

    sns.violinplot(
        data=data,
        x=y,
        y=x,
        order=order,
        hue=x,
        hue_order=order,
        palette=_shades(_family_color(family), len(order)),
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

    _finalize(fig, y, family)


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


def _chrf_acceptance_plot(data: pd.DataFrame, family: str | None = None):
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
    _finalize(fig, "chrf_acceptance", family)

def _speedup_threshold_alpha(c: float, gammas=range(2, 6)) -> tuple[float, int] | None:
    """Acceptance rate at which the speed-up factor first reaches 1.

    Solves f = 1 = (1 - a^(g+1)) / ((1 - a)(g*c + 1)) for a, where c is the
    ratio of mean draft to mean verifier forward-pass times. f is increasing in
    a (f(0) = 1/(g*c+1) < 1), so bisection finds the unique root per gamma; the
    optimal gamma is the one with the lowest threshold, since any alpha above it
    yields f > 1 for at least one gamma in the range.
    """
    def f(alpha: float, gamma: int) -> float:
        return (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (gamma * c + 1))

    best: tuple[float, int] | None = None
    for gamma in gammas:
        # Ceiling of f as alpha -> 1 is (g+1)/(g*c+1); unreachable if <= 1.
        if (gamma + 1) / (gamma * c + 1) <= 1:
            continue
        lo, hi = 0.0, 1.0 - 1e-6
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid, gamma) < 1:
                lo = mid
            else:
                hi = mid
        root = (lo + hi) / 2
        if best is None or root < best[0]:
            best = (root, gamma)
    return best


def _task_acceptance_scatter(
    data: pd.DataFrame,
    family: str | None = None,
    distill_setting: str = "Distilled (task)",
    filename: str = "task_acceptance_scatter",
):
    settings_to_show = ["Baseline", distill_setting]
    setting_to_color = SETTING_COLOR

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
    if pivoted.empty or distill_setting not in set(pivoted["setting"]):
        logger.info(f"[{family}] no {distill_setting} runs with both tasks; skipping {filename}")
        return

    fig, ax = plt.subplots(figsize=(4, 3))

    x_lo, x_hi = pivoted["translation"].min(), pivoted["translation"].max()
    y_lo, y_hi = pivoted["story_gen"].min(), pivoted["story_gen"].max()
    x_span, y_span = x_hi - x_lo, y_hi - y_lo
    x_lims = (max(0, x_lo - 0.05 * x_span), min(1, x_hi + 0.05 * x_span))
    y_lims = (max(0, y_lo - 0.08 * y_span), min(1, y_hi + 0.08 * y_span))
    diag_lo = min(x_lims[0], y_lims[0])
    diag_hi = max(x_lims[1], y_lims[1])
    # ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="#BFBFBF", linewidth=0.8, linestyle="--", zorder=0)

    # Acceptance rate above which speculative decoding is a net win (f > 1) on
    # each task. The cost ratio c is the mean draft / mean verifier
    # forward-pass time across the runs shown here.
    timings = data[data["setting"].isin(settings_to_show)]
    mu_d = timings["average_draft_time"].mean()
    mu_v = timings["average_verifier_time"].mean()
    threshold = _speedup_threshold_alpha(mu_d / mu_v) if mu_v > 0 and mu_d > 0 else None
    if threshold is not None:
        alpha_star, gamma_star = threshold
        logger.info(
            f"[{family}] {filename}: c={mu_d / mu_v:.4f}, optimal gamma={gamma_star}, "
            f"f>1 above alpha={alpha_star:.4f}"
        )
        ax.axvline(alpha_star, color="black", linestyle="--", linewidth=0.8, zorder=1)
        ax.axhline(alpha_star, color="black", linestyle="--", linewidth=0.8, zorder=1)
        # Keep the threshold in frame even when every run sits on one side of it.
        pad = 0.02
        x_lims = (min(x_lims[0], alpha_star - pad), max(x_lims[1], alpha_star + pad))
        y_lims = (min(y_lims[0], alpha_star - pad), max(y_lims[1], alpha_star + pad))
    else:
        logger.info(f"[{family}] {filename}: no gamma in [2, 5] can reach f > 1; skipping threshold lines")

    paired = pivoted.pivot(index="language", columns="setting", values=["translation", "story_gen"])
    for lang, row in paired.iterrows():
        if pd.isna(row[("translation", "Baseline")]) or pd.isna(row[("translation", distill_setting)]):
            continue
        ax.annotate(
            "",
            xy=(row[("translation", distill_setting)], row[("story_gen", distill_setting)]),
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
    handles, _ = ax.get_legend_handles_labels()
    if threshold is not None:
        handles.append(
            plt.Line2D([], [], color="black", linestyle="--", linewidth=0.8,
                       label="Positive speed-up threshold")
        )
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=10,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(settings_to_show),
        borderaxespad=0.1,
    )
    _style_spines(ax)
    _finalize(fig, filename, family)


TASK_TO_TITLE = {"translation": "Translation", "story_gen": "Story Generation"}


def _baseline_family_rows(data: pd.DataFrame, task: str) -> pd.DataFrame:
    """Baseline runs at each family's comparison draft size, for one task."""
    return pd.concat([
        data[
            (data["family"] == fam)
            & (data["task"] == task)
            & (data["setting"] == "Baseline")
            & (data["model_size"] == FAMILIES[fam]["baseline_size"])
        ]
        for fam in sorted(data["family"].unique())
    ])


def _ngram_vs_distilled_scatter(data: pd.DataFrame, task: str, filename: str, family: str = "qwen"):
    """Acceptance rate against speed-up factor for the two cheap drafters: the
    n-gram model and, per language, whichever distilled model does better. The
    distilled dot is coloured by which distillation won, so the figure shows
    both how the two approaches compare and which flavour of distillation the
    language preferred."""
    distill_settings = ["Distilled (task)", "Distilled (general)"]
    d = data[(data["family"] == family) & (data["task"] == task)]
    cols = ["language", "setting", "sentence_avg_acceptance_rate", "speedup_factor"]

    ngram = d[d["setting"] == "N-Gram"].dropna(subset=["speedup_factor"])[cols]
    distilled = d[
        d["setting"].isin(distill_settings)
        & (d["model_size"] == FAMILIES[family]["baseline_size"])
    ].dropna(subset=["speedup_factor"])[cols]
    if ngram.empty and distilled.empty:
        logger.info(f"[{family}] no n-gram or distilled {task} runs; skipping {filename}")
        return

    # "Best" distilled model per language, by speed-up factor.
    best = distilled.loc[distilled.groupby("language")["speedup_factor"].idxmax()]
    won = best.groupby("setting").size().to_dict()
    logger.info(f"[{family}] {filename}: best distillation per language: {won}")

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.axhline(1.0, color="black", linewidth=1.0, zorder=1)

    # Join each language's two drafters so the pair is readable at a glance.
    paired = ngram.set_index("language").join(
        best.set_index("language"), lsuffix="_ngram", rsuffix="_best", how="inner"
    )
    for _, row in paired.iterrows():
        ax.plot(
            [row["sentence_avg_acceptance_rate_ngram"], row["sentence_avg_acceptance_rate_best"]],
            [row["speedup_factor_ngram"], row["speedup_factor_best"]],
            color="#999999", alpha=0.25, linewidth=0.6, zorder=1,
        )

    groups = [("N-Gram", ngram)] + [(s, best[best["setting"] == s]) for s in distill_settings]
    for setting, sub in groups:
        if sub.empty:
            continue
        ax.scatter(
            sub["sentence_avg_acceptance_rate"], sub["speedup_factor"],
            color=SETTING_COLOR[setting], label=setting,
            s=40, zorder=3, edgecolors="black", linewidths=0.4,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["language"],
                (row["sentence_avg_acceptance_rate"], row["speedup_factor"]),
                fontsize=7, xytext=(3, 3), textcoords="offset points",
            )
        logger.info(
            f"[{family}] {filename}: {setting} n={len(sub)}, "
            f"avg α={sub['sentence_avg_acceptance_rate'].mean():.4f}, "
            f"avg f={sub['speedup_factor'].mean():.4f}"
        )

    ax.set_xlabel(f"Acceptance Rate ({TASK_TO_TITLE.get(task, task)})")
    ax.set_ylabel(KEY_TO_TITLE["speedup_factor"])
    ax.legend(
        frameon=False, fontsize=8, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=3, borderaxespad=0.1,
        columnspacing=1.0, handletextpad=0.3,
    )
    _style_spines(ax)
    _finalize(fig, filename, family)


def _resourcedness_order(present: list[str], counts: pd.DataFrame) -> list[str]:
    """Languages left to right in the same order the resourcedness plots lay
    them out: the unknown counts (-1) first, then ascending token count. A
    language absent from the counts table is treated as unknown."""
    words = counts.set_index("language_code")["words"]
    unknown = [lang for lang in present if words.get(lang, -1) <= 0]
    known = sorted((lang for lang in present if words.get(lang, -1) > 0), key=lambda l: words[l])
    return unknown + known


def _baseline_speedup_bars(data: pd.DataFrame, counts: pd.DataFrame, task: str, filename: str):
    """Baseline speed-up factor per language, Qwen and Llama as paired bars.
    Bars are anchored at f = 1 (break-even), so a run that is slower than
    autoregressive decoding reads as a bar below the line. Languages are ordered
    by resourcedness to match the acceptance-vs-resourcedness scatters."""
    d = _baseline_family_rows(data, task)
    if d.empty:
        logger.info(f"no baseline {task} runs; skipping {filename}")
        return

    values = d.groupby(["language", "family"])["speedup_factor"].mean()
    stds = d.groupby(["language", "family"])["speedup_factor_std"].mean()
    present = _resourcedness_order(
        [lang for lang in langs if lang in set(values.index.get_level_values("language"))],
        counts,
    )
    families = [f for f in FAMILY_STYLE if f in set(values.index.get_level_values("family"))]

    fig, ax = plt.subplots(figsize=(8, 1.6))
    width = 0.8 / len(families)
    xs = np.arange(len(present))
    for j, fam in enumerate(families):
        offset = -0.4 + (j + 0.5) * width
        idx = [i for i, lang in enumerate(present) if (lang, fam) in values.index]
        ys = np.array([values[(present[i], fam)] for i in idx])
        errs = np.array([stds.get((present[i], fam), np.nan) for i in idx])
        ax.bar(
            xs[idx] + offset, ys - 1.0, width=width, bottom=1.0,
            color=FAMILY_STYLE[fam]["color"], label=FAMILY_STYLE[fam]["label"],
            edgecolor="#333333", linewidth=0.4,
        )
        ax.errorbar(
            xs[idx] + offset, ys, yerr=np.nan_to_num(errs),
            fmt="none", ecolor="#444444", capsize=2, linewidth=0.6, capthick=0.6,
        )
        below = [present[i] for i, y in zip(idx, ys) if y < 1]
        logger.info(
            f"{filename}: {fam} avg f={ys.mean():.4f}, min={ys.min():.4f}, max={ys.max():.4f}"
            + (f", below 1.0: {below}" if below else "")
        )

    ax.axhline(1.0, color="black", linewidth=1.0, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(present)
    ax.set_xlim(-0.5, len(present) - 0.5)

    # Allow 0.25 steps so the region below break-even still gets a labelled tick.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(which="minor", axis="y", color="#E5E5E5", linewidth=0.5, alpha=0.7)

    _style_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel(f"{KEY_TO_TITLE['speedup_factor']}\n{TASK_TO_TITLE.get(task, task)}")
    ax.legend(
        frameon=False, fontsize=10, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=len(families), borderaxespad=0.1,
    )
    _finalize(fig, filename)


def _scatter_with_fits(
    ax,
    points: dict[str, list[tuple[str, float, float]]],
    fit_langs: set[str],
    relation: str,
    filename: str,
) -> list[Bbox]:
    """Scatter one series per model family — points[fam] holds (language, x, y)
    triples — with a colour-matched least-squares fit over the languages in
    fit_langs and each family's R^2 in whichever bottom corner the trend leaves
    empty. `relation` names the x-vs-y pairing for the log line. Returns
    display-space boxes for the markers and the R^2 text, so the caller can lay
    out labels around them."""
    fits = []
    for fam, pts in points.items():
        if not pts:
            continue
        style = FAMILY_STYLE[fam]
        ax.scatter(
            [p[1] for p in pts], [p[2] for p in pts],
            marker=style["marker"], color=style["color"], label=style["label"],
            s=40, zorder=3, edgecolors="black", linewidths=0.4,
        )
        fit_pts = [p for p in pts if p[0] in fit_langs]
        if len(fit_pts) > 2:
            fx = np.array([p[1] for p in fit_pts])
            fy = np.array([p[2] for p in fit_pts])
            r = np.corrcoef(fx, fy)[0, 1]
            slope, intercept = np.polyfit(fx, fy, 1)
            logger.info(
                f"{filename}: {relation} ({fam}): r={r:.4f}, "
                f"slope={slope:.4f}, n={len(fit_pts)}"
            )
            x_fit = np.array([fx.min(), fx.max()])
            ax.plot(x_fit, slope * x_fit + intercept, color=style["color"], linewidth=1.2, zorder=2)
            fits.append((fam, r ** 2, slope))

    # Reserve headroom for the language labels the caller stacks above each
    # point, so they stay inside the axes instead of running into the legend.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.16 * (hi - lo))

    ax.figure.canvas.draw()
    marker_boxes = [
        Bbox.from_bounds(*(ax.transData.transform((px, py)) - 5), 10, 10)
        for pts in points.values()
        for _, px, py in pts
    ]

    r2_texts = [
        ax.text(
            0, 0, f"{FAMILY_STYLE[fam]['label']} R$^2$ = {r2:.2f}",
            transform=ax.transAxes, fontsize=8,
            color=FAMILY_STYLE[fam]["color"], zorder=4,
        )
        for fam, r2, _ in fits
    ]
    if r2_texts:
        # A rising trend tends to leave the lower right and upper left empty, and
        # vice versa, but points still land there — so try every corner in that
        # order of preference and take the first that no marker overlaps.
        falling = np.mean([f[2] for f in fits]) < 0
        corners = [
            (0.02, "left", "bottom"), (0.98, "right", "top"),
            (0.02, "left", "top"), (0.98, "right", "bottom"),
        ] if falling else [
            (0.98, "right", "bottom"), (0.02, "left", "top"),
            (0.98, "right", "top"), (0.02, "left", "bottom"),
        ]
        best = None
        for cx, ha, va in corners:
            for k, txt in enumerate(r2_texts):
                row = len(r2_texts) - 1 - k if va == "bottom" else k
                cy = 0.04 + 0.09 * row if va == "bottom" else 0.96 - 0.09 * row
                txt.set_position((cx, cy))
                txt.set_ha(ha)
                txt.set_va(va)
            ax.figure.canvas.draw()
            boxes = [t.get_window_extent() for t in r2_texts]
            hits = sum(any(b.overlaps(m) for m in marker_boxes) for b in boxes)
            if best is None or hits < best[0]:
                best = (hits, cx, ha, va)
            if hits == 0:
                break
        hits, cx, ha, va = best  # type:ignore[misc]
        if hits:
            # Every corner is occupied, so make an empty one: push the axis limit
            # out until the nearest marker clears the text band. Preferred over
            # letting the text hide a data point.
            band = 0.09 * len(r2_texts) + 0.14
            lo, hi = ax.get_ylim()
            ys = [py for pts in points.values() for _, _, py in pts]
            if va == "bottom":
                ax.set_ylim((min(ys) - band * hi) / (1 - band), hi)
            else:
                ax.set_ylim(lo, (max(ys) - band * lo) / (1 - band))
            ax.figure.canvas.draw()
            marker_boxes = [
                Bbox.from_bounds(*(ax.transData.transform((px, py)) - 5), 10, 10)
                for pts in points.values()
                for _, px, py in pts
            ]
        for k, txt in enumerate(r2_texts):
            row = len(r2_texts) - 1 - k if va == "bottom" else k
            txt.set_position((cx, 0.04 + 0.09 * row if va == "bottom" else 0.96 - 0.09 * row))
            txt.set_ha(ha)
            txt.set_va(va)
        ax.figure.canvas.draw()

    return marker_boxes + [t.get_window_extent() for t in r2_texts]


def _place_labels(ax, items: list[tuple[str, float, float]], obstacles: list[Bbox]):
    """Annotate (text, x, y) items, lifting each label until it clears the
    markers and the labels already placed. Each step jumps straight above the
    highest thing it hit — stepping by a fixed amount instead runs out of
    attempts where several points nearly coincide (haw/ibo)."""
    placed = list(obstacles)
    px_to_pt = 72 / ax.figure.dpi
    for text, x, y in items:
        txt = ax.annotate(
            text, (x, y), fontsize=7, xytext=(0, 6),
            textcoords="offset points", ha="center",
        )
        ceiling = ax.get_window_extent().y1
        dy = 6
        for _ in range(8):
            box = txt.get_window_extent()
            hits = [p for p in placed if box.overlaps(p)]
            if not hits:
                break
            dy += (max(p.y1 for p in hits) - box.y0 + 2) * px_to_pt
            txt.set_position((0, dy))
            if txt.get_window_extent().y1 > ceiling:
                # Out of room above (a tall stack, or the R^2 block acting as a
                # wall) — drop below the point and clear downwards instead.
                dy = -6 - box.height * px_to_pt
                txt.set_position((0, dy))
                for _ in range(8):
                    box = txt.get_window_extent()
                    hits = [p for p in placed if box.overlaps(p)]
                    if not hits:
                        break
                    dy -= (box.y1 - min(p.y0 for p in hits) + 2) * px_to_pt
                    txt.set_position((0, dy))
                break
        placed.append(txt.get_window_extent())


def _resourcedness_plot(
    values: pd.Series,
    counts: pd.DataFrame,
    ylabel: str,
    filename: str,
    relation: str,
    column: str = "words",
    xlabel: str = "FineWeb Words (log$_{10}$)",
):
    """Any per-(language, family) metric against language resourcedness, both
    model families on one axes (colour + marker shape per family). Resourcedness
    is the log10 FineWeb token count; languages counted as -1 (absent from
    FineWeb) are laid out side by side in a separate region on the left, where
    the x-axis is blank because no count is defined for them."""
    if column not in counts.columns:
        logger.info(f"{filename}: no '{column}' column in the counts file; skipping")
        return
    words = counts.set_index("language_code")[column]
    present = [lang for lang in langs if lang in set(values.index.get_level_values("language"))]
    missing = [lang for lang in present if lang not in words.index]
    if missing:
        logger.warning(f"{filename}: no FineWeb count for {missing}; dropping")
    present = [lang for lang in present if lang in words.index]

    known = [lang for lang in present if words[lang] > 0]
    unknown = [lang for lang in present if words[lang] <= 0]
    if not known:
        logger.info(f"{filename}: no languages with a FineWeb count; skipping")
        return

    x = {lang: float(np.log10(words[lang])) for lang in known}
    lo, hi = min(x.values()), max(x.values())
    span = (hi - lo) or 1.0
    # Unknown-count languages sit left of the counted ones, evenly spaced, with
    # a gap wide enough to read as a separate region.
    spacing, gap = span * 0.07, span * 0.20
    for i, lang in enumerate(unknown):
        x[lang] = lo - gap - (len(unknown) - 1 - i) * spacing

    families = [f for f in FAMILY_STYLE if f in set(values.index.get_level_values("family"))]

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    points = {
        fam: [(lang, x[lang], values[(lang, fam)]) for lang in present if (lang, fam) in values.index]
        for fam in families
    }
    obstacles = _scatter_with_fits(ax, points, set(known), relation, filename)
    # One label per language: both families share an x here, so a per-point
    # label would just duplicate the code.
    _place_labels(
        ax,
        [
            (lang, x[lang], max(values[(lang, fam)] for fam in families if (lang, fam) in values.index))
            for lang in sorted(present, key=lambda l: x[l])
        ],
        obstacles,
    )

    if unknown:
        # Visual break marking where the x-axis stops being meaningful.
        ax.axvline(lo - gap / 2, color="#666666", linestyle=":", linewidth=1.4, zorder=2)
        left = min(x[lang] for lang in unknown) - spacing
    else:
        left = lo - span * 0.08
    ax.set_xlim(left, hi + span * 0.08)

    # Ticks only over the counted region, so the unknown region reads as blank.
    ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
    if len(ticks) < 2:
        ticks = np.round(np.linspace(lo, hi, 3), 1)
    ax.set_xticks(ticks)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(
        frameon=False, fontsize=10, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=len(families), borderaxespad=0.1,
    )
    _style_spines(ax)
    _finalize(fig, filename)


def _resourcedness_acceptance_plot(data: pd.DataFrame, counts: pd.DataFrame, task: str, filename: str):
    """Baseline acceptance rate against language resourcedness."""
    d = _baseline_family_rows(data, task)
    if d.empty:
        logger.info(f"no baseline {task} runs; skipping {filename}")
        return
    _resourcedness_plot(
        d.groupby(["language", "family"])["sentence_avg_acceptance_rate"].mean(),
        counts,
        f"Acceptance Rate ({TASK_TO_TITLE.get(task, task)})",
        filename,
        "log tokens vs acceptance",
    )


def _resourcedness_divergence_plot(
    divergences: pd.DataFrame, counts: pd.DataFrame, metric: str, filename: str,
    column: str = "words", xlabel: str = "FineWeb Words (log$_{10}$)",
):
    """Measured target/draft divergence against language resourcedness, laid out
    exactly like the acceptance version."""
    if metric not in divergences.columns:
        logger.info(f"divergence file lacks '{metric}'; skipping {filename}")
        return
    div = divergences.dropna(subset=[metric])
    dropped = len(divergences) - len(div)
    if dropped:
        logger.warning(f"{filename}: dropped {dropped} row(s) with no {metric} value")
    if div.empty:
        logger.info(f"no {metric} divergence values; skipping {filename}")
        return
    _resourcedness_plot(
        div.set_index(["language", "family"])[metric],
        counts,
        METRIC_TO_TITLE.get(metric, metric),
        filename,
        f"log {column} vs {metric}",
        column=column,
        xlabel=xlabel,
    )


def load_fertility() -> pd.DataFrame:
    """Per-(language, family) tokenizer fertility from src/compute_fertility.py."""
    path = Path("viz") / "fertility.csv"
    if not path.exists():
        logger.warning(f"no {path}; skipping fertility plots. Generate it with src.compute_fertility")
        return pd.DataFrame(columns=["language", "family", "fertility"])
    # The file carries both language_code and a full-name `language` column;
    # drop the latter first or the rename collides into a duplicate label.
    df = (
        pd.read_csv(path)
        .drop(columns=["language"], errors="ignore")
        .rename(columns={"language_code": "language"})
    )
    logger.info(f"Loaded {len(df)} fertility rows from {path}")
    return df


def _fertility_plot(
    values: pd.Series,
    fertility: pd.DataFrame,
    ylabel: str,
    filename: str,
    relation: str,
):
    """Any per-(language, family) metric against tokenizer fertility.

    Fertility is an inverse resourcedness proxy — tokens per character, so a
    language whose script is missing from the vocabulary falls back to UTF-8
    bytes and lands above 1.0. Unlike the FineWeb counts it is defined for every
    language, so there is no unknown region, and it is tokenizer-specific, so
    each family sits at its own x."""
    if fertility.empty:
        logger.info(f"no fertility data; skipping {filename}")
        return
    fert = fertility.set_index(["language", "family"])["fertility"]
    families = [f for f in FAMILY_STYLE if f in set(values.index.get_level_values("family"))]
    present = [lang for lang in langs if lang in set(values.index.get_level_values("language"))]
    points = {
        fam: [
            (lang, float(fert[(lang, fam)]), values[(lang, fam)])
            for lang in present
            if (lang, fam) in fert.index and (lang, fam) in values.index
        ]
        for fam in families
    }
    missing = [
        lang for lang in present
        if not any((lang, fam) in fert.index for fam in families)
    ]
    if missing:
        logger.warning(f"{filename}: no fertility value for {missing}; dropping")
    if not any(points.values()):
        logger.info(f"{filename}: no languages with both a fertility value and a metric; skipping")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    # Join each language's two family points before drawing them. Most languages
    # tokenize almost identically under both vocabularies, so the pair sits on
    # top of itself and one label serves both; where the connector is long (amh)
    # it is showing a genuine difference in script coverage between tokenizers.
    by_lang: dict[str, list[tuple[float, float]]] = {}
    for pts in points.values():
        for lang, px, py in pts:
            by_lang.setdefault(lang, []).append((px, py))
    for pair in by_lang.values():
        if len(pair) == 2:
            ax.plot(
                [pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]],
                color="#999999", alpha=0.35, linewidth=0.6, zorder=1,
            )

    # Every language enters the fit — fertility has no undefined values.
    obstacles = _scatter_with_fits(ax, points, set(present), relation, filename)
    # One label per language, on its upper point, instead of one per marker.
    _place_labels(
        ax,
        sorted(
            [(lang, max(pair, key=lambda p: p[1])[0], max(p[1] for p in pair))
             for lang, pair in by_lang.items()],
            key=lambda item: item[1],
        ),
        obstacles,
    )

    ax.set_xlabel("Tokenizer Fertility (tokens / char)")
    ax.set_ylabel(ylabel)
    ax.legend(
        frameon=False, fontsize=10, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=len(families), borderaxespad=0.1,
    )
    _style_spines(ax)
    _finalize(fig, filename)


def _fertility_acceptance_plot(data: pd.DataFrame, fertility: pd.DataFrame, task: str, filename: str):
    """Baseline acceptance rate against tokenizer fertility."""
    d = _baseline_family_rows(data, task)
    if d.empty:
        logger.info(f"no baseline {task} runs; skipping {filename}")
        return
    _fertility_plot(
        d.groupby(["language", "family"])["sentence_avg_acceptance_rate"].mean(),
        fertility,
        f"Acceptance Rate ({TASK_TO_TITLE.get(task, task)})",
        filename,
        "fertility vs acceptance",
    )


def _fertility_divergence_plot(divergences: pd.DataFrame, fertility: pd.DataFrame, metric: str, filename: str):
    """Measured target/draft divergence against tokenizer fertility. Note both
    axes derive from the same tokenization, so this is a diagnostic of that
    confound rather than an independent relationship."""
    div = divergences.dropna(subset=[metric])
    if div.empty:
        logger.info(f"no {metric} divergence values; skipping {filename}")
        return
    _fertility_plot(
        div.set_index(["language", "family"])[metric],
        fertility,
        METRIC_TO_TITLE.get(metric, metric),
        filename,
        f"fertility vs {metric}",
    )


# Wrapped because these are y-axis labels on a short axes.
METRIC_TO_TITLE = {
    "kl": "KL Divergence / token\n(target || draft)",
    "lk": "Total Variation Dist. / token\n(target || draft)",
    "kl_per_char": "KL Divergence / char\n(target || draft)",
    "tvd_per_char": "Total Variation Dist. / char\n(target || draft)",
    "bits_per_char_target": "Target Bits / char",
    "bits_per_char_draft": "Draft Bits / char",
}


def _bpc_divergence_plot(
    divergences: pd.DataFrame,
    metric: str,
    filename: str,
    bpc_column: str = "bits_per_char_target",
):
    """Per-character divergence against the target model's bits per character.

    Both axes are per character, so neither is affected by how the tokenizer
    splits the text — the confound that makes the per-token version track
    script coverage. Bits per char stands in for resourcedness: how well the
    target model actually models the language."""
    needed = [metric, bpc_column]
    if any(c not in divergences.columns for c in needed):
        logger.info(f"divergence file lacks {needed}; skipping {filename}")
        return
    div = divergences.dropna(subset=needed)
    if div.empty:
        logger.info(f"no {metric}/{bpc_column} values (re-run src.measure_divergence); skipping {filename}")
        return

    families = [f for f in FAMILY_STYLE if f in set(div["family"])]
    present = [lang for lang in langs if lang in set(div["language"])]
    idx = div.set_index(["language", "family"])
    points = {
        fam: [
            (lang, float(idx.loc[(lang, fam), bpc_column]), float(idx.loc[(lang, fam), metric]))
            for lang in present
            if (lang, fam) in idx.index
        ]
        for fam in families
    }
    if not any(points.values()):
        logger.info(f"{filename}: no languages to plot; skipping")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    obstacles = _scatter_with_fits(ax, points, set(present), f"bits/char vs {metric}", filename)
    _place_labels(
        ax,
        sorted([(l, px, py) for pts in points.values() for l, px, py in pts], key=lambda i: i[1]),
        obstacles,
    )

    ax.set_xlabel(METRIC_TO_TITLE.get(bpc_column, bpc_column).replace("\n", " "))
    ax.set_ylabel(METRIC_TO_TITLE.get(metric, metric))
    ax.legend(
        frameon=False, fontsize=10, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=len(families), borderaxespad=0.1,
    )
    _style_spines(ax)
    _finalize(fig, filename)


DIVERGENCE_METRICS = [
    "kl", "lk", "kl_per_char", "tvd_per_char",
    "bits_per_char_target", "bits_per_char_draft",
]


def load_divergences() -> pd.DataFrame:
    """Read the viz/divergences_<P>_<Q> files into (language, family, <metrics>).

    Current files from src/measure_divergence.py carry a header and a
    language_code column. Older ones were headerless with the full language
    name, so fall back to mapping names onto codes for those. The model family
    comes from the filename, which carries both model keys."""
    records = []
    for path in sorted(Path("viz").glob("divergences_*")):
        if path.suffix not in {".csv", ".tsv"}:
            continue
        family = _detect_family(path.stem)
        df = pd.read_csv(path, sep=None, engine="python")
        if "language_code" in df.columns:
            df = df.rename(columns={"language_code": "language"})
            if "language" in df.columns and df.columns.tolist().count("language") > 1:
                df = df.loc[:, ~df.columns.duplicated()]
            df = df.drop(columns=[c for c in ["n_examples", "n_chars"] if c in df.columns])
        else:
            # Legacy headerless: language name, kl, lk.
            ref = pd.read_csv(Path(__file__).resolve().parent / "data" / "reference_table_bilingual.csv")
            name_to_code = dict(zip(ref["Language"].str.strip(), ref["Code"].str.strip()))
            df = pd.read_csv(path, header=None, names=["language_name", "kl", "lk"], sep=None, engine="python")
            df["language"] = df["language_name"].astype(str).str.strip().map(name_to_code)
            unknown = df[df["language"].isna()]["language_name"].tolist()
            if unknown:
                logger.warning(f"{path.name}: unrecognised language(s) {unknown}; skipping")
            df = df.dropna(subset=["language"]).drop(columns=["language_name"])
            logger.info(f"{path.name}: legacy headerless format; per-character metrics unavailable")
        df["family"] = family
        records.append(df)
        logger.info(f"Loaded {len(df)} divergence rows from {path.name} as family={family}")
    if not records:
        return pd.DataFrame(columns=["language", "family", *DIVERGENCE_METRICS])
    out = pd.concat(records, ignore_index=True)
    for m in DIVERGENCE_METRICS:
        if m not in out.columns:
            out[m] = np.nan
    return out


def _size_scaling_plot(data: pd.DataFrame, y: str, filename: str, family: str | None = None):
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
    colors = _shades(_family_color(family), len(lang_order))
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
    _finalize(fig, filename, family)


def create_graphs(data: pd.DataFrame):
    for family in sorted(data["family"].unique()):
        logger.info(f"Creating graphs for model family: {family}")
        _create_family_graphs(data[data["family"] == family], family)


def _create_family_graphs(data: pd.DataFrame, family: str):
    cfg = FAMILIES[family]
    baseline_size = cfg["baseline_size"]
    # The setting-comparison plots compare Baseline vs. Distilled at a fixed
    # draft size (matching the distilled draft) alongside N-Gram.
    comparison_sizes = [baseline_size, 'N-Gram']

    # _bar_plot(data,             "Tokens / Second (Spec)", "tps_spec", family)
    translation_data = data[(data['task'] == 'translation') & (data['model_size'].isin(comparison_sizes))]
    if translation_data.empty:
        logger.info(f"[{family}] no translation runs; skipping translation plots")
    else:
        _bar_plot(translation_data, "sentence_avg_tokens_per_second", "sentence_std_tokens_per_second", "translation_tps", family)
        _bar_plot(translation_data, "speedup_factor", "speedup_factor_std", "translation_speedup", family, show_legend=False)
        _bar_plot(translation_data, "sentence_avg_acceptance_rate", "sentence_std_acceptance_rate", "translation_acceptance", family)
        _chrf_acceptance_plot(translation_data, family)

    story_data = data[(data['task'] == 'story_gen')  & (data['model_size'].isin(comparison_sizes))]
    if story_data.empty:
        logger.info(f"[{family}] no story_gen runs; skipping story plots")
    else:
        _bar_plot(story_data, "sentence_avg_tokens_per_second", "sentence_std_tokens_per_second", "story_tps", family)
        _bar_plot(story_data, "speedup_factor", "speedup_factor_std", "story_speedup", family)
        _bar_plot(story_data, "sentence_avg_acceptance_rate", "sentence_std_acceptance_rate", "story_acceptance", family)

    # Task-acceptance scatter needs both translation and story runs.
    if not translation_data.empty and not story_data.empty:
        scatter_data = data[data['model_size'].isin(comparison_sizes)]
        _task_acceptance_scatter(scatter_data, family)
        _task_acceptance_scatter(
            scatter_data,
            family,
            distill_setting="Distilled (general)",
            filename="task_acceptance_scatter_general",
        )

    if translation_data.empty:
        return

    forward_pass_data = data[
        (data["task"] == "translation") & (data["setting"] == "Baseline")
    ][["model_size", "average_draft_time", "draft_time_std"]]
    verifier_data = (
        data[(data["task"] == "translation") & (data["setting"] == "Baseline") & (data["model_size"] == baseline_size)]
        .drop(columns=["average_draft_time", "draft_time_std"])
        .rename(
            columns={
                "average_verifier_time": "average_draft_time",
                "verifier_time_std": "draft_time_std",
            }
        )[["model_size", "average_draft_time", "draft_time_std"]]
    )
    verifier_data["model_size"] = cfg["verifier_size"]
    ngram_pass_data = data[(data["task"] == "translation") & (data["setting"] == "N-Gram")][
        ["model_size", "average_draft_time", "draft_time_std"]
    ]
    _violin_plot(
        pd.concat([forward_pass_data, verifier_data, ngram_pass_data]),  # type:ignore
        "model_size",
        "average_draft_time",
        "draft_time_std",
        family,
    )

    # Size scaling only applies where multiple baseline draft sizes were run.
    n_baseline_sizes = data[(data["task"] == "translation") & (data["setting"] == "Baseline")]["model_size"].nunique()
    if n_baseline_sizes > 1:
        _size_scaling_plot(data, "sentence_avg_acceptance_rate", "size_scaling_acceptance", family)
        _size_scaling_plot(data, "speedup_factor", "size_scaling_speedup", family)
    else:
        logger.info(f"[{family}] only one baseline draft size; skipping size-scaling plots")


# eval_kl.py writes one teacher||student KL file per distillation setup, and the
# teacher/student template is baked into the run rather than the file, so the
# files are per family. The unqualified name is the original Qwen run.
KL_RESULTS_FILES = {
    "qwen": ["kl_results_qwen.csv", "kl_results.csv"],
    "llama": ["kl_results_llama.csv"],
}


def _load_kl_results(family: str) -> pd.DataFrame | None:
    """Teacher||student KL for one family's distilled students, or None if that
    family has no file. Never falls back to another family's file: the values
    are specific to a teacher/student pair."""
    names = KL_RESULTS_FILES.get(family, [f"kl_results_{family}.csv"])
    for name in names:
        path = Path("viz") / name
        if path.exists():
            logger.info(f"[{family}] KL results from {path}")
            return pd.read_csv(path)
    logger.warning(
        f"[{family}] no teacher||student KL file (looked for {names}); skipping the "
        f"Pinsker plot. Generate one with src/tasks/distillation/eval_kl.py using this "
        f"family's --teacher-short and --model-template."
    )
    return None


def _pinsker_plot(kl_df: pd.DataFrame, spec_df: pd.DataFrame, family: str | None = None):
    kl_df = kl_df.rename(columns={"language_code": "language"})

    sizes = [FAMILIES[family]["baseline_size"], "N-Gram"] if family in FAMILIES else None
    distilled_spec = spec_df[
        (spec_df["task"] == "translation") &
        (spec_df["setting"] == "Distilled (task)")
    ]
    if sizes:
        distilled_spec = distilled_spec[distilled_spec["model_size"].isin(sizes)]
    distilled_spec = distilled_spec.copy()
    distilled_spec["distill_type"] = distilled_spec["setting"].map({
        "Distilled (task)": "translation",
        "Distilled (general)": "general",
    })

    merged = distilled_spec.groupby(["language", "distill_type"])["sentence_avg_acceptance_rate"].mean().reset_index()
    merged = merged.merge(kl_df[["language", "kl_divergence"]], on="language", how="inner")

    if merged.empty:
        logger.warning(f"Pinsker plot ({family}): no data after merging distill and spec decode runs")
        return

    # Pinsker bound curve: acceptance >= 1 - sqrt(KL/2)
    kl_max = merged["kl_divergence"].max() + 0.3
    kl_range = np.linspace(0, kl_max, 300)
    pinsker_bound = np.maximum(0.0, 1.0 - np.sqrt(kl_range / 2))

    type_to_color  = {"translation": PALETTE[1], "general": PALETTE[4]}
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
    _finalize(fig, "pinsker_bound", family)


if __name__ == "__main__":
    spec_data = load_real_data()
    spec_data = spec_data[spec_data['language'] != 'zh']
    families = sorted(spec_data["family"].unique())
    for family in families:
        kl_data = _load_kl_results(family)
        if kl_data is not None:
            _pinsker_plot(kl_data, spec_data[spec_data["family"] == family], family)
    create_graphs(spec_data)

    fineweb_counts = pd.read_csv("viz/fineweb_counts.csv")
    for task, name in [("translation", "translation"), ("story_gen", "story")]:
        _resourcedness_acceptance_plot(spec_data, fineweb_counts, task, f"resourcedness_acceptance_{name}")
        _baseline_speedup_bars(spec_data, fineweb_counts, task, f"baseline_speedup_{name}")
        for family in families:
            _ngram_vs_distilled_scatter(spec_data, task, f"ngram_vs_distilled_{name}", family)

    fertility_data = load_fertility()
    for task, name in [("translation", "translation"), ("story_gen", "story")]:
        _fertility_acceptance_plot(spec_data, fertility_data, task, f"fertility_acceptance_{name}")

    divergence_data = load_divergences()
    for metric in ["kl", "lk"]:
        _resourcedness_divergence_plot(divergence_data, fineweb_counts, metric, f"resourcedness_divergence_{metric}")
        _fertility_divergence_plot(divergence_data, fertility_data, metric, f"fertility_divergence_{metric}")

    # Per-character versions: neither axis depends on how the tokenizer splits
    # the text, so the script-coverage confound in the per-token plots is gone.
    for metric in ["kl_per_char", "tvd_per_char"]:
        _bpc_divergence_plot(divergence_data, metric, f"bpc_divergence_{metric}")
        _resourcedness_divergence_plot(
            divergence_data, fineweb_counts, metric, f"bytes_divergence_{metric}",
            column="bytes", xlabel="FineWeb Bytes (log$_{10}$)",
        )
