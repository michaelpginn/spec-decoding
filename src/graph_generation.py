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
from matplotlib.transforms import Bbox
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


def _combined_speedup_plot(data: pd.DataFrame, task: str = "translation", filename: str = "combined_speedup"):
    """Speed-up factor for both model families on one chart. Laid out like the
    grouped bar charts (languages on x, settings dodged and colour-coded), but
    each setting shows a filled circle for Qwen and an open circle for Llama
    joined by a connecting line."""
    y = "speedup_factor"
    frames = []
    for fam in sorted(data["family"].unique()):
        bs = FAMILIES[fam]["baseline_size"]
        frames.append(
            data[(data["family"] == fam) & (data["task"] == task) & (data["model_size"].isin([bs, "N-Gram"]))]
        )
    d = pd.concat(frames)
    values = d.groupby(["language", "setting", "family"])[y].mean()

    present_langs = [lang for lang in langs if lang in set(d["language"])]
    setting_color = dict(zip(SETTINGS, ['#8C8C8C', *PALETTE[1:len(SETTINGS)]]))
    families = sorted(d["family"].unique())
    # Qwen filled, Llama open (falls back gracefully for any family order).
    family_style = {
        fam: ({"facecolor": "fill", "label": fam.capitalize()} if fam == "qwen" else {"facecolor": "open", "label": fam.capitalize()})
        for fam in families
    }

    fig, ax = plt.subplots(figsize=(8, 1.6))
    group_width = 0.8
    slot = group_width / len(SETTINGS)
    for j, setting in enumerate(SETTINGS):
        offset = -group_width / 2 + (j + 0.5) * slot
        color = setting_color[setting]
        for i, lang in enumerate(present_langs):
            x = i + offset
            pts = {
                fam: values.get((lang, setting, fam))
                for fam in families
                if (lang, setting, fam) in values.index
            }
            if len(pts) == 2:
                ys = list(pts.values())
                ax.plot([x, x], ys, color=color, linewidth=0.8, zorder=2)
            for fam, val in pts.items():
                if family_style[fam]["facecolor"] == "fill":
                    ax.scatter([x], [val], facecolor=color, edgecolor="#333333", linewidths=0.4, s=20, zorder=3)
                else:
                    ax.scatter([x], [val], facecolor="white", edgecolor=color, linewidths=1.0, s=20, zorder=3)

    ax.set_xticks(range(len(present_langs)))
    ax.set_xticklabels(present_langs)
    ax.set_xlim(-0.5, len(present_langs) - 0.5)

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='y', which='minor', length=0)
    ax.grid(which='minor', axis='y', color='#E5E5E5', linewidth=0.5, alpha=0.7)

    _style_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel(KEY_TO_TITLE[y])

    # Model legend (filled Qwen / open Llama) drawn inside the axes as an added
    # artist; the setting-colour legend is the primary one on top, matching the
    # bar charts and so it is reliably kept by tight-layout / tight bbox.
    model_handles = [
        plt.Line2D([], [], marker='o', markerfacecolor='#333333' if family_style[f]["facecolor"] == "fill" else 'white',
                   markeredgecolor='#333333', color='none', markersize=6, label=family_style[f]["label"])
        for f in families
    ]
    model_legend = ax.legend(
        handles=model_handles, frameon=False, fontsize=9,
        loc='lower left', bbox_to_anchor=(0.0, 0.0), handletextpad=0.3, borderaxespad=0.2,
    )
    ax.add_artist(model_legend)
    setting_handles = [
        plt.Line2D([], [], marker='o', color=setting_color[s], linestyle='none', markersize=6, label=s)
        for s in SETTINGS
    ]
    ax.legend(
        handles=setting_handles,
        frameon=False, fontsize=10, loc='lower center',
        bbox_to_anchor=(0.5, 1.0), ncol=len(SETTINGS), title=None, borderaxespad=0.1,
    )

    _finalize(fig, filename)


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
    setting_to_color = {
        "Baseline": "#8C8C8C",
        "Distilled (task)": PALETTE[2],
        "Distilled (general)": PALETTE[1],
    }

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


def _resourcedness_acceptance_plot(data: pd.DataFrame, counts: pd.DataFrame, task: str, filename: str):
    """Baseline acceptance rate vs. language resourcedness, both model families
    on one axes (colour + marker shape per family). Resourcedness is the log10
    FineWeb token count; languages counted as -1 (absent from FineWeb) are laid
    out side by side in a separate region on the left, where the x-axis is
    blank because no count is defined for them."""
    frames = []
    for fam in sorted(data["family"].unique()):
        frames.append(data[
            (data["family"] == fam)
            & (data["task"] == task)
            & (data["setting"] == "Baseline")
            & (data["model_size"] == FAMILIES[fam]["baseline_size"])
        ])
    d = pd.concat(frames)
    if d.empty:
        logger.info(f"no baseline {task} runs; skipping {filename}")
        return

    words = counts.set_index("language_code")["words"]
    values = d.groupby(["language", "family"])["sentence_avg_acceptance_rate"].mean()
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

    fam_style = {
        "qwen": {"marker": "o", "color": PALETTE[0], "label": "Qwen"},
        "llama": {"marker": "s", "color": PALETTE[1], "label": "Llama"},
    }
    families = [f for f in ["qwen", "llama"] if f in set(values.index.get_level_values("family"))]

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for fam in families:
        style = fam_style[fam]
        pts = [(x[lang], values[(lang, fam)]) for lang in present if (lang, fam) in values.index]
        if not pts:
            continue
        ax.scatter(
            [p[0] for p in pts], [p[1] for p in pts],
            marker=style["marker"], color=style["color"], label=style["label"],
            s=40, zorder=3, edgecolors="black", linewidths=0.4,
        )
        fam_known = [(x[lang], values[(lang, fam)]) for lang in known if (lang, fam) in values.index]
        if len(fam_known) > 2:
            r = np.corrcoef([p[0] for p in fam_known], [p[1] for p in fam_known])[0, 1]
            logger.info(f"{filename}: Pearson r (log tokens vs acceptance, {fam}) = {r:.4f} (n={len(fam_known)})")

    # Label each language once, above its highest point. Some languages have
    # near-identical token counts (haw/ibo), so nudge labels up until they clear
    # the ones already placed.
    fig.canvas.draw()
    # Seed the occupied regions with the markers themselves, so a label never
    # lands on a neighbouring point.
    placed = [
        Bbox.from_bounds(*(ax.transData.transform((x[lang], values[(lang, fam)])) - 5), 10, 10)
        for lang in present
        for fam in families
        if (lang, fam) in values.index
    ]
    for lang in sorted(present, key=lambda l: x[l]):
        ys = [values[(lang, fam)] for fam in families if (lang, fam) in values.index]
        txt = ax.annotate(
            lang, (x[lang], max(ys)),
            fontsize=7, xytext=(0, 6), textcoords="offset points", ha="center",
        )
        dy = 6
        for _ in range(6):
            if not any(txt.get_window_extent().overlaps(p) for p in placed):
                break
            dy += 8
            txt.set_position((0, dy))
        placed.append(txt.get_window_extent())

    if unknown:
        # Visual break marking where the x-axis stops being meaningful.
        ax.axvline(lo - gap / 2, color="#BFBFBF", linestyle=":", linewidth=0.8, zorder=0)
        left = min(x[lang] for lang in unknown) - spacing
    else:
        left = lo - span * 0.08
    ax.set_xlim(left, hi + span * 0.08)

    # Ticks only over the counted region, so the unknown region reads as blank.
    ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
    if len(ticks) < 2:
        ticks = np.round(np.linspace(lo, hi, 3), 1)
    ax.set_xticks(ticks)

    ax.set_xlabel("FineWeb Tokens (log$_{10}$)")
    ax.set_ylabel(f"Acceptance Rate ({TASK_TO_TITLE.get(task, task)})")
    ax.legend(
        frameon=False, fontsize=10, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=len(families), borderaxespad=0.1,
    )
    _style_spines(ax)
    _finalize(fig, filename)


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


def _pinsker_plot(kl_df: pd.DataFrame, spec_df: pd.DataFrame, family: str | None = None):
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
        logger.warning(f"Pinsker plot ({family}): no data after merging distill and spec decode runs")
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
    _finalize(fig, "pinsker_bound", family)


if __name__ == "__main__":
    kl_data = pd.read_csv("viz/kl_results.csv")
    spec_data = load_real_data()
    spec_data = spec_data[spec_data['language'] != 'zh']
    for family in sorted(spec_data["family"].unique()):
        _pinsker_plot(kl_data, spec_data[spec_data["family"] == family], family)
    create_graphs(spec_data)
    _combined_speedup_plot(spec_data)

    fineweb_counts = pd.read_csv("viz/fineweb_counts.csv")
    for task, name in [("translation", "translation"), ("story_gen", "story")]:
        _resourcedness_acceptance_plot(spec_data, fineweb_counts, task, f"resourcedness_acceptance_{name}")
