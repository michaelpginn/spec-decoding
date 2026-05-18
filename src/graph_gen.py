import random

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#E5E5E5'
plt.rcParams['axes.linewidth'] = 0.8

def fake_data(langs):
    len_langs = len(langs)
    gamma = 5
    c = 0.15
    base_tps = 40.0
    alpha = np.array([random.random() for _ in range(len_langs)])
    block_efficiency = np.array([random.random() for _ in range(len_langs)])

    draft_ratio = np.array(np.random.uniform(0, gamma, len_langs).tolist())

    speedup = (1 - alpha**(gamma + 1)) / ((1 - alpha) * (gamma * c + 1))

    tps_spec = base_tps * speedup
    tps_auto = np.full(len_langs, base_tps)

    bleu_auto = np.array([random.random() for _ in range(len_langs)])

    bleu_spec = bleu_auto + np.random.uniform(-0.001, 0.001, len_langs)
    return {
        "efficiency": block_efficiency,
        "ratio": draft_ratio,
        "tps spec": tps_spec,
        "tps auto":tps_auto,
        "bleu auto": bleu_auto,
        "bleu spec": bleu_spec,
        "speedup": speedup,
        "draft ratio": draft_ratio
    }

defualt_x = ["amh","ber","chr","grn","haw","ibo","npi","oci","que","yor","zgh","zh"]
def graphs(
    title:list[str]=["Wall-Clock Speed", "Speculative Decoding Efficiency", "BLEU"],
    x_title:str="Languages",
    y_title:str="Values",
    data:dict[str,list[int | float]] | None=None,
    x_axis:list[str]=defualt_x,
    figure_title:list[str]=["Placeholder_wall_clock_speed", "Placeholder_speculative_decoding_efficiency", "Placeholder_BLEU"],
    alpha:int|float=5,
    gamma = 5
):
    if data is None and x_axis==defualt_x:
        data = fake_data(x_axis)
        width = 0.35
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        rects1 = ax1.bar(np.array(x_axis) - width/2, data["tps auto"], width, label='Autoregressive Baseline', color='#34495e')
        rects2 = ax1.bar(np.array(x_axis) + width/2, data["tps spec"], width, label='Speculative Decoding', color='#3498db')
        ax1.set_ylabel(y_title, color='#2c3e50', fontweight='bold')
        ax1.set_xlabel(x_title, fontweight='bold', labelpad=10)
        ax1.set_xticks(np.array(x_axis))
        ax1.set_xticklabels(x_axis)
        ax1.tick_params(axis='y', labelcolor='#2c3e50')
        ax1.legend(loc='upper left', framealpha=0.9)

        ax2 = ax1.twinx()
        ax2.grid(False) # Turn off secondary grid to keep ggplot structure clean
        ax2.plot(np.array(x_axis), data["speedup"], color='#e74c3c', marker='o', linewidth=2.5, markersize=8, label='Speedup Factor')
        ax2.set_ylabel(y_title, color='#e74c3c', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        # Format labels as 1.2x multipliers
        ax2.set_yticklabels([f'{val:.1f}×' for val in ax2.get_yticks()])

        plt.title(title[0], fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()

        plt.savefig(f"{figure_title[0]}.png")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        rects3 = ax.bar(np.array(x_axis) - width, alpha, width, label=r'Acceptance Rate ($\alpha$)', color='#2ecc71')
        rects4 = ax.bar(np.array(x_axis), data["efficiency"], width, label='Block Efficiency', color='#27ae60')

        ax.set_ylabel('Efficiency Rate / Probability', fontweight='bold')
        ax.set_xticks(np.array(x_axis))
        ax.set_xticklabels(x_axis)
        ax.set_ylim(0, 1.0)

        ax3 = ax.twinx()
        ax3.grid(False)
        rects5 = ax3.bar(np.array(x_axis) + width, data["draft ratio"], width, label='Draft-to-Output Ratio', color='#f39c12')
        ax3.set_ylabel(r'Computational Overhead Ratio (1 to $\gamma$)', color='#d35400', fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='#d35400')
        ax3.set_ylim(1.0, gamma + 0.5)

        lines, labels = ax.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines + lines3, labels + labels3, loc='upper right', framealpha=0.9)

        plt.title(title[1], fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()

        plt.savefig(f"{figure_title[1]}.png")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        # Side-by-side verification bars proving distribution identity
        rects6 = ax.bar(np.array(x_axis) - width/2, data["bleu auto"], width, label='Autoregressive Baseline', color='#9b59b6')
        rects7 = ax.bar(np.array(x_axis) + width/2, data["bleu spec"], width, label='Speculative Decoding', color='#8e44ad')

        ax.set_ylabel('Corpus BLEU Score (SacreBLEU)', fontweight='bold')
        ax.set_xticks(np.array(x_axis))
        ax.set_xticklabels(x_axis)
        ax.set_ylim(0, 0.55) # Cap near realistic bounds
        ax.legend(loc='upper right', framealpha=0.9)

        plt.title(title[2], fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()

        plt.savefig(f"{figure_title[2]}.png")

if __name__ == "__main__":
    graph = graphs()
