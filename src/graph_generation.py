import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# BLEU, tps spec, speedup, acceptance rate, toks/second

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#E5E5E5'
plt.rcParams['axes.linewidth'] = 0.8

column_width_inch = 7.7 / 2.54
fig_height_inch = column_width_inch * 0.72

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 10
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

    # Dictionary construction
    return {
        "tps spec": tps_spec,
        "tps auto": tps_auto,
        "bleu auto": bleu_auto,
        "bleu spec": bleu_spec,
        "speedup": speedup,
        "acceptance rate": acceptance_rate
    }

def placeholder_graphs():
    langs = ["amh","ber","chr","grn","haw","ibo","npi","oci","que","yor","zgh","zh"]
    models = {
        "Baseline": fake_data(langs),
        "Task-Specific Distill": fake_data(langs),
        "General Domain Distill": fake_data(langs),
        "N-Gram": fake_data(langs),
    }

    fig, ax = plt.subplots(figsize=(column_width_inch, fig_height_inch), linewidth=4, edgecolor='black')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#333333')
