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
    return {"efficiency": block_efficiency, "ratio": draft_ratio, "tps spec": tps_spec, "tps auto":tps_auto, "bleu": bleu_spec}

defualt_x = ["amh","ber","chr","grn","haw","ibo","npi","oci","que","yor","zgh","zh"]
def graphs(
    title:list[str]=["Wall-Clock Speed", "Speculative Decoding Efficiency", "BLEU"],
    x_title:str="Languages",
    y_title:str="Values",
    data:dict[str,list[float]] | None=None,
    x_axis:list[str]=defualt_x,
):
    if data is None:
        data = fake_data(x_axis)

        for item in
