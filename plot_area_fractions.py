import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams["svg.fonttype"] = "none"

OUTPUT_ROOT = Path(r"D:\2026_Harshani\Analysis_outputs")
PLOTS_DIR = OUTPUT_ROOT / "plots"

df = pd.read_csv(OUTPUT_ROOT / "area_fraction_summary.csv")

TREATMENT_ORDER = ["Alk", "Alk_CP", "DES", "DES_CP", "Soy"]

CSV = pd.read_csv(OUTPUT_ROOT / "area_fraction_per_image.csv")

PHASE_ORDER_3 = ["Pore", "Non-aggregated protein", "Aggregated protein"]
PHASE_ORDER_2 = ["Pore", "Protein"]

COLORS = {
    "Pore": "#d62728",
    "Non-aggregated protein": "#ff7f0e",
    "Aggregated protein": "#2c7fbb",
    "Protein": "#2ca02c",
}

g = CSV.groupby(["treatment", "label"])["area_fraction"].agg(["mean", "sem", "std"]).reset_index()

fig, ax = plt.subplots(figsize=(9, 3.5))
y_pos = np.arange(len(TREATMENT_ORDER))
bar_h = 0.55

left = np.zeros(len(TREATMENT_ORDER))
handles = []
labels_added = []

for t_idx, treatment in enumerate(TREATMENT_ORDER):
    n_labels = 3 if treatment in ("Alk", "Alk_CP") else 2
    phases = PHASE_ORDER_3 if n_labels == 3 else PHASE_ORDER_2
    for i, phase in enumerate(phases):
        label_val = i + 1
        row = g[(g["treatment"] == treatment) & (g["label"] == label_val)]
        if len(row) == 0:
            continue
        mean_val = row["mean"].values[0]
        sem_val = row["sem"].values[0]
        ax.barh(
            y_pos[t_idx],
            mean_val,
            left=left[t_idx],
            height=bar_h,
            color=COLORS[phase],
            edgecolor="white",
            linewidth=0.5,
        )
        if mean_val > 0:
            ax.errorbar(
                left[t_idx] + mean_val,
                y_pos[t_idx],
                xerr=sem_val,
                color="black",
                capsize=2,
                capthick=1,
                lw=1,
            )
        left[t_idx] += mean_val
        if phase not in labels_added:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=COLORS[phase]))
            labels_added.append(phase)

ax.set_yticks(y_pos)
ax.set_yticklabels(TREATMENT_ORDER)
ax.set_xlabel("Area fraction")
ax.set_xlim(0, 1.15)
ax.legend(handles, labels_added, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

outfile = PLOTS_DIR / "area_fractions_all_treatments.svg"
fig.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {outfile}")