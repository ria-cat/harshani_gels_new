import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

plt.rcParams["svg.fonttype"] = "none"

OUTPUT_ROOT = Path(r"D:\2026_Harshani\Analysis_outputs")
PLOTS_DIR = OUTPUT_ROOT / "plots"

ELEMENT_SIZE = 0.3629  # um/px
STEP = ELEMENT_SIZE * 2  # 0.7258 um — one radius step in diameter

# ============================================================
# Load
# ============================================================
pdf_df = pd.read_csv(OUTPUT_ROOT / "granulometry_PDFs.csv")
summary_df = pd.read_csv(OUTPUT_ROOT / "granulometry_summary.csv")

treatments = sorted(pdf_df["treatment"].unique())  # sorted ensures deterministic order


# ============================================================
# 1. Pooled distributions per treatment
# ============================================================
def pooled_distribution(treatment, pdf_df, step):
    sub = pdf_df[pdf_df["treatment"] == treatment]
    filenames = sub["filename"].unique()
    max_d = sub["diameter_um"].max()
    common = np.arange(0, max_d + step, step)

    interp = []
    for fn in filenames:
        f = sub[sub["filename"] == fn].sort_values("diameter_um")
        p = np.interp(common, f["diameter_um"], f["pdf"], left=0, right=0)
        interp.append(p)

    arr = np.array(interp)
    return {
        "treatment": treatment,
        "diameter_um": common,
        "mean_pdf": np.mean(arr, axis=0),
        "sem_pdf": np.std(arr, axis=0, ddof=1) / np.sqrt(len(filenames)),
        "n": len(filenames),
    }


pooled_records = []
for t in treatments:
    res = pooled_distribution(t, pdf_df, STEP)
    for d, m, s in zip(res["diameter_um"], res["mean_pdf"], res["sem_pdf"]):
        pooled_records.append(
            {"treatment": t, "diameter_um": d, "mean_pdf": m, "sem_pdf": s}
        )

pooled_df = pd.DataFrame(pooled_records)
pooled_df.to_csv(OUTPUT_ROOT / "granulometry_pooled_distributions.csv", index=False)
print("Saved granulometry_pooled_distributions.csv")


# ============================================================
# 2. Replicate-level summary statistics
# ============================================================
metrics = ["mean", "sd", "D10", "D50", "D90"]
stat_rows = []
for t in treatments:
    sub = summary_df[summary_df["treatment"] == t]
    row = {"treatment": t, "n_images": len(sub)}
    for m in metrics:
        row[f"{m}_mean"] = sub[m].mean()
        row[f"{m}_sem"] = sub[m].sem()
        row[f"{m}_std"] = sub[m].std()
    stat_rows.append(row)

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(OUTPUT_ROOT / "granulometry_replicate_stats.csv", index=False)
print("Saved granulometry_replicate_stats.csv")
print()
print(stat_df.to_string(index=False))


# ============================================================
# 3. DES vs DES_CP comparison plot
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

comparison = ["DES", "DES_CP"]
colors = {"DES": "tab:blue", "DES_CP": "tab:orange"}

for t in comparison:
    sub = pooled_df[pooled_df["treatment"] == t]
    ax1.plot(sub["diameter_um"], sub["mean_pdf"], color=colors[t], lw=2, label=t)
    ax1.fill_between(
        sub["diameter_um"],
        (sub["mean_pdf"] - sub["sem_pdf"]).clip(0),
        (sub["mean_pdf"] + sub["sem_pdf"]).clip(0),
        color=colors[t],
        alpha=0.2,
    )

ax1.set_xlabel("Equivalent diameter (um)")
ax1.set_ylabel("Normalized distribution")
ax1.set_title("Pooled pore thickness distribution")
ax1.legend(frameon=False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Right panel: bar plot of D50 with SEM
d50_vals = {t: summary_df[summary_df["treatment"] == t]["D50"] for t in comparison}
means = [d50_vals[t].mean() for t in comparison]
sems = [d50_vals[t].sem() for t in comparison]

bars = ax2.bar(comparison, means, yerr=sems, capsize=5, color=[colors[t] for t in comparison])
ax2.set_ylabel("D50 (um)")
ax2.set_title("Median pore thickness")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# T-test
t_stat, p_val = scipy_stats.ttest_ind(d50_vals["DES"], d50_vals["DES_CP"])
ax2.text(
    0.5,
    0.95,
    f"t-test: p = {p_val:.4f}",
    transform=ax2.transAxes,
    ha="center",
    va="top",
    fontsize=9,
)

fig.tight_layout()
out = PLOTS_DIR / "DES_vs_DES_CP_comparison.svg"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out}")


# ============================================================
# 4. Per-treatment pooled distribution plots
# ============================================================
for t in treatments:
    sub = pooled_df[pooled_df["treatment"] == t]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sub["diameter_um"], sub["mean_pdf"], lw=2, color="tab:red")
    ax.fill_between(
        sub["diameter_um"],
        (sub["mean_pdf"] - sub["sem_pdf"]).clip(0),
        (sub["mean_pdf"] + sub["sem_pdf"]).clip(0),
        color="tab:red",
        alpha=0.2,
    )
    ax.set_xlabel("Equivalent diameter (um)")
    ax.set_ylabel("Normalized distribution")
    ax.set_title(f"{t} — pooled pore thickness (n=8 images)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = PLOTS_DIR / f"{t}_pooled_distribution.svg"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


print("\nAll done!")