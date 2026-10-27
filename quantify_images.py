import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import morphology
from scipy.ndimage import distance_transform_edt
from joblib import Parallel, delayed
from datetime import datetime
import traceback
import re

plt.rcParams["svg.fonttype"] = "none"

# ============================================================
# Configuration
# ============================================================
INPUT_ROOT = Path(r"D:\2026_Harshani\Segmented")
OUTPUT_ROOT = Path(r"D:\2026_Harshani\Analysis_outputs")
ELEMENT_SIZE = 0.3629  # um/px
SKIP_DIRS = {"Skeletons", "Skeleton_CSV"}

TREATMENT_LABELS = {
    "Alk": [1, 2, 3],
    "Alk_CP": [1, 2, 3],
    "DES": [1, 2],
    "DES_CP": [1, 2],
    "Soy": [1, 2],
}

# ============================================================
# Utility functions
# ============================================================
def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def weighted_percentile(x, w, q):
    cdf = np.cumsum(w)
    return np.interp(q / 100.0, cdf, x)


def granulometry_stats(diameters, pdf):
    pdf = pdf / pdf.sum()
    mean = np.sum(diameters * pdf)
    sd = np.sqrt(np.sum(pdf * (diameters - mean) ** 2))
    return {
        "mean": mean,
        "sd": sd,
        "D10": weighted_percentile(diameters, pdf, 10),
        "D50": weighted_percentile(diameters, pdf, 50),
        "D90": weighted_percentile(diameters, pdf, 90),
    }


def _volume_at_radius(mask, r):
    if r == 0:
        return np.sum(mask)
    footprint = morphology.disk(r, decomposition="sequence")
    transformed = morphology.opening(mask, footprint)
    return np.sum(transformed)


# ============================================================
# Granulometry 2D (pore phase = label 1)
# ============================================================
def granulometry_2d_pore(binary_mask, element_size):
    # Invert for pore phase: we want to open the background of label 1
    mask = ~binary_mask

    # Use distance transform to find the largest inscribed radius
    # This is the exact radius where the last pore vanishes under opening
    dt = distance_transform_edt(mask)
    max_r = int(np.max(dt) * 1.1) + 2
    max_r = min(max_r, min(mask.shape) // 2)

    radii = np.arange(max_r + 1)

    volumes = np.asarray(
        Parallel(n_jobs=-2, backend="loky")(
            delayed(_volume_at_radius)(mask, int(r))
            for r in radii
        ),
        dtype=float,
    )

    # Truncate after volumes hit zero
    nonzero = np.where(volumes > 0)[0]
    if len(nonzero) == 0:
        cutoff = 1
    else:
        cutoff = nonzero[-1] + 2
        cutoff = min(cutoff, len(volumes))

    radii = radii[:cutoff]
    volumes = volumes[:cutoff]

    delta = np.zeros_like(volumes)
    delta[1:] = volumes[:-1] - volumes[1:]

    if delta.sum() > 0:
        pdf = delta / delta.sum()
    else:
        pdf = delta

    diameters = 2 * radii * element_size

    pdf_stats = pdf[1:]
    diameter_stats = diameters[1:]

    if pdf_stats.sum() > 0:
        pdf_stats = pdf_stats / pdf_stats.sum()
        stats = granulometry_stats(diameter_stats, pdf_stats)
    else:
        stats = {
            "mean": np.nan,
            "sd": np.nan,
            "D10": np.nan,
            "D50": np.nan,
            "D90": np.nan,
        }

    return {
        "diameters": diameters,
        "pdf": pdf,
        "cdf": np.cumsum(pdf),
        "volumes": volumes,
        "stats": stats,
    }


# ============================================================
# Plotting helpers
# ============================================================
def save_individual_plot(diameters, pdf, title, outfile):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(diameters, pdf, "-o", color="tab:red", lw=2, ms=3)
    ax.set_xlabel("Equivalent diameter (um)")
    ax.set_ylabel("Normalized distribution")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_combined_treatment_plot(treatment, pdf_df, outfile):
    filenames = pdf_df["filename"].unique()
    fig, ax = plt.subplots(figsize=(7, 4))
    for fn in filenames:
        subset = pdf_df[pdf_df["filename"] == fn]
        label = Path(fn).stem
        ax.plot(subset["diameter_um"], subset["pdf"], lw=1.5, alpha=0.7, label=label)
    ax.set_xlabel("Equivalent diameter (um)")
    ax.set_ylabel("Normalized distribution")
    ax.set_title(f"{treatment} - pore thickness distribution")
    ax.legend(frameon=False, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# File discovery
# ============================================================
def discover_files(root):
    files = []
    for tif_file in root.rglob("*.tif"):
        parent = tif_file.parent
        skip = any(s in parent.parts for s in SKIP_DIRS)
        if skip:
            continue

        treatment = parent.name
        if treatment not in TREATMENT_LABELS:
            log(f"Unknown treatment folder: {treatment}, skipping")
            continue

        stem = tif_file.stem
        m = re.search(r"R(\d+)_T(\d+)", stem)
        if not m:
            log(f"Cannot parse replicate info from: {tif_file.name}, skipping")
            continue

        bio_rep = int(m.group(1))
        tech_rep = int(m.group(2))

        files.append(
            {
                "path": tif_file,
                "treatment": treatment,
                "bio_rep": bio_rep,
                "tech_rep": tech_rep,
                "filename": tif_file.name,
            }
        )
    return files


# ============================================================
# Main
# ============================================================
def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)

    files = discover_files(INPUT_ROOT)
    log(f"Found {len(files)} TIFF files across {len(TREATMENT_LABELS)} treatments")

    area_rows = []
    granulometry_rows = []
    distribution_records = []

    for f in files:
        try:
            log(f"Processing: {f['treatment']}/{f['filename']}")
            img = np.array(Image.open(f["path"])).astype(np.uint8)

            labels = TREATMENT_LABELS[f["treatment"]]
            total_pixels = img.size

            # --- Area fraction per label ---
            for label in labels:
                count = np.sum(img == label)
                area_rows.append(
                    {
                        "treatment": f["treatment"],
                        "bio_rep": f["bio_rep"],
                        "tech_rep": f["tech_rep"],
                        "filename": f["filename"],
                        "label": label,
                        "pixel_count": int(count),
                        "area_fraction": count / total_pixels,
                    }
                )

            # --- Granulometry 2D on label 1 (pore phase) ---
            pore_mask = img == 1
            res = granulometry_2d_pore(pore_mask, ELEMENT_SIZE)

            for d, pdf, cdf in zip(res["diameters"], res["pdf"], res["cdf"]):
                distribution_records.append(
                    {
                        "treatment": f["treatment"],
                        "bio_rep": f["bio_rep"],
                        "tech_rep": f["tech_rep"],
                        "filename": f["filename"],
                        "diameter_um": d,
                        "pdf": pdf,
                        "cdf": cdf,
                    }
                )

            row = {
                "treatment": f["treatment"],
                "bio_rep": f["bio_rep"],
                "tech_rep": f["tech_rep"],
                "filename": f["filename"],
            }
            row.update(res["stats"])
            granulometry_rows.append(row)

            # Individual plot
            base = Path(f["filename"]).stem
            individual_out = plots_dir / f"{f['treatment']}_{base}_distribution.svg"
            save_individual_plot(
                res["diameters"],
                res["pdf"],
                f"{f['treatment']} {base}",
                individual_out,
            )
            log(f"  Saved plot: {individual_out.name}")

        except Exception as e:
            log(f"ERROR processing {f['filename']}: {e}")
            traceback.print_exc()
            continue

    # ============================================================
    # Save area fraction CSVs
    # ============================================================
    area_df = pd.DataFrame(area_rows)
    area_df.to_csv(OUTPUT_ROOT / "area_fraction_per_image.csv", index=False)
    log("Saved area_fraction_per_image.csv")

    area_summary = (
        area_df.groupby(["treatment", "label"])["area_fraction"]
        .agg(["mean", "sem", "std", "count"])
        .reset_index()
    )
    area_summary.to_csv(OUTPUT_ROOT / "area_fraction_summary.csv", index=False)
    log("Saved area_fraction_summary.csv")

    # ============================================================
    # Save granulometry CSVs
    # ============================================================
    pdf_df = pd.DataFrame(distribution_records)
    pdf_df.to_csv(OUTPUT_ROOT / "granulometry_PDFs.csv", index=False)
    log("Saved granulometry_PDFs.csv")

    summary_df = pd.DataFrame(granulometry_rows)
    summary_df.to_csv(OUTPUT_ROOT / "granulometry_summary.csv", index=False)
    log("Saved granulometry_summary.csv")

    # ============================================================
    # Combined per-treatment plots
    # ============================================================
    for treatment in TREATMENT_LABELS:
        treat_pdfs = pdf_df[pdf_df["treatment"] == treatment]
        if len(treat_pdfs) == 0:
            continue
        outfile = plots_dir / f"{treatment}_combined_distribution.svg"
        save_combined_treatment_plot(treatment, treat_pdfs, outfile)
        log(f"Saved combined plot: {outfile.name}")

    log("All done!")


if __name__ == "__main__":
    main()