"""
Visualization module for the precision agriculture optimization system.

Generates all charts required by the project:
  1. Algorithm performance comparison (bar chart)
  2. Resource usage comparison table / chart
  3. Convergence curves for each algorithm
  4. Feature importance (crop yield predictor)
  5. Predicted yield distribution
  6. Climate vs yield scatter plots
  7. Summary results table (console + CSV)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for headless environments)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Colour palette consistent across all charts
PALETTE = {
    "Genetic Algorithm":  "#2196F3",   # blue
    "PSO":                "#4CAF50",   # green
    "Linear Programming": "#FF9800",   # orange
    "QAOA":               "#9C27B0",   # purple
}

RESULTS_DIR = Path("results")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# 1. Algorithm performance comparison
# ------------------------------------------------------------------

def plot_algorithm_comparison(
    results: list[dict],
    output_dir: str = "results",
) -> str:
    """Bar chart comparing predicted yield and normalised cost per algorithm."""
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "algorithm_comparison.png"

    names  = [r["algorithm"] for r in results]
    yields = [r["predicted_yield_t_ha"] for r in results]
    costs  = [r["normalised_cost"] * 100 for r in results]   # → %
    obj    = [abs(r["objective_value"]) for r in results]

    colours = [PALETTE.get(n, "#607D8B") for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Algorithm Performance Comparison\n"
        "Precision Agriculture Resource Allocation",
        fontsize=13, fontweight="bold",
    )

    # Yield
    bars = axes[0].bar(names, yields, color=colours, edgecolor="white", width=0.55)
    axes[0].set_title("Predicted Crop Yield (t/ha)", fontweight="bold")
    axes[0].set_ylabel("tons / ha")
    axes[0].set_ylim(0, max(yields) * 1.25)
    for bar, val in zip(bars, yields):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    # Cost
    bars = axes[1].bar(names, costs, color=colours, edgecolor="white", width=0.55)
    axes[1].set_title("Normalised Resource Cost (%)", fontweight="bold")
    axes[1].set_ylabel("cost %")
    axes[1].set_ylim(0, max(costs) * 1.25)
    for bar, val in zip(bars, costs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    # Objective (higher = better for maximisation visualisation)
    bars = axes[2].bar(names, obj, color=colours, edgecolor="white", width=0.55)
    axes[2].set_title("|Objective Value| (higher = better)", fontweight="bold")
    axes[2].set_ylabel("|f(x)|")
    axes[2].set_ylim(0, max(obj) * 1.25)
    for bar, val in zip(bars, obj):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 2. Resource usage comparison
# ------------------------------------------------------------------

def plot_resource_usage(
    results: list[dict],
    output_dir: str = "results",
) -> str:
    """Grouped bar chart for water, fertiliser, energy, land per algorithm."""
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "resource_usage.png"

    resource_keys = [
        ("irrigation_water_l_ha", "Water (L/ha)"),
        ("fertilizer_n_kg_ha",    "N Fertiliser (kg/ha)"),
        ("energy_kwh_ha",         "Energy (kWh/ha)"),
        ("land_area_ha",          "Land (ha)"),
    ]

    n_res  = len(resource_keys)
    n_alg  = len(results)
    x      = np.arange(n_res)
    width  = 0.18
    offsets = np.linspace(-(n_alg - 1) / 2, (n_alg - 1) / 2, n_alg) * width

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_title(
        "Resource Usage per Algorithm",
        fontsize=13, fontweight="bold",
    )

    for i, res in enumerate(results):
        vals    = [res[k] for k, _ in resource_keys]
        colour  = PALETTE.get(res["algorithm"], "#607D8B")
        bars    = ax.bar(
            x + offsets[i], vals, width * 0.85,
            label=res["algorithm"], color=colour, edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in resource_keys])
    ax.set_ylabel("Resource Usage")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 3. Convergence curves
# ------------------------------------------------------------------

def plot_convergence(
    results: list[dict],
    output_dir: str = "results",
) -> str:
    """Plot objective-value convergence curves for all algorithms."""
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "convergence_curves.png"

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(
        "Convergence Curves — Objective Value vs Iteration",
        fontsize=13, fontweight="bold",
    )

    for res in results:
        hist   = res.get("convergence", [])
        if not hist:
            continue
        colour = PALETTE.get(res["algorithm"], "#607D8B")
        # Running minimum for monotone curve
        running_min = np.minimum.accumulate(hist)
        ax.plot(
            range(len(running_min)),
            running_min,
            label=res["algorithm"],
            color=colour,
            linewidth=2,
        )

    ax.set_xlabel("Iteration / Generation")
    ax.set_ylabel("Best Objective Value")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 4. Feature importance
# ------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_dir: str = "results",
    top_n: int = 15,
) -> str:
    """Horizontal bar chart of top-N feature importances."""
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "feature_importance.png"

    df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        df["feature"][::-1],
        df["importance"][::-1],
        color="#2196F3",
        edgecolor="white",
    )
    ax.set_title(
        f"Top-{top_n} Feature Importances — Crop Yield Predictor",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Importance Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 5. Yield distribution
# ------------------------------------------------------------------

def plot_yield_distribution(
    crop_df: pd.DataFrame,
    output_dir: str = "results",
) -> str:
    """KDE + histogram of crop yields split by crop type."""
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "yield_distribution.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Crop Yield Distribution", fontsize=13, fontweight="bold")

    # Overall distribution
    sns.histplot(
        crop_df["crop_yield_t_ha"],
        kde=True,
        ax=axes[0],
        color="#2196F3",
        bins=30,
    )
    axes[0].set_title("Overall Yield Distribution")
    axes[0].set_xlabel("Yield (t/ha)")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # By crop type
    if "crop_type" in crop_df.columns:
        sns.boxplot(
            data=crop_df,
            x="crop_type",
            y="crop_yield_t_ha",
            ax=axes[1],
            palette="Set2",
        )
        axes[1].set_title("Yield by Crop Type")
        axes[1].set_xlabel("Crop Type")
        axes[1].set_ylabel("Yield (t/ha)")
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 6. Climate vs yield scatter
# ------------------------------------------------------------------

def plot_climate_vs_yield(
    climate_df: pd.DataFrame,
    crop_df: pd.DataFrame,
    output_dir: str = "results",
) -> str:
    """Scatter plots of key climate variables against crop yield."""
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "climate_vs_yield.png"

    climate_vars = [
        ("temperature_mean_c", "Temperature (°C)"),
        ("rainfall_mm",         "Rainfall (mm)"),
        ("humidity_pct",        "Humidity (%)"),
        ("solar_radiation_mj_m2", "Solar Radiation (MJ/m²)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Climate Variables vs Crop Yield",
        fontsize=13, fontweight="bold",
    )

    yield_vals = crop_df["crop_yield_t_ha"].values
    axes_flat  = axes.flatten()

    for i, (var, label) in enumerate(climate_vars):
        if var not in climate_df.columns:
            continue
        ax = axes_flat[i]
        ax.scatter(
            climate_df[var].values, yield_vals,
            alpha=0.35, s=8, color="#2196F3",
        )
        # Trend line
        z   = np.polyfit(climate_df[var].values, yield_vals, 1)
        p   = np.poly1d(z)
        xs  = np.linspace(climate_df[var].min(), climate_df[var].max(), 200)
        ax.plot(xs, p(xs), "r--", linewidth=1.5, label="trend")

        ax.set_xlabel(label)
        ax.set_ylabel("Yield (t/ha)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 7. Summary results table
# ------------------------------------------------------------------

def save_results_table(
    results: list[dict],
    output_dir: str = "results",
) -> str:
    """Save algorithm comparison table as CSV and print to console."""
    _ensure_dir(Path(output_dir))
    out_csv = Path(output_dir) / "algorithm_results.csv"

    cols = [
        "algorithm",
        "irrigation_water_l_ha",
        "fertilizer_n_kg_ha",
        "fertilizer_p_kg_ha",
        "fertilizer_k_kg_ha",
        "energy_kwh_ha",
        "land_area_ha",
        "predicted_yield_t_ha",
        "normalised_cost",
        "objective_value",
        "elapsed_seconds",
    ]

    rows = []
    for r in results:
        rows.append({c: r.get(c, "N/A") for c in cols})

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON RESULTS")
    print("=" * 70)
    print(
        df[["algorithm", "irrigation_water_l_ha",
            "predicted_yield_t_ha", "normalised_cost",
            "objective_value", "elapsed_seconds"]]
        .to_string(index=False)
    )
    print("=" * 70)
    print(f"  Saved: {out_csv}")

    return str(out_csv)
