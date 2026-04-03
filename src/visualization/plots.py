"""
Visualization module for the precision agriculture optimization system.

Generates all charts required by the project:
  1.  Algorithm performance comparison (bar chart)
  2.  Resource usage comparison table / chart
  3.  Convergence curves for each algorithm
  4.  Feature importance (crop yield predictor)
  5.  Predicted yield distribution
  6.  Climate vs yield scatter plots
  7.  Summary results table (console + CSV)
  8.  Actual vs Predicted yield scatter (new)
  9.  Residual plot (new)
  10. Statistical summary bar chart (mean ± std) (new)
  11. ANOVA / t-test p-value heatmap (new)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
# Agg backend works on every platform (Windows, Mac, Linux, servers).
# Charts are always saved as PNG files in the results/ folder.
# On your PC just open the PNG files from results/ to view them.
matplotlib.use("Agg")
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
        colour = PALETTE.get(res["algorithm"], "#607D8B")
        if not hist:
            # No convergence history — draw a single horizontal line at final objective
            final_val = res.get("objective_value", 0)
            ax.axhline(
                final_val, color=colour, linewidth=2,
                linestyle="--", label=f"{res['algorithm']} (single point)",
            )
            continue
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


# ------------------------------------------------------------------
# 8. Actual vs Predicted yield scatter
# ------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    stats_dict: dict,
    output_dir: str = "results",
) -> str:
    """Scatter plot of actual vs predicted crop yield with perfect-fit line."""
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "actual_vs_predicted.png"

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.45, s=18, color="#2196F3", label="Samples")

    # Perfect prediction line
    lo = min(y_true.min(), y_pred.min()) - 0.1
    hi = max(y_true.max(), y_pred.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.8, label="Perfect fit (y=x)")

    ax.set_xlabel("Actual Yield (t/ha)")
    ax.set_ylabel("Predicted Yield (t/ha)")
    ax.set_title(
        f"Actual vs Predicted Crop Yield\n"
        f"R²={stats_dict['r2']}  RMSE={stats_dict['rmse']} t/ha  "
        f"MAE={stats_dict['mae']} t/ha",
        fontweight="bold",
    )
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 9. Residual plot
# ------------------------------------------------------------------

def plot_residuals(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    stats_dict: dict,
    output_dir: str = "results",
) -> str:
    """Residual histogram + residuals vs predicted scatter."""
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "residuals.png"

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Residual Analysis   "
        f"Mean={stats_dict['resid_mean']}  Std={stats_dict['resid_std']}  "
        f"Shapiro-Wilk p={stats_dict['shapiro_wilk_p']}",
        fontsize=11, fontweight="bold",
    )

    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=axes[0], color="#FF9800", bins=30)
    axes[0].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_title("Distribution of Residuals")
    axes[0].set_xlabel("Residual (t/ha)")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.4, s=14, color="#9C27B0")
    axes[1].axhline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted Yield (t/ha)")
    axes[1].set_ylabel("Residual (t/ha)")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 10. Statistical summary: mean ± std per algorithm
# ------------------------------------------------------------------

def plot_statistical_summary(
    alg_stats: dict,
    output_dir: str = "results",
) -> str:
    """
    Bar chart showing key metrics per algorithm with descriptive stats overlay.
    Displays: yield, water, cost, runtime — with value labels.
    """
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "statistical_summary.png"

    rows  = alg_stats["per_algorithm"]
    names = [r["algorithm"] for r in rows]
    colours = [PALETTE.get(n, "#607D8B") for n in names]

    metrics = [
        ("predicted_yield_t_ha",  "Predicted Yield (t/ha)"),
        ("irrigation_water_l_ha", "Water Usage (L/ha)"),
        ("normalised_cost",       "Resource Cost (%)"),
        ("elapsed_seconds",       "Runtime (s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Statistical Summary — All Algorithms\n"
        f"Yield mean={alg_stats['descriptive']['yield_mean']} t/ha  "
        f"std={alg_stats['descriptive']['yield_std']} t/ha  "
        f"range={alg_stats['descriptive']['yield_range']} t/ha",
        fontsize=12, fontweight="bold",
    )

    for ax, (key, label) in zip(axes.flatten(), metrics):
        vals = [r[key] for r in rows]
        bars = ax.bar(names, vals, color=colours, edgecolor="white", width=0.55)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.30 if max(vals) > 0 else 1)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )
        ax.tick_params(axis="x", rotation=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 11. ANOVA / t-test p-value heatmap
# ------------------------------------------------------------------

def plot_statistical_tests(
    alg_stats: dict,
    output_dir: str = "results",
) -> str:
    """Heatmap of pairwise t-test p-values + ANOVA result annotation."""
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "statistical_tests.png"

    ttests = alg_stats["pairwise_ttests"]
    anova  = alg_stats["anova"]

    # Collect algorithm names
    names = list(dict.fromkeys(
        [t["algorithm_A"] for t in ttests] +
        [t["algorithm_B"] for t in ttests]
    ))
    n = len(names)

    # Build p-value matrix (diagonal = 1.0 = not significant vs itself)
    p_matrix = np.ones((n, n))
    d_matrix = np.zeros((n, n))   # yield diff matrix

    idx = {name: i for i, name in enumerate(names)}
    for t in ttests:
        i, j = idx[t["algorithm_A"]], idx[t["algorithm_B"]]
        p_matrix[i, j] = t["p_value"]
        p_matrix[j, i] = t["p_value"]
        d_matrix[i, j] = t["yield_diff"]
        d_matrix[j, i] = -t["yield_diff"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Inferential Statistics\n"
        f"One-Way ANOVA: F={anova.get('f_statistic','N/A')}  "
        f"p={anova.get('p_value','N/A')}  "
        f"Significant={anova.get('significant','N/A')}",
        fontsize=11, fontweight="bold",
    )

    # p-value heatmap
    im0 = axes[0].imshow(p_matrix, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_yticks(range(n)); axes[0].set_yticklabels(names)
    axes[0].set_title("Pairwise t-test p-values\n(green=p<0.05 significant)", fontweight="bold")
    for i in range(n):
        for j in range(n):
            sig = "*" if p_matrix[i, j] < 0.05 and i != j else ""
            axes[0].text(j, i, f"{p_matrix[i,j]:.3f}{sig}",
                         ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im0, ax=axes[0], label="p-value")

    # Yield difference heatmap
    vabs = max(abs(d_matrix.max()), abs(d_matrix.min()), 0.001)
    im1 = axes[1].imshow(d_matrix, vmin=-vabs, vmax=vabs, cmap="RdBu", aspect="auto")
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_yticks(range(n)); axes[1].set_yticklabels(names)
    axes[1].set_title("Yield Difference (t/ha)\nrow − column", fontweight="bold")
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f"{d_matrix[i,j]:+.3f}",
                         ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im1, ax=axes[1], label="Δ yield (t/ha)")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 12. Learning curve (train vs validation R²)
# ------------------------------------------------------------------

def plot_learning_curve(
    lc_stats: dict,
    output_dir: str = "results",
) -> str:
    """
    Line chart showing train and validation R² as training size grows.
    Reveals over/under-fitting and whether more data would help.
    """
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "learning_curve.png"

    ts   = lc_stats["train_sizes_abs"]
    tr_m = np.array(lc_stats["train_scores_mean"])
    tr_s = np.array(lc_stats["train_scores_std"])
    va_m = np.array(lc_stats["val_scores_mean"])
    va_s = np.array(lc_stats["val_scores_std"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(ts, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color="#2196F3")
    ax.fill_between(ts, va_m - va_s, va_m + va_s, alpha=0.15, color="#4CAF50")
    ax.plot(ts, tr_m, "o-", color="#2196F3", label="Train R²",      linewidth=2)
    ax.plot(ts, va_m, "s-", color="#4CAF50", label="Validation R²", linewidth=2)

    ax.set_xlabel("Training Set Size (samples)")
    ax.set_ylabel("R² Score")
    ax.set_title(
        "Learning Curve — Crop Yield Predictor\n"
        f"Final train R²={tr_m[-1]:.4f}  val R²={va_m[-1]:.4f}  "
        f"gap={tr_m[-1]-va_m[-1]:.4f}",
        fontweight="bold",
    )
    ax.legend(frameon=False)
    ax.set_ylim(max(0, min(va_m) - 0.1), 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)


# ------------------------------------------------------------------
# 13. Per-crop RMSE / R² bar charts (bias / fairness analysis)
# ------------------------------------------------------------------

def plot_per_crop_errors(
    per_crop_df: "pd.DataFrame",
    output_dir: str = "results",
) -> str:
    """
    Side-by-side bar charts of RMSE and R² for each crop type.
    Highlights model bias toward/against certain crops.
    """
    _ensure_dir(Path(output_dir))
    out = Path(output_dir) / "per_crop_errors.png"

    crops = per_crop_df["crop_type"].tolist()
    rmse  = per_crop_df["rmse"].tolist()
    r2    = per_crop_df["r2"].tolist()
    mae   = per_crop_df["mae"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Per-Crop Bias / Fairness Analysis\n"
        "Model accuracy broken down by crop type",
        fontsize=12, fontweight="bold",
    )

    crop_colours = plt.cm.Set2(np.linspace(0, 1, len(crops)))

    # RMSE
    bars = axes[0].bar(crops, rmse, color=crop_colours, edgecolor="white", width=0.6)
    axes[0].set_title("RMSE (t/ha) — lower=better", fontweight="bold")
    axes[0].set_ylabel("RMSE (t/ha)")
    for bar, val in zip(bars, rmse):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # MAE
    bars = axes[1].bar(crops, mae, color=crop_colours, edgecolor="white", width=0.6)
    axes[1].set_title("MAE (t/ha) — lower=better", fontweight="bold")
    axes[1].set_ylabel("MAE (t/ha)")
    for bar, val in zip(bars, mae):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # R²
    bars = axes[2].bar(crops, r2, color=crop_colours, edgecolor="white", width=0.6)
    axes[2].set_title("R² Score — higher=better", fontweight="bold")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim(0, 1.05)
    for bar, val in zip(bars, r2):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return str(out)
