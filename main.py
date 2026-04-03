"""
Hybrid Quantum-Classical Algorithms for Optimizing Resource Allocation
in Precision Agriculture under Climate Variability
======================================================================

Entry point — runs the full pipeline:

  Step 1 : Generate synthetic datasets (climate, soil, crop)
  Step 2 : Preprocess and split data (leakage-free: split first)
  Step 3 : Train crop yield prediction model
  Step 4 : Statistical validation + learning curves + per-crop bias analysis
  Step 5 : Formulate the resource allocation optimization problem
           (ML model wired directly into objective function)
  Step 6 : Run classical optimizers (GA, PSO, LP)
  Step 7 : Run quantum optimizer (QAOA)
  Step 8 : Out-of-distribution (OOD) scenario testing
  Step 9 : Compare results and generate output charts (13 charts, 5 CSVs)

Usage
-----
    python main.py                  # full run (all algorithms)
    python main.py --skip-qaoa      # skip QAOA (faster, no Qiskit needed)
    python main.py --n-samples 500  # use 500 data samples
    python main.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantum-Classical Agriculture Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of synthetic data samples to generate",
    )
    parser.add_argument(
        "--skip-qaoa", action="store_true",
        help="Skip QAOA (useful when Qiskit is not installed)",
    )
    parser.add_argument(
        "--qaoa-p", type=int, default=2,
        help="QAOA circuit depth (p layers)",
    )
    parser.add_argument(
        "--ga-generations", type=int, default=200,
        help="GA number of generations",
    )
    parser.add_argument(
        "--pso-iters", type=int, default=200,
        help="PSO number of iterations",
    )
    parser.add_argument(
        "--temperature", type=float, default=28.0,
        help="Climate context: mean temperature (°C)",
    )
    parser.add_argument(
        "--rainfall", type=float, default=120.0,
        help="Climate context: seasonal rainfall (mm)",
    )
    parser.add_argument(
        "--soil-nitrogen", type=float, default=80.0,
        help="Soil nitrogen content (kg/ha)",
    )
    parser.add_argument(
        "--soil-ph", type=float, default=6.5,
        help="Soil pH",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for output charts and CSV",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose console output",
    )
    return parser.parse_args()


def banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def main() -> None:
    args = parse_args()
    t_total = time.perf_counter()

    # ── Imports (here so --help works without all deps installed) ────
    from src.data.data_generator import generate_all_datasets
    from src.data.preprocessor   import run_preprocessing
    from src.models.crop_yield_predictor import CropYieldPredictor
    from src.models.statistical_validation import (
        compute_prediction_statistics,
        print_prediction_statistics,
        compute_algorithm_statistics,
        print_algorithm_statistics,
        save_statistics_csv,
        compute_per_crop_statistics,
        print_per_crop_statistics,
        compute_learning_curve_stats,
    )
    from src.optimization.problem        import ProblemConfig
    from src.optimization.genetic_algorithm import GeneticAlgorithmOptimizer
    from src.optimization.pso             import PSOOptimizer
    from src.optimization.linear_programming import LinearProgrammingOptimizer
    from src.visualization.plots import (
        plot_algorithm_comparison,
        plot_resource_usage,
        plot_convergence,
        plot_feature_importance,
        plot_yield_distribution,
        plot_climate_vs_yield,
        save_results_table,
        plot_actual_vs_predicted,
        plot_residuals,
        plot_statistical_summary,
        plot_statistical_tests,
        plot_learning_curve,
        plot_per_crop_errors,
    )

    # ── Step 1 : Data generation ─────────────────────────────────────
    banner("STEP 1 — Generate Datasets")
    import pandas as pd
    import numpy as np
    climate_df, soil_df, crop_df = generate_all_datasets(
        output_dir="data/raw",
        n=args.n_samples,
    )

    # ── Step 2 : Preprocessing (leakage-free) ────────────────────────
    banner("STEP 2 — Preprocess Data (split first, fit on train only)")
    train_df, test_df, preprocessor = run_preprocessing(
        raw_dir="data/raw",
        proc_dir="data/processed",
    )

    # ── Step 3 : Crop yield predictor ────────────────────────────────
    banner("STEP 3 — Train Crop Yield Predictor")
    predictor = CropYieldPredictor()
    predictor.fit(train_df, verbose=args.verbose)
    metrics = predictor.evaluate(test_df, verbose=args.verbose)
    predictor.save(f"{args.output_dir}/crop_yield_model.joblib")
    importance_df = predictor.feature_importances()

    # ── Step 4 : Statistical validation ─────────────────────────────
    banner("STEP 4 — Statistical Validation")
    X_test = test_df.drop(columns=["crop_yield_t_ha"])
    y_true = test_df["crop_yield_t_ha"].values
    y_pred = predictor.predict(X_test)

    pred_stats = compute_prediction_statistics(
        y_true, y_pred, label="Gradient Boosting / Random Forest"
    )
    print_prediction_statistics(pred_stats)

    # Learning curves (uses the best model architecture on full train data)
    print("\n  Computing learning curves (this may take ~30s)...")
    X_train_all = train_df.drop(columns=["crop_yield_t_ha"])
    y_train_all = train_df["crop_yield_t_ha"].values
    lc_stats = compute_learning_curve_stats(
        predictor.best_model, X_train_all, y_train_all, cv=5, n_points=8
    )
    print(
        f"  Learning curve — final train R²={lc_stats['train_scores_mean'][-1]:.4f}  "
        f"val R²={lc_stats['val_scores_mean'][-1]:.4f}  "
        f"gap={lc_stats['train_scores_mean'][-1]-lc_stats['val_scores_mean'][-1]:.4f}"
    )

    # Per-crop bias analysis (extract original crop_type labels from test set)
    # crop_type is OHE encoded; recover the original label from the OHE columns
    ohe_crop_cols = [c for c in test_df.columns if c.startswith("crop_type_")]
    if ohe_crop_cols:
        crop_labels_test = (
            test_df[ohe_crop_cols]
            .idxmax(axis=1)
            .str.replace("crop_type_", "", regex=False)
            .values
        )
    else:
        crop_labels_test = np.array(["unknown"] * len(test_df))

    per_crop_df = compute_per_crop_statistics(y_true, y_pred, crop_labels_test)
    print_per_crop_statistics(per_crop_df)

    # ── Step 5 : Problem configuration ──────────────────────────────
    banner("STEP 5 — Formulate Optimization Problem (ML-connected objective)")
    config = ProblemConfig(
        temperature   = args.temperature,
        rainfall_mm   = args.rainfall,
        soil_nitrogen = args.soil_nitrogen,
        soil_ph       = args.soil_ph,
    )
    print(f"  Climate context : temp={config.temperature}°C, "
          f"rain={config.rainfall_mm}mm")
    print(f"  Soil context    : N={config.soil_nitrogen}kg/ha, "
          f"pH={config.soil_ph}")
    print(f"  Decision vars   : {config.n_vars}  |  n_bits={config.n_bits}  "
          f"|  QUBO size={config.n_vars * config.n_bits} qubits")
    print(f"  Yield model     : ML (Gradient Boosting / Random Forest)")

    # Shared kwargs: pass ML model + preprocessor to all optimizers
    ml_kwargs = dict(ml_model=predictor, preprocessor=preprocessor)

    all_results: list[dict] = []

    # ── Step 6a : Genetic Algorithm ──────────────────────────────────
    banner("STEP 6a — Genetic Algorithm (DEAP)")
    ga = GeneticAlgorithmOptimizer(
        config=config,
        n_generations=args.ga_generations,
        **ml_kwargs,
    )
    ga_result = ga.optimise(verbose=args.verbose)
    all_results.append(ga_result)

    # ── Step 6b : PSO ────────────────────────────────────────────────
    banner("STEP 6b — Particle Swarm Optimization (PySwarms)")
    pso = PSOOptimizer(
        config=config,
        n_iters=args.pso_iters,
        **ml_kwargs,
    )
    pso_result = pso.optimise(verbose=args.verbose)
    all_results.append(pso_result)

    # ── Step 6c : Linear Programming / NLP ──────────────────────────
    banner("STEP 6c — Linear / Nonlinear Programming (SciPy)")
    lp = LinearProgrammingOptimizer(config=config, **ml_kwargs)
    lp_result = lp.optimise(verbose=args.verbose)
    all_results.append(lp_result)

    # ── Step 7 : QAOA ────────────────────────────────────────────────
    if not args.skip_qaoa:
        banner("STEP 7 — QAOA (Qiskit Aer Simulator)")
        try:
            from src.optimization.qaoa import QAOAOptimizer, QISKIT_AVAILABLE
            if not QISKIT_AVAILABLE:
                print(
                    "\n  [WARNING] Qiskit / qiskit-aer is NOT installed.\n"
                    "  Install with:  pip install qiskit qiskit-aer\n"
                    "  Running classical fallback for QAOA slot...\n"
                )
            qaoa = QAOAOptimizer(
                config=config, p_layers=args.qaoa_p, **ml_kwargs
            )
            qaoa_result = qaoa.optimise(verbose=args.verbose)
            all_results.append(qaoa_result)
            print(
                f"\n  QAOA qiskit_used={qaoa_result.get('qiskit_used')} | "
                f"qubits={qaoa_result.get('n_qubits')} | "
                f"p={qaoa_result.get('p_layers')} | "
                f"trials={qaoa_result.get('n_trials')} | "
                f"approx_ratio={qaoa_result.get('approximation_ratio')}"
            )
        except Exception as exc:
            print(
                f"\n  [ERROR] QAOA failed: {exc}\n"
                f"  Fix:  pip install qiskit>=1.0 qiskit-aer>=0.14\n"
                f"  The other 3 algorithm results will still be plotted.\n"
            )
    else:
        print("\n  [QAOA skipped via --skip-qaoa flag]")

    # ── Step 8 : OOD scenario testing ───────────────────────────────
    banner("STEP 8 — Out-of-Distribution (OOD) Scenario Testing")
    ood_scenarios = [
        {"name": "Extreme heat",    "temperature": 42.0, "rainfall_mm": 40.0},
        {"name": "Cold season",     "temperature": 10.0, "rainfall_mm": 200.0},
        {"name": "Drought",         "temperature": 30.0, "rainfall_mm": 15.0},
        {"name": "High rainfall",   "temperature": 25.0, "rainfall_mm": 290.0},
    ]
    from src.optimization.problem import AgricultureOptimizationProblem
    print(f"\n  {'Scenario':<20s}  {'Temp':>6s}  {'Rain':>7s}  "
          f"{'Yield(ML)':>10s}  {'Yield(Analytical)':>18s}")
    print("  " + "-" * 70)
    for sc in ood_scenarios:
        sc_config = ProblemConfig(
            temperature=sc["temperature"],
            rainfall_mm=sc["rainfall_mm"],
            soil_nitrogen=args.soil_nitrogen,
            soil_ph=args.soil_ph,
        )
        x_mid    = (sc_config.lower_bounds + sc_config.upper_bounds) / 2.0
        prob_ml  = AgricultureOptimizationProblem(
            sc_config, ml_model=predictor, preprocessor=preprocessor
        )
        prob_ana = AgricultureOptimizationProblem(sc_config)
        y_ml     = prob_ml._predict_yield_ml(x_mid)
        y_ana    = prob_ana._predict_yield_analytical(x_mid)
        print(
            f"  {sc['name']:<20s}  {sc['temperature']:>5.1f}°C  "
            f"{sc['rainfall_mm']:>6.0f}mm  {y_ml:>10.3f} t/ha  "
            f"{y_ana:>18.3f} t/ha"
        )

    # ── Step 9 : Compare & visualise ────────────────────────────────
    banner("STEP 9 — Results & Visualization (13 charts)")

    # Reload raw CSVs for EDA plots
    raw_climate = pd.read_csv("data/raw/climate_data.csv", index_col="sample_id")
    raw_crop    = pd.read_csv("data/raw/crop_data.csv",    index_col="sample_id")

    # Compute algorithm-level statistics + inferential tests
    alg_stats = compute_algorithm_statistics(all_results)
    print_algorithm_statistics(alg_stats)

    # --- Original 6 charts ---
    plot_algorithm_comparison(all_results, output_dir=args.output_dir)
    plot_resource_usage(all_results,       output_dir=args.output_dir)
    plot_convergence(all_results,          output_dir=args.output_dir)
    plot_feature_importance(importance_df, output_dir=args.output_dir)
    plot_yield_distribution(raw_crop,      output_dir=args.output_dir)
    plot_climate_vs_yield(raw_climate, raw_crop, output_dir=args.output_dir)
    save_results_table(all_results,        output_dir=args.output_dir)

    # --- 4 statistical charts ---
    plot_actual_vs_predicted(y_true, y_pred, pred_stats, output_dir=args.output_dir)
    plot_residuals(y_true, y_pred, pred_stats,            output_dir=args.output_dir)
    plot_statistical_summary(alg_stats,                   output_dir=args.output_dir)
    plot_statistical_tests(alg_stats,                     output_dir=args.output_dir)

    # --- 2 new charts (fix/improve) ---
    plot_learning_curve(lc_stats,             output_dir=args.output_dir)
    plot_per_crop_errors(per_crop_df,          output_dir=args.output_dir)

    # Save all statistics to CSV (now includes per-crop table)
    save_statistics_csv(pred_stats, alg_stats, output_dir=args.output_dir,
                        per_crop_df=per_crop_df)

    # Final recommendation
    best = min(all_results, key=lambda r: r["objective_value"])
    banner("FINAL RECOMMENDATION")
    print(f"  Best algorithm   : {best['algorithm']}")
    print(f"  Irrigation water : {best['irrigation_water_l_ha']} L/ha")
    print(f"  N Fertiliser     : {best['fertilizer_n_kg_ha']} kg/ha")
    print(f"  P Fertiliser     : {best['fertilizer_p_kg_ha']} kg/ha")
    print(f"  K Fertiliser     : {best['fertilizer_k_kg_ha']} kg/ha")
    print(f"  Energy           : {best['energy_kwh_ha']} kWh/ha")
    print(f"  Land area        : {best['land_area_ha']} ha")
    print(f"  Predicted yield  : {best['predicted_yield_t_ha']} t/ha  "
          f"[ML model]")
    print(f"  Normalised cost  : {best['normalised_cost']*100:.1f}%")
    print(f"\n  Yield predictor  : {metrics['best_model']}")
    print(f"  Test R²          : {metrics['test_r2']}")
    print(f"  Test RMSE        : {metrics['test_rmse']} t/ha")

    # Per-crop summary in final report
    print(f"\n  Per-crop R² summary:")
    for _, row in per_crop_df.iterrows():
        print(f"    {row['crop_type']:10s}  R²={row['r2']:.4f}  "
              f"RMSE={row['rmse']:.4f}  n={row['n_samples']}")

    elapsed = time.perf_counter() - t_total
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"  Output charts: {args.output_dir}/  (13 PNG files)")
    print(f"  Output CSVs  : {args.output_dir}/  (5 CSV files)")
    print()


if __name__ == "__main__":
    main()
