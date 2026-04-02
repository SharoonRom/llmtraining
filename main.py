"""
Hybrid Quantum-Classical Algorithms for Optimizing Resource Allocation
in Precision Agriculture under Climate Variability
======================================================================

Entry point — runs the full pipeline:

  Step 1 : Generate synthetic datasets (climate, soil, crop)
  Step 2 : Preprocess and split data
  Step 3 : Train crop yield prediction model
  Step 4 : Formulate the resource allocation optimization problem
  Step 5 : Run classical optimizers (GA, PSO, LP)
  Step 6 : Run quantum optimizer (QAOA)
  Step 7 : Compare results and generate output charts

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
    )

    # ── Step 1 : Data generation ─────────────────────────────────────
    banner("STEP 1 — Generate Datasets")
    import pandas as pd
    climate_df, soil_df, crop_df = generate_all_datasets(
        output_dir="data/raw",
        n=args.n_samples,
    )

    # ── Step 2 : Preprocessing ───────────────────────────────────────
    banner("STEP 2 — Preprocess Data")
    train_df, test_df, preprocessor = run_preprocessing(
        raw_dir="data/raw",
        proc_dir="data/processed",
    )

    # ── Step 3 : Crop yield predictor ────────────────────────────────
    banner("STEP 3 — Train Crop Yield Predictor")
    import numpy as np
    predictor = CropYieldPredictor()
    predictor.fit(train_df, verbose=args.verbose)
    metrics = predictor.evaluate(test_df, verbose=args.verbose)
    predictor.save(f"{args.output_dir}/crop_yield_model.joblib")
    importance_df = predictor.feature_importances()

    # Compute full statistical validation on test set
    X_test = test_df.drop(columns=["crop_yield_t_ha"])
    y_true = test_df["crop_yield_t_ha"].values
    y_pred = predictor.predict(X_test)
    pred_stats = compute_prediction_statistics(y_true, y_pred, label="Gradient Boosting / Random Forest")
    print_prediction_statistics(pred_stats)

    # ── Step 4 : Problem configuration ──────────────────────────────
    banner("STEP 4 — Formulate Optimization Problem")
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
    print(f"  Decision vars   : {config.n_vars}  |  bounds set")

    all_results: list[dict] = []

    # ── Step 5a : Genetic Algorithm ───────────────────────────────────
    banner("STEP 5a — Genetic Algorithm (DEAP)")
    ga = GeneticAlgorithmOptimizer(
        config=config,
        n_generations=args.ga_generations,
    )
    ga_result = ga.optimise(verbose=args.verbose)
    all_results.append(ga_result)

    # ── Step 5b : PSO ─────────────────────────────────────────────────
    banner("STEP 5b — Particle Swarm Optimization (PySwarms)")
    pso = PSOOptimizer(
        config=config,
        n_iters=args.pso_iters,
    )
    pso_result = pso.optimise(verbose=args.verbose)
    all_results.append(pso_result)

    # ── Step 5c : Linear Programming / NLP ───────────────────────────
    banner("STEP 5c — Linear / Nonlinear Programming (SciPy)")
    lp = LinearProgrammingOptimizer(config=config)
    lp_result = lp.optimise(verbose=args.verbose)
    all_results.append(lp_result)

    # ── Step 6 : QAOA ────────────────────────────────────────────────
    if not args.skip_qaoa:
        banner("STEP 6 — QAOA (Qiskit Aer Simulator)")
        try:
            from src.optimization.qaoa import QAOAOptimizer, QISKIT_AVAILABLE
            if not QISKIT_AVAILABLE:
                print(
                    "\n  [WARNING] Qiskit / qiskit-aer is NOT installed.\n"
                    "  Install with:  pip install qiskit qiskit-aer\n"
                    "  Running classical fallback for QAOA slot...\n"
                )
            qaoa        = QAOAOptimizer(config=config, p_layers=args.qaoa_p)
            qaoa_result = qaoa.optimise(verbose=args.verbose)
            all_results.append(qaoa_result)
            print(
                f"\n  QAOA qiskit_used={qaoa_result.get('qiskit_used')} | "
                f"qubits={qaoa_result.get('n_qubits')} | "
                f"p={qaoa_result.get('p_layers')}"
            )
        except Exception as exc:
            print(
                f"\n  [ERROR] QAOA failed: {exc}\n"
                f"  Cause: most likely Qiskit is not installed or there is a\n"
                f"  version mismatch.  Fix:  pip install qiskit>=1.0 qiskit-aer>=0.14\n"
                f"  The other 3 algorithm results will still be plotted.\n"
            )
    else:
        print("\n  [QAOA skipped via --skip-qaoa flag]")

    # ── Step 7 : Compare & visualise ─────────────────────────────────
    banner("STEP 7 — Results & Visualization")

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

    # --- 4 new statistical charts ---
    plot_actual_vs_predicted(y_true, y_pred, pred_stats, output_dir=args.output_dir)
    plot_residuals(y_true, y_pred, pred_stats,            output_dir=args.output_dir)
    plot_statistical_summary(alg_stats,                   output_dir=args.output_dir)
    plot_statistical_tests(alg_stats,                     output_dir=args.output_dir)

    # Save all statistics to CSV
    save_statistics_csv(pred_stats, alg_stats, output_dir=args.output_dir)

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
    print(f"  Predicted yield  : {best['predicted_yield_t_ha']} t/ha")
    print(f"  Normalised cost  : {best['normalised_cost']*100:.1f}%")
    print(f"\n  Yield predictor  : {metrics['best_model']}")
    print(f"  Test R²          : {metrics['test_r2']}")
    print(f"  Test RMSE        : {metrics['test_rmse']} t/ha")

    elapsed = time.perf_counter() - t_total
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"  Output charts: {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()
