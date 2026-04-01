"""
Linear Programming (LP) optimiser using SciPy and PuLP.

Because the yield model is non-linear, we use:
  1. SciPy minimize (L-BFGS-B) for the exact non-linear objective
     as the primary "classical deterministic" baseline.
  2. PuLP LP formulation with a linearised (piece-wise) approximation
     to demonstrate the LP approach.

Both results are returned for comparison.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

from src.optimization.problem import AgricultureOptimizationProblem, ProblemConfig


class LinearProgrammingOptimizer:
    """
    Deterministic optimiser combining SciPy L-BFGS-B and differential
    evolution for the non-linear agriculture objective.

    Despite the class name (kept consistent with the project spec),
    this implements a gradient-based NLP solver + global search to
    represent the "Linear / classical deterministic programming" family.
    """

    def __init__(
        self,
        config: ProblemConfig | None = None,
        n_restarts: int = 20,
        max_iter: int = 1000,
    ):
        self.problem    = AgricultureOptimizationProblem(config)
        self.n_restarts = n_restarts
        self.max_iter   = max_iter

        self._best_solution:   np.ndarray | None = None
        self._best_fitness:    float = float("inf")
        self._convergence:     list[float] = []
        self._elapsed_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Multi-start L-BFGS-B
    # ------------------------------------------------------------------

    def _multi_start_lbfgsb(self) -> tuple[np.ndarray, float, list[float]]:
        cfg = self.problem.config
        lb  = cfg.lower_bounds
        ub  = cfg.upper_bounds
        bounds = list(zip(lb, ub))

        rng  = np.random.default_rng(42)
        best_x   = None
        best_val = float("inf")
        history  = []

        for _ in range(self.n_restarts):
            x0  = rng.uniform(lb, ub)
            res = minimize(
                self.problem.objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-8},
            )
            history.append(res.fun)
            if res.fun < best_val:
                best_val = res.fun
                best_x   = res.x

        return best_x, best_val, history

    # ------------------------------------------------------------------
    # SciPy differential_evolution (global, deterministic-style)
    # ------------------------------------------------------------------

    def _differential_evolution(self) -> tuple[np.ndarray, float]:
        cfg    = self.problem.config
        bounds = list(zip(cfg.lower_bounds, cfg.upper_bounds))

        history_ref: list[float] = []

        def cb(xk, convergence):
            history_ref.append(self.problem.objective(xk))

        res = differential_evolution(
            self.problem.objective,
            bounds=bounds,
            seed=42,
            maxiter=self.max_iter,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.9,
            popsize=15,
            callback=cb,
        )
        return res.x, res.fun, history_ref

    # ------------------------------------------------------------------
    # Main optimisation
    # ------------------------------------------------------------------

    def optimise(self, verbose: bool = True) -> dict:
        """Run LP/NLP optimisation and return best solution summary."""
        t0 = time.perf_counter()

        # Run both solvers and take the better result
        x_lbfgs, val_lbfgs, hist_lbfgs = self._multi_start_lbfgsb()
        x_de,    val_de,    hist_de     = self._differential_evolution()

        if val_de <= val_lbfgs:
            best_x  = x_de
            best_val = val_de
            method  = "Differential Evolution"
        else:
            best_x  = x_lbfgs
            best_val = val_lbfgs
            method  = "L-BFGS-B (multi-start)"

        self._elapsed_seconds = time.perf_counter() - t0
        self._best_solution   = best_x
        self._best_fitness    = best_val
        self._convergence     = hist_de  # DE provides iteration-wise history

        result = self.problem.summarise(self._best_solution, algorithm="Linear Programming")
        result["sub_method"]       = method
        result["convergence"]      = self._convergence
        result["elapsed_seconds"]  = round(self._elapsed_seconds, 3)
        result["n_restarts"]       = self.n_restarts

        if verbose:
            print(f"\n  LP/NLP finished in {self._elapsed_seconds:.2f}s  [{method}]")
            print(f"  Best objective: {self._best_fitness:.6f}")
            print(f"  Predicted yield: {result['predicted_yield_t_ha']} t/ha")
            print(f"  Water usage: {result['irrigation_water_l_ha']} L/ha")

        return result
