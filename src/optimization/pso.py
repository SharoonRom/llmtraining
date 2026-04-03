"""
Particle Swarm Optimisation (PSO) using PySwarms.

Implements global-best PSO (pyswarms.single.GlobalBestPSO) for the
agriculture resource allocation problem.

Hyperparameters follow the classic Clerc-Kennedy constriction model:
  c1 (cognitive), c2 (social), w (inertia weight).
"""

from __future__ import annotations

import time
import numpy as np
import pyswarms as ps
from pyswarms.single import GlobalBestPSO

from src.optimization.problem import AgricultureOptimizationProblem, ProblemConfig


class PSOOptimizer:
    """
    Particle Swarm Optimisation for agriculture resource allocation.

    Parameters
    ----------
    config      : Problem configuration
    n_particles : Swarm size
    n_iters     : Number of iterations
    c1          : Cognitive parameter (personal best attraction)
    c2          : Social parameter (global best attraction)
    w           : Inertia weight
    """

    def __init__(
        self,
        config: ProblemConfig | None = None,
        n_particles: int = 100,
        n_iters: int = 200,
        c1: float = 1.494,
        c2: float = 1.494,
        w: float  = 0.729,
        ml_model=None,
        preprocessor=None,
    ):
        self.problem     = AgricultureOptimizationProblem(
            config, ml_model=ml_model, preprocessor=preprocessor
        )
        self.n_particles = n_particles
        self.n_iters     = n_iters
        self.options     = {"c1": c1, "c2": c2, "w": w}

        self._best_solution:   np.ndarray | None = None
        self._best_fitness:    float = float("inf")
        self._convergence:     list[float] = []
        self._elapsed_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Cost function (PySwarms expects shape [n_particles, n_dims] → [n_particles])
    # ------------------------------------------------------------------

    def _cost_fn(self, X: np.ndarray) -> np.ndarray:
        """Vectorised cost function for PySwarms."""
        return self.problem.objective_batch(X)

    # ------------------------------------------------------------------
    # Main optimisation
    # ------------------------------------------------------------------

    def optimise(self, verbose: bool = True) -> dict:
        """Run PSO and return best solution summary."""
        np.random.seed(42)
        cfg = self.problem.config
        lb  = cfg.lower_bounds
        ub  = cfg.upper_bounds

        bounds = (lb, ub)

        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=cfg.n_vars,
            options=self.options,
            bounds=bounds,
        )

        t0 = time.perf_counter()

        # iternow controls verbosity inside PySwarms
        best_cost, best_pos = optimizer.optimize(
            self._cost_fn,
            iters=self.n_iters,
            verbose=False,
        )

        self._elapsed_seconds = time.perf_counter() - t0
        self._best_solution   = best_pos
        self._best_fitness    = best_cost
        self._convergence     = list(optimizer.cost_history)

        result = self.problem.summarise(self._best_solution, algorithm="PSO")
        result["convergence"]     = self._convergence
        result["elapsed_seconds"] = round(self._elapsed_seconds, 3)
        result["n_iterations"]    = self.n_iters
        result["n_particles"]     = self.n_particles

        if verbose:
            print(f"\n  PSO finished in {self._elapsed_seconds:.2f}s")
            print(f"  Best objective: {self._best_fitness:.6f}")
            print(f"  Predicted yield: {result['predicted_yield_t_ha']} t/ha")
            print(f"  Water usage: {result['irrigation_water_l_ha']} L/ha")

        return result
