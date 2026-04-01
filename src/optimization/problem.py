"""
Optimization problem formulation for precision agriculture resource allocation.

Objective
---------
Maximise predicted crop yield while minimising total resource usage cost,
subject to hard constraints on available water, fertiliser, energy, and land.

Decision variables (continuous)
--------------------------------
  x[0] : irrigation_water_l_ha     in [water_min, water_max]
  x[1] : fertilizer_n_kg_ha        in [fert_n_min, fert_n_max]
  x[2] : fertilizer_p_kg_ha        in [fert_p_min, fert_p_max]
  x[3] : fertilizer_k_kg_ha        in [fert_k_min, fert_k_max]
  x[4] : energy_kwh_ha             in [energy_min, energy_max]
  x[5] : land_area_ha              in [land_min, land_max]

Objective (to maximise, so algorithms minimise the negative)
------------------------------------------------------------
  f(x) = w_yield * yield_model(x)
        - w_cost * normalised_cost(x)

QUBO encoding
-------------
Variables are discretised into n_bits binary variables each for QAOA.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ProblemConfig:
    """Bounds and weights for the optimisation problem."""

    # --- decision variable bounds ---
    water_min:   float = 400.0
    water_max:   float = 2000.0
    fert_n_min:  float = 10.0
    fert_n_max:  float = 120.0
    fert_p_min:  float = 5.0
    fert_p_max:  float = 60.0
    fert_k_min:  float = 10.0
    fert_k_max:  float = 80.0
    energy_min:  float = 50.0
    energy_max:  float = 400.0
    land_min:    float = 0.5
    land_max:    float = 10.0

    # --- objective weights ---
    w_yield: float = 0.70   # weight on yield maximisation
    w_cost:  float = 0.30   # weight on resource cost minimisation

    # --- per-unit costs (arbitrary units for comparison) ---
    cost_water:   float = 0.002   # per litre
    cost_fert_n:  float = 0.80    # per kg
    cost_fert_p:  float = 0.60
    cost_fert_k:  float = 0.40
    cost_energy:  float = 0.15    # per kWh
    cost_land:    float = 50.0    # per ha

    # --- climate context (set at runtime) ---
    temperature:  float = 28.0
    rainfall_mm:  float = 120.0
    soil_nitrogen:float = 80.0
    soil_ph:      float = 6.5

    # --- QUBO settings ---
    n_bits: int = 3   # bits per decision variable → 2^n_bits levels


    @property
    def n_vars(self) -> int:
        return 6

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [
            (self.water_min,  self.water_max),
            (self.fert_n_min, self.fert_n_max),
            (self.fert_p_min, self.fert_p_max),
            (self.fert_k_min, self.fert_k_max),
            (self.energy_min, self.energy_max),
            (self.land_min,   self.land_max),
        ]

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.array([b[0] for b in self.bounds])

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.array([b[1] for b in self.bounds])


class AgricultureOptimizationProblem:
    """
    Encapsulates the objective function, constraints, and QUBO construction.
    """

    def __init__(self, config: ProblemConfig | None = None):
        self.config = config or ProblemConfig()

    # ------------------------------------------------------------------
    # Yield model (physics-inspired, matching data_generator)
    # ------------------------------------------------------------------

    def predict_yield(self, x: np.ndarray) -> float:
        """
        Estimate crop yield (tons/ha) from decision variable vector x.

        x = [water, fert_n, fert_p, fert_k, energy, land]
        """
        cfg = self.config
        water, fert_n, fert_p, fert_k, energy, land = x

        temp_stress   = 1.0 - min(abs(cfg.temperature - 25.0) / 15.0, 1.0)
        rain_stress   = min(cfg.rainfall_mm / 200.0, 1.0)
        water_stress  = min(water / 1500.0, 1.0)
        n_stress      = min((fert_n + cfg.soil_nitrogen) / 200.0, 1.0)
        ph_stress     = 1.0 - min(abs(cfg.soil_ph - 6.5) / 1.5, 1.0)
        energy_factor = min(energy / 300.0, 1.0)

        combined = (
            0.22 * temp_stress  +
            0.18 * rain_stress  +
            0.22 * water_stress +
            0.20 * n_stress     +
            0.10 * ph_stress    +
            0.08 * energy_factor
        )

        base_yield = 4.5  # tons/ha (generic crop)
        return float(np.clip(base_yield * combined, 0.1, 12.0))

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------

    def resource_cost(self, x: np.ndarray) -> float:
        """Total resource cost (normalised to [0, 1])."""
        cfg = self.config
        water, fert_n, fert_p, fert_k, energy, land = x

        raw_cost = (
            cfg.cost_water  * water  +
            cfg.cost_fert_n * fert_n +
            cfg.cost_fert_p * fert_p +
            cfg.cost_fert_k * fert_k +
            cfg.cost_energy * energy +
            cfg.cost_land   * land
        )

        # Normalise by maximum possible cost
        x_max = cfg.upper_bounds
        max_cost = (
            cfg.cost_water  * x_max[0] +
            cfg.cost_fert_n * x_max[1] +
            cfg.cost_fert_p * x_max[2] +
            cfg.cost_fert_k * x_max[3] +
            cfg.cost_energy * x_max[4] +
            cfg.cost_land   * x_max[5]
        )
        return float(raw_cost / max_cost)

    # ------------------------------------------------------------------
    # Objective (minimise → negate the "good" objective)
    # ------------------------------------------------------------------

    def objective(self, x: np.ndarray) -> float:
        """
        Returns scalar to MINIMISE.
        Lower is better (higher yield, lower cost).
        """
        cfg = self.config
        y   = self.predict_yield(x)
        c   = self.resource_cost(x)
        # Normalise yield to [0, 1]
        y_norm = y / 12.0
        return -(cfg.w_yield * y_norm - cfg.w_cost * c)

    def objective_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised objective for population-based optimisers."""
        return np.array([self.objective(x) for x in X])

    # ------------------------------------------------------------------
    # Constraint checking
    # ------------------------------------------------------------------

    def is_feasible(self, x: np.ndarray) -> bool:
        lb = self.config.lower_bounds
        ub = self.config.upper_bounds
        return bool(np.all(x >= lb) and np.all(x <= ub))

    def clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.config.lower_bounds, self.config.upper_bounds)

    # ------------------------------------------------------------------
    # QUBO helpers for QAOA
    # ------------------------------------------------------------------

    def encode_to_binary(self, x: np.ndarray) -> np.ndarray:
        """
        Encode continuous decision variables to binary (Gray-code style).
        Each variable uses n_bits bits → total n_vars * n_bits bits.
        """
        cfg   = self.config
        lb    = cfg.lower_bounds
        ub    = cfg.upper_bounds
        levels = 2 ** cfg.n_bits - 1

        bits = []
        for i, xi in enumerate(x):
            level = int(round((xi - lb[i]) / (ub[i] - lb[i]) * levels))
            level = max(0, min(levels, level))
            for b in range(cfg.n_bits - 1, -1, -1):
                bits.append((level >> b) & 1)
        return np.array(bits, dtype=int)

    def decode_from_binary(self, bits: np.ndarray) -> np.ndarray:
        """Decode binary vector back to continuous decision variables."""
        cfg    = self.config
        lb     = cfg.lower_bounds
        ub     = cfg.upper_bounds
        levels = 2 ** cfg.n_bits - 1
        x      = np.zeros(cfg.n_vars)

        for i in range(cfg.n_vars):
            start = i * cfg.n_bits
            chunk = bits[start:start + cfg.n_bits]
            level = int("".join(str(b) for b in chunk), 2)
            x[i]  = lb[i] + (level / levels) * (ub[i] - lb[i])
        return x

    def build_qubo_matrix(self) -> np.ndarray:
        """
        Build a QUBO matrix Q where the energy E = x^T Q x is to be minimised.

        Strategy: sample objective at all 2^(n_bits) levels for each variable,
        fit quadratic cross-terms, and assemble Q.

        For n_vars=6, n_bits=3 → 18-qubit problem.
        """
        cfg   = self.config
        n     = cfg.n_vars * cfg.n_bits
        Q     = np.zeros((n, n))
        lb    = cfg.lower_bounds
        ub    = cfg.upper_bounds
        levels = 2 ** cfg.n_bits - 1

        # Centre-point continuous values
        x_mid = (lb + ub) / 2.0

        # Linear diagonal terms from individual variable contributions
        for i in range(cfg.n_vars):
            for b in range(cfg.n_bits):
                bit_idx = i * cfg.n_bits + b
                bit_val = 2 ** (cfg.n_bits - 1 - b)
                x_test  = x_mid.copy()
                x_test[i] = lb[i] + (bit_val / levels) * (ub[i] - lb[i])
                x_base  = x_mid.copy()
                delta_obj = self.objective(x_test) - self.objective(x_base)
                Q[bit_idx, bit_idx] += delta_obj

        # Quadratic cross-terms between variables (first-order approximation)
        for i in range(cfg.n_vars):
            for j in range(i + 1, cfg.n_vars):
                bit_i = i * cfg.n_bits
                bit_j = j * cfg.n_bits
                x_test = x_mid.copy()
                x_test[i] = ub[i]
                x_test[j] = ub[j]
                x_hi_i = x_mid.copy(); x_hi_i[i] = ub[i]
                x_hi_j = x_mid.copy(); x_hi_j[j] = ub[j]
                cross = (
                    self.objective(x_test)
                    - self.objective(x_hi_i)
                    - self.objective(x_hi_j)
                    + self.objective(x_mid)
                )
                Q[bit_i, bit_j] += cross
                Q[bit_j, bit_i] += cross

        return Q

    # ------------------------------------------------------------------
    # Result packaging
    # ------------------------------------------------------------------

    def summarise(self, x: np.ndarray, algorithm: str = "Unknown") -> dict:
        """Return a dict of results for the given solution vector."""
        y = self.predict_yield(x)
        c = self.resource_cost(x)
        obj = self.objective(x)
        return {
            "algorithm":             algorithm,
            "irrigation_water_l_ha": round(x[0], 1),
            "fertilizer_n_kg_ha":   round(x[1], 2),
            "fertilizer_p_kg_ha":   round(x[2], 2),
            "fertilizer_k_kg_ha":   round(x[3], 2),
            "energy_kwh_ha":        round(x[4], 1),
            "land_area_ha":         round(x[5], 2),
            "predicted_yield_t_ha": round(y, 3),
            "normalised_cost":      round(c, 4),
            "objective_value":      round(obj, 6),
        }
