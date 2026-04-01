"""
Unit and integration tests for the precision agriculture optimization system.

Run with:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.problem import ProblemConfig, AgricultureOptimizationProblem
from src.data.data_generator  import (
    generate_climate_dataset,
    generate_soil_dataset,
    generate_crop_dataset,
)
from src.data.preprocessor import AgriculturalPreprocessor


# ─────────────────────────────────────────────────────────────────────
# Problem formulation tests
# ─────────────────────────────────────────────────────────────────────

class TestProblemConfig:
    def test_n_vars(self):
        cfg = ProblemConfig()
        assert cfg.n_vars == 6

    def test_bounds_lengths(self):
        cfg = ProblemConfig()
        assert len(cfg.bounds) == 6
        assert len(cfg.lower_bounds) == 6
        assert len(cfg.upper_bounds) == 6

    def test_bounds_order(self):
        cfg = ProblemConfig()
        assert np.all(cfg.lower_bounds < cfg.upper_bounds)


class TestAgricultureOptimizationProblem:
    def setup_method(self):
        self.problem = AgricultureOptimizationProblem()
        cfg = self.problem.config
        # Use midpoint of bounds as a valid test vector
        self.x_mid = (cfg.lower_bounds + cfg.upper_bounds) / 2.0

    def test_predict_yield_range(self):
        y = self.problem.predict_yield(self.x_mid)
        assert 0.0 < y < 12.0, f"yield {y} out of expected range"

    def test_resource_cost_range(self):
        c = self.problem.resource_cost(self.x_mid)
        assert 0.0 <= c <= 1.0, f"cost {c} out of [0,1]"

    def test_objective_is_negative(self):
        """The objective should be ≤ 0 (we minimise negative utility)."""
        obj = self.problem.objective(self.x_mid)
        assert obj < 0, f"expected negative objective, got {obj}"

    def test_objective_batch_shape(self):
        X = np.tile(self.x_mid, (5, 1))
        vals = self.problem.objective_batch(X)
        assert vals.shape == (5,)

    def test_is_feasible_midpoint(self):
        assert self.problem.is_feasible(self.x_mid)

    def test_is_infeasible_out_of_bounds(self):
        x_bad = self.x_mid.copy()
        x_bad[0] = -999.0
        assert not self.problem.is_feasible(x_bad)

    def test_clip_to_bounds(self):
        x_bad  = np.zeros(6)
        x_clip = self.problem.clip_to_bounds(x_bad)
        assert np.all(x_clip >= self.problem.config.lower_bounds)
        assert np.all(x_clip <= self.problem.config.upper_bounds)

    def test_binary_encode_decode_roundtrip(self):
        x     = self.x_mid
        bits  = self.problem.encode_to_binary(x)
        x_rec = self.problem.decode_from_binary(bits)
        # Expect close (not exact — discretisation error)
        np.testing.assert_allclose(x, x_rec, rtol=0.05, atol=50)

    def test_qubo_matrix_shape(self):
        cfg = self.problem.config
        Q   = self.problem.build_qubo_matrix()
        n   = cfg.n_vars * cfg.n_bits
        assert Q.shape == (n, n)

    def test_summarise_keys(self):
        s = self.problem.summarise(self.x_mid, algorithm="Test")
        for key in [
            "algorithm", "irrigation_water_l_ha", "predicted_yield_t_ha",
            "normalised_cost", "objective_value",
        ]:
            assert key in s


# ─────────────────────────────────────────────────────────────────────
# Dataset generation tests
# ─────────────────────────────────────────────────────────────────────

class TestDataGenerator:
    def test_climate_shape(self):
        df = generate_climate_dataset(n=50)
        assert len(df) == 50
        assert "temperature_mean_c" in df.columns
        assert "rainfall_mm" in df.columns

    def test_soil_shape(self):
        df = generate_soil_dataset(n=50)
        assert len(df) == 50
        assert "soil_nitrogen_kg_ha" in df.columns
        assert "soil_ph" in df.columns

    def test_crop_shape(self):
        clim = generate_climate_dataset(n=50)
        soil = generate_soil_dataset(n=50)
        crop = generate_crop_dataset(clim, soil)
        assert len(crop) == 50
        assert "crop_yield_t_ha" in crop.columns
        assert "crop_type" in crop.columns

    def test_yield_positive(self):
        clim = generate_climate_dataset(n=100)
        soil = generate_soil_dataset(n=100)
        crop = generate_crop_dataset(clim, soil)
        assert (crop["crop_yield_t_ha"] > 0).all()

    def test_no_missing_values(self):
        df = generate_climate_dataset(n=50)
        assert df.isnull().sum().sum() == 0


# ─────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ─────────────────────────────────────────────────────────────────────

class TestPreprocessor:
    def setup_method(self):
        clim = generate_climate_dataset(n=200)
        soil = generate_soil_dataset(n=200)
        crop = generate_crop_dataset(clim, soil)
        raw  = pd.concat([clim, soil, crop], axis=1)
        self.raw = raw

    def test_fit_transform_split_ratio(self):
        prep = AgriculturalPreprocessor(test_size=0.30)
        train, test = prep.fit_transform(self.raw)
        total = len(train) + len(test)
        assert abs(len(test) / total - 0.30) < 0.05

    def test_no_missing_after_preprocess(self):
        prep = AgriculturalPreprocessor()
        train, test = prep.fit_transform(self.raw)
        assert train.isnull().sum().sum() == 0
        assert test.isnull().sum().sum() == 0

    def test_target_column_preserved(self):
        prep = AgriculturalPreprocessor()
        train, _ = prep.fit_transform(self.raw)
        assert "crop_yield_t_ha" in train.columns

    def test_numeric_features_in_range(self):
        prep = AgriculturalPreprocessor()
        train, test = prep.fit_transform(self.raw)
        for col in prep.numeric_features:
            if col in train.columns:
                assert train[col].min() >= -0.01
                assert train[col].max() <=  1.01


# ─────────────────────────────────────────────────────────────────────
# Optimizer smoke tests (small budgets)
# ─────────────────────────────────────────────────────────────────────

class TestGeneticAlgorithm:
    def test_runs_and_returns_valid_result(self):
        from src.optimization.genetic_algorithm import GeneticAlgorithmOptimizer
        ga = GeneticAlgorithmOptimizer(
            pop_size=20,
            n_generations=10,
        )
        result = ga.optimise(verbose=False)
        assert "predicted_yield_t_ha" in result
        assert result["predicted_yield_t_ha"] > 0
        assert "convergence" in result
        assert len(result["convergence"]) == 10


class TestPSO:
    def test_runs_and_returns_valid_result(self):
        from src.optimization.pso import PSOOptimizer
        pso = PSOOptimizer(n_particles=10, n_iters=10)
        result = pso.optimise(verbose=False)
        assert "predicted_yield_t_ha" in result
        assert result["predicted_yield_t_ha"] > 0


class TestLinearProgramming:
    def test_runs_and_returns_valid_result(self):
        from src.optimization.linear_programming import LinearProgrammingOptimizer
        lp = LinearProgrammingOptimizer(n_restarts=3, max_iter=50)
        result = lp.optimise(verbose=False)
        assert "predicted_yield_t_ha" in result
        assert result["predicted_yield_t_ha"] > 0


# ─────────────────────────────────────────────────────────────────────
# Crop yield predictor smoke test
# ─────────────────────────────────────────────────────────────────────

class TestCropYieldPredictor:
    def test_fit_predict(self):
        from src.models.crop_yield_predictor import CropYieldPredictor

        clim = generate_climate_dataset(n=300)
        soil = generate_soil_dataset(n=300)
        crop = generate_crop_dataset(clim, soil)
        raw  = pd.concat([clim, soil, crop], axis=1)

        prep = AgriculturalPreprocessor()
        train, test = prep.fit_transform(raw)

        predictor = CropYieldPredictor(n_estimators=20)
        predictor.fit(train, verbose=False)
        metrics = predictor.evaluate(test, verbose=False)

        assert metrics["test_r2"] > 0.5, f"R² too low: {metrics['test_r2']}"
        assert metrics["test_rmse"] > 0

    def test_feature_importances(self):
        from src.models.crop_yield_predictor import CropYieldPredictor

        clim = generate_climate_dataset(n=200)
        soil = generate_soil_dataset(n=200)
        crop = generate_crop_dataset(clim, soil)
        raw  = pd.concat([clim, soil, crop], axis=1)

        prep = AgriculturalPreprocessor()
        train, _ = prep.fit_transform(raw)

        predictor = CropYieldPredictor(n_estimators=10)
        predictor.fit(train, verbose=False)
        imp_df = predictor.feature_importances()

        assert "feature" in imp_df.columns
        assert "importance" in imp_df.columns
        assert len(imp_df) > 0
