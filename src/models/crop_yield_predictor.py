"""
Crop yield prediction model.

Trains a Random Forest Regressor on the preprocessed dataset and provides:
  - fit / predict interface
  - Feature importance ranking
  - Cross-validated performance metrics (R², RMSE, MAE)
  - Persisted model (joblib)
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import cross_val_score

TARGET = "crop_yield_t_ha"


class CropYieldPredictor:
    """
    Ensemble-based crop yield prediction model.

    Two models are trained and compared:
      1. Random Forest Regressor
      2. Gradient Boosting Regressor

    The best model by cross-validated R² is retained.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        random_state: int = 42,
    ):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.random_state  = random_state

        self.rf_model  = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.gb_model  = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            random_state=random_state,
        )

        self.best_model = None
        self.best_name  = ""
        self.feature_names: list[str] = []
        self.metrics: dict = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        verbose: bool = True,
    ) -> "CropYieldPredictor":
        """Fit both models; keep the better one."""
        if TARGET not in train_df.columns:
            raise ValueError(f"Target column '{TARGET}' not found in training data.")

        X = train_df.drop(columns=[TARGET])
        y = train_df[TARGET]
        self.feature_names = list(X.columns)

        t0 = time.perf_counter()

        # 5-fold CV for model selection
        cv_rf = cross_val_score(
            self.rf_model, X, y, cv=5, scoring="r2", n_jobs=-1
        ).mean()
        cv_gb = cross_val_score(
            self.gb_model, X, y, cv=5, scoring="r2", n_jobs=-1
        ).mean()

        if cv_rf >= cv_gb:
            self.best_model = self.rf_model
            self.best_name  = "Random Forest"
            cv_score = cv_rf
        else:
            self.best_model = self.gb_model
            self.best_name  = "Gradient Boosting"
            cv_score = cv_gb

        self.best_model.fit(X, y)
        self._fitted = True
        elapsed = time.perf_counter() - t0

        self.metrics["cv_r2"]           = round(float(cv_rf), 4)
        self.metrics["cv_r2_gb"]        = round(float(cv_gb), 4)
        self.metrics["best_model"]      = self.best_name
        self.metrics["best_cv_r2"]      = round(float(cv_score), 4)
        self.metrics["train_time_s"]    = round(elapsed, 3)

        if verbose:
            print(f"  RF  CV R² = {cv_rf:.4f}")
            print(f"  GB  CV R² = {cv_gb:.4f}")
            print(f"  → Best model: {self.best_name}  (CV R² = {cv_score:.4f})")

        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_df: pd.DataFrame,
        verbose: bool = True,
    ) -> dict:
        """Evaluate on test set; return metrics dict."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        X_test = test_df.drop(columns=[TARGET])
        y_test = test_df[TARGET].values
        y_pred = self.best_model.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = mean_absolute_error(y_test, y_pred)

        self.metrics.update({
            "test_r2":   round(float(r2),   4),
            "test_rmse": round(rmse,         4),
            "test_mae":  round(float(mae),   4),
        })

        if verbose:
            print(f"  Test  R²   = {r2:.4f}")
            print(f"  Test  RMSE = {rmse:.4f} t/ha")
            print(f"  Test  MAE  = {mae:.4f} t/ha")

        return self.metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.best_model.predict(X)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        imp = self.best_model.feature_importances_
        df  = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "results/crop_yield_model.joblib") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "CropYieldPredictor":
        return joblib.load(path)
