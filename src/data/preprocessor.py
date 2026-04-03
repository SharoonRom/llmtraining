"""
Data preprocessing pipeline for the precision agriculture system.

Steps:
  1. Load raw CSV datasets
  2. Merge climate + soil + crop into a unified feature matrix
  3. Train/test split FIRST (prevents data leakage)
  4. Handle missing values (median imputation, fit on train only)
  5. Remove outliers (IQR method, applied to train only)
  6. Encode categorical variables (OneHot via pd.get_dummies, fit on train)
  7. Normalise / scale numeric features (MinMaxScaler, fit on train only)

Key fixes vs original:
  - Split happens BEFORE any fitting — no data leakage
  - Imputer/scaler fitted exclusively on training split
  - LabelEncoder replaced with pd.get_dummies (OneHotEncoding)
    eliminates ordinal bias for nominal categories
  - transform_single() method enables ML inference at optimization time
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

RANDOM_SEED = 42

TARGET_COL = "crop_yield_t_ha"

# Feature groups used in the yield-prediction model
CLIMATE_FEATURES = [
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "rainfall_mm",
    "humidity_pct",
    "solar_radiation_mj_m2",
    "wind_speed_m_s",
]

SOIL_FEATURES = [
    "soil_moisture_pct",
    "soil_nitrogen_kg_ha",
    "soil_phosphorus_kg_ha",
    "soil_potassium_kg_ha",
    "soil_ph",
    "soil_organic_matter_pct",
]

RESOURCE_FEATURES = [
    "irrigation_water_l_ha",
    "fertilizer_n_kg_ha",
    "fertilizer_p_kg_ha",
    "fertilizer_k_kg_ha",
    "energy_kwh_ha",
    "land_area_ha",
]

CATEGORICAL_FEATURES = ["crop_type", "soil_type"]


class AgriculturalPreprocessor:
    """Full preprocessing pipeline with data-leakage prevention."""

    def __init__(self, test_size: float = 0.30, random_state: int = RANDOM_SEED):
        self.test_size    = test_size
        self.random_state = random_state
        self.scaler       = MinMaxScaler()
        self.numeric_imputer = SimpleImputer(strategy="median")

        # Populated during fit_transform
        self.numeric_features: list[str] = []   # numeric cols used for scaling
        self.feature_columns:  list[str] = []   # all feature cols (model input order)
        self.ohe_columns:      list[str] = []   # OHE binary column names
        self.cat_fill_values:  dict      = {}   # mode per categorical col (from train)
        self._raw_numeric_cols: list[str] = []  # numeric cols before OHE
        self._ohe_train_cols:   list[str] = []  # all columns after OHE on train
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_raw(
        self,
        climate_path: str = "data/raw/climate_data.csv",
        soil_path:    str = "data/raw/soil_data.csv",
        crop_path:    str = "data/raw/crop_data.csv",
    ) -> pd.DataFrame:
        """Load and merge the three raw CSV files."""
        climate = pd.read_csv(climate_path, index_col="sample_id")
        soil    = pd.read_csv(soil_path,    index_col="sample_id")
        crop    = pd.read_csv(crop_path,    index_col="sample_id")

        merged = pd.concat([climate, soil, crop], axis=1)
        print(f"Loaded merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
        return merged

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fixed pipeline: split FIRST on raw data, then fit on train only.

        Order of operations
        -------------------
        1. Train / test split on RAW unprocessed data
        2. Fit median imputer on train — transform both
        3. Remove outliers from train only (IQR 3×)
        4. OneHot encode categoricals (categories derived from train)
        5. Fit MinMaxScaler on train numeric cols — transform both

        Returns
        -------
        (train_df, test_df)  both fully preprocessed DataFrames
        """
        df = df.copy()

        # ── Step 1: split BEFORE any transformation ───────────────────
        train_raw, test_raw = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # Column types (determined from merged structure, same for both splits)
        num_all  = train_raw.select_dtypes(include=np.number).columns.tolist()
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in train_raw.columns]
        self._raw_numeric_cols = [c for c in num_all if c != TARGET_COL]

        # ── Step 2: impute missing values — fit on train only ─────────
        if num_all:
            self.numeric_imputer.fit(train_raw[num_all])
            train_raw = train_raw.copy()
            test_raw  = test_raw.copy()
            train_raw[num_all] = self.numeric_imputer.transform(train_raw[num_all])
            test_raw[num_all]  = self.numeric_imputer.transform(test_raw[num_all])

        for col in cat_cols:
            mode_val = train_raw[col].mode()[0]
            self.cat_fill_values[col] = mode_val
            train_raw[col] = train_raw[col].fillna(mode_val)
            test_raw[col]  = test_raw[col].fillna(mode_val)

        mt = train_raw.isnull().sum().sum()
        ms = test_raw.isnull().sum().sum()
        print(f"Missing values after imputation — train: {mt}, test: {ms}")

        # ── Step 3: remove outliers from train only ───────────────────
        train_raw = self._remove_outliers(train_raw)

        # ── Step 4: OneHot encode categoricals ────────────────────────
        train_enc = self._fit_ohe(train_raw, cat_cols)
        test_enc  = self._apply_ohe(test_raw, cat_cols)

        # ── Step 5: scale numerics — fit on train only ────────────────
        self.numeric_features = [
            c for c in self._raw_numeric_cols
            if c in train_enc.columns
        ]

        self.scaler.fit(train_enc[self.numeric_features])
        train_enc[self.numeric_features] = self.scaler.transform(
            train_enc[self.numeric_features]
        )
        test_enc[self.numeric_features] = self.scaler.transform(
            test_enc[self.numeric_features]
        )

        # Store column ordering for optimization-time transforms
        self.feature_columns = [c for c in train_enc.columns if c != TARGET_COL]

        self._fitted = True
        print(f"Split → train: {len(train_enc)} rows | test: {len(test_enc)} rows")
        print(f"Feature columns after OHE: {len(self.feature_columns)}")

        return train_enc, test_enc

    def transform_single(self, row_dict: dict) -> pd.DataFrame:
        """
        Transform a single raw feature dict for ML inference during optimization.

        Parameters
        ----------
        row_dict : dict mapping feature names → raw (unscaled) values.
                   Categorical values should be strings e.g. "maize", "loam".

        Returns
        -------
        DataFrame with shape (1, n_features) in training column order,
        ready for model.predict().
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")

        # Initialise all feature columns to 0 (handles missing / OHE defaults)
        row: dict = {col: 0.0 for col in self.feature_columns}

        for k, v in row_dict.items():
            if k in CATEGORICAL_FEATURES:
                ohe_col = f"{k}_{v}"
                if ohe_col in row:
                    row[ohe_col] = 1.0
                # unknown category → all OHE cols for that feature remain 0
            elif k in row:
                row[k] = float(v)

        df_row = pd.DataFrame([row])

        # Scale numeric features only (OHE binary cols stay 0/1)
        num_present = [c for c in self.numeric_features if c in df_row.columns]
        df_row[num_present] = self.scaler.transform(df_row[num_present])

        return df_row[self.feature_columns]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted pipeline to a new DataFrame (no refitting)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        df = df.copy()
        num_cols = [c for c in self.numeric_imputer.feature_names_in_
                    if c in df.columns]
        if num_cols:
            df[num_cols] = self.numeric_imputer.transform(df[num_cols])
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        df = self._apply_ohe(df, cat_cols)
        df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])
        return df

    def get_X_y(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Split a processed DataFrame into features X and target y."""
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
        return X, y

    def save_processed(
        self,
        train_df: pd.DataFrame,
        test_df:  pd.DataFrame,
        output_dir: str = "data/processed",
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out / "train.csv")
        test_df.to_csv(out  / "test.csv")
        print(f"Saved processed data to {output_dir}/")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR-based outlier removal on numeric columns (excluding target)."""
        num_cols = [
            c for c in df.select_dtypes(include=np.number).columns
            if c != TARGET_COL
        ]
        before = len(df)
        mask = pd.Series([True] * len(df), index=df.index)
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask &= df[col].between(Q1 - 3.0 * IQR, Q3 + 3.0 * IQR)
        df = df[mask]
        print(f"Outlier removal: {before - len(df)} rows dropped ({len(df)} remain)")
        return df

    def _fit_ohe(self, df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
        """Fit OneHot encoding on training data via pd.get_dummies."""
        if not cat_cols:
            self.ohe_columns = []
            self._ohe_train_cols = list(df.columns)
            return df
        df_enc = pd.get_dummies(df, columns=cat_cols, dtype=float)
        self.ohe_columns = [
            c for c in df_enc.columns
            if any(c.startswith(f"{cat}_") for cat in cat_cols)
        ]
        self._ohe_train_cols = list(df_enc.columns)
        return df_enc

    def _apply_ohe(self, df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
        """Apply OHE using categories from training (no new categories)."""
        if not cat_cols:
            return df
        df_enc = pd.get_dummies(df, columns=cat_cols, dtype=float)
        # Add columns for any training categories absent in this split
        for col in self.ohe_columns:
            if col not in df_enc.columns:
                df_enc[col] = 0.0
        # Drop columns for categories not seen during training
        extra = [
            c for c in df_enc.columns
            if any(c.startswith(f"{cat}_") for cat in cat_cols)
            and c not in self.ohe_columns
        ]
        if extra:
            df_enc = df_enc.drop(columns=extra)
        # Reorder to match training column order exactly
        for col in self._ohe_train_cols:
            if col not in df_enc.columns:
                df_enc[col] = 0.0
        return df_enc[self._ohe_train_cols]


def run_preprocessing(
    raw_dir:  str = "data/raw",
    proc_dir: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, AgriculturalPreprocessor]:
    """Convenience function: load, preprocess, and save datasets."""
    prep = AgriculturalPreprocessor()
    raw  = prep.load_raw(
        climate_path=f"{raw_dir}/climate_data.csv",
        soil_path   =f"{raw_dir}/soil_data.csv",
        crop_path   =f"{raw_dir}/crop_data.csv",
    )
    train_df, test_df = prep.fit_transform(raw)
    prep.save_processed(train_df, test_df, output_dir=proc_dir)
    return train_df, test_df, prep


if __name__ == "__main__":
    run_preprocessing()
