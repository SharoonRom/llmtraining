"""
Data preprocessing pipeline for the precision agriculture system.

Steps:
  1. Load raw CSV datasets
  2. Merge climate + soil + crop into a unified feature matrix
  3. Handle missing values (median imputation for numeric, mode for categorical)
  4. Remove outliers (IQR method)
  5. Encode categorical variables
  6. Normalise / scale numeric features (MinMaxScaler)
  7. Split into train (70%) / test (30%) sets
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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
    """Full preprocessing pipeline for the agriculture dataset."""

    def __init__(self, test_size: float = 0.30, random_state: int = RANDOM_SEED):
        self.test_size    = test_size
        self.random_state = random_state
        self.scaler       = MinMaxScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.numeric_imputer  = SimpleImputer(strategy="median")
        self.numeric_features: list[str] = []
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
        Full pipeline: clean → encode → scale → split.

        Returns
        -------
        (train_df, test_df)  both fully preprocessed DataFrames
        """
        df = df.copy()
        df = self._handle_missing(df)
        df = self._remove_outliers(df)
        df = self._encode_categoricals(df)

        self.numeric_features = [
            c for c in df.select_dtypes(include=np.number).columns
            if c != TARGET_COL
        ]

        df = self._scale(df, fit=True)

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self._fitted = True
        print(
            f"Split → train: {len(train_df)} rows | test: {len(test_df)} rows"
        )
        return train_df, test_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted pipeline to new data (no refitting)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        df = df.copy()
        df = self._handle_missing(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._scale(df, fit=False)
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

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        if num_cols:
            df[num_cols] = self.numeric_imputer.fit_transform(df[num_cols])

        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        missing_after = df.isnull().sum().sum()
        print(f"Missing values after imputation: {missing_after}")
        return df

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

    def _encode_categoricals(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = le.transform(df[col].astype(str))
        return df

    def _scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        num_cols = [c for c in self.numeric_features if c in df.columns]
        if fit:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
        return df


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
