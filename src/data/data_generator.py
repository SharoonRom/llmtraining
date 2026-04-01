"""
Synthetic dataset generator for precision agriculture simulation.

Generates realistic climate, crop, and soil datasets modelled after:
  - NASA POWER climate variables
  - Kaggle crop yield datasets
  - FAO/SoilGrids soil databases
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 1000

CROP_TYPES = ["wheat", "maize", "rice", "soybean", "cotton"]

# Realistic climate ranges per season
CLIMATE_PARAMS = {
    "temperature_mean_c":    (15.0, 38.0),   # °C
    "temperature_max_c":     (20.0, 45.0),
    "temperature_min_c":     (5.0,  28.0),
    "rainfall_mm":           (10.0, 300.0),
    "humidity_pct":          (30.0, 90.0),
    "solar_radiation_mj_m2": (8.0,  25.0),
    "wind_speed_m_s":        (0.5,  8.0),
}

SOIL_PARAMS = {
    "soil_moisture_pct":     (10.0, 45.0),
    "soil_nitrogen_kg_ha":   (20.0, 180.0),
    "soil_phosphorus_kg_ha": (5.0,  60.0),
    "soil_potassium_kg_ha":  (50.0, 300.0),
    "soil_ph":               (5.5,  8.0),
    "soil_organic_matter_pct": (1.0, 5.5),
}

RESOURCE_PARAMS = {
    "irrigation_water_l_ha":   (400.0,  2000.0),
    "fertilizer_n_kg_ha":      (10.0,   120.0),
    "fertilizer_p_kg_ha":      (5.0,    60.0),
    "fertilizer_k_kg_ha":      (10.0,   80.0),
    "energy_kwh_ha":           (50.0,   400.0),
    "land_area_ha":            (0.5,    10.0),
}


def _uniform(low: float, high: float, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(low, high, size)


def _yield_model(
    temp: np.ndarray,
    rain: np.ndarray,
    nitrogen: np.ndarray,
    water: np.ndarray,
    ph: np.ndarray,
    crop_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Physics-inspired yield model (tons/ha) with crop-specific base yields."""
    base_yields = np.array([3.5, 5.0, 4.5, 2.8, 1.8])   # wheat/maize/rice/soybean/cotton

    # Normalised stress factors in [0, 1]
    temp_opt   = 25.0
    temp_stress = 1.0 - np.clip(np.abs(temp - temp_opt) / 15.0, 0, 1)

    rain_stress  = np.clip(rain / 200.0, 0, 1)
    n_stress     = np.clip(nitrogen / 150.0, 0, 1)
    water_stress = np.clip(water / 1500.0, 0, 1)
    ph_stress    = 1.0 - np.clip(np.abs(ph - 6.5) / 1.5, 0, 1)

    combined = (
        0.25 * temp_stress +
        0.20 * rain_stress +
        0.20 * n_stress +
        0.25 * water_stress +
        0.10 * ph_stress
    )

    base = base_yields[crop_idx]
    noise = rng.normal(0, 0.15, len(temp))
    yield_t_ha = np.clip(base * combined + noise, 0.3, 12.0)
    return yield_t_ha


def generate_climate_dataset(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic climate dataset (NASA POWER style)."""
    rng = np.random.default_rng(seed)
    data = {}
    for col, (lo, hi) in CLIMATE_PARAMS.items():
        data[col] = _uniform(lo, hi, n, rng)

    # Season encoding (1=spring,2=summer,3=autumn,4=winter)
    data["season"] = rng.integers(1, 5, n)
    # Year range
    data["year"]   = rng.integers(2000, 2024, n)
    # Latitude (sub-tropical / tropical belt)
    data["latitude"]  = _uniform(-35.0, 55.0, n, rng)
    data["longitude"] = _uniform(-120.0, 150.0, n, rng)

    df = pd.DataFrame(data)
    df.index.name = "sample_id"
    return df


def generate_soil_dataset(n: int = N_SAMPLES, seed: int = RANDOM_SEED + 1) -> pd.DataFrame:
    """Generate synthetic soil dataset (FAO/SoilGrids style)."""
    rng = np.random.default_rng(seed)
    data = {}
    for col, (lo, hi) in SOIL_PARAMS.items():
        data[col] = _uniform(lo, hi, n, rng)

    data["soil_type"] = rng.choice(
        ["loam", "clay", "sandy", "silt", "peat"], n
    )
    df = pd.DataFrame(data)
    df.index.name = "sample_id"
    return df


def generate_crop_dataset(
    climate_df: pd.DataFrame,
    soil_df: pd.DataFrame,
    seed: int = RANDOM_SEED + 2,
) -> pd.DataFrame:
    """Generate synthetic crop dataset combining climate and soil inputs."""
    rng = np.random.default_rng(seed)
    n = len(climate_df)

    crop_idx = rng.integers(0, len(CROP_TYPES), n)

    resource_data = {}
    for col, (lo, hi) in RESOURCE_PARAMS.items():
        resource_data[col] = _uniform(lo, hi, n, rng)

    yield_t_ha = _yield_model(
        temp=climate_df["temperature_mean_c"].values,
        rain=climate_df["rainfall_mm"].values,
        nitrogen=soil_df["soil_nitrogen_kg_ha"].values,
        water=resource_data["irrigation_water_l_ha"],
        ph=soil_df["soil_ph"].values,
        crop_idx=crop_idx,
        rng=rng,
    )

    df = pd.DataFrame(resource_data)
    df["crop_type"]     = [CROP_TYPES[i] for i in crop_idx]
    df["crop_yield_t_ha"] = yield_t_ha
    df.index.name = "sample_id"
    return df


def generate_all_datasets(
    output_dir: str = "data/raw",
    n: int = N_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate and save climate, soil, and crop datasets.

    Returns
    -------
    climate_df, soil_df, crop_df
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n} samples for climate, soil, and crop datasets...")

    climate_df = generate_climate_dataset(n=n)
    soil_df    = generate_soil_dataset(n=n)
    crop_df    = generate_crop_dataset(climate_df, soil_df)

    climate_df.to_csv(out / "climate_data.csv")
    soil_df.to_csv(out / "soil_data.csv")
    crop_df.to_csv(out / "crop_data.csv")

    print(f"  Saved: {out}/climate_data.csv  ({len(climate_df)} rows)")
    print(f"  Saved: {out}/soil_data.csv     ({len(soil_df)} rows)")
    print(f"  Saved: {out}/crop_data.csv     ({len(crop_df)} rows)")

    return climate_df, soil_df, crop_df


if __name__ == "__main__":
    generate_all_datasets()
