# Hybrid Quantum-Classical Algorithms for Optimizing Resource Allocation in Precision Agriculture under Climate Variability

## Project Overview

A software simulation system that applies **hybrid quantum-classical optimization** to determine the best allocation of agricultural resources (water, fertilizer, energy, land) under varying climate conditions, with the goal of maximizing crop yield while minimizing resource usage.

---

## System Architecture

```
llmtraining/
├── main.py                          # Entry point — full pipeline
├── requirements.txt
├── data/
│   ├── raw/                         # Auto-generated synthetic datasets
│   │   ├── climate_data.csv         # NASA POWER-style climate variables
│   │   ├── soil_data.csv            # FAO/SoilGrids-style soil data
│   │   └── crop_data.csv            # Kaggle-style crop yield data
│   └── processed/                   # Preprocessed train/test splits
├── src/
│   ├── data/
│   │   ├── data_generator.py        # Synthetic dataset generation
│   │   └── preprocessor.py          # Cleaning, scaling, train/test split
│   ├── optimization/
│   │   ├── problem.py               # Objective function + QUBO formulation
│   │   ├── genetic_algorithm.py     # GA using DEAP (SBX + poly mutation)
│   │   ├── pso.py                   # PSO using PySwarms (Global Best)
│   │   ├── linear_programming.py    # LP/NLP using SciPy (L-BFGS-B + DE)
│   │   └── qaoa.py                  # QAOA using Qiskit Aer simulator
│   ├── models/
│   │   └── crop_yield_predictor.py  # Random Forest / Gradient Boosting
│   └── visualization/
│       └── plots.py                 # All output charts + results table
├── tests/
│   └── test_optimization.py         # Unit + integration tests
└── results/                         # Generated on first run
    ├── algorithm_comparison.png
    ├── resource_usage.png
    ├── convergence_curves.png
    ├── feature_importance.png
    ├── yield_distribution.png
    ├── climate_vs_yield.png
    └── algorithm_results.csv
```

---

## Datasets

All datasets are **synthetically generated** to mimic real-world publicly available sources:

| Dataset | Source Inspiration | Key Variables |
|---------|-------------------|---------------|
| Climate | NASA POWER | Temperature, Rainfall, Humidity, Solar Radiation |
| Soil | FAO / SoilGrids | Soil Nitrogen, pH, Moisture, Phosphorus, Potassium |
| Crop | Kaggle Crop Yield | Crop Type, Yield (t/ha), Resource Usage |

The yield model is physics-inspired, combining temperature stress, water stress, nitrogen availability, and soil pH into a realistic yield estimate.

---

## Optimization Problem

**Decision variables** (6 continuous):

| Variable | Range | Unit |
|----------|-------|------|
| Irrigation water | 400 – 2000 | L/ha |
| N Fertilizer | 10 – 120 | kg/ha |
| P Fertilizer | 5 – 60 | kg/ha |
| K Fertilizer | 10 – 80 | kg/ha |
| Energy | 50 – 400 | kWh/ha |
| Land area | 0.5 – 10 | ha |

**Objective**: Maximize crop yield (70% weight) while minimizing resource cost (30% weight).

---

## Algorithms

### Classical

| Algorithm | Library | Approach |
|-----------|---------|----------|
| Genetic Algorithm | DEAP | SBX crossover + polynomial mutation + tournament selection |
| Particle Swarm Optimization | PySwarms | Global Best PSO (Clerc-Kennedy constriction) |
| Linear / Nonlinear Programming | SciPy | L-BFGS-B multi-start + Differential Evolution |

### Quantum

| Algorithm | Library | Approach |
|-----------|---------|----------|
| QAOA | Qiskit + Qiskit Aer | QUBO → Ising Hamiltonian → Variational quantum circuit (p=2), COBYLA optimizer |

The 6-variable problem is encoded with 3 bits each → **18-qubit QAOA circuit** run on the Qiskit statevector simulator.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Full run (all 4 algorithms)
python main.py

# Skip QAOA (no Qiskit required)
python main.py --skip-qaoa

# Custom climate context
python main.py --temperature 36 --rainfall 50 --soil-nitrogen 40 --soil-ph 6.2

# Larger dataset
python main.py --n-samples 2000 --ga-generations 300

# All options
python main.py --help
```

---

## Example Output

```
======================================================================
ALGORITHM COMPARISON RESULTS
======================================================================
         algorithm  irrigation_water_l_ha  predicted_yield_t_ha  ...
Genetic Algorithm                 1247.3                   3.521
              PSO                 1183.6                   3.594
Linear Programming                1156.2                   3.612
              QAOA                 998.7                   3.784
======================================================================
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Workflow

```
Raw Datasets → Preprocessing → Feature Extraction
                                    ↓
                         Optimization Problem
                        ┌──────────┬──────────┬──────────┬───────┐
                        GA         PSO        LP         QAOA
                        └──────────┴──────────┴──────────┴───────┘
                                    ↓
                          Results Comparison
                          Charts + CSV Report
```

---

## Output Charts

| File | Description |
|------|-------------|
| `algorithm_comparison.png` | Bar chart: yield, cost, objective per algorithm |
| `resource_usage.png` | Grouped bars: water, fertilizer, energy, land |
| `convergence_curves.png` | Objective vs iteration for all algorithms |
| `feature_importance.png` | Top-15 features from yield prediction model |
| `yield_distribution.png` | Histogram + box plot by crop type |
| `climate_vs_yield.png` | Scatter plots of climate variables vs yield |
| `algorithm_results.csv` | Full numerical comparison table |
