# How to Run — CC-DLM-MFisheries

This guide explains, step by step, how to set up the environment, prepare the data, train the Mixture-of-Experts (MoE) model, make catch predictions, and run the downstream analyses.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Project Structure](#3-project-structure)
4. [Required Data Files](#4-required-data-files)
5. [Step 1 – Prepare the Historical Data](#5-step-1--prepare-the-historical-data)
6. [Step 2 – Prepare Future Temperature Data](#6-step-2--prepare-future-temperature-data)
7. [Step 3 – Train the MoE Model](#7-step-3--train-the-moe-model)
8. [Step 4 – Forecast Catch for a Single Species](#8-step-4--forecast-catch-for-a-single-species)
9. [Step 5 – Batch Forecasting (all species / clusters)](#9-step-5--batch-forecasting-all-species--clusters)
10. [Step 6 – SHAP Explainability Analysis](#10-step-6--shap-explainability-analysis)
11. [Step 7 – Temperature Sensitivity Analysis](#11-step-7--temperature-sensitivity-analysis)
12. [Step 8 – Economic Analysis](#12-step-8--economic-analysis)
13. [Common Issues & Troubleshooting](#13-common-issues--troubleshooting)

---

## 1. Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.11 |
| TensorFlow / Keras | 2.x |
| pandas | 1.5+ |
| numpy | 1.24+ |
| scikit-learn | 1.2+ |
| matplotlib | 3.7+ |
| seaborn | 0.12+ |
| shap | 0.42+ |
| joblib | 1.3+ |
| jupyter | 1.0+ |

A GPU is optional but strongly recommended for training. Training on CPU is supported but will be significantly slower.

---

## 2. Installation

### 2.1 Clone the repository

```bash
git clone https://github.com/cedointercultural/CC-DLM-MFisheries.git
cd CC-DLM-MFisheries
```

### 2.2 Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 2.3 Install dependencies

```bash
pip install -r requeriments.txt
```

> **Note:** the dependency file is named `requeriments.txt` (note the non-standard spelling in the repository). If any packages are missing, install them individually, e.g. `pip install shap joblib`.

### 2.4 (Optional) Register the environment as a Jupyter kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name cc-dlm --display-name "CC-DLM-MFisheries"
jupyter notebook
```

---

## 3. Project Structure

```
CC-DLM-MFisheries/
├── data/
│   ├── data.csv                   # Main merged historical dataset (you must supply this)
│   ├── data_merge.py              # Script to merge raw fisheries CSVs
│   ├── futuretemp/                # Raw future-temperature CSVs per cluster / year
│   └── futuretemp1/               # Alternative future-temperature scenario CSVs
├── scripts/
│   ├── species_forecaster.py      # Core forecasting class (SpeciesForecaster)
│   ├── prepare_future_data.py     # Helper to reformat future-temperature DataFrames
│   ├── shap_analysis.py           # SHAP analysis functions
│   ├── sensitivity_analysis.py    # Temperature sensitivity sweep
│   ├── merge_results.py           # Merge per-species results into a single CSV
│   └── species_classifier.py      # Classify species by habitat group
├── modelos_moe/                   # Trained models (*.h5) and scalers (*.pkl) — generated
├── resultados_moe/                # Prediction outputs — generated
├── train_moe/                     # Training history CSVs — generated
├── Prepare_data.ipynb             # Step 1: Data preparation
├── notebook5.ipynb                # Step 3: MoE model training (main training notebook)
├── Forecast_one_species.ipynb     # Step 4: Single-species forecasting
├── INTEGRATE.ipynb                # Batch forecasting across all species
├── SHAP2.ipynb                    # Step 6: SHAP analysis
├── SHAP_Analysis.ipynb            # Alternative SHAP notebook
├── analisis_económico.ipynb       # Step 8: Economic analysis
├── diagnostic.ipynb               # Model diagnostics
├── forecast_sp.py                 # CLI wrapper for SpeciesForecaster
├── shap_analysis_script.py        # CLI wrapper for SHAP analysis
├── requeriments.txt               # Python dependencies
└── HOW_TO_RUN.md                  # This file
```

---

## 4. Required Data Files

Before running any notebooks or scripts, ensure the following files exist:

| File | Description |
|---|---|
| `data/data.csv` | Main dataset: historical landings merged with oceanographic variables. Must contain the columns listed below. |
| `data/futuretemp/*.csv` | Per-cluster future ocean-temperature projections. Column names: `year`, `month`, `cluster`, `depth_m`, `thetao`. |
| `future_temp.csv` | Processed future-temperature file (generated in Step 2). |

### Required columns in `data/data.csv`

```
year, month_no, species, Cluster_Label, landed_w_kg,
mean_temp_30m, mean_temp_10m,
thetao_sfc=6, thetao_sfc=7.92956018447876, thetao_sfc=9.572997093200684,
thetao_sfc=11.40499973297119, thetao_sfc=13.46714019775391,
thetao_sfc=15.8100700378418, thetao_sfc=18.49555969238281,
thetao_sfc=21.59881973266602, thetao_sfc=25.21141052246094,
thetao_sfc=29.44473075866699
```

---

## 5. Step 1 – Prepare the Historical Data

Open and run **`Prepare_data.ipynb`** cell by cell.

This notebook:
- Reads raw temperature CSV files from `data/futuretemp/`
- Pivots long-format depth/temperature rows into wide-format columns (`depth_<value>`)
- Saves transformed files under `data/forecast_data/`

```bash
jupyter notebook Prepare_data.ipynb
```

> After running this notebook, verify that files named `transformed_*.csv` exist in `data/forecast_data/`.

---

## 6. Step 2 – Prepare Future Temperature Data

Run the merge script to combine all transformed future-temperature CSVs into a single file:

```bash
cd data
python data_merge.py
cd ..
```

This creates `data/future_data.csv`.  
Then rename / copy it so the forecasting notebooks can find it:

```bash
cp data/future_data.csv future_temp.csv
```

The expected columns in `future_temp.csv` after renaming are:

```
year, month, Cluster_Label,
mean_temp_30m, mean_temp_10m,
thetao_sfc=6, thetao_sfc=7.92956018447876, ...
```

---

## 7. Step 3 – Train the MoE Model

Open **`notebook5.ipynb`** — this is the primary training notebook for the Mixture-of-Experts (MoE) architecture.

```bash
jupyter notebook notebook5.ipynb
```

### What the notebook does

1. Loads `data/data.csv` and drops rows with missing `Cluster_Label`.
2. Constructs a `date` column from `year` + `month_no`.
3. Selects the 14 model features (landed weight, cluster label, depth-temperature columns).
4. For each `(species, Cluster_Label)` pair that has at least 57 distinct months of data, it:
   - Scales features with `MinMaxScaler`.
   - Builds a **Mixture-of-Experts** model with three experts:
     - **LSTM** — captures temporal patterns.
     - **CNN (Conv1D)** — captures local patterns in the time window.
     - **MLP (Dense)** — captures non-linear relationships.
   - A gating network (Dense → Softmax) assigns a weight to each expert's output.
   - Trains with early stopping (patience = 10, max 50 epochs, batch size = 1).
5. Saves each trained model and its scaler:
   - `modelos_moe/<SPECIES>_cluster_<N>_moe_model.h5`
   - `modelos_moe/<SPECIES>_cluster_<N>_moe_scaler.pkl`
6. Saves training history to `train_moe/<SPECIES>_cluster_<N>_training_history.csv`.

### Training a single species/cluster from the command line

```python
# At the top of a Python script or interactive session:
import pandas as pd, numpy as np, os, joblib
from keras.models import Model
# ... (copy the helper functions from notebook5.ipynb)

data = pd.read_csv('data/data.csv')
data = data.dropna(subset=['Cluster_Label'])
data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month_no'].astype(str))

species_name   = 'BANDERA'   # change as needed
cluster_label  = 1           # change as needed

group = data[
    (data['species'] == species_name) &
    (data['Cluster_Label'] == cluster_label)
].sort_values('date')

model, scaler = train_and_save_moe_model(group, species_name, cluster_label)
```

---

## 8. Step 4 – Forecast Catch for a Single Species

Once a model has been trained, open **`Forecast_one_species.ipynb`**:

```bash
jupyter notebook Forecast_one_species.ipynb
```

Set the target species and cluster at the top of the notebook:

```python
specie        = 'BANDERA'   # must match the saved model name
cluster_label = 1
```

The notebook will:
1. Load the corresponding model (`modelos_moe/BANDERA_cluster_1_moe_model.h5`) and scaler.
2. Filter historical data for that species/cluster.
3. Load `future_temp.csv` and filter by cluster.
4. Iteratively predict monthly catch values, feeding each prediction back as input.
5. Compute bootstrap confidence intervals (99% by default).
6. Plot the forecast against historical observations.

### Using the `SpeciesForecaster` class directly

```python
from pathlib import Path
from scripts.species_forecaster import ForecastConfig, SpeciesForecaster
import pandas as pd

config       = ForecastConfig()          # default look_back=6
SPECIES      = 'BANDERA'
CLUSTER      = 1

forecaster = SpeciesForecaster(
    model_path  = Path(f'modelos_moe/{SPECIES}_cluster_{CLUSTER}_moe_model.h5'),
    scaler_path = Path(f'modelos_moe/{SPECIES}_cluster_{CLUSTER}_moe_scaler.pkl'),
    config      = config
)

data          = pd.read_csv('data/data.csv', low_memory=False)
filtered_data = forecaster.filter_data(data, SPECIES, CLUSTER)

future_temps  = pd.read_csv('future_temp.csv')
future_temps  = future_temps[future_temps['Cluster_Label'] == CLUSTER].copy()
future_temps['date'] = pd.to_datetime(
    future_temps['year'].astype(str) + '-' + future_temps['month'].astype(str)
)
future_temps.sort_values('date', inplace=True)

predictions, lower_bound, upper_bound = forecaster.make_predictions(filtered_data, future_temps)
print("Predictions:", predictions)
```

Or via the CLI wrapper:

```bash
python forecast_sp.py
```

---

## 9. Step 5 – Batch Forecasting (all species / clusters)

Open **`INTEGRATE.ipynb`** to run predictions for every trained model automatically.

```bash
jupyter notebook INTEGRATE.ipynb
```

Individual result CSVs are written to `resultados_moe/`. To combine them into a single file:

```bash
python scripts/merge_results.py
```

> **Note:** `merge_results.py` contains hardcoded Windows paths. Edit the `input_folder` and `output_file` variables to match your local directory before running.

---

## 10. Step 6 – SHAP Explainability Analysis

Open **`SHAP2.ipynb`** (or `SHAP_Analysis.ipynb`):

```bash
jupyter notebook SHAP2.ipynb
```

Or run the standalone script:

```bash
python shap_analysis_script.py
```

The script calls `scripts/shap_analysis.py::evaluate_all_models`, which:
- Iterates over all `*.h5` files in `modelos_moe/`
- Computes SHAP values with a `GradientExplainer`
- Appends results to `shap_analysis_results.csv`

To run SHAP analysis for a single model programmatically:

```python
from scripts.shap_analysis import run_shap_analysis, prepare_future_data
import pandas as pd

future_data = pd.read_csv('future_temp.csv')
prepared_data, _ = prepare_future_data(future_data.copy(), cluster=1)

result = run_shap_analysis(
    species     = 'BANDERA',
    cluster     = 1,
    prepared_data = prepared_data,
    model_path  = 'modelos_moe/BANDERA_cluster_1_moe_model.h5',
    scaler_path = 'modelos_moe/BANDERA_cluster_1_moe_scaler.pkl'
)
print(result)
```

---

## 11. Step 7 – Temperature Sensitivity Analysis

```bash
python scripts/sensitivity_analysis.py
```

This sweeps ocean temperature from 15 °C to 32 °C in 1 °C steps for every trained model and saves:
- `sensitivity_analysis/<SPECIES>_cluster_<N>.csv` — numerical results
- `sensitivity_analysis/<SPECIES>_cluster_<N>.png` — plots

To analyse a single species/cluster:

```python
from scripts.sensitivity_analysis import run_analysis
results = run_analysis(species='BANDERA', cluster=1)
```

---

## 12. Step 8 – Economic Analysis

Open **`analisis_económico.ipynb`**:

```bash
jupyter notebook analisis_económico.ipynb
```

This notebook reads the merged prediction results and applies price-per-kg data to produce economic projections by functional group (e.g. demersal, pelagic, invertebrate).

---

## 13. Common Issues & Troubleshooting

### `KeyError: "['mean_temp_30m', ...] not in index"`

The raw `data.csv` file may use older column names (e.g. `0.49402499_m` instead of `thetao_sfc=6`). Run `Prepare_data.ipynb` to produce the updated column names. The correct names expected by the MoE model are:

```
mean_temp_30m, mean_temp_10m,
thetao_sfc=6, thetao_sfc=7.92956018447876, ...
```

### `OSError: No such file or directory: 'modelos_moe/..._moe_model.h5'`

The model has not been trained yet. Complete **Step 3** first.

### `FileNotFoundError: future_temp.csv`

Run **Step 2** to generate `future_temp.csv` from the raw future-temperature CSVs.

### TensorFlow GPU warnings on startup

The `E tensorflow ... Unable to register cuDNN factory` messages are harmless and appear when CUDA drivers are absent. Training will fall back to CPU automatically.

### `UserWarning: X does not have valid feature names, but MinMaxScaler was fitted with feature names`

This is a scikit-learn warning that occurs when a NumPy array (without column names) is passed to a scaler that was fitted on a DataFrame. It does not affect correctness and can be suppressed with:

```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

### `merge_results.py` fails with a Windows path error

Edit the two path variables at the top of `scripts/merge_results.py`:

```python
input_folder = 'resultados_moe'          # relative path
output_file  = 'resultado_unificado.csv' # relative path
```

### Memory errors during training

Reduce `batch_size` (default is 1) or `epochs` in `train_and_save_moe_model()`, or process fewer species/cluster pairs at a time.
