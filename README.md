# Privacy-Preserving Transfer Learning (PPTL)

This repository contains the implementation code for the PPTL framework, a metadata-free transfer learning approach for building energy forecasting under full data anonymization.

## Overview

AI-driven energy forecasting for buildings faces a structural deadlock: data heterogeneity demands diverse training data from many buildings, yet privacy regulations prevent cross-building data aggregation. Conventional transfer learning relies on metadata (e.g., building type, climate zone) for source selection—information stripped away by anonymization.

The **PPTL framework** resolves this by learning similarity directly from anonymized time-series dynamics. Three modular components work in sequence:

1. **Encoder** ([TS2Vec](https://github.com/yuezhihan/ts2vec)): An unsupervised contrastive encoder maps multivariate time series into a representation space where buildings with similar operational patterns cluster naturally.
2. **Strategy controller**: Cosine distance in the learned space ranks candidate sources, enabling data-driven source selection without metadata.
3. **Forecaster** ([TiDE](https://arxiv.org/abs/2304.08424)): A lightweight MLP-based encoder–decoder is pretrained on the most similar sources and fine-tuned on the target building.

**Key results** on 89 real-world buildings:
- Median MSE reductions of **27–31%** over no-transfer baselines
- Improvements in **99.2%** of configurations (353/356)
- Only **0.51%** of the communication bandwidth compared to federated learning

## Citation

If you use this code, please cite:

```
W. Choi, S. Lee, M. Langtry, R. Choudhary, "Privacy-preserving transfer learning
framework for building energy forecasting with fully anonymized data,"
Applied Energy, 2026.
```

## Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Experimental Workflow](#experimental-workflow)
- [Scripts Documentation](#scripts-documentation)
- [Hardcoded File Paths](#hardcoded-file-paths)
- [TS2Vec Library](#ts2vec-library)
- [Output Structure](#output-structure)
- [Usage Examples](#usage-examples)

## Installation

### Prerequisites

- Python 3.10 (Python 3.11 is not supported)
- CUDA-compatible GPU (recommended)
- `uv` package manager

### Install Dependencies

```bash
uv sync
```

This will install all required dependencies including PyTorch, Darts, Optuna, and other necessary packages as specified in `pyproject.toml`.

## Dataset Setup

The experiments use the Cambridge University Estates Building Energy Archive dataset. Follow these steps to set up the dataset:

```bash
cd datasets
git clone https://github.com/EECi/Cambridge-Estates-Building-Energy-Archive.git

# Reset to the specific commit used in the paper's experiments
git reset --hard b2f5d4e
```

The dataset should be located at `datasets/Cambridge-Estates-Building-Energy-Archive/` relative to the repository root.

## Experimental Workflow

The PPTL framework follows a 4-step workflow, preceded by a one-time hyperparameter tuning step (Step 0). Steps must be executed in order:

### Step 0: Hyperparameter Tuning (One-time Prerequisite)

**Script:** `scripts/tune_hyperparameter.py`

Performs hyperparameter optimization for the TiDE forecaster using Optuna (400 trials with Tree-structured Parzen Estimator and Asynchronous Successive Halving pruning).

**Output:** SQLite database at `output/assets/tide-hypertune.db`.

### Step 1: Unsupervised Encoder Training

**Script:** `scripts/train_encoder.py`

Trains TS2Vec contrastive encoders for each target building. For each target, an encoder is trained on data from all 88 other buildings (leave-one-out), constructing the representation space used for similarity assessment.

**Output:** Encoder weights at `output/assets/weights/encoder_b{bid}.pt` for each building.

### Step 2: Similarity-Based Source Selection

**Script:** `scripts/calculate_similarity.py`

Generates representation vectors and computes cosine distances between the target building (Jan–Feb 2010) and each source building (Jan–Feb 2009). The 1-year temporal gap tests generalization robustness.

**Output:** JSON file at `output/assets/similarities.json`.

### Step 3: Forecaster Pretraining

**Script:** `scripts/train_tide.py`

Pretrains TiDE forecasters on source buildings selected by similarity ranking. Supports two source selection strategies (paper terminology in parentheses):
- `best` (**Closest**): Uses the $N_{S^*}$ most similar sources
- `worst` (**Farthest**): Uses the $N_{S^*}$ least similar sources
- `all`: Uses all 88 source buildings

The paper systematically tests $N_{S^*} \in \{2, 4, 8, 16\}$.

**Output:** Pretrained weights at `output/assets/weights/tide_bid_{bid}_{mode}_{n_sources}.pt`.

### Step 4: Fine-tuning and Evaluation

**Script:** `scripts/transfer_tide.py`

Fine-tunes the pretrained TiDE model on the target building's data (Jan–Feb 2010) and evaluates on the test period (Mar–Apr 2010).

- For transfer learning modes (`best`, `worst`, `all`): Learning rate is scaled to 1/10 of the pretraining rate
- For `none` mode (No-TL baseline): Learning rate is unscaled

**Output:** Results in SQLite database at `output/assets/transfer_learning.db`.

## Scripts Documentation

### 1. `scripts/tune_hyperparameter.py`

**Usage:**
```bash
python scripts/tune_hyperparameter.py <device_id>
```

**Arguments:**
- `device_id` (positional): GPU device ID (e.g., 0, 1)

**Data Ranges:**
- Training: `2009-01-01` to `2009-10-01`
- Validation: `2009-10-01` to `2010-01-01`

### 2. `scripts/train_encoder.py`

**Usage:**
```bash
python scripts/train_encoder.py
```

**Arguments:** None (processes all 89 buildings automatically)

**TS2Vec Configuration:**
- Hidden dimensions: 64
- Output dimensions: 320
- Max train length: 3000
- Training iterations: 200
- Batch size: 16

### 3. `scripts/calculate_similarity.py`

**Usage:**
```bash
python scripts/calculate_similarity.py
```

**Arguments:** None (processes all buildings automatically)

**Data Ranges:**
- Target building: `2010-01-01` to `2010-03-01`
- Source buildings: `2009-01-01` to `2009-03-01`

### 4. `scripts/train_tide.py`

**Usage:**
```bash
python scripts/train_tide.py --bid <building_id> --mode <mode> --n-sources <n> --device <device_id>
```

**Arguments:**
- `--bid`: Target building ID (integer)
- `--mode`: Source selection strategy — `best` (Closest), `worst` (Farthest), or `all`
- `--n-sources`: Number of source buildings $N_{S^*}$ to use (required for `best` and `worst`)
- `--device`: GPU device ID (default: 0)

**TiDE Hyperparameters (selected via Optuna):**
- Input chunk length: 168 (7 days × 24 hours)
- Output chunk length: 24 (1 day)
- Batch size: 256
- Hidden size: 256
- Encoder/Decoder layers: 1 / 1
- Decoder output dim: 8
- Temporal decoder hidden: 32
- Dropout: 0.3981
- Learning rate: 5.3954 × 10⁻⁴

### 5. `scripts/transfer_tide.py`

**Usage:**
```bash
python scripts/transfer_tide.py --bid <building_id> --mode <mode> --n-sources <n> --device <device_id>
```

**Arguments:**
- `--bid`: Target building ID (integer)
- `--mode`: Transfer mode — `best` (Closest), `worst` (Farthest), `all`, or `none` (No-TL baseline)
- `--n-sources`: Number of source buildings used in pretraining (default: 0)
- `--device`: GPU device ID (default: 0)

**Output database schema:**
- `bid`: Building ID
- `mode`: Transfer learning mode
- `n_sources`: Number of source buildings ($N_{S^*}$)
- `last_val_loss` / `best_val_loss`: Validation losses
- `last_test_loss` / `best_test_loss`: Test losses (MSE)
- `run_id`: MLFlow run ID

### 6. `scripts/visualize_forecast.py`

Visualize the forecast of a single fine-tuned TiDE checkpoint against the ground truth. **Requires Steps 1–4** to have been completed for the target building.

**Usage:**
```bash
uv run python scripts/visualize_forecast.py --bid <building_id> --mode <mode> [--n <n_sources>] [--output <path>]
```

**Arguments:**
- `--bid`: Target building ID (integer)
- `--mode`: Transfer mode — `best` (Closest), `worst` (Farthest), or `none` (No-TL)
- `--n`: Number of source buildings (default: 0, ignored if mode=none)
- `--output`: Custom output PNG path (optional)

**Prerequisites:**
- Fine-tuned checkpoint must exist in `output/assets/tide_transfer/`
- Dataset must be available


## Hardcoded File Paths

All scripts resolve paths relative to the script file location. The following table documents key paths:

| Path | Used In | Purpose |
|------|---------|---------|
| `../datasets/Cambridge-Estates-Building-Energy-Archive` | All scripts | Dataset root |
| `../output/assets/weights/encoder_b{bid}.pt` | `train_encoder.py`, `calculate_similarity.py` | Encoder weights |
| `../output/assets/similarities.json` | `calculate_similarity.py`, `train_tide.py` | Similarity scores |
| `../output/assets/weights/tide_bid_{bid}_{mode}_{n_sources}.pt` | `train_tide.py`, `transfer_tide.py` | Pretrained TiDE weights |
| `../output/assets/tide-hypertune.db` | `tune_hyperparameter.py` | Optuna study database |
| `../output/assets/transfer_learning.db` | `transfer_tide.py` | Transfer learning results |
| `../output/assets/tide_transfer/` | `transfer_tide.py` | Fine-tuning checkpoints |
| `../ts2vec` | `train_encoder.py`, `calculate_similarity.py` | TS2Vec library |

**Note:** Output directories are created automatically when scripts are executed.

## TS2Vec Library

The `ts2vec/` directory contains a modified version of the TS2Vec codebase from the [official repository](https://github.com/yuezhihan/ts2vec). Only library version compatibility issues were resolved; no functional changes were made.

## Output Structure

```
output/assets/
├── weights/
│   ├── encoder_b{bid}.pt                          # TS2Vec encoder weights
│   └── tide_bid_{bid}_{mode}_{n_sources}.pt       # Pretrained TiDE weights
├── tide_transfer/                                  # Fine-tuning checkpoints
├── similarities.json                               # Building similarity scores
├── tide-hypertune.db                               # Optuna study database
├── transfer_learning.db                            # Transfer learning results
└── forecast_b{bid}_*.png                           # Single-model forecast plots
```

## Usage Examples

### Complete Workflow

```bash
# Step 0: Hyperparameter tuning (one-time)
python scripts/tune_hyperparameter.py 0

# Step 1: Train encoders for all buildings
python scripts/train_encoder.py

# Step 2: Calculate similarities
python scripts/calculate_similarity.py

# Step 3: Pretrain TiDE (example: building 0, Closest 4 sources)
python scripts/train_tide.py --bid 0 --mode best --n-sources 4 --device 0

# Step 4: Fine-tune and evaluate
python scripts/transfer_tide.py --bid 0 --mode best --n-sources 4 --device 0

# No-TL baseline comparison
python scripts/transfer_tide.py --bid 0 --mode none --device 0

# Visualize a single model's forecast
uv run python scripts/visualize_forecast.py --bid 0 --mode best --n 4
```

### Batch Processing (Shell Scripts)

```bash
# Pretrain all buildings with Closest / Farthest sources
bash scripts/train_tide_best.sh
bash scripts/train_tide_worst.sh

# Transfer learning for all buildings
bash scripts/transfer_tide_best_worst.sh  # Closest + Farthest
bash scripts/transfer_tide_none.sh        # No-TL baselines
```

### Querying Results

```python
import sqlite3
conn = sqlite3.connect('output/assets/transfer_learning.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM transfer_learning LIMIT 10')
for row in cursor.fetchall():
    print(row)
```

## Repository Structure

```
PPTL/
├── datasets/                      # Dataset directory
│   └── Cambridge-Estates-Building-Energy-Archive/
├── scripts/                       # Main experiment scripts
│   ├── tune_hyperparameter.py     # Step 0: Hyperparameter tuning
│   ├── train_encoder.py           # Step 1: TS2Vec encoder training
│   ├── calculate_similarity.py    # Step 2: Cosine similarity calculation
│   ├── train_tide.py              # Step 3: TiDE pretraining
│   ├── transfer_tide.py           # Step 4: Fine-tuning and evaluation
│   ├── visualize_forecast.py      # Single-model forecast visualization
│   └── *.sh                       # Batch processing scripts
├── utils/                         # Utility functions
│   └── data.py                    # Data loading and preprocessing
├── ts2vec/                        # TS2Vec library (modified)
├── output/                        # Output directory (auto-created)
│   └── assets/
├── pyproject.toml                 # Project dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Notes

- Scripts can be executed from any directory (paths are resolved relative to the script file)
- GPU is required for training (CUDA device)
- The dataset must be properly set up before running any scripts
- Scripts use fixed random seeds for reproducibility
- MLFlow is used for experiment tracking
- Early stopping is configured in all training scripts to prevent overfitting

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
