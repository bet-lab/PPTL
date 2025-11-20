# Privacy-Preserving Transfer Learning

This repository contains the implementation code for Privacy-Preserving Transfer Learning research. The codebase follows a 5-step experimental workflow for training time series forecasting models with transfer learning capabilities.

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

The experiments use the Cambridge-Estates-Building-Energy-Archive dataset. Follow these steps to set up the dataset:

```bash
cd datasets
git clone https://github.com/EECi/Cambridge-Estates-Building-Energy-Archive.git

# Reset to the specific commit point where experiments were conducted
git reset --hard b2f5d4e
```

The dataset should be located at `datasets/Cambridge-Estates-Building-Energy-Archive/` relative to the repository root.

## Experimental Workflow

The experimental process consists of 5 sequential steps that must be executed in order:

### Step 1: Hyperparameter Tuning

**Script:** `scripts/tune_hyperparameter.py`

Performs hyperparameter optimization for the TiDE model using Optuna. This step searches for optimal hyperparameters including hidden size, number of encoder/decoder layers, dropout rate, learning rate, and other model configurations.

**Prerequisites:** Dataset must be downloaded and set up.

**Output:** SQLite database containing hyperparameter search results at `output/assets/tide-hypertune.db`.

### Step 2: Time Series Encoder Training

**Script:** `scripts/train_encoder.py`

Trains TS2Vec encoders for each building in the dataset. For each target building, an encoder is trained using data from all other buildings (excluding the target building itself). This creates building-specific encoders that learn representations of time series patterns.

**Prerequisites:** Dataset must be downloaded and set up.

**Output:** Encoder model weights saved at `output/assets/weights/encoder_b{bid}.pt` for each building ID.

### Step 3: Cosine Similarity Calculation

**Script:** `scripts/calculate_similarity.py`

Computes cosine similarities between building pairs using the trained encoders. For each target building, it calculates similarity scores with all other buildings based on their encoded representations. The similarities are used to select source buildings for transfer learning.

**Prerequisites:** 
- Step 2 must be completed (encoder weights must exist)

**Output:** JSON file containing similarity scores at `output/assets/similarities.json`.

### Step 4: TiDE Pre-training

**Script:** `scripts/train_tide.py`

Pre-trains TiDE (Time-series Dense Encoder) models using source building data selected based on similarity scores. The script supports three modes:
- `best`: Uses the N most similar buildings
- `worst`: Uses the N least similar buildings  
- `all`: Uses all available buildings

**Prerequisites:**
- Step 1 must be completed (hyperparameters should be tuned, though hardcoded values are used)
- Step 3 must be completed (similarity scores must exist)

**Output:** Pre-trained TiDE model weights at `output/assets/weights/tide_bid_{bid}_{mode}_{n_targets}.pt`.

### Step 5: Transfer Learning

**Script:** `scripts/transfer_tide.py`

Performs transfer learning by fine-tuning the pre-trained TiDE models on target building data. The script evaluates model performance on validation and test sets and stores results in a SQLite database.

**Prerequisites:**
- Step 4 must be completed (pre-trained TiDE models must exist)

**Output:** 
- Fine-tuned model checkpoints in `output/assets/tide_transfer/`
- Results stored in SQLite database at `output/assets/transfer_learning.db`

## Scripts Documentation

### 1. `scripts/tune_hyperparameter.py`

**Purpose:** Hyperparameter optimization for TiDE model using Optuna.

**Usage:**
```bash
cd scripts
python tune_hyperparameter.py <device_id>
```

**Arguments:**
- `device_id` (positional): GPU device ID to use (e.g., 0, 1)

**Hardcoded Paths:**
- Dataset: `../datasets/Cambridge-Estates-Building-Energy-Archive`
- Output database: `../output/assets/tide-hypertune.db`

**Data Ranges:**
- Training: `2009-01-01` to `2009-10-01`
- Validation: `2009-10-01` to `2010-01-01`

**Notes:**
- Uses all buildings in the dataset for hyperparameter search
- Performs 200 trials by default
- Results are stored in an Optuna study database

### 2. `scripts/train_encoder.py`

**Purpose:** Train TS2Vec encoders for each building.

**Usage:**
```bash
cd scripts
python train_encoder.py
```

**Arguments:** None (processes all buildings automatically)

**Hardcoded Paths:**
- Dataset: `../datasets/Cambridge-Estates-Building-Energy-Archive`
- Encoder weights output: `../output/assets/weights/encoder_b{bid}.pt`

**Model Configuration:**
- Input dimensions: Number of features (NONTARGETS + TARGETS)
- Hidden dimensions: 64
- Output dimensions: 320
- Max train length: 3000
- Training iterations: 200
- Device: CUDA

**Data Ranges:**
- Training data: `2009-01-01` to `2010-01-01` (excludes target building)

**Notes:**
- For each building, trains an encoder using data from all other buildings
- Uses fixed random seed (42) for reproducibility

### 3. `scripts/calculate_similarity.py`

**Purpose:** Calculate cosine similarities between building pairs.

**Usage:**
```bash
cd scripts
python calculate_similarity.py
```

**Arguments:** None (processes all buildings automatically)

**Hardcoded Paths:**
- Dataset: `../datasets/Cambridge-Estates-Building-Energy-Archive`
- Encoder weights input: `../output/assets/weights/encoder_b{bid}.pt`
- Similarity output: `../output/assets/similarities.json`

**Data Ranges:**
- Target building: `2010-01-01` to `2010-03-01`
- Source buildings: `2009-01-01` to `2009-03-01`

**Output Format:**
The output JSON file has the following structure:
```json
{
  "<building_id>": {
    "bid": [<list of building IDs sorted by similarity>],
    "similarity": [<list of similarity scores in ascending order>]
  }
}
```

**Notes:**
- Similarities are calculated using full series encoding
- Results are sorted in ascending order (most similar = lower cosine distance)

### 4. `scripts/train_tide.py`

**Purpose:** Pre-train TiDE models using source building data.

**Usage:**
```bash
cd scripts
python train_tide.py --bid <building_id> --mode <mode> --n-targets <n> --device <device_id>
```

**Arguments:**
- `--bid`: Target building ID (integer)
- `--mode`: Selection mode - `best`, `worst`, or `all` (string)
- `--n-targets`: Number of source buildings to use (integer, required for `best` and `worst` modes)
- `--device`: GPU device ID (integer, default: 0)

**Hardcoded Paths:**
- Dataset: `../datasets/Cambridge-Estates-Building-Energy-Archive`
- Similarity file input: `../output/assets/similarities.json`
- Model weights output: `../output/assets/weights/tide_bid_{bid}_{mode}_{n_targets}.pt`

**Data Ranges:**
- Source buildings training: `2009-01-01` to `2010-01-01`
- Target building validation: `2010-01-01` to `2010-03-01`

**Model Hyperparameters (hardcoded, from Step 1 results):**
- Input chunk length: 168 (7 days × 24 hours)
- Output chunk length: 24 (1 day)
- Batch size: 256
- Hidden size: 256
- Num encoder layers: 1
- Num decoder layers: 1
- Decoder output dim: 8
- Temporal decoder hidden: 32
- Dropout: 0.3981
- Learning rate: 0.00053954
- Use layer norm: False
- Use reversible instance norm: True

**Notes:**
- Early stopping with patience of 5 epochs
- Uses MLFlow logger for experiment tracking
- Validation uses target building data

### 5. `scripts/transfer_tide.py`

**Purpose:** Perform transfer learning by fine-tuning pre-trained TiDE models.

**Usage:**
```bash
cd scripts
python transfer_tide.py --bid <building_id> --mode <mode> --n-targets <n> --device <device_id>
```

**Arguments:**
- `--bid`: Target building ID (integer)
- `--mode`: Transfer learning mode - `best`, `worst`, `all`, or `none` (string)
- `--n-targets`: Number of source buildings used in pre-training (integer, default: 0)
- `--device`: GPU device ID (integer, default: 0)

**Hardcoded Paths:**
- Dataset: `../datasets/Cambridge-Estates-Building-Energy-Archive`
- Pre-trained model input: `../output/assets/weights/tide_bid_{bid}_{mode}_{n_targets}.pt`
- Checkpoint directory: `../output/assets/tide_transfer`
- Results database: `../output/assets/transfer_learning.db`

**Data Ranges:**
- Training: `2010-01-01` to `2010-03-01`
- Validation: `2010-03-01` to `2010-05-01`
- Test: `2010-03-01` to `2010-05-01`

**Transfer Learning Configuration:**
- For `none` mode: Learning rate scale = 1.0 (no pre-trained model loaded)
- For other modes: Learning rate scale = 0.1 (uses pre-trained model)
- Early stopping patience: 10 epochs

**Output:**
The script saves results to a SQLite database with the following schema:
- `id`: Auto-increment primary key
- `bid`: Building ID
- `mode`: Transfer learning mode
- `n_targets`: Number of source buildings
- `last_val_loss`: Last validation loss
- `best_val_loss`: Best validation loss
- `last_test_loss`: Last test loss (MSE)
- `best_test_loss`: Best test loss (MSE)
- `run_id`: MLFlow run ID

**Notes:**
- Evaluates both best and last checkpoints
- Uses historical forecasts with stride=1 for test evaluation
- Test evaluation uses MSE (Mean Squared Error)

## Hardcoded File Paths

All scripts use relative paths from the `scripts/` directory. The following table documents all hardcoded file paths:

| Path | Used In | Purpose |
|------|---------|---------|
| `../datasets/Cambridge-Estates-Building-Energy-Archive` | All scripts | Root directory of the dataset |
| `../output/assets/weights/encoder_b{bid}.pt` | `train_encoder.py`, `calculate_similarity.py` | Encoder model weights (input/output) |
| `../output/assets/similarities.json` | `calculate_similarity.py`, `train_tide.py` | Building similarity scores (input/output) |
| `../output/assets/weights/tide_bid_{bid}_{mode}_{n_targets}.pt` | `train_tide.py`, `transfer_tide.py` | Pre-trained TiDE model weights (input/output) |
| `../output/assets/tide-hypertune.db` | `tune_hyperparameter.py` | Optuna study database for hyperparameter tuning |
| `../output/assets/transfer_learning.db` | `transfer_tide.py` | SQLite database for transfer learning results |
| `../output/assets/tide_transfer/` | `transfer_tide.py` | Directory for TiDE transfer learning checkpoints |
| `../ts2vec` | `train_encoder.py`, `calculate_similarity.py` | TS2Vec library directory (added to Python path) |

**Important Notes:**
- All paths are relative to the `scripts/` directory
- The `output/assets/` directory structure must exist before running scripts
- Building IDs (`{bid}`) are integers (e.g., 0, 1, 2, ...)
- Mode values are strings: `best`, `worst`, `all`, or `none`
- `{n_targets}` is an integer representing the number of source buildings

## TS2Vec Library

The `ts2vec/` directory contains a modified version of the TS2Vec codebase from the official repository. 

**Source:** The code was copied from the official TS2Vec repository and only library version compatibility issues were resolved. No functional changes were made to the core TS2Vec implementation.

**Modifications:** Only dependency version updates to ensure compatibility with the rest of the codebase.

**Original Repository:** [TS2Vec Official Repository](https://github.com/yuezhihan/ts2vec)

## Output Structure

The `output/assets/` directory structure is as follows:

```
output/assets/
├── weights/
│   ├── encoder_b{bid}.pt          # TS2Vec encoder weights for each building
│   └── tide_bid_{bid}_{mode}_{n_targets}.pt  # Pre-trained TiDE model weights
├── tide_transfer/                 # Checkpoint directory for transfer learning
│   └── [fine-tuned checkpoint files]
├── similarities.json              # Building similarity scores
├── tide-hypertune.db              # Optuna hyperparameter tuning database
└── transfer_learning.db           # Transfer learning results database
```

## Usage Examples

### Complete Workflow Example

Execute the following commands in sequence from the repository root:

```bash
# Step 1: Hyperparameter tuning
cd scripts
python tune_hyperparameter.py 0

# Step 2: Train encoders for all buildings
python train_encoder.py

# Step 3: Calculate similarities
python calculate_similarity.py

# Step 4: Pre-train TiDE models (example for building 0 with best 2 sources)
python train_tide.py --bid 0 --mode best --n-targets 2 --device 0

# Step 5: Transfer learning (example for building 0)
python transfer_tide.py --bid 0 --mode best --n-targets 2 --device 0
```

### Using Shell Scripts

The repository includes shell scripts for batch processing:

**Pre-training scripts:**
- `scripts/train_tide_best.sh`: Pre-train TiDE models using best similar buildings for all buildings
- `scripts/train_tide_worst.sh`: Pre-train TiDE models using worst similar buildings for all buildings

**Transfer learning scripts:**
- `scripts/transfer_tide_best_worst.sh`: Perform transfer learning with both best and worst modes for all buildings
- `scripts/transfer_tide_none.sh`: Perform transfer learning without pre-training (baseline) for all buildings

Example usage:
```bash
cd scripts
bash train_tide_best.sh      # Pre-train all models with best sources
bash transfer_tide_best_worst.sh  # Transfer learning for all buildings
```

### Querying Results

To view transfer learning results from the database:

```bash
cd scripts
python -c "
import sqlite3
conn = sqlite3.connect('../output/assets/transfer_learning.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM transfer_learning LIMIT 10')
for row in cursor.fetchall():
    print(row)
"
```

## Repository Structure

```
PPTL/
├── datasets/                      # Dataset directory
│   └── Cambridge-Estates-Building-Energy-Archive/
├── scripts/                       # Main experiment scripts
│   ├── tune_hyperparameter.py
│   ├── train_encoder.py
│   ├── calculate_similarity.py
│   ├── train_tide.py
│   ├── transfer_tide.py
│   └── *.sh                       # Batch processing scripts
├── utils/                         # Utility functions
│   └── data.py                    # Data loading and preprocessing
├── ts2vec/                        # TS2Vec library (modified)
├── output/                        # Output directory (created during execution)
│   └── assets/
│       ├── weights/
│       ├── tide_transfer/
│       └── *.db, *.json
├── pyproject.toml                # Project dependencies
└── README.md                     # This file
```

## Notes

- All scripts should be executed from the `scripts/` directory
- GPU is required for training (CUDA device)
- The dataset must be properly set up before running any scripts
- Scripts use fixed random seeds for reproducibility
- MLFlow is used for experiment tracking (logs are saved automatically)
- Early stopping is configured in all training scripts to prevent overfitting
