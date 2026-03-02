# Privacy-Preserving Transfer Learning (PPTL)

> **Privacy-Preserving Transfer Learning Framework for Building Energy Forecasting with Fully Anonymized Data**
>
> Wonjun Choi · Sangwon Lee · Max Langtry · Ruchi Choudhary
>
> _Applied Energy_, 2026

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.apenergy.2026.127600-blue)](https://doi.org/10.1016/j.apenergy.2026.127600)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet.svg)](https://github.com/astral-sh/uv)

---

## 📖 Abstract

AI-driven forecasting offers a promising solution for optimal building energy control, yet is constrained by scarce labeled data and strict privacy regulations. While transfer learning can alleviate data scarcity by leveraging data from other buildings, conventional approaches rely on metadata — such as building type, climate zone, or occupancy schedules — that is unavailable in fully anonymized datasets.

The PPTL framework resolves this deadlock by learning similarity directly from anonymized time-series dynamics. Using an unsupervised contrastive encoder, the framework maps each building's dynamics to high-dimensional representation vectors learned solely from temporal patterns. Cosine distance between representations guides source selection to pretrain a lightweight forecaster, which is then fine-tuned on limited target data. Leave-one-out experiments on 89 real-world buildings validate that learned similarity strongly correlates with transfer performance.

---

## ✨ Key Results

| Metric                                                | Value             |
| :---------------------------------------------------- | :---------------- |
| Median MSE reduction vs. No-TL baseline               | 27–31%            |
| Configurations improved over No-TL baseline           | 99.2% (353 / 356) |
| Maximum degradation vs. No-TL baseline (only 3 cases) | 2.2%              |
| Communication bandwidth vs. federated learning        | 0.51%             |

---

## 🏗️ Framework Architecture

Three modular components work in sequence to enable metadata-free transfer learning:

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│   🔍 Encoder    │ ──▶ │ 🎯 Strategy          │ ──▶ │  📈 Forecaster   │
│   (TS2Vec)      │     │    Controller         │     │  (TiDE)          │
│                 │     │                      │     │                  │
│ Unsupervised    │     │ Cosine distance in   │     │ Lightweight      │
│ contrastive     │     │ the learned space    │     │ MLP-based        │
│ encoder maps    │     │ ranks candidate      │     │ encoder–decoder  │
│ multivariate    │     │ sources → data-      │     │ pretrained on    │
│ time series to  │     │ driven selection     │     │ selected sources │
│ a representation│     │ without metadata     │     │ and fine-tuned   │
│ space           │     │                      │     │ on target        │
└─────────────────┘     └──────────────────────┘     └──────────────────┘
```

- **Encoder** — [TS2Vec](https://github.com/yuezhihan/ts2vec): An unsupervised contrastive learning encoder that captures intrinsic temporal dynamics — diurnal cycling, seasonal periodicity, and load-shape dynamics — by enforcing _contextual consistency_. Rather than relying on data augmentation (which can distort physical signatures), TS2Vec augments the temporal context via timestamp masking, forcing the model to learn robust operational patterns.

- **Strategy Controller**: Generates representation vectors for all buildings, then ranks sources by cosine distance to the target. Cosine distance is chosen for three reasons: (1) mathematical consistency with the encoder's contrastive loss, (2) alignment with percentile-based data normalization, and (3) computational efficiency ($O(1)$ per pair vs. DTW's $O(L^2)$).

- **Forecaster** — [TiDE](https://arxiv.org/abs/2304.08424): A Time-series Dense Encoder with $O(L)$ linear scaling and full parallel computation. TiDE integrates residual connections and covariate projections to capture both linear trends and complex nonlinear dependencies, while offering significant speed advantages over sequential architectures like RNNs and lower complexity than Transformers' $O(L^2)$.

---

## 🔬 Contributions

1. **Metadata-free transfer learning framework** — Enables effective transfer learning using exclusively anonymized time-series data, establishing a data-native methodology that bypasses reliance on metadata.

2. **Representation distance as a transferability proxy** — Establishes that cosine distance in the learned representation space serves as an objective proxy for transfer success, replacing heuristic-based judgments with data-driven verification.

3. **Negative transfer as a manageable engineering risk** — Characterizes the trade-off between source quantity and similarity, identifying a distinct performance sweet spot and transforming negative transfer from an unpredictable risk into a systematic engineering decision.

4. **Scalable deployment complementing federated learning** — Requires only 0.51% of the communication bandwidth compared to federated learning while offloading all computation to the server, enabling deployment on legacy building systems.

---

## 🆚 Comparison with Federated Learning

| Dimension                 | Federated Learning                                             | PPTL                                                        |
| :------------------------ | :------------------------------------------------------------- | :---------------------------------------------------------- |
| **Privacy approach**      | Structural locality (raw data stays on client)                 | Regulatory compliance (identifiers stripped before pooling) |
| **Communication**         | High — continuous sync over many rounds (~608 MB)              | Minimal — single upload/download cycle (~3.1 MB)            |
| **Client computation**    | Heavy — iterative local gradient computation (GPU required)    | Negligible — all training offloaded to server               |
| **Non-IID robustness**    | Vulnerable — performance degrades with divergent distributions | Robust by design — automatically selects similar sources    |
| **Model personalization** | Generic global model (averaged behavior)                       | Target-specific model (fine-tuned per building)             |
| **Scalability**           | Bottlenecked by edge network reliability                       | Bounded by server storage/compute                           |

> FL and PPTL are complementary, not competing. PPTL's similarity-based clustering can enhance FL by grouping clients into operationally compatible cohorts, directly addressing FL's non-IID vulnerability.

---

## 📊 Dataset

The experiments use the [Cambridge University Estates Building Energy Archive](https://github.com/EECi/Cambridge-Estates-Building-Energy-Archive) — a fully anonymized dataset spanning 24 years (2000–2023) of hourly electricity usage, weather observations, and metadata for ~120 buildings at the University of Cambridge. Due to privacy, all buildings are identified only by randomized numerical indices with no metadata.

A 16-month interval `[2009-01-01, 2010-05-01)` was curated to maximize gap-free coverage, yielding 89 buildings:

| Period       | Role                                   | Duration  |
| :----------- | :------------------------------------- | :-------- |
| Jan–Dec 2009 | Source data (pretraining)              | 12 months |
| Jan–Feb 2010 | Target data (fine-tuning & similarity) | 2 months  |
| Mar–Apr 2010 | Test data (evaluation)                 | 2 months  |

Features: 10 nontarget covariates (cyclical time encodings, weather variables) + 1 target feature (hourly electricity usage normalized via percentile-based transform).

---

## 🧪 Experimental Workflow

The PPTL framework follows a 4-step sequential pipeline, preceded by a one-time hyperparameter tuning step. Each step must be executed in order.

### Step 0 · Hyperparameter Tuning _(one-time prerequisite)_

**Script:** `scripts/tune_hyperparameter.py`

Performs hyperparameter optimization for the TiDE forecaster using Optuna (400 trials with Tree-structured Parzen Estimator and Asynchronous Successive Halving pruning).

```bash
uv run python scripts/tune_hyperparameter.py <device_id>
```

**Output:** `output/assets/tide-hypertune.db`

---

### Step 1 · Unsupervised Encoder Training

**Script:** `scripts/train_encoder.py`

Trains TS2Vec contrastive encoders for each target building. For each target, an encoder is trained on data from all 88 other buildings (leave-one-out), constructing the representation space used for similarity assessment.

```bash
uv run python scripts/train_encoder.py
```

| Parameter           | Value |
| :------------------ | :---- |
| Hidden dimensions   | 64    |
| Output dimensions   | 320   |
| Max train length    | 3000  |
| Training iterations | 200   |
| Batch size          | 16    |

**Output:** `output/assets/weights/encoder_b{bid}.pt`

---

### Step 2 · Similarity-Based Source Selection

**Script:** `scripts/calculate_similarity.py`

Generates representation vectors and computes cosine distances between each target (Jan–Feb 2010) and source (Jan–Feb 2009). The 1-year temporal gap tests generalization robustness, ensuring the learned similarity is not merely a reflection of contemporaneous patterns.

```bash
uv run python scripts/calculate_similarity.py
```

**Output:** `output/assets/similarities.json`

---

### Step 3 · Forecaster Pretraining

**Script:** `scripts/train_tide.py`

Pretrains TiDE forecasters on source buildings selected by similarity ranking.

```bash
uv run python scripts/train_tide.py --bid <building_id> --mode <mode> --n-sources <n> --device <device_id>
```

**Source selection strategies** (paper terminology in parentheses):

- `best` (**Closest**) — Top $N_{S^*}$ most similar sources
- `worst` (**Farthest**) — Bottom $N_{S^*}$ least similar sources
- `all` — All 88 source buildings

The paper systematically tests $N_{S^*} \in \{2, 4, 8, 16\}$.

<details>
<summary><strong>TiDE Hyperparameters (selected via Optuna)</strong></summary>

| Parameter                | Value                   |
| :----------------------- | :---------------------- |
| Input chunk length       | 168 (7 days × 24 hours) |
| Output chunk length      | 24 (1 day)              |
| Batch size               | 256                     |
| Hidden size              | 256                     |
| Encoder / Decoder layers | 1 / 1                   |
| Decoder output dim       | 8                       |
| Temporal decoder hidden  | 32                      |
| Dropout                  | 0.3981                  |
| Learning rate            | 5.3954 × 10⁻⁴           |

</details>

**Output:** `output/assets/weights/tide_bid_{bid}_{mode}_{n_sources}.pt`

---

### Step 4 · Fine-tuning and Evaluation

**Script:** `scripts/transfer_tide.py`

Fine-tunes the pretrained TiDE model on the target building's data (Jan–Feb 2010) and evaluates on the test period (Mar–Apr 2010).

```bash
uv run python scripts/transfer_tide.py --bid <building_id> --mode <mode> --n-sources <n> --device <device_id>
```

- Transfer modes (`best`, `worst`, `all`): Learning rate scaled to 1/10 of the pretraining rate
- No-TL baseline (`none`): Learning rate unscaled

<details>
<summary><strong>Output Database Schema</strong></summary>

| Column                              | Description                            |
| :---------------------------------- | :------------------------------------- |
| `bid`                               | Building ID                            |
| `mode`                              | Transfer learning mode                 |
| `n_sources`                         | Number of source buildings ($N_{S^*}$) |
| `last_val_loss` / `best_val_loss`   | Validation losses                      |
| `last_test_loss` / `best_test_loss` | Test losses (MSE)                      |
| `run_id`                            | MLFlow run ID                          |

</details>

**Output:** `output/assets/transfer_learning.db`

---

### Visualization

**Script:** `scripts/visualize_forecast.py`

Visualize the forecast of a single fine-tuned TiDE checkpoint against the ground truth. Requires Steps 1–4 to have been completed for the target building.

```bash
uv run python scripts/visualize_forecast.py --bid <building_id> --mode <mode> [--n <n_sources>] [--output <path>]
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 (Python 3.11 is not supported)
- CUDA-compatible GPU (recommended)
- [`uv`](https://github.com/astral-sh/uv) package manager

### Installation

```bash
uv sync
```

### Dataset Setup

```bash
cd datasets
git clone https://github.com/EECi/Cambridge-Estates-Building-Energy-Archive.git
cd Cambridge-Estates-Building-Energy-Archive

# Reset to the specific commit used in the paper
git reset --hard b2f5d4e
```

### Complete Workflow Example

```bash
# Step 0: Hyperparameter tuning (one-time)
uv run python scripts/tune_hyperparameter.py 0

# Step 1: Train encoders for all buildings
uv run python scripts/train_encoder.py

# Step 2: Calculate similarities
uv run python scripts/calculate_similarity.py

# Step 3: Pretrain TiDE (example: building 0, Closest 4 sources)
uv run python scripts/train_tide.py --bid 0 --mode best --n-sources 4 --device 0

# Step 4: Fine-tune and evaluate
uv run python scripts/transfer_tide.py --bid 0 --mode best --n-sources 4 --device 0

# No-TL baseline comparison
uv run python scripts/transfer_tide.py --bid 0 --mode none --device 0

# Visualize a single model's forecast
uv run python scripts/visualize_forecast.py --bid 0 --mode best --n 4
```

### Batch Processing

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

conn = sqlite3.connect("output/assets/transfer_learning.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM transfer_learning LIMIT 10")
for row in cursor.fetchall():
    print(row)
```

---

## 📁 Repository Structure

```
PPTL_codes/
├── scripts/                       # Main experiment scripts
│   ├── tune_hyperparameter.py     # Step 0: Hyperparameter tuning
│   ├── train_encoder.py           # Step 1: TS2Vec encoder training
│   ├── calculate_similarity.py    # Step 2: Cosine similarity calculation
│   ├── train_tide.py              # Step 3: TiDE pretraining
│   ├── transfer_tide.py           # Step 4: Fine-tuning & evaluation
│   ├── visualize_forecast.py      # Forecast visualization
│   └── *.sh                       # Batch processing shell scripts
├── utils/                         # Utility functions
│   └── data.py                    # Data loading & preprocessing
├── ts2vec/                        # TS2Vec library (modified for compatibility)
├── datasets/                      # Dataset directory
│   └── Cambridge-Estates-.../     #   └─ Cloned dataset repository
├── output/                        # Output directory (auto-created)
│   └── assets/
│       ├── weights/               #   ├─ Encoder & TiDE weights
│       ├── tide_transfer/         #   ├─ Fine-tuning checkpoints
│       ├── similarities.json      #   ├─ Building similarity scores
│       ├── tide-hypertune.db      #   ├─ Optuna study database
│       ├── transfer_learning.db   #   └─ Transfer learning results
│       └── forecast_b{bid}_*.png  #       Forecast visualization plots
├── pyproject.toml                 # Project dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## 🔧 Hardcoded File Paths

All scripts resolve paths relative to the script file location. Key paths:

| Path                                                            | Used In                                       | Purpose                   |
| :-------------------------------------------------------------- | :-------------------------------------------- | :------------------------ |
| `../datasets/Cambridge-Estates-Building-Energy-Archive`         | All scripts                                   | Dataset root              |
| `../output/assets/weights/encoder_b{bid}.pt`                    | `train_encoder.py`, `calculate_similarity.py` | Encoder weights           |
| `../output/assets/similarities.json`                            | `calculate_similarity.py`, `train_tide.py`    | Similarity scores         |
| `../output/assets/weights/tide_bid_{bid}_{mode}_{n_sources}.pt` | `train_tide.py`, `transfer_tide.py`           | Pretrained TiDE weights   |
| `../output/assets/tide-hypertune.db`                            | `tune_hyperparameter.py`                      | Optuna study database     |
| `../output/assets/transfer_learning.db`                         | `transfer_tide.py`                            | Transfer learning results |
| `../output/assets/tide_transfer/`                               | `transfer_tide.py`                            | Fine-tuning checkpoints   |
| `../ts2vec`                                                     | `train_encoder.py`, `calculate_similarity.py` | TS2Vec library            |

> **Note:** Output directories are created automatically when scripts are executed.

---

## 📝 Notes

- Scripts can be executed from any directory (paths are resolved relative to the script file)
- GPU is required for training (CUDA device)
- The dataset must be properly set up before running any scripts
- Scripts use fixed random seeds for reproducibility
- MLFlow is used for experiment tracking
- Early stopping is configured in all training scripts to prevent overfitting

---

## 📜 Citation

If you use this code, please cite:

```bibtex
@article{choi2026pptl,
  title   = {Privacy-Preserving Transfer Learning Framework for Building
             Energy Forecasting with Fully Anonymized Data},
  author  = {Choi, Wonjun and Lee, Sangwon and Langtry, Max and Choudhary, Ruchi},
  journal = {Applied Energy},
  year    = {2026},
  doi     = {10.1016/j.apenergy.2026.127600}
}
```

## TS2Vec Library

The `ts2vec/` directory contains a modified version of the TS2Vec codebase from the [official repository](https://github.com/yuezhihan/ts2vec). Only library version compatibility issues were resolved; no functional changes were made.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
