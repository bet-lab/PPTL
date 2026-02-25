"""
End-to-end PPTL pipeline test for building 4.

Steps:
  1. Train TS2Vec encoder (leave-one-out for building 4)
  2. Calculate cosine similarity for building 4
  3. Pretrain TiDE for Closest/Farthest {2,4,8,16}
  4. Fine-tune (transfer) for Closest/Farthest {2,4,8,16} + No-TL baseline
  5. Generate forecasting visualization for all 9 cases
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(1, str(_SCRIPT_DIR.parent / 'ts2vec'))

import random
import json
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.data import (
    CSVLoader, remove_anomaly, preprocess, between,
    TARGETS, NONTARGETS, make_time_series_dict,
)
from ts2vec.ts2vec import TS2Vec
from darts.models import TiDEModel
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.spatial.distance import cosine


# ──────────────────── Configuration ────────────────────
TARGET_BID = 4
DEVICE = 0
N_SOURCES_LIST = [2, 4, 8, 16]
MODES = ['best', 'worst']

DATA_ROOT = str(_SCRIPT_DIR.parent / 'datasets' / 'Cambridge-Estates-Building-Energy-Archive')
OUTPUT_DIR = _SCRIPT_DIR.parent / 'output' / 'assets'
WEIGHTS_DIR = OUTPUT_DIR / 'weights'

csv_loader = CSVLoader(DATA_ROOT)


# ──────────────────── Step 1: Train encoder ────────────────────
def train_encoder_for_bid(target_bid):
    print(f'\n{"="*60}')
    print(f'STEP 1: Training TS2Vec encoder for building {target_bid}')
    print(f'{"="*60}')

    weight_path = WEIGHTS_DIR / f'encoder_b{target_bid}.pt'
    if weight_path.exists():
        print(f'  -> Encoder weight already exists: {weight_path}')
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = TS2Vec(
        input_dims=len(NONTARGETS + TARGETS),
        device='cuda',
        hidden_dims=64,
        output_dims=320,
        max_train_length=3000,
    )

    Xs = []
    for bid in csv_loader.building_ids:
        if bid == target_bid:
            continue
        df = csv_loader.load(bid)
        df = remove_anomaly(df)
        df, _ = preprocess(df, '2009-01-01', '2010-01-01')
        X = between(df[NONTARGETS + TARGETS], '2009-01-01', '2010-01-01').values[np.newaxis, ...]
        Xs.append(X)

    X = np.concatenate(Xs, axis=0)
    print(f'  Training data shape: {X.shape}')
    model.fit(X, n_iters=200, verbose=True)
    model.save(str(weight_path))
    print(f'  -> Saved to {weight_path}')


# ──────────────────── Step 2: Calculate similarity ────────────────────
def calculate_similarity_for_bid(target_bid):
    print(f'\n{"="*60}')
    print(f'STEP 2: Calculating similarity for building {target_bid}')
    print(f'{"="*60}')

    sim_path = OUTPUT_DIR / 'similarities.json'

    model = TS2Vec(
        input_dims=len(NONTARGETS + TARGETS),
        device='cuda',
        hidden_dims=64,
        output_dims=320,
        max_train_length=3000,
    )
    model.load(str(WEIGHTS_DIR / f'encoder_b{target_bid}.pt'))

    # Target representation
    df = csv_loader.load(target_bid)
    df = remove_anomaly(df)
    df, _ = preprocess(df, '2010-01-01', '2010-03-01')
    X0 = between(df[NONTARGETS + TARGETS], '2010-01-01', '2010-03-01').values[np.newaxis, ...]
    Z0 = model.encode(X0, encoding_window='full_series')

    sims = []
    for b in csv_loader.building_ids:
        df = csv_loader.load(b)
        df = remove_anomaly(df)
        df, _ = preprocess(df, '2009-01-01', '2009-03-01')
        X = between(df[NONTARGETS + TARGETS], '2009-01-01', '2009-03-01').values[np.newaxis, ...]
        Z = model.encode(X, encoding_window='full_series')
        similarity = cosine(Z0[0], Z[0])
        sims.append((similarity, b))

    sims.sort()
    result = {
        'bid': [b for _, b in sims],
        'similarity': [s for s, _ in sims],
    }

    # Load existing or create new
    if sim_path.exists():
        with open(sim_path) as f:
            sim_dict = json.load(f)
    else:
        sim_dict = {}

    sim_dict[str(target_bid)] = result
    with open(sim_path, 'w') as f:
        json.dump(sim_dict, f, indent=4)

    print(f'  Top 5 closest: {result["bid"][:5]} (dist: {[f"{s:.4f}" for s in result["similarity"][:5]]})')
    print(f'  Top 5 farthest: {result["bid"][-5:]} (dist: {[f"{s:.4f}" for s in result["similarity"][-5:]]})')
    print(f'  -> Saved to {sim_path}')

    return result


# ──────────────────── Step 3: Pretrain TiDE ────────────────────
def create_tide_model(device, lr_scale=1.0, save_checkpoints=False, work_dir=None, model_name=None):
    kwargs = dict(
        input_chunk_length=7 * 24,
        output_chunk_length=24,
        batch_size=256,
        hidden_size=256,
        num_encoder_layers=1,
        num_decoder_layers=1,
        decoder_output_dim=8,
        temporal_decoder_hidden=32,
        dropout=0.3981,
        use_layer_norm=False,
        use_reversible_instance_norm=True,
        optimizer_kwargs={'lr': 0.00053954 * lr_scale},
        pl_trainer_kwargs={
            'log_every_n_steps': 500,
            'max_epochs': -1,
            'accelerator': 'gpu',
            'devices': [device],
            'enable_progress_bar': True,
            'callbacks': [
                EarlyStopping(monitor="val_loss", patience=5, mode='min'),
            ],
        },
    )
    if save_checkpoints:
        kwargs['force_reset'] = True
        kwargs['save_checkpoints'] = True
        kwargs['work_dir'] = str(work_dir)
        kwargs['model_name'] = model_name
    return TiDEModel(**kwargs)


def get_source_bids(target_bid, mode, n_sources):
    with open(OUTPUT_DIR / 'similarities.json') as f:
        data = json.load(f)
    bids = data[str(target_bid)]['bid']
    bids_filtered = [b for b in bids if b != target_bid]
    if mode == 'best':
        return bids_filtered[:n_sources]
    elif mode == 'worst':
        return bids_filtered[-n_sources:]
    else:
        raise ValueError(f'Invalid mode: {mode}')


def pretrain_tide(target_bid, mode, n_sources):
    weight_path = WEIGHTS_DIR / f'tide_bid_{target_bid}_{mode}_{n_sources}.pt'
    if weight_path.exists():
        print(f'    -> Already exists: {weight_path.name}')
        return

    source_bids = get_source_bids(target_bid, mode, n_sources)
    print(f'    Source buildings: {source_bids}')

    series = []
    future_covariates = []
    for bid in source_bids:
        data = make_time_series_dict(
            bid=bid, csv_loader=csv_loader,
            train_range=('2009-01-01', '2010-01-01'),
        )
        series.append(data['train_series'])
        future_covariates.append(data['train_future_covariates'])

    # Target as validation
    val_data = make_time_series_dict(
        bid=target_bid, csv_loader=csv_loader,
        train_range=('2010-01-01', '2010-03-01'),
    )

    model = create_tide_model(DEVICE)
    model.fit(
        series=series,
        future_covariates=future_covariates,
        val_series=val_data['train_series'],
        val_future_covariates=val_data['train_future_covariates'],
        verbose=True,
        num_loader_workers=4,
    )
    model.save(str(weight_path))
    print(f'    -> Saved to {weight_path.name}')


def pretrain_all(target_bid):
    print(f'\n{"="*60}')
    print(f'STEP 3: Pretraining TiDE for building {target_bid}')
    print(f'{"="*60}')

    for mode in MODES:
        for n in N_SOURCES_LIST:
            print(f'\n  [{mode.upper()} {n}]')
            pretrain_tide(target_bid, mode, n)


# ──────────────────── Step 4: Transfer learning ────────────────────
def transfer_and_evaluate(target_bid, mode, n_sources):
    """Run transfer learning and return results dict with forecasts."""
    data = make_time_series_dict(
        bid=target_bid, csv_loader=csv_loader,
        train_range=('2010-01-01', '2010-03-01'),
        val_range=('2010-03-01', '2010-05-01'),
        test_range=('2010-03-01', '2010-05-01'),
    )

    tide_transfer_dir = OUTPUT_DIR / 'tide_transfer'
    tide_transfer_dir.mkdir(parents=True, exist_ok=True)

    if mode != 'none':
        model = create_tide_model(
            DEVICE, lr_scale=0.1, save_checkpoints=True,
            work_dir=str(tide_transfer_dir),
            model_name=f'test_b{target_bid}_{mode}_{n_sources}',
        )
        weight_path = WEIGHTS_DIR / f'tide_bid_{target_bid}_{mode}_{n_sources}.pt'
        model.load_weights(str(weight_path))
    else:
        model = create_tide_model(
            DEVICE, lr_scale=1.0, save_checkpoints=True,
            work_dir=str(tide_transfer_dir),
            model_name=f'test_b{target_bid}_none_0',
        )

    model.fit(
        series=data['train_series'],
        future_covariates=data['train_future_covariates'],
        val_series=data['val_series'],
        val_future_covariates=data['val_future_covariates'],
        verbose=True,
        num_loader_workers=4,
        epochs=-1,
    )

    # Load best checkpoint
    model_name = f'test_b{target_bid}_{mode}_{n_sources}'
    dummy_model = TiDEModel.load_from_checkpoint(
        model_name=model_name,
        work_dir=str(tide_transfer_dir),
        best=True,
    )
    with TemporaryDirectory() as tmp_dir:
        dummy_model.save(f'{tmp_dir}/model.pt')
        model.load_weights(f'{tmp_dir}/model.pt')

    # Historical forecasts
    results = model.historical_forecasts(
        series=data['test_series'],
        future_covariates=data['test_future_covariates'],
        forecast_horizon=24,
        stride=24,
        retrain=False,
        enable_optimization=True,
        last_points_only=False,
    )

    # Compute MSE
    data_df = data['test_series'].pd_dataframe()
    mse_list = []
    for r in results:
        pred = r.pd_dataframe()
        real = data_df.loc[pred.index]
        mse = np.mean((pred.values - real.values) ** 2)
        mse_list.append(mse)
    test_mse = np.mean(mse_list)

    # Get rolling 24h forecast for visualization (stride=1 for smooth plot)
    viz_results = model.historical_forecasts(
        series=data['test_series'],
        future_covariates=data['test_future_covariates'],
        forecast_horizon=24,
        stride=24,
        retrain=False,
        enable_optimization=True,
        last_points_only=True,
    )

    return {
        'mode': mode,
        'n_sources': n_sources,
        'test_mse': test_mse,
        'forecasts': viz_results,
        'ground_truth': data['test_series'],
        'scaler': data['scaler'],
    }


def run_all_transfers(target_bid):
    print(f'\n{"="*60}')
    print(f'STEP 4: Transfer learning for building {target_bid}')
    print(f'{"="*60}')

    all_results = {}

    # No-TL baseline first
    print(f'\n  [NO-TL (baseline)]')
    result = transfer_and_evaluate(target_bid, 'none', 0)
    label = 'No-TL'
    all_results[label] = result
    print(f'    MSE: {result["test_mse"]:.6f}')

    for mode in MODES:
        for n in N_SOURCES_LIST:
            mode_name = 'Closest' if mode == 'best' else 'Farthest'
            label = f'{mode_name} {n}'
            print(f'\n  [{label}]')
            result = transfer_and_evaluate(target_bid, mode, n)
            all_results[label] = result
            print(f'    MSE: {result["test_mse"]:.6f}')

    return all_results


# ──────────────────── Step 5: Visualization ────────────────────
def plot_forecasts(target_bid, all_results):
    print(f'\n{"="*60}')
    print(f'STEP 5: Generating visualizations')
    print(f'{"="*60}')

    # Get ground truth and scaler from any result
    any_result = next(iter(all_results.values()))
    gt_series = any_result['ground_truth']
    scaler = any_result['scaler']

    gt_df = gt_series.pd_dataframe()

    # Inverse transform ground truth
    gt_values = scaler.inverse_transform(gt_df.values.reshape(-1, 1)).flatten()
    gt_index = gt_df.index

    # --- Plot 1: All forecasts comparison ---
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

    # Top: Closest strategies
    ax = axes[0]
    ax.plot(gt_index, gt_values, 'k-', linewidth=0.8, alpha=0.7, label='Ground Truth')

    colors_closest = ['#006D77', '#42999B', '#83C5BE', '#B8D8D8']
    for i, n in enumerate(N_SOURCES_LIST):
        label = f'Closest {n}'
        if label in all_results:
            result = all_results[label]
            fc_df = result['forecasts'].pd_dataframe()
            fc_values = scaler.inverse_transform(fc_df.values.reshape(-1, 1)).flatten()
            ax.plot(fc_df.index, fc_values, '-', color=colors_closest[i],
                    linewidth=0.8, alpha=0.9, label=f'{label} (MSE={result["test_mse"]:.4f})')

    notl = all_results.get('No-TL')
    if notl:
        fc_df = notl['forecasts'].pd_dataframe()
        fc_values = scaler.inverse_transform(fc_df.values.reshape(-1, 1)).flatten()
        ax.plot(fc_df.index, fc_values, '--', color='#E29578',
                linewidth=0.8, alpha=0.8, label=f'No-TL (MSE={notl["test_mse"]:.4f})')

    ax.set_title(f'Building {target_bid} — Closest Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Electricity Usage [kWh]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: Farthest strategies
    ax = axes[1]
    ax.plot(gt_index, gt_values, 'k-', linewidth=0.8, alpha=0.7, label='Ground Truth')

    colors_farthest = ['#E76F51', '#F4A261', '#E9C46A', '#FFE5B4']
    for i, n in enumerate(N_SOURCES_LIST):
        label = f'Farthest {n}'
        if label in all_results:
            result = all_results[label]
            fc_df = result['forecasts'].pd_dataframe()
            fc_values = scaler.inverse_transform(fc_df.values.reshape(-1, 1)).flatten()
            ax.plot(fc_df.index, fc_values, '-', color=colors_farthest[i],
                    linewidth=0.8, alpha=0.9, label=f'{label} (MSE={result["test_mse"]:.4f})')

    if notl:
        fc_df = notl['forecasts'].pd_dataframe()
        fc_values = scaler.inverse_transform(fc_df.values.reshape(-1, 1)).flatten()
        ax.plot(fc_df.index, fc_values, '--', color='#264653',
                linewidth=0.8, alpha=0.8, label=f'No-TL (MSE={notl["test_mse"]:.4f})')

    ax.set_title(f'Building {target_bid} — Farthest Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Electricity Usage [kWh]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: MSE comparison bar chart
    ax = axes[2]
    labels = []
    mses = []
    colors = []

    # No-TL
    labels.append('No-TL')
    mses.append(all_results['No-TL']['test_mse'])
    colors.append('#888888')

    for n in N_SOURCES_LIST:
        label = f'Closest {n}'
        labels.append(label)
        mses.append(all_results[label]['test_mse'])
        colors.append('#006D77')

    for n in N_SOURCES_LIST:
        label = f'Farthest {n}'
        labels.append(label)
        mses.append(all_results[label]['test_mse'])
        colors.append('#E76F51')

    bars = ax.bar(labels, mses, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=all_results['No-TL']['test_mse'], color='#888888', linestyle='--',
               linewidth=1, alpha=0.7, label='No-TL baseline')
    ax.set_ylabel('Test MSE (normalized)')
    ax.set_title(f'Building {target_bid} — MSE Comparison', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add MSE values on bars
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{mse:.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f'test_b{target_bid}_forecasts.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> Saved forecast plot: {plot_path}')

    # --- Print summary table ---
    print(f'\n  {"="*50}')
    print(f'  RESULTS SUMMARY (Building {target_bid})')
    print(f'  {"="*50}')
    baseline_mse = all_results['No-TL']['test_mse']
    print(f'  {"Configuration":<18} {"MSE":>10} {"Relative":>10}')
    print(f'  {"-"*40}')
    for label in ['No-TL'] + [f'Closest {n}' for n in N_SOURCES_LIST] + [f'Farthest {n}' for n in N_SOURCES_LIST]:
        mse = all_results[label]['test_mse']
        rel = mse / baseline_mse * 100
        print(f'  {label:<18} {mse:>10.6f} {rel:>9.1f}%')

    return plot_path


# ──────────────────── Main ────────────────────
if __name__ == '__main__':
    print(f'PPTL End-to-End Test — Building {TARGET_BID}')
    print(f'GPU: cuda:{DEVICE}')
    print(f'Configs: Closest/Farthest × N_S* ∈ {N_SOURCES_LIST} + No-TL')
    print(f'Total models to train: {len(MODES) * len(N_SOURCES_LIST)} pretrain + {len(MODES) * len(N_SOURCES_LIST) + 1} fine-tune = {len(MODES) * len(N_SOURCES_LIST) * 2 + 1}')

    # Step 1
    train_encoder_for_bid(TARGET_BID)

    # Step 2
    calculate_similarity_for_bid(TARGET_BID)

    # Step 3
    pretrain_all(TARGET_BID)

    # Step 4
    all_results = run_all_transfers(TARGET_BID)

    # Step 5
    plot_path = plot_forecasts(TARGET_BID, all_results)

    print(f'\n{"="*60}')
    print('ALL DONE!')
    print(f'{"="*60}')
