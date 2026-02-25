"""
Simple single-model forecasting visualization.

Usage:
    uv run python scripts/visualize_forecast.py --bid 4 --mode best --n 2
    uv run python scripts/visualize_forecast.py --bid 4 --mode none
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(1, str(_SCRIPT_DIR.parent / 'ts2vec'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts.models import TiDEModel
from utils.data import CSVLoader, make_time_series_dict

DATA_ROOT = str(_SCRIPT_DIR.parent / 'datasets' / 'Cambridge-Estates-Building-Energy-Archive')
OUTPUT_DIR = _SCRIPT_DIR.parent / 'output' / 'assets'
TRANSFER_DIR = OUTPUT_DIR / 'tide_transfer'

csv_loader = CSVLoader(DATA_ROOT)


def main():
    parser = argparse.ArgumentParser(description='Visualize a single TiDE forecast')
    parser.add_argument('--bid', type=int, required=True, help='Target building ID')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['best', 'worst', 'none'],
                        help='Transfer mode (best=closest, worst=farthest, none=no-TL)')
    parser.add_argument('--n', type=int, default=0,
                        help='Number of source buildings (ignored if mode=none)')
    parser.add_argument('--output', type=str, default=None, help='Output PNG path')
    args = parser.parse_args()

    if args.mode == 'none':
        model_name = f'test_b{args.bid}_none_0'
        title = f'Building {args.bid} — No Transfer Learning'
    else:
        model_name = f'test_b{args.bid}_{args.mode}_{args.n}'
        mode_label = 'Closest' if args.mode == 'best' else 'Farthest'
        title = f'Building {args.bid} — {mode_label} {args.n}'

    # Load model
    print(f'Loading checkpoint: {model_name}')
    model = TiDEModel.load_from_checkpoint(
        model_name=model_name,
        work_dir=str(TRANSFER_DIR),
        best=True,
    )

    # Prepare test data
    data = make_time_series_dict(
        bid=args.bid, csv_loader=csv_loader,
        train_range=('2010-01-01', '2010-03-01'),
        val_range=('2010-03-01', '2010-05-01'),
        test_range=('2010-03-01', '2010-05-01'),
    )
    scaler = data['scaler']

    # Run inference
    print('Running historical forecasts ...')
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
    gt_df = data['test_series'].pd_dataframe()
    mse_list = []
    for r in results:
        pred = r.pd_dataframe()
        real = gt_df.loc[pred.index]
        mse_list.append(np.mean((pred.values - real.values) ** 2))
    test_mse = np.mean(mse_list)
    print(f'Test MSE: {test_mse:.6f}')

    # Concatenate forecast blocks
    fc_df = pd.concat([r.pd_dataframe() for r in results]).sort_index()
    fc_df = fc_df[~fc_df.index.duplicated(keep='first')]

    # Inverse transform
    gt_values = scaler.inverse_transform(gt_df.values.reshape(-1, 1)).flatten()
    fc_values = scaler.inverse_transform(fc_df.values.reshape(-1, 1)).flatten()

    # Plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(gt_df.index, gt_values, 'k-', linewidth=0.8, alpha=0.6, label='Ground Truth')
    ax.plot(fc_df.index, fc_values, '-', color='#006D77', linewidth=0.8,
            label=f'Forecast (MSE={test_mse:.4f})')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Electricity Usage [kWh]')
    ax.set_xlabel('Date')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = args.output or str(OUTPUT_DIR / f'forecast_b{args.bid}_{model_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
