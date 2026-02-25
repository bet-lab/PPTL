import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

import sqlite3
from tempfile import TemporaryDirectory
from argparse import ArgumentParser

import numpy as np

from darts.models import TiDEModel
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.data import CSVLoader, make_time_series_dict


parser = ArgumentParser()
parser.add_argument('--bid', type=int)
parser.add_argument('--mode', choices=['best', 'worst', 'all', 'none'])
parser.add_argument('--n-sources', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()


def create_database():
    """Create a database for saving transfer learning results.
    Columns contain the following information:
    - id: Unique id.
    - bid: Building id.
    - mode: Transfer learning mode.
    - last_val_loss: Last validation loss.
    - best_val_loss: Best validation loss.
    - run_id: MLFlow run id.
    """
    assets_dir = _SCRIPT_DIR.parent / 'output' / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / 'tide_transfer').mkdir(exist_ok=True)

    with sqlite3.connect(str(assets_dir / 'transfer_learning.db')) as conn:
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transfer_learning (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bid INTEGER,
            mode TEXT,
            n_sources INTEGER,
            last_val_loss REAL,
            best_val_loss REAL,
            last_test_loss REAL,
            best_test_loss REAL,
            run_id TEXT
        )
        ''')

        conn.commit()


def save_to_database(
    bid, mode, n_sources,
    last_val_loss, best_val_loss,
    last_test_loss, best_test_loss,
    run_id,
):
    """Save transfer learning results to the database.
    
    Parameters
    ----------
    bid : int
        Building id.
    
    mode : str
        Transfer learning mode.
    
    val_loss : float
        Validation loss.
    """

    db_path = _SCRIPT_DIR.parent / 'output' / 'assets' / 'transfer_learning.db'
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO transfer_learning
            (bid, mode, n_sources, last_val_loss, best_val_loss, last_test_loss, best_test_loss, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (bid, mode, n_sources, last_val_loss, best_val_loss, last_test_loss, best_test_loss, run_id)
        )
        conn.commit()


def create_model(device, lr_scale, work_dir=str(_SCRIPT_DIR.parent / 'output' / 'assets' / 'tide_transfer')):
    return TiDEModel(
        input_chunk_length=7 * 24,
        output_chunk_length=24,
        batch_size=256,

        # Last update 2024-07-03.
        hidden_size=256,
        num_encoder_layers=1,
        num_decoder_layers=1,
        decoder_output_dim=8,
        temporal_decoder_hidden=32,
        dropout=0.3981,
        use_layer_norm=False,
        use_reversible_instance_norm=True,

        force_reset=True,
        save_checkpoints=True,
        work_dir=work_dir,
        model_name='tide_bid_{}_{}_{}'.format(args.bid, args.mode, args.n_sources),
        optimizer_kwargs={
            'lr': 0.00053954 * lr_scale,
        },
        pl_trainer_kwargs={
            'log_every_n_steps': 500,
            'max_epochs': -1,
            'accelerator': 'gpu',
            'devices': [device],
            'callbacks': [
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    mode='min',
                ),
            ],
            'logger': MLFlowLogger(experiment_name='TiDE2'),
        },
    )


def main():
    csv_loader = CSVLoader(str(_SCRIPT_DIR.parent / 'datasets' / 'Cambridge-Estates-Building-Energy-Archive'))

    # Transfer learning (fine-tuning on target building).
    data = make_time_series_dict(
        bid=args.bid,
        csv_loader=csv_loader,
        train_range=('2010-01-01', '2010-03-01'),
        val_range=('2010-03-01', '2010-05-01'),
        test_range=('2010-03-01', '2010-05-01'),
    )

    if args.mode != 'none':
        # Smaller learning rate for transfer learning.
        model = create_model(device=args.device, lr_scale=0.1)
        weights_path = _SCRIPT_DIR.parent / 'output' / 'assets' / 'weights' / 'tide_bid_{}_{}_{}.pt'.format(args.bid, args.mode, args.n_sources)
        model.load_weights(str(weights_path))
    else:
        model = create_model(device=args.device, lr_scale=1)

    model.fit(
        series=data['train_series'],
        future_covariates=data['train_future_covariates'],
        val_series=data['val_series'],
        val_future_covariates=data['val_future_covariates'],
        verbose=True,
        num_loader_workers=8,
        epochs=-1,
    )

    # Extract losses from final training run.
    last_val_loss = model.trainer.callback_metrics['val_loss'].cpu().numpy().item()
    best_val_loss = model.trainer.early_stopping_callback.best_score.cpu().numpy().item()

    # Load best model via checkpoint to extract weights without static attribute mismatch.
    dummy_model = TiDEModel.load_from_checkpoint(
        model_name='tide_bid_{}_{}_{}'.format(args.bid, args.mode, args.n_sources),
        work_dir=str(_SCRIPT_DIR.parent / 'output' / 'assets' / 'tide_transfer'),
        best=True,
    )

    with TemporaryDirectory() as tmp_dir:
        dummy_model.save(f'{tmp_dir}//model.pt')
        model.load_weights(f'{tmp_dir}//model.pt')

    results = model.historical_forecasts(
        series=data['test_series'],
        future_covariates=data['test_future_covariates'],
        forecast_horizon=24,
        stride=1,
        retrain=False,
        enable_optimization=True,
        last_points_only=False,
    )

    data_df = data['test_series'].pd_dataframe()
    mse_list = []


    for r in results:
        pred = r.pd_dataframe()
        real = data_df.loc[pred.index]

        mse = np.mean((pred - real) ** 2)
        mse_list.append(mse)

    best_test_loss = np.mean(mse_list)

    # Load last model.
    dummy_model = TiDEModel.load_from_checkpoint(
        model_name='tide_bid_{}_{}_{}'.format(args.bid, args.mode, args.n_sources),
        work_dir=str(_SCRIPT_DIR.parent / 'output' / 'assets' / 'tide_transfer'),
        best=False,
    )

    with TemporaryDirectory() as tmp_dir:
        dummy_model.save(f'{tmp_dir}//model.pt')
        model.load_weights(f'{tmp_dir}//model.pt')

    results = model.historical_forecasts(
        series=data['test_series'],
        future_covariates=data['test_future_covariates'],
        forecast_horizon=24,
        stride=1,
        retrain=False,
        enable_optimization=True,
        last_points_only=False,
    )

    data_df = data['test_series'].pd_dataframe()
    mse_list = []

    for r in results:
        pred = r.pd_dataframe()
        real = data_df.loc[pred.index]

        mse = np.mean((pred - real) ** 2)
        mse_list.append(mse)

    last_test_loss = np.mean(mse_list)

    # Save to database.
    save_to_database(
        args.bid, args.mode, args.n_sources,
        last_val_loss, best_val_loss,
        last_test_loss, best_test_loss,
        model.trainer.logger.run_id
    )


if __name__ == '__main__':
    create_database()
    print(args)
    main()