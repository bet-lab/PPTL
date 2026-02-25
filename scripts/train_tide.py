import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
import json
from argparse import ArgumentParser
from darts.models import TiDEModel
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.data import CSVLoader, make_time_series_dict


parser = ArgumentParser()
parser.add_argument('--bid', type=int)
parser.add_argument('--mode', choices=['best', 'worst', 'all'])
parser.add_argument('--n-sources', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()


def create_model(device):
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
        optimizer_kwargs={
            'lr': 0.00053954,
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


def get_source_bids(bid, mode, n_sources):
    # Import json.
    with open('../output/assets/similarities.json') as f:
        data = json.load(f)

    bids = data[str(bid)]['bid']

    # Remove the current bid.
    bids.remove(bid)

    if mode == 'best':
        # Best 3 bids.
        return bids[:n_sources]
    elif mode == 'worst':
        # Worst (farthest) bids.
        return bids[-n_sources:]
    elif mode == 'all':
        return bids
    else:
        raise ValueError('Invalid mode.')


def main():
    if args.mode == 'all' and args.n_sources > 0:
        raise ValueError('Invalid mode and n_sources combination.')

    csv_loader = CSVLoader('../datasets/Cambridge-Estates-Building-Energy-Archive')

    series = []
    future_covariates = []
    
    bids = get_source_bids(bid=args.bid, mode=args.mode, n_sources=args.n_sources)

    for bid in bids:
        data = make_time_series_dict(
            bid=bid,
            csv_loader=csv_loader,
            train_range=('2009-01-01', '2010-01-01'),
        )

        series.append(data['train_series'])
        future_covariates.append(data['train_future_covariates'])

    # Target training data as validation data.
    data = make_time_series_dict(
        bid=args.bid,
        csv_loader=csv_loader,
        train_range=('2010-01-01', '2010-03-01'),
    )

    val_series = data['train_series']
    val_future_covariates = data['train_future_covariates']

    model = create_model(device=args.device)
    model.fit(
        series=series,
        future_covariates=future_covariates,
        val_series=val_series,
        val_future_covariates=val_future_covariates,
        verbose=True,
        num_loader_workers=8,
    )

    # Just save last model.
    model.save(
        '../output/assets/weights/tide_bid_{}_{}_{}.pt'
        .format(args.bid, args.mode, args.n_sources)
    )


if __name__ == '__main__':
    print(args)
    main()