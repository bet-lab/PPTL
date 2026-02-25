import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_SCRIPT_DIR.parent / 'ts2vec'))
import random
import numpy as np
import torch
from utils.data import (
    CSVLoader, remove_anomaly, preprocess, between, TARGETS, NONTARGETS
)
from ts2vec.ts2vec import TS2Vec


csv_loader = CSVLoader('../datasets/Cambridge-Estates-Building-Energy-Archive')


def bid_2_X(bid, start, end):
    """Return X for a building id and a range.
    
    Parameters
    ----------
    bid : str
        Building id.

    range_ : tuple[str, str]
        Range of the data.

    Returns
    -------
    X : np.ndarray
        X for the building id and the range. The shape is (1, n_timesteps, n_features).
    """
    df = csv_loader.load(bid)
    df = remove_anomaly(df)
    df, _ = preprocess(df, start, end)

    X = between(df[NONTARGETS + TARGETS], start, end).values[np.newaxis, ...]

    return X


def make_X(target_bid):
    """Make X from the dataset. X is a 3D tensor of shape 
    (n_instances, n_timesteps, n_features). It is used to train TS2Vec.
    """
    bids = csv_loader.building_ids

    Xs = []
    for bid in bids:
        if bid == target_bid:
            continue

        X = bid_2_X(bid, '2009-01-01', '2010-01-01')
        Xs.append(X)

    X = np.concatenate(Xs, axis=0)

    return X


def train_encoder(target_bid):
    model = TS2Vec(
        input_dims=len(NONTARGETS + TARGETS),
        device='cuda',
        hidden_dims=64,
        output_dims=320,
        max_train_length=3000,
    )

    X = make_X(target_bid)

    # This function returns loss_log. But it is not used here.
    model.fit(
        X,
        n_iters=200,
        verbose=True,
    )

    return model


if __name__ == "__main__":
    weights_dir = _SCRIPT_DIR.parent / 'output' / 'assets' / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    for bid in csv_loader.building_ids:
        print(f'Training encoder for building {bid}...')

        # Set random seed for reproducibility.
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        model = train_encoder(bid)
        model.save(str(weights_dir / f'encoder_b{bid}.pt'))