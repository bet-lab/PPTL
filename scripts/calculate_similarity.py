import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_SCRIPT_DIR.parent / 'ts2vec'))
import json
import numpy as np
from scipy.spatial.distance import cosine
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


def calculate_similarities(model, bid):
    """Calculate similarity between the building id and the others.
    
    Parameters
    ----------
    bid : str
        Building id.
    """
    # Target range.
    X0 = bid_2_X(bid, '2010-01-01', '2010-03-01')
    Z0 = model.encode(X0, encoding_window='full_series')

    sims = []
    for b in csv_loader.building_ids:
        X = bid_2_X(b, '2009-01-01', '2009-03-01')
        Z = model.encode(X, encoding_window='full_series')

        similarity = cosine(Z0[0], Z[0])
        print(f'{bid} vs {b}: {similarity:.5f}, {Z.shape = }')
        sims.append((similarity, b))

    sims.sort()

    bids = [b for _, b in sims]
    sims = [s for s, _ in sims]

    return {'bid': bids, 'similarity': sims}


if __name__ == '__main__':
    output_dir = _SCRIPT_DIR.parent / 'output' / 'assets'
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TS2Vec(
        input_dims=len(NONTARGETS + TARGETS),
        device='cuda',
        hidden_dims=64,
        output_dims=320,
        max_train_length=3000,
    )
    
    sim_dict = {}
    for bid in csv_loader.building_ids:
        print(bid)
        model.load(f'../output/assets/weights/encoder_b{bid}.pt')
        sims = calculate_similarities(model, bid)
        sim_dict[bid] = sims

    # Save as json.
    with open('../output/assets/similarities.json', 'w') as f:
        json.dump(sim_dict, f, indent=4)
   