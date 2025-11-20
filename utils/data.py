from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from darts import TimeSeries


DATA_YEARS = (
    # 2008,
    2009,
    2010,
)


TARGETS = [
    'equipment load [kWh]',
]


NONTARGETS = [
    'month_cos',
    'month_sin',
    'hour_cos',
    'hour_sin',
    'day_type',
    'air_temperature [degC]',
    'rltv_hum [%]',
    'glbl_irad_amt [W/m2]',
    'irradiance_direct [W/m2]',
    'irradiance_diffuse [W/m2]',
]


def set_data_years(years):
    """Set data years to be used for training.

    Parameters
    ----------
    years: tuple[int]
        Tuple of years to be used for training.
    """
    global DATA_YEARS
    DATA_YEARS = years


class CSVLoader:
    def __init__(self, data_root):
        """Simple data loading helper. This is used for loading CSV files for
        each building. Only 2008, 2009, and 2010 data are used.

        Parameters
        ----------
        data_root: str
            Path of Cambridge-Estates-Building-Energy-Archive directory.
        """
        self.data_root = Path(data_root)

    @property
    def building_ids(self):
        root = self.data_root / 'building_data/processed_data'

        bids = []
        for path in root.glob('UCam_Building_*'):
            invalid = False
            for year in DATA_YEARS:
                if not (path / f'electricity/{year}.csv').exists():
                    invalid = True
                    break

            if invalid:
                continue

            bids.append(int(str(path.name).split('_')[-1][1:]))

        return sorted(bids)

    def load(self, building_id):
        dfs = []

        for year in DATA_YEARS:
            df = pd.read_csv(
                self.data_root / f'building_data/processed_data/UCam_Building_b{building_id}/electricity/{year}.csv',
                index_col='datetime',
                parse_dates=True,
            )
            
            weather = pd.read_csv(
                self.data_root / f'aux_data/MetOffice Weather Data/processed_data/bedford/{year}.csv',
                index_col='datetime',
                parse_dates=True,
            )

            df[weather.columns] = weather

            ninja = pd.read_csv(
                self.data_root / f'aux_data/RenewablesNinja Generation Data/processed_data/cambridge_52-194_0-131/{year}.csv',
                index_col='datetime',
                parse_dates=True,
            )

            df[ninja.columns] = ninja

            dfs.append(df)

        df = pd.concat(dfs)

        return df


def remove_anomaly(df):
    """
    Remove anomaly from the data frame.
    Fill the missing values using interpolation.

    Parameters
    ----------
    df: pd.DataFrame
        Data frame to be used.

    Returns
    -------
    df: pd.DataFrame
        Data frame without anomaly.
    """
    df = df.copy()

    col = 'equipment load [kWh]'

    year = 365 * 24
    n_years = len(df) // year
    if n_years < len(df) / year:
        n_years += 1

    anomaly = np.zeros(len(df), dtype=bool)
    for i in range(n_years):
        # 10% quantile for each year.
        data = df[col].iloc[i * year : (i + 1) * year].copy()

        q = data.quantile(0.1)
        thresh = 0.5 * q

        anomaly[i * year : (i + 1) * year] = data < thresh
    
    df[col].values[anomaly] = np.nan
    df = df.interpolate(method='linear', limit_direction='both')

    return df


def between(df, start, end):
    """Get data between start and end dates. end date is exclusive.
    This is better than using df.loc[start:end] because the first day of a month
    is always 1 but the last day is not fixed, so it is better to use '<' for the end date.

    Parameters
    ----------
    df: pd.DataFrame
        Data frame to be used.

    start: str
        Start date.
        The format should be 'YYYY-MM-DD'.

    end: str
        End date.
        The format should be 'YYYY-MM-DD'.

    Returns
    -------
    df: pd.DataFrame
        Data frame between start and end dates.
    """
    return df[(df.index >= start) & (df.index < end)]


def preprocess(df, start, end):
    """
    Convert the data frame to the format that can be used for training.
    Remove anomaly before using this function.

    Parameters
    ----------
    df: pd.DataFrame
        Data frame to be used.

    start: str
        Start date for target value normalization.
        The format should be 'YYYY-MM-DD'.

    end: str
        End date for target value normalization.
        The format should be 'YYYY-MM-DD'.

    Returns
    -------
    df: pd.DataFrame
        Preprocessed data frame.

    scaler: RobustScaler
        Scaler used for target value normalization.
        This is used for inverse transformation.
        Quantile range is set to (25.0, 95.0).
    """
    df = df.copy()

    # Add sin, cos of time stamps.
    # Month, Hour
    
    a = 2 * np.pi / 12
    df['month_cos'] = np.cos(df.index.month * a)
    df['month_sin'] = np.sin(df.index.month * a)

    a = 2 * np.pi / 24
    df['hour_cos'] = np.cos(df.index.hour * a)
    df['hour_sin'] = np.sin(df.index.hour * a)

    # Just normalize with maximum value. Monday = 0, ..., Sunday = 6.
    df['day_type'] = df.index.weekday / 6

    data = (
        between(df, start, end)['equipment load [kWh]']
        .values.reshape(-1, 1)
    )
    
    # Larger upper quantile value is used to consider peak values.
    scaler = RobustScaler(quantile_range=(25.0, 95.0))
    scaler.fit(data)
    
    df['equipment load [kWh]'] = scaler.transform(
        df['equipment load [kWh]'].values.reshape(-1, 1)
    ).reshape(-1)

    # Global irradiance.
    df['glbl_irad_amt [W/m2]'] /= 1000

    # Normalize direct solar radiation.
    df['irradiance_direct [W/m2]'] /= 1000

    # Normalize diffuse solar radiation.
    df['irradiance_diffuse [W/m2]'] /= 400

    # Normalize outdoor temperature.
    df['air_temperature [degC]'] /= 50

    # Normalize relative humidity.
    df['rltv_hum [%]'] /= 100

    return df, scaler


def make_time_series_dict(
    bid,
    csv_loader,
    train_range,
    val_range=None,
    test_range=None,
):
    """
    Make time series data for the given building id. Scale target values
    using training data.

    Parameters
    ----------
    bid: int
        Building id.

    train_range: tuple[str, str]
        Tuple of start and end date for training data.
        The format should be 'YYYY-MM-DD'.

    val_range: tuple[str, str]
        Tuple of start and end date for validation data. No validation
        data is used if None.

    test_range: tuple[str, str]
        Tuple of start and end date for test data. No test data is used
        if None.
    
    Returns
    -------
    data: dict
        Dictionary containing train, val, test and scaler data.
        Keys are given as follows:
        - 'train_series'
        - 'train_future_covariates'
        - 'val_series'
        - 'val_future_covariates'
        - 'test_series'
        - 'test_future_covariates'
        - 'scaler'
    """
    static_covariates = pd.DataFrame(
        data={
            f'b{b}': [int(b == bid)]
            for b in csv_loader.building_ids
        },
    )

    # Construct training data.
    df = csv_loader.load(bid)
    df = remove_anomaly(df)
    df, scaler = preprocess(df, *train_range)

    data = {}

    train_series = TimeSeries.from_dataframe(
        between(df, *train_range),
        value_cols=TARGETS,
        static_covariates=static_covariates,
    )

    train_future_covariates = TimeSeries.from_dataframe(
        between(df, *train_range),
        value_cols=NONTARGETS,
        static_covariates=static_covariates,
    )

    data['train_series'] = train_series
    data['train_future_covariates'] = train_future_covariates
    
    # Construct validation data.
    if val_range is not None:
        val_series = TimeSeries.from_dataframe(
            between(df, *val_range),
            value_cols=TARGETS,
            static_covariates=static_covariates,
        )

        val_future_covariates = TimeSeries.from_dataframe(
            between(df, *val_range),
            value_cols=NONTARGETS,
            static_covariates=static_covariates,
        )

        data['val_series'] = val_series
        data['val_future_covariates'] = val_future_covariates

    # Construct test data.
    if test_range is not None:
        test_series = TimeSeries.from_dataframe(
            between(df, *test_range),
            value_cols=TARGETS,
            static_covariates=static_covariates,
        )

        test_future_covariates = TimeSeries.from_dataframe(
            between(df, *test_range),
            value_cols=NONTARGETS,
            static_covariates=static_covariates,
        )

        data['test_series'] = test_series
        data['test_future_covariates'] = test_future_covariates

    data['scaler'] = scaler

    return data
