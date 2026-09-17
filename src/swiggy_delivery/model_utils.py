import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def build_ordinal_encoder(categories: list) -> OrdinalEncoder:
    """Ordinal encoder that leaves missing values as NaN.

    A magic-number sentinel (e.g. -999) would sit far outside every other
    feature's scaled range and get treated as a real distance by a
    downstream KNN imputer. NaN lets that imputer treat it as missing.
    """
    return OrdinalEncoder(
        categories=categories,
        encoded_missing_value=np.nan,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )


def naive_median_predictions(y_train: pd.Series, n: int) -> np.ndarray:
    return np.full(n, y_train.median())


def time_based_split(df: pd.DataFrame, date_col: str, test_size: float = 0.2):
    sorted_df = df.sort_values(date_col)
    cutoff_index = int(len(sorted_df) * (1 - test_size))
    cutoff_date = sorted_df[date_col].iloc[cutoff_index]

    train = sorted_df.loc[sorted_df[date_col] < cutoff_date]
    test = sorted_df.loc[sorted_df[date_col] >= cutoff_date]
    return train, test
