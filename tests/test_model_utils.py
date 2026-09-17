import numpy as np
import pandas as pd

from model_utils import naive_median_predictions, time_based_split


def test_time_based_split_puts_latest_dates_in_test_set():
    df = pd.DataFrame({
        "order_date": pd.to_datetime(
            ["2022-02-01", "2022-02-02", "2022-02-03", "2022-02-04", "2022-02-05"]
        ),
        "value": [1, 2, 3, 4, 5],
    })

    train, test = time_based_split(df, date_col="order_date", test_size=0.2)

    assert test["order_date"].tolist() == [pd.Timestamp("2022-02-05")]
    assert train["order_date"].max() < test["order_date"].min()


def test_time_based_split_keeps_all_rows_of_a_shared_date_together():
    df = pd.DataFrame({
        "order_date": pd.to_datetime(
            ["2022-02-01", "2022-02-01", "2022-02-01", "2022-02-02", "2022-02-02"]
        ),
        "value": [1, 2, 3, 4, 5],
    })

    train, test = time_based_split(df, date_col="order_date", test_size=0.2)

    shared_dates = set(train["order_date"]) & set(test["order_date"])
    assert shared_dates == set()


def test_naive_median_predictions_repeats_train_median_for_every_test_row():
    y_train = pd.Series([10, 20, 30])

    predictions = naive_median_predictions(y_train, n=4)

    assert np.array_equal(predictions, np.array([20, 20, 20, 20]))


def test_build_ordinal_encoder_represents_missing_values_as_nan_not_a_magic_number():
    from model_utils import build_ordinal_encoder

    encoder = build_ordinal_encoder(categories=[["low", "medium", "high"]])
    encoded = encoder.fit_transform(pd.DataFrame({"traffic": ["low", np.nan, "high"]}))

    assert np.isnan(encoded[1, 0])
