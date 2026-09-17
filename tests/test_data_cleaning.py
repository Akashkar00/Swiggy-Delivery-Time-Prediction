import pandas as pd

from swiggy_delivery.data_cleaning import time_of_day


def test_midnight_hour_is_after_midnight():
    result = time_of_day(pd.Series([0]))

    assert result.tolist() == ["after_midnight"]
