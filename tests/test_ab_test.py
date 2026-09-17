import numpy as np

from swiggy_delivery.ab_test import compare_model_errors


def test_compare_model_errors_picks_the_model_with_consistently_lower_error():
    rng = np.random.default_rng(0)
    errors_a = rng.uniform(1, 5, size=50)
    errors_b = errors_a + rng.uniform(2, 4, size=50)  # b is consistently worse

    result = compare_model_errors(errors_a, errors_b, model_a_name="a", model_b_name="b")

    assert result["significant"] is True
    assert result["better_model"] == "a"
    assert result["p_value"] < 0.05


def test_compare_model_errors_reports_no_winner_when_difference_is_not_significant():
    rng = np.random.default_rng(1)
    errors_a = rng.uniform(1, 5, size=10)
    errors_b = rng.uniform(1, 5, size=10)

    result = compare_model_errors(errors_a, errors_b)

    assert result["significant"] is False
    assert result["better_model"] is None
