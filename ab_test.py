import numpy as np
from scipy import stats


def compare_model_errors(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    alpha: float = 0.05,
) -> dict:
    """Paired significance test on two models' per-row absolute errors.

    Uses the Wilcoxon signed-rank test: non-parametric, appropriate for
    paired error samples that are unlikely to be normally distributed.
    """
    _, p_value = stats.wilcoxon(errors_a, errors_b)
    p_value = float(p_value)
    significant = bool(p_value < alpha)

    mean_error_a = float(errors_a.mean())
    mean_error_b = float(errors_b.mean())
    better_model = None
    if significant:
        better_model = model_a_name if mean_error_a < mean_error_b else model_b_name

    return {
        "mean_error_a": mean_error_a,
        "mean_error_b": mean_error_b,
        "p_value": p_value,
        "significant": significant,
        "better_model": better_model,
    }
