import numpy as np
from typing import Tuple
import logging
from scipy.stats import t

gsim_logger = logging.getLogger("gsim")


def mean_and_ci(
    values,
    alpha: float = 0.05,
    weights=None,
) -> Tuple[float, float]:
    """
    Compute the sample mean and half-width of the confidence interval (CI) for
    the mean.

    Uses Student-t if SciPy is available; otherwise falls back to a Gaussian
    (CLT) approximation using NumPy only.

    Args:

        values: array-like. Each entry is a sample of a random variable.

        alpha : float. Significance level (e.g. 0.05 for 95% CI)

        weights: array-like of the same length as `values` or None. If
        provided, the weighted mean

            sum_i weights[i] * values[i] / sum_i weights[i]

        is computed instead of the plain mean. The weighted mean is a ratio
        estimator; its CI half-width is obtained via the delta-method
        (Taylor) linearization of the ratio, treating the pairs (values[i],
        weights[i]) as i.i.d. With unit weights, this reduces exactly to the
        unweighted case.

    Returns:

    sample_mean : float

    half_width : float
        Half-width of the (1 - alpha) confidence interval. This means that
        Prob(|sample_mean - true_mean| < half_width) = 1 - alpha.
    """
    values = np.asarray(values, dtype=float)
    num_values = values.size

    if num_values < 2:
        raise ValueError("At least two samples are required")

    crit = t.ppf(1.0 - alpha / 2.0, df=num_values - 1)

    if weights is None:
        mean = values.mean()
        std = values.std(ddof=1)
        half_width = crit * std / np.sqrt(num_values)
        return mean, half_width
    else:

        weights = np.asarray(weights, dtype=float)
        if weights.shape != values.shape:
            raise ValueError("weights must have the same shape as values")
        sum_weights = weights.sum()
        if sum_weights <= 0:
            raise ValueError(
                "The weights must not sum to a non-positive value")

        mean = float(np.sum(weights * values) / sum_weights)
        # Delta-method linearization of the ratio estimator: the i-th linearized
        # residual is (w_i * l_i - mean * w_i) / mean(w).
        v_residuals = (weights * values - mean * weights) / weights.mean()
        std = v_residuals.std(ddof=1)
        half_width = crit * std / np.sqrt(num_values)

        return mean, half_width
