import numpy as np
from typing import Tuple
import logging
import warnings
from scipy.stats import t

gsim_logger = logging.getLogger("gsim")


def mean_and_ci_clt(
    values,
    weights=None,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute the sample mean and the confidence interval (CI) for the mean.

    The CI is centered at the sample mean and its half-width is

        t_{1 - alpha/2, N - 1} * s / sqrt(N),

    where N is the number of values, s is their sample standard deviation,
    and t_{1 - alpha/2, N - 1} is the (1 - alpha/2)-quantile of the Student-t
    distribution with N - 1 degrees of freedom.

    Assumptions: the entries of `values` (or, if `weights` is provided, the
    pairs (values[i], weights[i])) are independent and identically
    distributed (i.i.d.). If they are not, e.g. because they are correlated,
    the returned CI may be wrong, typically too narrow. For non-Gaussian
    values, which is the general case, the CI relies on the central limit
    theorem: it is approximate and improves as the number of values grows.
    With weights, the delta-method linearization (see below) adds further
    approximation error. The CI is exact if the values are i.i.d. Gaussian
    and no weights are provided.

    Args:

        values: array-like. Each entry is a sample of a random variable.

        weights: array-like of the same length as `values` or None. If
        provided, the weighted mean

            sum_i weights[i] * values[i] / sum_i weights[i]

        is computed instead of the plain mean. The weighted mean is a ratio
        estimator; its CI half-width is obtained via the delta-method
        (Taylor) linearization of the ratio. With unit weights, this reduces
        exactly to the unweighted case.

        alpha : float. Significance level (e.g. 0.05 for a 95% CI).

    Returns:

    sample_mean : float

    (lower, upper) : tuple of floats
        Bounds of the (1 - alpha) CI. Under the assumptions above,
        Prob(lower < true_mean < upper) = 1 - alpha (approximately, unless the
        CI is exact).
    """
    values = np.asarray(values, dtype=float)
    num_values = values.size

    if num_values < 2:
        raise ValueError("At least two samples are required")

    crit = t.ppf(1.0 - alpha / 2.0, df=num_values - 1)

    if weights is None:
        mean = float(values.mean())
        std = values.std(ddof=1)
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

    half_width = float(crit * std / np.sqrt(num_values))
    return mean, (mean - half_width, mean + half_width)


def mean_and_ci(
    values,
    alpha: float = 0.05,
    weights=None,
) -> Tuple[float, float]:
    """
    Deprecated: use `mean_and_ci_clt`, which returns the CI as a tuple
    (lower, upper).

    Returns the sample mean and the half-width of the CI of
    `mean_and_ci_clt`, i.e. half the difference between its endpoints.
    """
    warnings.warn(
        "`mean_and_ci` is deprecated; use `mean_and_ci_clt`, which returns "
        "the CI as a tuple (lower, upper).",
        DeprecationWarning,
        stacklevel=2)
    mean, (lower, upper) = mean_and_ci_clt(values, weights=weights, alpha=alpha)
    return mean, (upper - lower) / 2
