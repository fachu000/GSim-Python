import numpy as np
from typing import Tuple
import logging
from scipy.stats import t

gsim_logger = logging.getLogger("gsim")


def mean_and_ci(
    values,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute the sample mean and half-width of the confidence interval (CI) for
    the mean.

    Uses Student-t if SciPy is available; otherwise falls back to a Gaussian
    (CLT) approximation using NumPy only.

    Args:
        
        values: array-like. Each entry is a sample of a random variable.
        
        alpha : float. Significance level (e.g. 0.05 for 95% CI)

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

    mean = values.mean()
    std = values.std(ddof=1)

    crit = t.ppf(1.0 - alpha / 2.0, df=num_values - 1)
    half_width = crit * std / np.sqrt(num_values)

    return mean, half_width
