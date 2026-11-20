import numpy as np
import pytest

from gsim.include.utils.statistics import mean_and_ci, mean_and_ci_clt


def test_mean_and_ci_clt_unit_weights_matches_unweighted():
    """mean_and_ci_clt with unit weights must reduce exactly to the
    unweighted case."""
    l_vals = [1.0, 2.0, 3.0, 10.0]
    mean_unweighted, ci_unweighted = mean_and_ci_clt(l_vals)
    mean_weighted, ci_weighted = mean_and_ci_clt(l_vals, weights=[1.0] * 4)
    assert np.isclose(mean_unweighted, mean_weighted)
    assert np.allclose(ci_unweighted, ci_weighted)


def test_mean_and_ci_clt_weighted_mean():
    l_vals = [1.0, 2.0, 3.0, 4.0]
    l_weights = [1.0, 1.0, 1.0, 7.0]
    mean, (lower, upper) = mean_and_ci_clt(l_vals, weights=l_weights)
    assert np.isclose(mean, 3.4)
    assert lower < mean < upper
    assert np.isclose(mean - lower, upper - mean)


def test_mean_and_ci_is_deprecated_and_returns_the_half_width():
    l_vals = [1.0, 2.0, 3.0, 10.0]
    mean_clt, (lower, upper) = mean_and_ci_clt(l_vals)
    with pytest.warns(DeprecationWarning):
        mean, hci = mean_and_ci(l_vals)
    assert np.isclose(mean, mean_clt)
    assert np.isclose(hci, (upper - lower) / 2)
