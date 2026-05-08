import numpy as np

from gsim.gfigure import GFigure, hist_bin_edges_to_xy


def test_hist_bin_edges_to_xy_uses_counts_when_density_false():
    hist = np.array([2, 4])
    bin_edges = np.array([0.0, 1.0, 3.0])

    _, yaxis = hist_bin_edges_to_xy(hist, bin_edges, density=False)

    assert np.array_equal(yaxis, np.array([0.0, 2.0, 2.0, 4.0, 4.0, 0.0]))


def test_hist_bin_edges_to_xy_normalizes_when_density_true():
    hist = np.array([2, 4])
    bin_edges = np.array([0.0, 1.0, 3.0])

    _, yaxis = hist_bin_edges_to_xy(hist, bin_edges, density=True)

    assert np.allclose(
        yaxis, np.array([0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
                         0.0]))
