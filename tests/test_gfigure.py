import matplotlib

matplotlib.use("Agg")

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


def test_gfigure_supports_secondary_y_axis():
    gfig = GFigure(xlabel="x", ylabel="left", secondary_ylabel="right")
    gfig.add_curve([0, 1, 2], [0, 1, 0], legend="left", styles="b-")
    gfig.add_curve([0, 1, 2], [10, 20, 30],
                   legend="right",
                   styles="r--",
                   secondary_y=True)

    gfig.plot()

    subplot = gfig.l_subplots[0]
    assert subplot.secondary_axes is not None
    assert subplot.axes.get_ylabel() == "left"
    assert subplot.secondary_axes.get_ylabel() == "right"
    assert len(subplot.axes.lines) == 1
    assert len(subplot.secondary_axes.lines) == 1
