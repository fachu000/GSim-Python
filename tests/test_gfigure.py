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


# ---------------------------------------------------------------------------
# Broadcasting of xaxis, yaxis, ylower, yupper, and whiskers (see
# GFigure.__init__) and GFigure.add_whiskers_curve
# ---------------------------------------------------------------------------

import pickle

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gsim.gfigure import WhiskerSpec

L_WHISKERS = [
    WhiskerSpec(0., 1., 2., 3., 4.),
    WhiskerSpec(1., 2., 3., 4., 5.),
    WhiskerSpec(2., 3., 4., 5., 6.)
]


def _curves(G):
    return G.l_subplots[0].l_curves


def _only_curve(G):
    l_curves = _curves(G)
    assert len(l_curves) == 1
    return l_curves[0]


# --- one curve


def test_one_curve_default_xaxis():
    curve = _only_curve(GFigure(yaxis=[10., 20., 30.]))
    assert curve.xaxis == [0, 1, 2]
    assert curve.yaxis == [10., 20., 30.]
    assert curve.whiskers is None


def test_one_curve_with_bands_and_whiskers():
    G = GFigure(xaxis=[1., 3., 5.],
                yaxis=[10., 20., 30.],
                ylower=[9., 19., 29.],
                yupper=np.array([11., 21., 31.]),
                whiskers=L_WHISKERS)
    curve = _only_curve(G)
    assert curve.xaxis == [1., 3., 5.]
    assert curve.ylower == [9., 19., 29.]
    assert curve.yupper == [11., 21., 31.]
    assert curve.whiskers == L_WHISKERS


def test_whiskers_only_default_xaxis():
    curve = _only_curve(GFigure(whiskers=L_WHISKERS, legend="w"))
    assert curve.yaxis is None
    assert len(curve) == 3
    assert curve.xaxis == [0, 1, 2]
    assert curve.whiskers == L_WHISKERS
    assert curve.legend_str == "w"


def test_bands_only():
    curve = _only_curve(
        GFigure(xaxis=np.array([2., 4.]), ylower=[0., 1.], yupper=[1., 2.]))
    assert curve.yaxis is None
    assert curve.xaxis == [2., 4.]
    assert curve.ylower == [0., 1.]


def test_no_curve_when_nothing_given():
    assert GFigure().l_subplots == []
    assert _curves(GFigure(xlabel="x")) == []


def test_errors_one_curve():
    with pytest.raises(ValueError):
        GFigure(xaxis=[0, 1, 2])  # nothing to draw
    with pytest.raises(ValueError):
        GFigure(ylower=[0., 1.])  # a band without yaxis needs yupper
    with pytest.raises(ValueError):
        GFigure(yaxis=[1., 2.], whiskers=L_WHISKERS)  # 2 vs 3 points
    with pytest.raises(ValueError):
        GFigure(xaxis=[0, 1], whiskers=L_WHISKERS)
    with pytest.raises(ValueError):
        GFigure(yaxis=[1., 2., 3.], ylower=[0., 1.])
    with pytest.raises(TypeError):
        GFigure(yaxis=[1., 2., 3.], whiskers=[(0., 1., 2., 3., 4.)] * 3)


# --- several curves


def test_several_curves_list_of_lists_ragged_default_xaxis():
    G = GFigure(yaxis=[[1., 2., 3.], [4., 5.]])
    l_curves = _curves(G)
    assert len(l_curves) == 2
    assert l_curves[0].xaxis == [0, 1, 2]
    assert l_curves[1].xaxis == [0, 1]
    assert l_curves[1].yaxis == [4., 5.]


def test_several_curves_2d_array_shared_xaxis():
    m_y = np.arange(6.).reshape(2, 3)
    G = GFigure(xaxis=[10., 20., 30.], yaxis=m_y)
    l_curves = _curves(G)
    assert len(l_curves) == 2
    assert all(curve.xaxis == [10., 20., 30.] for curve in l_curves)
    assert l_curves[1].yaxis == [3., 4., 5.]


def test_several_curves_2d_xaxis():
    m_x = np.array([[0., 1.], [10., 11.]])
    G = GFigure(xaxis=m_x, yaxis=[[1., 2.], [3., 4.]])
    l_curves = _curves(G)
    assert l_curves[0].xaxis == [0., 1.]
    assert l_curves[1].xaxis == [10., 11.]


def test_1d_yaxis_broadcast_to_2d_xaxis():
    G = GFigure(xaxis=[[0., 1.], [10., 11.]], yaxis=[5., 6.])
    l_curves = _curves(G)
    assert len(l_curves) == 2
    assert all(curve.yaxis == [5., 6.] for curve in l_curves)


def test_1d_bands_and_whiskers_broadcast_to_several_curves():
    G = GFigure(yaxis=[[1., 2., 3.], [4., 5., 6.]],
                ylower=[0., 0., 0.],
                whiskers=L_WHISKERS)
    l_curves = _curves(G)
    assert len(l_curves) == 2
    assert all(curve.ylower == [0., 0., 0.] for curve in l_curves)
    assert all(curve.whiskers == L_WHISKERS for curve in l_curves)


def test_2d_whiskers_one_list_per_curve():
    l_whiskers_2 = [
        WhiskerSpec(w.low_whisker + 10, w.box_bottom + 10, w.line_inside + 10,
                    w.box_top + 10, w.high_whisker + 10) for w in L_WHISKERS
    ]
    G = GFigure(yaxis=[[1., 2., 3.], [4., 5., 6.]],
                whiskers=[L_WHISKERS, l_whiskers_2])
    l_curves = _curves(G)
    assert l_curves[0].whiskers == L_WHISKERS
    assert l_curves[1].whiskers == l_whiskers_2


def test_whiskers_only_several_curves():
    G = GFigure(whiskers=[L_WHISKERS, L_WHISKERS], legend=["a", "b"])
    l_curves = _curves(G)
    assert len(l_curves) == 2
    assert all(curve.yaxis is None for curve in l_curves)
    assert [curve.legend_str for curve in l_curves] == ["a", "b"]


def test_errors_several_curves():
    with pytest.raises(ValueError):  # 2 vs 3 curves
        GFigure(yaxis=[[1., 2.], [3., 4.]], ylower=[[0., 0.]] * 3)
    with pytest.raises(ValueError):  # 1D broadcast to ragged curves
        GFigure(yaxis=[[1., 2., 3.], [4., 5.]], ylower=[0., 0., 0.])
    with pytest.raises(ValueError):  # per-curve length mismatch
        GFigure(yaxis=[[1., 2., 3.], [4., 5.]],
                whiskers=[L_WHISKERS, L_WHISKERS])


def test_add_curve_appends_with_whiskers():
    G = GFigure(xaxis=[0, 1], yaxis=[1., 2.])
    G.add_curve(xaxis=[0, 1, 2], yaxis=[1., 2., 3.], whiskers=L_WHISKERS)
    l_curves = _curves(G)
    assert len(l_curves) == 2
    assert l_curves[0].whiskers is None
    assert l_curves[1].whiskers == L_WHISKERS


# --- plotting


def _lines_by_color(axes):
    d_lines = {}
    for line in axes.lines:
        d_lines.setdefault(line.get_color(), []).append(line)
    return d_lines


def test_plot_whiskers_one_box_per_point_in_curve_color():
    G = GFigure(xaxis=[0., 1., 2.],
                yaxis=[2., 3., 4.],
                whiskers=L_WHISKERS,
                styles='o#ff0000')
    G.plot()
    d_lines = _lines_by_color(plt.gca())
    # The curve line plus, per box, the box, two whiskers, two caps and the
    # median line.
    assert list(d_lines.keys()) == ['#ff0000']
    assert len(d_lines['#ff0000']) == 1 + 6 * len(L_WHISKERS)
    plt.close('all')


def test_plot_whiskers_of_each_curve_in_its_color():
    G = GFigure(yaxis=[[1., 2., 3.], [4., 5., 6.]],
                whiskers=[L_WHISKERS, L_WHISKERS])
    G.plot()
    d_lines = _lines_by_color(plt.gca())
    assert len(d_lines) == 2
    for l_lines in d_lines.values():
        assert len(l_lines) == 1 + 6 * len(L_WHISKERS)
    plt.close('all')


def test_plot_whiskers_only_and_bands_only():
    G = GFigure(whiskers=L_WHISKERS)
    G.add_curve(xaxis=[0, 1, 2], ylower=[0., 0., 0.], yupper=[1., 1., 1.])
    G.plot()
    axes = plt.gca()
    assert len(axes.lines) == 2 + 6 * len(L_WHISKERS)
    assert len(axes.collections) == 1  # the band
    assert len(_lines_by_color(axes)) == 2
    plt.close('all')


def test_curve_without_xaxis_and_yaxis_plots():
    # A `Curve` built directly, without the broadcasting of `Subplot`, may
    # have `xaxis=None`.
    from gsim.gfigure import Curve
    curve = Curve(yaxis=None, whiskers=L_WHISKERS)
    assert len(curve) == 3
    curve.plot()
    assert len(plt.gca().lines) == 1 + 6 * len(L_WHISKERS)
    plt.close('all')
    curve = Curve(yaxis=None, ylower=[0., 1.], yupper=[1., 2.])
    assert len(curve) == 2
    curve.plot()
    assert len(plt.gca().collections) == 1
    plt.close('all')


def test_len_of_3d_curve_is_number_of_entries_of_zaxis():
    G = GFigure(zaxis=np.zeros((2, 3)))
    assert len(_only_curve(G)) == 6


def test_whiskers_survive_pickling():
    G = GFigure(yaxis=[1., 2., 3.], whiskers=L_WHISKERS)
    G2 = pickle.loads(pickle.dumps(G))
    assert _only_curve(G2).whiskers == L_WHISKERS


# --- add_whiskers_curve


def _default_stats(v):
    return WhiskerSpec(np.min(v), np.percentile(v, 25), np.median(v),
                       np.percentile(v, 75), np.max(v))


def test_add_whiskers_curve_broadcasts_1d_data_to_one_box():
    v_samples = np.arange(10.)
    G = GFigure()
    G.add_whiskers_curve(v_samples)
    curve = _only_curve(G)
    assert curve.yaxis is None
    assert curve.xaxis == [0]
    assert curve.whiskers == [_default_stats(v_samples)]

    G = GFigure()
    G.add_whiskers_curve(list(v_samples))  # a list of scalars
    assert _only_curve(G).whiskers == [_default_stats(v_samples)]


def test_add_whiskers_curve_2d_array_one_box_per_row():
    m_samples = np.random.randn(3, 10)
    G = GFigure()
    G.add_whiskers_curve(m_samples, xaxis=[5., 6., 7.])
    curve = _only_curve(G)
    assert curve.xaxis == [5., 6., 7.]
    assert curve.whiskers == [_default_stats(v_row) for v_row in m_samples]


def test_add_whiskers_curve_list_of_lists_of_different_lengths():
    l_samples = [[1., 2., 3.], [4., 5.], [6., 7., 8., 9.]]
    G = GFigure()
    G.add_whiskers_curve(l_samples)
    curve = _only_curve(G)
    assert curve.xaxis == [0, 1, 2]
    assert curve.whiskers == [
        _default_stats(np.array(l_row)) for l_row in l_samples
    ]


def test_add_whiskers_curve_custom_stats_and_yaxis():

    def f_stats(v):
        m, s = np.mean(v), np.std(v)
        return WhiskerSpec(m - 2 * s, m - s, m, m + s, m + 2 * s)

    m_samples = np.random.randn(2, 20)
    G = GFigure()
    G.add_whiskers_curve(m_samples,
                         yaxis=[1., 2.],
                         f_stats=f_stats,
                         legend="mean +- k std")
    curve = _only_curve(G)
    assert curve.yaxis == [1., 2.]
    assert curve.legend_str == "mean +- k std"
    assert curve.whiskers == [f_stats(v_row) for v_row in m_samples]


def test_add_whiskers_curve_wrong_xaxis_length():
    with pytest.raises(ValueError):
        GFigure().add_whiskers_curve(np.random.randn(3, 10), xaxis=[0, 1])
