"""
This module provides a layer over matplotlib that is used to construct and plot
figure representations. It affords an intuitive and compact syntax that allows
one to store and edit figures.  

The main idea is that a figure is a collection of subplots, and each subplot is
a collection of curves. Each curve can be 2D or 3D. 

The easiest way to learn how to use this module is to run the examples at the
end of this file. To do so, cd to the folder `gsim` and type:

python3 gfigure.py <figure_number>

where <figure_number> is an integer. See the code and possible values of
<figure_number> in `plot_example_figure` below.

The reference documentation of the arguments of GFigure and its functions
follows.

FIGURE
======

figsize: can be a tuple of format (width, height), e.g. (20., 10.). If
    None and the global `default_figsize` is not None, the value of the latter
    is used.

`layout`: can be "", "tight", or "constrained". See pyplot documentation.

        Since April 2022, layout='tight' is set by default.

One of `num_subplot_rows` or `num_subplot_columns` can be specified for figures
with multiple subplots. 

SUBPLOT ARGUMENTS:
=================

The first set of arguments allow the user to create a subplot when creating the
GFigure object.

title : str 

xlabel : str

ylabel : str

grid : bool

xlim : tuple, endpoints for the x axis.

ylim : 
    - tuple, endpoints for the y axis.

    - float: then the y-limits are set to (y_min - ylim * delta , y_max + ylim *
      delta), where delta = y_max - y_min, and y_min and y_max are the minimum
      and maximum values of the y-axis data to be plotted in the subplot. This
      is useful because matplotlib disables y-axis autoscaling when `xlim` is
      provided. 

zlim : tuple, endpoints for the z axis. Used e.g. for the color scale. 

logx: bool, whether to use a logarithmic scale for the x-axis.

logy: bool, whether to use a logarithmic scale for the y-axis.

yticks: None or 1D array like. If None, the default ticks are used. If 1D array
like, it specifies the ticks. yticks can be set to an empty list for no ticks.

legend_loc: str, it indicates the location of the legend. Example values:
    "lower left", "upper right", etc.

num_legend_cols: int, number of columns in the legend.

sharex: Set to true so that the x-axis is shared with the previous subplot. 

transpose_subplots: if True, the second subplot is placed at position (1,0), the
third at (2,0), etc. 

CURVE ARGUMENTS:
=================

1. 2D plots
    -----------

xaxis, yaxis, ylower, yupper, and whiskers:

These arguments specify one or more curves and their decorations. They are
broadcast with the same rule:

    - The last dimension of each of them is the number of points of the curve.
      When two or more are given, the length of their last dimension must
      coincide (curve by curve if there are several curves).

    - To specify one curve, these args must be 1D: a list of a numeric type, a
      1D np.ndarray, or a 1D tf.Tensor (a list of `WhiskerSpec` for `whiskers`).

    - To specify M curves, at least one of these args must be 2D: a list of M
      entries with the format of the 1D case (a different length per curve is
      OK), or an M x N np.ndarray or tf.Tensor, where each row corresponds to a
      curve. A 1D argument is broadcast to all the curves (e.g. a 1D `xaxis` is
      shared by all curves).

    - `xaxis` can also be None or [], in which case the x-axis of every curve is
      0, 1, ..., N-1.

    - `yaxis` can also be None: only the whiskers/bands are drawn. At least one
      of `yaxis`, `ylower`, `yupper`, and `whiskers` must be given (unless
      `xaxis` is also empty, in which case no curve is added).

ylower and yupper: specify a shaded area around the curve, used e.g. for
confidence bounds. The area between ylower and yaxis as well as the area between
yaxis and yupper are shaded. If `yaxis` is None, the area between ylower and
yupper is shaded, and both must be given.

whiskers: `WhiskerSpec` objects, one per point of the curve, drawn at `xaxis[n]`
as a box-and-whisker plot in the color of the curve (typically minimum, lower
quartile, median, upper quartile, and maximum of some samples; see
`GFigure.add_whiskers_curve` to obtain them from samples).

zaxis: None

mode: it can be 'plot' (default) or 'stem'

2. 3D plots
-----------

2a. Axes
--------

zaxis: M x N numpy array. When `mode` is 'imshow', the bottom left of the matrix
corresponds to the bottom left of the figure. 

There are 3 options:

- xaxis and yaxis are M x N numpy arrays. The (x,y) coordinates
    corresponding to zaxis[i,j] are xaxis[i,j] and yaxis[i,j].

- xaxis and yaxis are vectors of length N and M, respectively. The (x,y)
    coordinates corresponding to zaxis[i,j] are xaxis[j] and yaxis[i]. This is
    useful e.g. when we want the matrix to provide the values of a function on
    the first quadrant, where the bottom-left entry of the matrix would
    correspond to the origin and yaxis is thought of as a column vector whose
    bottom entry provides the y-coordinate of the origin. 

- xaxis and yaxis are None or []. In this case, it is understood that
    the user wants to visualize the entries of a matrix. Thus, the (x,y)
    coordinates corresponding to zaxis[i,j] are respectively j and i. Arguments
    xlabel and ylabel respectively correspond to columns and rows. 


2b. Rest of arguments
---------------------

mode: it can be 'imshow' (default), 'contour3D', or 'surface'.

zinterpolation: Supported values are 'none', 'antialiased', 'nearest',
'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
'lanczos'.

color_bar: If True, a color bar is created for the specified axis.

global_color_bar: if True, one color bar for the entire figure. 

global_color_bar_label: str indicating the label of the global color bar.

global_color_bar_position: vector with four entries.

aspect: can be 'square' or take the values in plt.imshow. It applies only to
imshow. 

3. Others
    ---------

styles: specifies the style argument to plot, similarly to MATLAB.
Possibilities:
    
    - str : this style is applied to all curves specified by `xaxis` and
    `yaxis`. It is a concatenation of the following items: 

        * marker style (e.g. '.','o','x')

        * line style (e.g. '-','--','-.')

        * color. The color can be:
            
            - a letter, as in MATLAB (e.g. 'k', 'b',
            'r') 
            
            - an hexadecimal number of the form  "#??????", where ?
            denotes an hexadecimal digit (e.g. '#2244FF'). The curve style needs
            to precede the color specification, e.g. 'o--#2244FF'.

            - the hash symbol followed by a natural number, e.g. '#3'. In this
            case, the number indicates the index of the color in the default
            matplotlib color cycle. The curve style needs to precede the color
            specification, e.g. 'o--#3'.

    - list of str : then styles[n] is applied to the n-th
    curve. Its length must be at least the number of curves.

legend : str, tuple of str, or list of str. If the str begins with "_",
    then that curve is not included in the legend.


ARGUMENTS FOR SPECIFYING HOW TO SUBPLOT:
========================================


`ind_active_subplot`: The index of the subplot that is created and where
    new curves will be added until a different value for the property of GFigure
    with the same name is specified. A value of 0 refers to the first subplot.

`num_subplot_rows` and `num_subplot_columns` determine the number of
    subplots in each column and row respectively. If None, their value is
    determined by the value of the other of these parameters and the number of
    specified subplots. If the number of specified subplots does not equal
    num_subplot_columns*num_subplot_rows, then the value of num_subplot_columns
    is determined from the number of subplots and num_subplot_rows.

    The values of the properties of GFigure with the same name can be specified
    subsequently.


"""

from collections.abc import Callable
import copy
import logging
import sys

from matplotlib.axes import Axes
from matplotlib.backend_bases import TimerBase
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

title_to_caption = False
default_figsize = None  # `None` lets plt choose

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
gsim_logger = logging.getLogger("gsim")
""" 
TODO: 

Replace lists of a numeric type in xaxis or yaxis with numpy
arrays. With lists it gets messy when using 3D plots. 

"""


def inspect_hist(data, hist_args={}):
    G = GFigure()
    G.add_histogram_curve(data, hist_args=hist_args)
    G.plot()


def hist_bin_edges_to_xy(hist, bin_edges, density=False):
    """Return step-plot coordinates for a histogram.

    If `density` is True, the y-axis values represent a density estimate.
    Otherwise, the y-axis values are the raw counts returned by
    `np.histogram`.
    """

    def duplicate_entries(v_in):
        """ If v_in = [v1,v2,...vN], this function returns [v1, v1, v2, v2, ..., vN, vN]."""
        return np.ravel(np.tile(v_in, (2, 1)).T)

    if density:
        v_bin_widths = bin_edges[1:] - bin_edges[:-1]
        v_y_interior = hist / np.sum(hist) / v_bin_widths
    else:
        v_y_interior = hist

    v_x = duplicate_entries(bin_edges)
    v_y = np.concatenate(([0], duplicate_entries(v_y_interior), [0]))
    return v_x, v_y


def is_number(num):
    #return isinstance(num, (int, float, complex, bool))
    # From https://stackoverflow.com/questions/500328/identifying-numeric-and-array-types-in-numpy

    if hasattr(num, "numpy"):
        num = num.numpy()

    if isinstance(num, np.ndarray):
        if num.size != 1:
            return False

    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(num, attr) for attr in attrs)


class WhiskerSpec:
    """Values drawn by a box-and-whisker plot at one point of a curve; see
    the argument `whiskers` of GFigure.__init__. Typically, `low_whisker`,
    `box_bottom`, `line_inside`, `box_top`, and `high_whisker` are the
    minimum, the lower quartile, the median, the upper quartile, and the
    maximum of some samples."""

    def __init__(self, low_whisker, box_bottom, line_inside, box_top,
                 high_whisker):
        self.low_whisker = float(low_whisker)
        self.box_bottom = float(box_bottom)
        self.line_inside = float(line_inside)
        self.box_top = float(box_top)
        self.high_whisker = float(high_whisker)

    def as_tuple(self):
        return (self.low_whisker, self.box_bottom, self.line_inside,
                self.box_top, self.high_whisker)

    def __eq__(self, other):
        return isinstance(other,
                          WhiskerSpec) and self.as_tuple() == other.as_tuple()

    def __repr__(self):
        return f"WhiskerSpec{self.as_tuple()}"


class Curve:

    def __init__(self,
                 xaxis=None,
                 yaxis=[],
                 zaxis=None,
                 zinterpolation='none',
                 ylower=[],
                 yupper=[],
                 whiskers=None,
                 style=None,
                 mode=None,
                 legend_str="",
                 aspect=None):
        """

        See GFigure.__init__ for more information.

        1. For 2D plots:
        ---------------

        xaxis : None or a list of a numeric type. In the latter case, its length 
            equals the length of yaxis.

        yaxis : list of a numeric type, or None if at least one of
            `ylower`, `yupper`, and `whiskers` is given.

        zaxis : None

        ylower, yupper: [] or lists of a numeric type with the same length as
        yaxis.

        whiskers: None or a list of `WhiskerSpec` with the same length as
        `yaxis`; see GFigure.__init__.

        mode : can be 'plot' or 'stem'

        aspect: can be 'square' or take the values in plt.imshow. It applies only
        to imshow. 

        2. For 3D plots:
        ----------------

        xaxis: M x N numpy array

        yaxis: M x N numpy array

        zaxis: M x N numpy array

        zinterpolation: see GFigure.__init__

        Other arguments
        ---------------

        style : see the docstring of GFigure
        """

        # Input check
        if zaxis is None:
            # 2D plot
            def is_given(arg):
                return arg is not None and len(arg)

            if yaxis is None:
                if not (is_given(ylower) or is_given(yupper)
                        or is_given(whiskers)):
                    raise TypeError(
                        "`yaxis` can be None only if `ylower`, `yupper`, or "
                        "`whiskers` is given")
                if is_given(ylower) != is_given(yupper):
                    raise ValueError("If `yaxis` is None, `ylower` and "
                                     "`yupper` must be both given or none")
            elif type(yaxis) != list:
                raise TypeError("`yaxis` must be a list of numeric entries")
            if type(xaxis) == list:
                pass
            elif xaxis is not None:
                raise TypeError(
                    "`xaxis` must be a list of numeric entries or None")
            s_lengths = {
                len(arg)
                for arg in (xaxis, yaxis, ylower, yupper, whiskers)
                if is_given(arg)
            }
            if len(s_lengths) > 1:
                raise ValueError(
                    "`xaxis`, `yaxis`, `ylower`, `yupper`, and `whiskers` "
                    f"must have the same length; got {s_lengths}")
            if whiskers is not None and not all(
                    isinstance(w, WhiskerSpec) for w in whiskers):
                raise TypeError("`whiskers` must be a list of `WhiskerSpec`")
        else:
            # 3D plot

            # zaxis
            if not isinstance(zaxis, np.ndarray):
                raise TypeError(f"Argument `zaxis` must be of class np.array.")
            if zaxis.ndim != 2:
                raise ValueError(f"Argument `zaxis` must be of dimension 2. ")

            num_rows, num_cols = zaxis.shape

            # xaxis and yaxis
            def is_empty(arg):
                return (arg is None) or ((type(arg) == list) and
                                         (len(arg) == 0))

            if is_empty(xaxis):
                assert is_empty(
                    yaxis), "If `xaxis` is empty, then `yaxis` must be empty"
            else:
                if isinstance(xaxis, np.ndarray):
                    assert isinstance(
                        yaxis, np.ndarray
                    ), "If `xaxis` is an `np.ndarray`, then `yaxis` must be an `np.ndarray`."

                    # At this point, both are arrays. Just check their dimensions
                    if xaxis.ndim == 1:
                        assert yaxis.ndim == 1, "If `xaxis.ndim` is 1, then `yaxis.ndim` must also be 1."
                        assert xaxis.shape == (
                            num_cols,
                        ), f"If `xaxis.ndim` is 1, then `xaxis.shape` must be ({num_cols},). "
                        assert yaxis.shape == (
                            num_rows,
                        ), f"If `yaxis.ndim` is 1, then `yaxis.shape` must be ({num_rows},). "
                    elif xaxis.ndim == 2:
                        assert yaxis.ndim == 2, "If `xaxis.ndim` is 2, then `yaxis.ndim` must also be 2."
                        assert xaxis.shape == (
                            num_rows, num_cols
                        ), f"If `xaxis.ndim` is 2, then `xaxis.shape` must be ({num_rows},{num_cols}). "
                        assert yaxis.shape == (
                            num_rows, num_cols
                        ), f"If `yaxis.ndim` is 2, then `yaxis.shape` must be ({num_rows},{num_cols}). "
                    else:
                        raise ValueError("`xaxis.ndim` must be either 1 or 2.")
                else:
                    raise TypeError(
                        f"If `xaxis` is not empty, it must be of class `np.ndarray`."
                    )

        if (style is not None) and (type(style) != str):
            raise TypeError("`style` must be of type str or None")
        if type(legend_str) != str:
            raise TypeError("`legend_str` must be of type str")

        # Common
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.mode = mode
        self.line = None  # Handle to the plotted line

        # 2D
        self.ylower = ylower
        self.yupper = yupper
        self.whiskers = whiskers
        self.style = style
        self.legend_str = legend_str

        # 3D
        self.zaxis = zaxis
        self.zinterpolation = zinterpolation
        self.image = None
        self.aspect = aspect

    def __repr__(self):
        return f"<Curve: legend_str = {self.legend_str}, num_points = {len(self)}>"

    def plot(self, **kwargs):

        if self.is_3D:
            self._plot_3D(**kwargs)
        else:
            self._plot_2D()

    @staticmethod
    def _split_style(style):
        """Returns `(style_without_color, d_color_kwargs)`, where
        `d_color_kwargs` is `{'color': <hex color>}` if `style` contains a
        color specification after '#' (either 6 hex digits or the index of a
        color in the default color cycle) and `{}` otherwise."""
        color_spec = style.split("#")[1] if "#" in style else None
        if color_spec:
            if len(color_spec) == 6:
                hex_color = "#" + color_spec
            else:
                # The default color cycle of matplotlib contains just 10
                # colors. Consider extending this.
                plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                hex_color = plt_colors[int(color_spec) % len(plt_colors)]
            kwargs = {'color': hex_color}
        else:
            kwargs = dict()
        return style.split("#")[0], kwargs

    def _plot_2D(self):

        # Decorations of the curve. `getattr` for curves pickled before the
        # attribute existed.
        def get_decoration(name):
            decoration = getattr(self, name, None)
            return decoration if (decoration is not None
                                  and len(decoration)) else None

        ylower = get_decoration("ylower")
        yupper = get_decoration("yupper")
        whiskers = get_decoration("whiskers")

        if self.yaxis is None:
            # Decorations only: a curve of NaNs takes a color from the color
            # cycle and provides the legend entry, but draws nothing.
            yaxis = [np.nan] * len(self)
        else:
            yaxis = self.yaxis

        if ((type(self.xaxis) == list) or
            (type(self.xaxis) == np.ndarray)) and len(self.xaxis):
            axis_args = (self.xaxis, yaxis)
        else:
            axis_args = (yaxis, )

        style = self.style if self.style else "-"

        if hasattr(self, 'mode') and (self.mode is not None) and (self.mode
                                                                  == 'stem'):

            def plot_fun(*args, **kwargs):
                return plt.stem(*args, **kwargs)

            # stem does not take 'color' as an argument, but the color may be
            # specified through `style`
            container = plot_fun(*axis_args, style, label=self.legend_str)
            color = container.markerline.get_color()
        else:
            # Get the color from self.style if present
            style, kwargs = Curve._split_style(style)

            if hasattr(self, 'line') and (self.line is not None):
                # The curve has been plotted before. Update the data.
                if len(axis_args) == 1:
                    self.line.set_ydata(axis_args[0])
                elif len(axis_args) == 2:
                    self.line.set_data(axis_args[0], axis_args[1])
                else:
                    raise ValueError("Invalid number of axis arguments.")
                self.line.set_label(self.legend_str)
            else:
                # First time the curve is plotted.
                self.line, = plt.plot(*axis_args,
                                      style,
                                      label=self.legend_str,
                                      **kwargs)
            color = self.line.get_color()

        # The decorations are plotted after the curve so that they take its
        # color rather than consuming further colors of the color cycle.
        def plot_band(lower, upper):
            if len(axis_args) == 2:
                plt.fill_between(self.xaxis,
                                 lower,
                                 upper,
                                 alpha=0.2,
                                 color=color)
            else:
                plt.fill_between(range(len(lower)),
                                 lower,
                                 upper,
                                 alpha=0.2,
                                 color=color)

        if self.yaxis is None:
            if ylower is not None and yupper is not None:
                plot_band(ylower, yupper)
        else:
            if ylower is not None:
                plot_band(ylower, self.yaxis)
            if yupper is not None:
                plot_band(self.yaxis, yupper)

        if whiskers is not None:
            self._plot_whiskers(whiskers, color)

    def _plot_whiskers(self, whiskers, color):
        """Draws one box-and-whisker plot per entry of `whiskers` at the
        corresponding point of the x-axis, in the given color."""
        if ((type(self.xaxis) == list) or
            (type(self.xaxis) == np.ndarray)) and len(self.xaxis):
            positions = np.array(self.xaxis, dtype=float)
        else:
            positions = np.arange(len(whiskers), dtype=float)
        # The boxes are half as wide as the smallest distance between
        # consecutive positions.
        v_spacing = np.diff(np.unique(positions))
        width = 0.5 * np.min(v_spacing) if len(v_spacing) else 0.5
        l_stats = [{
            'whislo': w.low_whisker,
            'q1': w.box_bottom,
            'med': w.line_inside,
            'q3': w.box_top,
            'whishi': w.high_whisker
        } for w in whiskers]
        d_line_props = {'color': color}
        # `manage_ticks=False` leaves the ticks as they are (by default,
        # `bxp` places a tick with a label at every box).
        plt.gca().bxp(l_stats,
                      positions=positions,
                      widths=width,
                      showfliers=False,
                      manage_ticks=False,
                      boxprops=d_line_props,
                      whiskerprops=d_line_props,
                      capprops=d_line_props,
                      medianprops=d_line_props)

    def _plot_3D(self, axes=None, interpolation="none", zlim=None):

        assert axes

        # Default mode
        if not hasattr(self, 'mode') or (self.mode is None):
            self.mode = 'imshow'

        len_y, len_x = self.zaxis.shape

        # xaxis and yaxis
        if not isinstance(self.xaxis, np.ndarray):
            v_x = np.arange(len_x)
            v_y = np.arange(len_y)
            m_X, m_Y = np.meshgrid(v_x, v_y)
            m_Z = self.zaxis
        else:
            if (self.xaxis.ndim == 1):
                m_X, m_Y = np.meshgrid(self.xaxis, self.yaxis)
                m_Z = self.zaxis
            else:
                m_X, m_Y = self.xaxis, self.yaxis
                m_Z = self.zaxis

        if self.mode == 'imshow':
            aspect = (m_X[-1, -1] - m_X[-1, 0]) / (m_Y[-1, 0] - m_Y[0, 0]) if (
                hasattr(self, "aspect") and self.aspect == "square") else None
            self.image = axes.imshow(
                m_Z,
                interpolation=self.zinterpolation,
                cmap='jet',
                # origin='lower',
                extent=[m_X[-1, 0], m_X[-1, -1], m_Y[-1, 0], m_Y[0, 0]],
                vmax=zlim[1] if zlim else None,
                aspect=aspect,
                vmin=zlim[0] if zlim else None)
        elif self.mode == 'contour3D':
            self.image = axes.contour3D(m_X, m_Y, m_Z, 50, cmap='plasma')
            if zlim is not None:
                axes.set_zlim(zlim[0], zlim[1])
        elif self.mode == 'surface':
            self.image = axes.plot_surface(m_X,
                                           m_Y,
                                           m_Z,
                                           rstride=1,
                                           cstride=1,
                                           cmap='viridis',
                                           edgecolor='none')
            if zlim is not None:
                axes.set_zlim(zlim[0], zlim[1])
        else:
            raise ValueError(f'Unrecognized 3D plotting mode. Got {self.mode}')

    @staticmethod
    def legend_is_empty(l_curves):
        for curve in l_curves:
            if curve.legend_str != "" and curve.legend_str[0] != "_":
                return False
        return True

    @property
    def projection(self):
        """This is used to create the axes.
        
        Note that plt is not consistent. The projection mode can be '3d'
        (lowercase), but the function is called 'contour3D'. 
        """

        if self.is_3D:
            if hasattr(self, 'mode') and (self.mode == 'contour3D'
                                          or self.mode == 'surface'):
                return '3d'
        return None

    @property
    def is_3D(self):
        return hasattr(self, "zaxis") and self.zaxis is not None

    def __len__(self):
        """Number of points of the curve. For 3D curves, the number of
        entries of `zaxis`. For 2D curves, the length of `xaxis`, `yaxis`,
        or of any decoration (`ylower`, `yupper`, `whiskers`), whichever is
        given (all of them have the same length); 0 if none is given."""
        if self.is_3D:
            return self.zaxis.size
        for name in ("yaxis", "xaxis", "ylower", "yupper", "whiskers"):
            arg = getattr(self, name, None)
            if arg is not None and len(arg):
                return len(arg)
        return 0


class VerticalLinesCurve(Curve):

    def __init__(self,
                 subplot: 'Subplot',
                 x_positions,
                 legend_str="",
                 style="k--"):
        super().__init__(xaxis=[],
                         yaxis=[],
                         legend_str=legend_str,
                         style=style)
        self.subplot = subplot  # Required to determine the y limits
        self.x_positions = x_positions

    def plot(self, **kwargs):

        if len(self.x_positions) == 0:
            # This is to allow placeholders for vertical lines. This is
            # necessary with animations: sometimes there are no vertical lines
            # to plot initially, but they may appear later.
            self.xaxis = [np.nan]
            self.yaxis = [np.nan]
        else:
            # Ensure the y-limits of the subplot are defined
            if self.subplot.ylim is None:
                try:
                    self.subplot.ylim = self.subplot.get_auto_ylims(.2)
                except ValueError:
                    self.subplot.ylim = (0, 1) if not self.subplot.logy else (
                        1e-1, 1)
            ymin, ymax = self.subplot.ylim

            self.xaxis = []
            self.yaxis = []
            for x in self.x_positions:
                self.xaxis += [x, x, np.nan]
                self.yaxis += [ymin, ymax, np.nan]

        super()._plot_2D()


class Subplot:

    def __init__(self,
                 title="",
                 xlabel="",
                 ylabel="",
                 zlabel="",
                 color_bar=False,
                 grid=True,
                 xlim=None,
                 ylim=None,
                 zlim=None,
                 xticks=None,
                 xticklabels=None,
                 num_xticks_decimal_places=None,
                 yticks=None,
                 legend_loc=None,
                 create_curves=True,
                 num_legend_cols=1,
                 sharex=None,
                 logx=False,
                 logy=False,
                 **kwargs):
        """
      For a description of the arguments, see GFigure.__init__

      """

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.color_bar = color_bar
        self.grid = grid
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.xticks = xticks
        # Labels of the ticks in `xticks`, e.g. names of categories placed
        # at integer positions. Requires `xticks`.
        self.xticklabels = xticklabels
        self.num_xticks_decimal_places = num_xticks_decimal_places
        self.yticks = yticks
        self.legend_loc = legend_loc
        self.num_legend_cols = num_legend_cols
        self.l_curves: list[Curve] = []
        self.sharex = sharex
        self.logx = logx
        self.logy = logy
        self.axes: Axes | None = None  # Handle to the subplot axis
        if create_curves:
            self.add_curve(**kwargs)

    def __repr__(self):
        return f"<Subplot object with title=\"{self.title}\", len(self.l_curves)={len(self.l_curves)} curves>"

    def is_empty(self):

        return not any([self.title, self.xlabel, self.ylabel, self.l_curves])

    def update_properties(self, **kwargs):

        if "title" in kwargs:
            self.title = kwargs["title"]
        if "xlabel" in kwargs:
            self.xlabel = kwargs["xlabel"]
        if "ylabel" in kwargs:
            self.ylabel = kwargs["ylabel"]
        if "zlabel" in kwargs:
            self.ylabel = kwargs["zlabel"]

    def add_curve(self,
                  xaxis=[],
                  yaxis=[],
                  zaxis=None,
                  zinterpolation="bilinear",
                  ylower=[],
                  yupper=[],
                  whiskers=None,
                  styles=[],
                  mode=None,
                  legend=tuple(),
                  aspect=None):
        """
      Adds one or multiple curves to `self`. See documentation of GFigure.__init__
      """

        if zaxis is None:
            # 2D figure
            self.l_curves += Subplot._l_2D_curves_from_input_args(
                xaxis,
                yaxis,
                ylower,
                yupper,
                styles,
                legend,
                mode=mode,
                whiskers=whiskers)
        else:
            # 3D figure
            self.l_curves.append(
                Curve(xaxis=xaxis,
                      yaxis=yaxis,
                      zaxis=zaxis,
                      zinterpolation=zinterpolation,
                      mode=mode,
                      aspect=aspect))

    def add_vertical_lines(self, *args, **kwargs):
        self.l_curves.append(VerticalLinesCurve(subplot=self, *args, **kwargs))

    def _l_2D_curves_from_input_args(xaxis,
                                     yaxis,
                                     ylower,
                                     yupper,
                                     styles,
                                     legend,
                                     mode,
                                     whiskers=None):
        """Returns a list of `Curve` objects built from the arguments of
        `add_curve`, broadcast as described in GFigure.__init__."""

        l_num_pts = Subplot._infer_num_pts(xaxis, yaxis, ylower, yupper,
                                           whiskers)
        num_curves = len(l_num_pts)
        if num_curves == 0:
            return []

        l_xaxis = Subplot._broadcast_curve_arg(xaxis, l_num_pts)
        l_xaxis = [
            list(range(num_pts)) if xax is None else xax
            for xax, num_pts in zip(l_xaxis, l_num_pts)
        ]
        l_yaxis = Subplot._broadcast_curve_arg(yaxis, l_num_pts)
        l_ylower = Subplot._broadcast_curve_arg(ylower, l_num_pts)
        l_yupper = Subplot._broadcast_curve_arg(yupper, l_num_pts)
        l_whiskers = Subplot._broadcast_curve_arg(whiskers, l_num_pts)

        # Process style input.
        l_style = Subplot._list_from_style_argument(styles)
        if len(l_style) == 0:
            l_style = [None] * num_curves
        elif len(l_style) == 1:
            l_style = l_style * num_curves
        else:
            assert len(l_style) >= num_curves, (
                "The length of `style` must be either 1 or no less than the "
                "number of curves")
            l_style = l_style[0:num_curves]

        # Process the legend
        assert ((type(legend) == tuple) or (type(legend) == list)
                or (type(legend) == str))
        if type(legend) == str:
            legend = [legend] * num_curves
        else:  # legend is tuple or list
            if len(legend) == 0:
                legend = [""] * num_curves
            else:
                if type(legend[0]) != str:
                    raise TypeError(
                        "`legend` must be an str, list of str, or tuple of str."
                    )
                if (len(legend) != num_curves):
                    raise ValueError(
                        f"len(legend)={len(legend)} should equal 0 or the "
                        f"number of curves={num_curves}")

        # Construct Curve objects
        l_curve = []
        for xax, yax, ylow, yup, whisk, stl, leg in zip(
                l_xaxis, l_yaxis, l_ylower, l_yupper, l_whiskers, l_style,
                legend):
            l_curve.append(
                Curve(xaxis=xax,
                      yaxis=yax,
                      ylower=[] if ylow is None else ylow,
                      yupper=[] if yup is None else yup,
                      whiskers=whisk,
                      style=stl,
                      legend_str=leg,
                      mode=mode))
        return l_curve

    @staticmethod
    def _unify_curve_arg(arg):
        """Returns `arg` (one of `xaxis`, `yaxis`, `ylower`, `yupper`, or
        `whiskers`; see GFigure.__init__) as None if it is None or empty, or
        as a list of lists otherwise: one inner list per curve, with the
        values (floats, or `WhiskerSpec` objects) of that curve."""

        def is_element(entry):
            return np.isscalar(entry) or isinstance(entry, WhiskerSpec)

        if hasattr(arg, "numpy"):  # Compatibility with TensorFlow
            arg = arg.numpy()
        if arg is None:
            return None
        if isinstance(arg, np.ndarray):
            if arg.size == 0:
                return None
            if arg.ndim == 1:
                return [[float(v) for v in arg]]
            if arg.ndim == 2:
                return [[float(v) for v in row] for row in arg]
            raise ValueError("Input arrays need to be of dimension 1 or 2")
        if isinstance(arg, (list, tuple)):
            if len(arg) == 0:
                return None
            if is_element(arg[0]):
                # 1D: a single curve
                if not all(is_element(entry) for entry in arg):
                    raise TypeError(
                        "The entries of a 1D argument must all be numbers "
                        "or `WhiskerSpec` objects")
                return [[
                    entry if isinstance(entry, WhiskerSpec) else float(entry)
                    for entry in arg
                ]]
            # 2D: one entry per curve
            ll_out = []
            for entry in arg:
                l_entry = Subplot._unify_curve_arg(entry)
                if l_entry is None:
                    l_entry = [[]]
                if len(l_entry) != 1:
                    raise ValueError(
                        "A 2D argument must be a list of 1D entries")
                ll_out.append(l_entry[0])
            return ll_out
        raise TypeError(
            "`xaxis`, `yaxis`, `ylower`, `yupper`, and `whiskers` must be "
            "None, lists, np.ndarrays, or tf.Tensors")

    @staticmethod
    def _infer_num_pts(xaxis, yaxis, ylower, yupper, whiskers):
        """Returns a list with the number of points of each curve specified
        by the arguments (the length of the list is the number of curves);
        see GFigure.__init__. It is empty if no curve is specified."""
        d_args = {
            'xaxis': Subplot._unify_curve_arg(xaxis),
            'yaxis': Subplot._unify_curve_arg(yaxis),
            'ylower': Subplot._unify_curve_arg(ylower),
            'yupper': Subplot._unify_curve_arg(yupper),
            'whiskers': Subplot._unify_curve_arg(whiskers),
        }
        d_given = {name: ll for name, ll in d_args.items() if ll is not None}
        if not d_given:
            return []
        if 'whiskers' in d_given and not all(
                isinstance(w, WhiskerSpec) for row in d_given['whiskers']
                for w in row):
            raise TypeError("`whiskers` must contain `WhiskerSpec` objects")
        if list(d_given.keys()) == ['xaxis']:
            raise ValueError("At least one of `yaxis`, `ylower`, `yupper`, "
                             "and `whiskers` must be given")

        # The number of curves is the number of entries of the 2D arguments,
        # which must agree.
        s_num_curves = {len(ll) for ll in d_given.values() if len(ll) > 1}
        if len(s_num_curves) > 1:
            raise ValueError("The arguments specify different numbers of "
                             f"curves: {s_num_curves}")
        num_curves = s_num_curves.pop() if s_num_curves else 1

        l_num_pts = []
        for ind_curve in range(num_curves):
            s_num_pts = {
                len(ll[ind_curve] if len(ll) > 1 else ll[0])
                for ll in d_given.values()
            }
            if len(s_num_pts) > 1:
                raise ValueError(
                    f"The arguments of curve {ind_curve} have different "
                    f"numbers of points: {s_num_pts}")
            l_num_pts.append(s_num_pts.pop())
        return l_num_pts

    @staticmethod
    def _broadcast_curve_arg(arg, l_num_pts):
        """Returns a list with one entry per curve (as many as entries in
        `l_num_pts`; see `_infer_num_pts`) with the values of `arg` for that
        curve: None if `arg` is None or empty, else a list of `l_num_pts[n]`
        values. A 1D `arg` is broadcast to all the curves."""
        ll_arg = Subplot._unify_curve_arg(arg)
        num_curves = len(l_num_pts)
        if ll_arg is None:
            return [None] * num_curves
        if len(ll_arg) == 1:
            ll_arg = ll_arg * num_curves
        if len(ll_arg) != num_curves:
            raise ValueError(
                f"Expected {num_curves} curves, got {len(ll_arg)}")
        for l_entry, num_pts in zip(ll_arg, l_num_pts):
            if len(l_entry) != num_pts:
                raise ValueError(f"Expected {num_pts} points, got "
                                 f"{len(l_entry)}")
        return [list(l_entry) for l_entry in ll_arg]

    def _list_from_style_argument(style_arg):
        """
      Returns a list of str. 
      """
        err_msg = "Style argument must be an str or list of str"
        if type(style_arg) == str:
            return [style_arg]
        elif type(style_arg) == list:
            for entry in style_arg:
                if type(entry) != str:
                    raise TypeError(err_msg)
            return copy.copy(style_arg)
        else:
            raise TypeError(err_msg)

    def plot(self, *subplot_args, **subplot_kwargs):

        if not hasattr(self, 'axes'):  # backwards compatibility
            self.axes = None

        if (self.axes is None):
            regenerating = False
            self.axes = plt.subplot(*subplot_args, **subplot_kwargs)
        else:
            regenerating = True

        for curve in self.l_curves:
            curve.plot(
                zlim=self.zlim
                if hasattr(self, "zlim") else None,  # backwards comp.
                axes=self.axes)

        if not Curve.legend_is_empty(self.l_curves):
            if not hasattr(self, "legend_loc"):
                self.legend_loc = None  # backwards compatibility
            if not hasattr(self, "num_legend_cols"):
                self.num_legend_cols = 1
            self.axes.legend(loc=self.legend_loc, ncol=self.num_legend_cols)

        # Axis labels
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        # X ticks
        if hasattr(self, "xticks") and self.xticks is not None:
            self.axes.set_xticks(self.xticks)
            if getattr(self, "xticklabels", None) is not None:
                self.axes.set_xticklabels(self.xticklabels)

        if hasattr(self, "num_xticks_decimal_places"
                   ) and self.num_xticks_decimal_places is not None:
            import matplotlib.ticker as ticker
            self.axes.xaxis.set_major_formatter(
                ticker.FormatStrFormatter(
                    f'%.{self.num_xticks_decimal_places}f'))

        # Y ticks
        if hasattr(self, "yticks") and self.yticks is not None:
            self.axes.set_yticks(self.yticks)

        if self.projection == '3d' and hasattr(self, 'zlabel') and self.zlabel:
            self.axes.set_zlabel(self.zlabel)

        # Color bar
        if hasattr(self, "color_bar") and self.color_bar:
            image = self.get_image()
            if image is None:
                raise ValueError(
                    "color_bar=True but no color figure was specified")
            cbar = plt.colorbar(image)  #, cax=cbar_ax)
            if self.zlabel:
                cbar.set_label(self.zlabel)

        if self.title:
            self.axes.set_title(self.title)

        if "grid" in dir(self):  # backwards compatibility
            self.axes.grid(self.grid)

        # Apply logarithmic scales
        if "logx" in dir(self) and self.logx:
            self.axes.set_xscale('log')
        if "logy" in dir(self) and self.logy:
            self.axes.set_yscale('log')

        if "xlim" in dir(self):  # backwards compatibility
            if self.xlim:
                if regenerating and type(self.xlim) == tuple and np.any(
                    [val is None for val in self.xlim]):
                    gsim_logger.warning(
                        "Regeneration not implemented for xlim with None values."
                    )
                if self.xlim[0] != self.xlim[
                        1]:  # Avoids a warning if limits are equal
                    self.axes.set_xlim(self.xlim)
            else:
                # Automatic x-limits
                self.axes.autoscale(enable=True, axis='x')
                self.axes.relim()
                self.axes.autoscale_view(scalex=True, scaley=False)

        if "ylim" in dir(self):  # backwards compatibility
            if self.ylim:
                if isinstance(self.ylim, float):
                    self.axes.set_ylim(self.get_auto_ylims(self.ylim))
                else:
                    if self.ylim[0] != self.ylim[
                            1]:  # Avoids a warning if limits are equal
                        self.axes.set_ylim(self.ylim)
            else:
                # Automatic y-limits
                if regenerating:
                    if not self.is_3D:  # For images, the following line does not work well, so we skip it
                        if len(self.l_curves):
                            try:
                                ylims = self.get_auto_ylims(.2)
                                if ylims[0] != ylims[
                                        1]:  # Avoids a warning if a curve is constant.
                                    self.axes.set_ylim(ylims)
                            except ValueError:
                                pass

        return self.axes

    def get_image(self):
        """Scans l_curves to see if one has defined the attribute "image". If
      so, it returns the value of this attribute, else it returns
      None.

      """
        for curve in self.l_curves:
            if curve.image:
                return curve.image
        return None

    @property
    def projection(self):
        """This is used to create the axes."""
        for curve in self.l_curves:
            if curve.projection == '3d':
                return '3d'
        return None

    @property
    def is_3D(self):
        """Returns true if at least one curve is 3D."""
        for curve in self.l_curves:
            if curve.is_3D:
                return True
        return False

    def get_auto_ylims(self, ylim_factor):
        """Returns automatic y-limits based on the curves in self.l_curves. 
        
        It first finds the minimum and maximum y-values among all 2D curves
        within the x-limits if specified.

        Then returns (y_min - ylim_factor * (y_max - y_min), y_max + ylim_factor *
        (y_max - y_min)).
        """
        assert isinstance(ylim_factor, float), "ylim_factor must be a float"

        y_min = sys.float_info.max
        y_max = -sys.float_info.max
        for curve in self.l_curves:
            # Only the values of `yaxis` are considered.
            if len(curve) == 0 or curve.yaxis is None:
                continue
            if not curve.is_3D:
                # 2D curve
                if curve.xaxis is None or (type(curve.xaxis) == list
                                           and len(curve.xaxis) == 0):
                    x_vals = np.arange(len(curve.yaxis))
                else:
                    x_vals = np.array(curve.xaxis)
                y_vals = np.array(curve.yaxis)

                # Consider only the data within the x-limits
                if "xlim" in dir(self) and self.xlim:
                    ind_within_xlim = np.where((x_vals >= self.xlim[0])
                                               & (x_vals <= self.xlim[1]))[0]
                    if len(ind_within_xlim) == 0:
                        continue
                    y_vals = y_vals[ind_within_xlim]

                if len([val for val in y_vals if not np.isnan(val)]) == 0:
                    continue
                y_min = min(y_min, np.nanmin(y_vals))
                y_max = max(y_max, np.nanmax(y_vals))
        if y_min == sys.float_info.max or y_max == -sys.float_info.max:
            raise ValueError(
                "Could not determine automatic y-limits; no 2D curves found.")
        y_range = y_max - y_min
        ylim = (y_min - ylim_factor * y_range, y_max + ylim_factor * y_range)

        if "logy" in dir(self) and self.logy:
            # We cannot pass negative y-limits to matplotlib when using a logarithmic scale.
            if ylim[0] <= 0:
                ylim = (y_min, y_max + ylim_factor * y_range)

        return ylim


class GFigure:
    str_caption = None

    def __init__(self,
                 *args,
                 figsize=None,
                 ind_active_subplot=0,
                 num_subplot_rows=None,
                 num_subplot_columns=1,
                 transpose_subplots=False,
                 global_color_bar=False,
                 global_color_bar_label="",
                 global_color_bar_position=[0.85, 0.35, 0.02, 0.5],
                 layout="tight",
                 **kwargs):

        # Create a subplot if the arguments specify one
        new_subplot = Subplot(*args, **kwargs)
        self.ind_active_subplot = ind_active_subplot
        self.l_subplots: list["Subplot | None"] = []
        if not new_subplot.is_empty():
            # List of axes to create subplots
            self.l_subplots = [None] * (self.ind_active_subplot + 1)
            self.l_subplots[self.ind_active_subplot] = new_subplot

        self.num_subplot_rows = num_subplot_rows
        self.num_subplot_columns = num_subplot_columns
        self.transpose_subplots = transpose_subplots
        self.figsize = figsize
        self.global_color_bar = global_color_bar
        self.global_color_bar_label = global_color_bar_label
        self.global_color_bar_position = global_color_bar_position

        self.plt_fig: Figure | None = None  # Handle to the plt.figure object

        # Animations
        self.f_update: Callable[
            [],
            bool] | None = None  # Function to update the figure. It returns True if the animation should stop.
        self.animation_interval = 50  # in milliseconds
        self._animation_has_started = False  # Whether the animation is running

        if layout == "" or layout == "tight":
            self.layout = layout
        else:
            raise ValueError("Invalid value of argument `layout`")

    def __repr__(self):
        return f"<GFigure object with len(self.l_subplots)={len(self.l_subplots)} subplots>"

    def add_curve(self, *args, ind_active_subplot=None, **kwargs):
        """
         Similar arguments to __init__ above.

        """

        # Modify ind_active_subplot only if provided
        if ind_active_subplot is not None:
            self.ind_active_subplot = ind_active_subplot

        self.select_subplot(self.ind_active_subplot, **kwargs)
        self.l_subplots[self.ind_active_subplot].add_curve(*args, **kwargs)

    def add_vertical_lines(self, *args, ind_active_subplot=None, **kwargs):
        """ 
        Wrapper for Subplot.add_vertical_lines.

        """

        # Modify ind_active_subplot only if provided
        if ind_active_subplot is not None:
            self.ind_active_subplot = ind_active_subplot

        self.select_subplot(self.ind_active_subplot)

        self.l_subplots[self.ind_active_subplot].add_vertical_lines(
            *args, **kwargs)

    def add_histogram_curve(self,
                            data,
                            *args,
                            hist_args={},
                            ind_active_subplot=None,
                            **kwargs):
        """ This works like add_curve, but it adds a curve with the histogram of
        `data`. See example below.
        
        Args:
            - `hist_args`: dictionary with the arguments passed to np.histogram. 
        
        """
        v_hist, v_bin_edges = np.histogram(data, **hist_args)
        density = hist_args.get("density", False)

        v_x, v_y = hist_bin_edges_to_xy(v_hist, v_bin_edges, density=density)
        self.add_curve(v_x,
                       v_y,
                       *args,
                       ind_active_subplot=ind_active_subplot,
                       **kwargs)

    def add_whiskers_curve(self,
                           data,
                           xaxis=[],
                           f_stats=None,
                           ind_active_subplot=None,
                           **kwargs):
        """Plots whiskers computed from the samples in `data` on the active
        subplot; cf. the argument `whiskers` of GFigure.__init__. Any other
        argument of `add_curve` (e.g. `yaxis`, `styles`, `legend`) can be
        passed through `kwargs`. See also the example below.

        Args:

            - `data`: samples summarized by each box. It can be a list of L
              lists/arrays (possibly of different lengths), an L x M array (one
              row per box), or a single list/1D array of samples, which is
              understood as `[data]` (a single box).

            - `xaxis`: None, [], or a list of length L; see GFigure.__init__.

            - `f_stats`: function that maps a 1D array of samples to a
              `WhiskerSpec`. By default, minimum, lower quartile, median, upper
              quartile, and maximum.
        """
        if f_stats is None:

            def f_stats(v_samples):
                return WhiskerSpec(np.min(v_samples),
                                   np.percentile(v_samples, 25),
                                   np.median(v_samples),
                                   np.percentile(v_samples, 75),
                                   np.max(v_samples))

        if hasattr(data, "numpy"):  # Compatibility with TensorFlow
            data = data.numpy()
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                l_samples = [data]
            elif data.ndim == 2:
                l_samples = list(data)
            else:
                raise ValueError("`data` must be of dimension 1 or 2")
        elif isinstance(data, (list, tuple)):
            if len(data) and np.isscalar(data[0]):
                l_samples = [data]
            else:
                l_samples = list(data)
        else:
            raise TypeError("`data` must be a list or an np.ndarray")

        whiskers = [
            f_stats(np.ravel(np.asarray(v_samples, dtype=float)))
            for v_samples in l_samples
        ]
        self.add_curve(xaxis=xaxis,
                       whiskers=whiskers,
                       ind_active_subplot=ind_active_subplot,
                       **kwargs)

    def next_subplot(self, **kwargs):
        # Creates a new subplot at the end of the list of axes. One can
        # specify subplot parameters; see GFigure.
        self.ind_active_subplot = len(self.l_subplots)
        if kwargs:
            self.l_subplots.append(Subplot(**kwargs))

    def select_subplot(self, ind_subplot, **kwargs):
        """Creates the `ind_subplot`-th subplot if it does not exist and
        selects it. Subplot keyword parameters can also be provided (see
        GFigure), but parameters to create a curve are ignored.        
        """

        self.ind_active_subplot = ind_subplot

        # Complete the list l_subplots if index self.ind_active_subplot does
        # not exist.
        if ind_subplot >= len(self.l_subplots):
            self.l_subplots += [None] * (self.ind_active_subplot -
                                         len(self.l_subplots) + 1)

        # Create if it does not exist
        if self.l_subplots[self.ind_active_subplot] is None:
            self.l_subplots[self.ind_active_subplot] = Subplot(
                create_curves=False, **kwargs)
        else:
            self.l_subplots[self.ind_active_subplot].update_properties(
                **kwargs)

    def plot(self):

        if "plt_fig" not in dir(self):  # backwards compatibility
            self.plt_fig = None

        # backwards compatibility
        if "figsize" not in dir(self):
            figsize = None
        else:
            figsize = self.figsize
        if figsize is None:
            figsize = default_figsize

        if self.plt_fig is None:
            # Create the figure only if it does not exist. If it exists, it is
            # because we are dynamically replotting a figure.
            self.plt_fig = plt.figure(figsize=figsize)

        # Determine the number of rows and columns for arranging the subplots
        num_axes = len(self.l_subplots)
        if self.num_subplot_rows is not None:
            self.num_subplot_columns = int(
                np.ceil(num_axes / self.num_subplot_rows))
        else:  # self.num_subplot_rows is None
            if self.num_subplot_columns is None:
                # Both are None. Just arrange thhe plots as a column
                self.num_subplot_columns = 1
                self.num_subplot_rows = num_axes
            else:
                self.num_subplot_rows = int(
                    np.ceil(num_axes / self.num_subplot_columns))

        # Process title
        if title_to_caption and (len(self.l_subplots) == 1):
            self.str_caption = self.l_subplots[0].title
            self.l_subplots[0].title = ""
            print("Caption: ", self.str_caption)

        # Unify zlimits if required
        if hasattr(self, "global_color_bar") and self.global_color_bar:
            self.unify_zlim_intervals()

        # Transpose the subplots if needed
        if hasattr(self, "transpose_subplots") and self.transpose_subplots:
            self.l_subplots = np.array(self.l_subplots).reshape(
                self.num_subplot_columns, self.num_subplot_rows).T.flatten()

        # Actual plotting operation
        for index, subplot in enumerate(self.l_subplots):
            if index > 0:
                prev_axis = axis
            if self.l_subplots[index] is not None:
                axis = self.l_subplots[index].plot(
                    self.num_subplot_rows,
                    self.num_subplot_columns,
                    index + 1,
                    sharex=prev_axis if
                    (index > 0 and hasattr(subplot, "sharex")
                     and subplot.sharex) else None,
                    projection=self.l_subplots[index].projection)

        # Layout
        if hasattr(self, "layout"):  # backwards compatibility
            if self.layout == "":
                pass
            elif self.layout == "tight":
                plt.tight_layout()
            else:
                raise ValueError("Invalid value of argument `layout`")

        # Color bar
        if hasattr(self, "global_color_bar") and self.global_color_bar:

            for subplot in self.l_subplots:
                image = subplot.get_image()
                if image:
                    break
            self.plt_fig.subplots_adjust(right=0.85)

            cbar_ax = self.plt_fig.add_axes(self.global_color_bar_position)
            cbar = self.plt_fig.colorbar(image, cax=cbar_ax)

            if self.global_color_bar_label:
                cbar.set_label(self.global_color_bar_label)

        # This is needed if we are updating the figure
        self.plt_fig.canvas.draw_idle()

        # Animations
        if hasattr(self, "f_update") and (self.f_update is not None):
            if not self._animation_has_started:
                self._animation_has_started = True
                timer = self.plt_fig.canvas.new_timer(
                    interval=self.animation_interval)

                def update():
                    # This wrapper allows f_update to receive the timer to
                    # stop the animation if needed.
                    assert self.f_update is not None
                    if self.f_update():
                        timer.stop()
                    self.plot()

                timer.add_callback(update)
                timer.start()

        return self.plt_fig

    @staticmethod
    def concatenate(it_gfigs,
                    num_subplot_rows=None,
                    num_subplot_columns=1) -> "GFigure":
        """Concatenates the subplots of a collection of GFigure objects.

     Args:
       it_gfigs: iterable that returns GFigures. 

       num_subplot_rows and num_subplot_columns: see GFigure.__init__()

     Returns: 
       gfig: an object of class GFigure.

     """

        l_subplots = [
            subplot for gfig in it_gfigs for subplot in gfig.l_subplots
        ]

        gfig = next(iter(it_gfigs))  # take the first
        gfig.l_subplots = l_subplots
        gfig.num_subplot_rows = num_subplot_rows
        gfig.num_subplot_columns = num_subplot_columns

        return gfig

    def export(self, base_filename):
        # Save figure to pdf
        filename_pdf = base_filename + ".pdf"
        print(f"Exporting GFigure as {filename_pdf}")
        plt.savefig(filename_pdf)

        # Save caption if applicable
        if hasattr(self, "str_caption") and self.str_caption is not None:
            basename_txt = base_filename + ".txt"
            print(f"Saving caption as {basename_txt}")
            with open(basename_txt, "w") as f:
                f.write(self.str_caption)

    @staticmethod
    def show():
        plt.show()

    def unify_zlim_intervals(self):
        """Sets the zlimits of all subplots to be the same. This is useful e.g. to have a global color bar."""

        # Unify zlim intervals
        def get_zlim(subplot):
            if subplot.zlim is not None:
                return subplot.zlim
            else:
                l_zvals = [
                    curve.zaxis.flatten() for curve in subplot.l_curves
                    if curve.zaxis is not None
                ]
                if len(l_zvals) == 0:
                    return [np.nan, np.nan]
                zvals = np.concatenate(l_zvals)

                return [np.nanmin(zvals), np.nanmax(zvals)]

        l_zlims = np.concatenate(
            [get_zlim(subplot) for subplot in self.l_subplots])
        zlims = [np.nanmin(l_zlims), np.nanmax(l_zlims)]
        for subplot in self.l_subplots:
            subplot.zlim = zlims

    @staticmethod
    def make_periodically_refreshing_figure(
            f_make_figure: Callable[[], "GFigure|None"],
            interval: int = 1000,  # ms
    ):
        """Creates a GFigure that refreshes itself automatically by calling
        `f_make_figure` every `interval` milliseconds.

        Args:

            - f_make_figure: function that returns a GFigure object. To stop the
              animation, return None. 

            - interval: refresh interval in milliseconds.

        Returns:
            - gfig: GFigure object that refreshes itself automatically.
        
        """

        def copy_plt_refs(gfig_src: GFigure, gfig_dest: GFigure):
            """Copies the plt figure, axes, and line references from gfig_src to
            gfig_dest. 
            """
            gfig_dest.plt_fig = gfig_src.plt_fig
            for subplot_src, subplot_dest in zip(gfig_src.l_subplots,
                                                 gfig_dest.l_subplots):
                if subplot_src is not None and subplot_dest is not None:
                    subplot_dest.axes = subplot_src.axes
                    for curve_src, curve_dest in zip(subplot_src.l_curves,
                                                     subplot_dest.l_curves):
                        curve_dest.line = curve_src.line

        gfig = f_make_figure()

        def f_update() -> bool:
            nonlocal gfig, f_make_figure
            new_fig = f_make_figure()
            if new_fig is None:
                return True  # Stop the animation
            copy_plt_refs(gfig, new_fig)
            gfig.l_subplots = new_fig.l_subplots
            return False  # Do not stop the animation

        gfig.f_update = f_update
        gfig.animation_interval = interval

        return gfig


def plot_example_figure(ind_example):
    """
    This is example code to learn how to use GFigure. See the description at the
    top of this file. 
    
    """

    v_x = np.linspace(0, 10, 20)
    v_y1 = v_x**2 - v_x + 3
    v_y2 = v_x**2 + v_x + 3
    v_y3 = v_x**2 - 2 * v_x - 10

    if ind_example == 1:
        # Example with a single curve, single subplot
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabolas",
                    legend="P1")

    elif ind_example == 2:
        # Example with three curves on one subplot
        # The style can be specified as in MATLAB
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabolas",
                    styles=".-#FF0000",
                    legend="P1")
        G.add_curve(xaxis=v_x, yaxis=v_y2, legend="P2", styles="o--k")
        G.add_curve(xaxis=v_x, yaxis=v_y3, legend="P3", styles="x:b")

    elif ind_example == 3:
        # Typical scheme where a simulation function produces each
        # curve.
        def my_simulation():
            coef = np.random.random()
            v_y_new = coef * v_y1
            G.add_curve(xaxis=v_x, yaxis=v_y_new, legend="coef = %.2f" % coef)

        """ One can specify the axis labels and title when the figure is
        created."""
        G = GFigure(xlabel="x", ylabel="f(x)", title="Parabola")
        for ind in range(0, 6):
            my_simulation()

    elif ind_example == 4:
        # Example with two subplots

        # As a shortcut, the first curve can be directly specified by passing
        # the necessary args to the constructor. Alternatively, one can
        # instantiate GFigure and then add the curves one by one with
        # `add_curve`.
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabolas",
                    legend="P1")
        G.add_curve(xaxis=v_x, yaxis=v_y2, legend="P2")
        G.next_subplot(xlabel="x")
        G.add_curve(xaxis=v_x,
                    yaxis=v_y2,
                    legend="P3",
                    mode="stem",
                    styles="ob")
        G.add_curve(xaxis=v_x,
                    yaxis=v_y3,
                    legend="P3",
                    mode="stem",
                    styles="xk")

    elif ind_example == 5:
        # Example with a large multiplot
        G = GFigure(num_subplot_rows=4)
        for ind in range(0, 12):
            G.select_subplot(ind, xlabel="x", ylabel="f(x)", title="Parabolas")
            G.add_curve(xaxis=v_x, yaxis=v_y1, legend="P1", styles="r")

    elif ind_example == 6:
        # Typical scheme where a simulation function produces each subplot
        def my_simulation():
            G.next_subplot(xlabel="x", ylabel="f(x)", title="Parabola")
            G.add_curve(xaxis=v_x, yaxis=v_y1, legend="P1", styles="r")

        """ Important not to specify axis labels or the title in the next line
        because that would create an initial subplot without curves
        and, therefore, function `next_subplot` will move to the
        second subplot of the figure the first time `my_simulation` is
        executed."""

        G = GFigure(num_subplot_rows=3)
        for ind in range(0, 6):
            my_simulation()

    elif ind_example == 7:
        # Colorplot of a function of 2 arguments.
        num_points_x = 30
        num_points_y = 30
        gridpoint_spacing = 1 / 30
        v_x_coords = np.arange(0, num_points_x) * gridpoint_spacing
        v_y_coords = np.arange(num_points_y - 1, -1,
                               step=-1) * gridpoint_spacing
        x_coords, y_coords = np.meshgrid(v_x_coords, v_y_coords, indexing='xy')

        def my_simulation():
            xroot = np.random.random()
            yroot = np.random.random()
            zaxis = (x_coords - xroot)**2 + (y_coords - yroot)**2
            G.next_subplot(xlabel="x",
                           ylabel="y",
                           zlabel="z",
                           grid=False,
                           color_bar=False,
                           zlim=(0, 1))
            G.add_curve(xaxis=x_coords, yaxis=y_coords, zaxis=zaxis)
            G.add_curve(xaxis=[xroot], yaxis=[yroot], styles="+w")

        G = GFigure(num_subplot_rows=3,
                    global_color_bar=True,
                    global_color_bar_label="z")

        for ind in range(0, 6):
            my_simulation()

    elif ind_example == 8:
        # Scatter plot
        v_x = np.linspace(0, 10, 200)
        v_y = v_x + np.random.normal(size=(len(v_x), ))
        G = GFigure(xaxis=v_x,
                    yaxis=v_y,
                    styles=['.'],
                    xlabel='x',
                    ylabel='y=x+w')

    elif ind_example == 9:
        # 3D plots
        G = GFigure(num_subplot_rows=3)

        # The three modes 'imshow' (default), 'contour3d', and 'surface' are
        # tested.

        # 1. Auto axes
        # Format as an image
        m_Z = np.reshape(np.arange(20 * 30), (20, 30))
        G.next_subplot(zaxis=m_Z,
                       xlabel='columns',
                       ylabel='rows',
                       aspect='square')
        G.next_subplot(zaxis=m_Z,
                       xlabel='columns',
                       ylabel='rows',
                       zlabel='value',
                       mode='contour3D')

        # 2. Semi-manual specification of the axes (most typical)
        xaxis = np.linspace(0, 20, 50)
        yaxis = np.flip(np.linspace(0, 10, 50))
        zaxis = yaxis[:, None]**2 + 10 * xaxis[None, :]
        G.next_subplot(
            xaxis=xaxis,
            yaxis=yaxis,
            zaxis=zaxis,
            xlabel='x',
            ylabel='y',
        )
        G.next_subplot(xaxis=xaxis,
                       yaxis=yaxis,
                       zaxis=zaxis,
                       xlabel='x',
                       ylabel='y',
                       zlabel='z',
                       mode='surface')

        # 3. Fully manual specification of the axes
        v_xaxis = np.linspace(0, 6, 50)
        v_yaxis = np.linspace(0, np.pi, 50)
        m_X, m_Y = np.meshgrid(v_xaxis, v_yaxis)
        m_Z = np.sin(m_Y) * m_X
        G.next_subplot(
            xaxis=m_X,
            yaxis=m_Y,
            zaxis=m_Z,
            xlabel='x',
            ylabel='y',
        )
        G.next_subplot(xaxis=m_X,
                       yaxis=m_Y,
                       zaxis=m_Z,
                       xlabel='x',
                       ylabel='y',
                       zlabel='z',
                       mode='contour3D')

    elif ind_example == 10:
        # Histogram
        v_data_norm = np.random.normal(size=(1000, ), loc=0, scale=1)
        v_data_exp = np.random.exponential(size=(1000, ), scale=1)

        G = GFigure(xlabel='Value', ylabel='Density')
        G.add_histogram_curve(data=v_data_norm,
                              styles='b',
                              hist_args={
                                  'bins': 50,
                                  'density': True
                              },
                              legend='Normal')
        G.add_histogram_curve(data=v_data_exp,
                              styles='r',
                              hist_args={
                                  'bins': 50,
                                  'density': True
                              },
                              legend='Exponential')

    elif ind_example == 11:
        # Example of xticks with a fixed number of decimal places
        v_x = np.linspace(0, 10, 10)
        v_y = v_x**2
        G = GFigure(xaxis=v_x,
                    yaxis=v_y,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabola",
                    num_xticks_decimal_places=3)
        G.next_subplot(xaxis=v_x,
                       yaxis=v_y,
                       xlabel="x",
                       ylabel="f(x)",
                       title="Parabola",
                       num_xticks_decimal_places=2,
                       xticks=v_x + 0.5)

    elif ind_example == 12:
        # Example of color specification (not colorp)
        G = GFigure(xlabel="x", ylabel="y")
        num_curves = 3
        v_x = np.linspace(0, 10, 100)
        for ind_curve in range(num_curves):
            v_mean = (ind_curve + 1) / num_curves * 2 * v_x
            v_y = v_mean + np.random.normal(scale=0.5, size=v_x.shape)
            G.add_curve(xaxis=v_x,
                        yaxis=v_y,
                        legend="Curve %d" % (ind_curve + 1),
                        styles=f'.#{ind_curve}')
            G.add_curve(xaxis=v_x,
                        yaxis=v_mean,
                        legend="Curve %d (mean)" % (ind_curve + 1),
                        styles=f'-#{ind_curve}')

        G.add_curve(xaxis=v_x,
                    yaxis=v_x,
                    styles='--#000000',
                    legend='Reference y=x')

    elif ind_example == 13:
        # By default, pyplot autoscale in the y-axis is disabled when providing
        # xlim.
        #
        # GFigure allows the specification of the y-limit via a float to restore
        # autoscaling.
        v_x = np.linspace(0, 40, 1000)
        v_y = 3 * np.sin(10 * v_x) + 10 * v_x
        G = GFigure(xaxis=v_x,
                    yaxis=v_y,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Without y-limit specification",
                    xlim=(2, 4))
        G.next_subplot(xaxis=v_x,
                       yaxis=v_y,
                       xlabel="x",
                       ylabel="f(x)",
                       title="With y-limit specification",
                       xlim=(2, 4),
                       ylim=0.2)  # enables autoscaling

    elif ind_example == 14:
        # Example with vertical lines
        v_x = np.linspace(0, 10, 100)
        v_y = np.sin(v_x)
        G = GFigure(xaxis=v_x,
                    yaxis=v_y,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Figure with vertical lines")
        G.add_vertical_lines(x_positions=[2, 4, 6, 8],
                             style='r--',
                             legend_str='Red vertical lines')
        G.add_vertical_lines(x_positions=[3, 5, 7, 9],
                             style='g--',
                             legend_str='Green vertical lines')

    elif ind_example == 15:
        # Example of figure that is updated over time (animation). For a figure
        # that is automatically recreated, see the example below.
        n = 2
        v_x = np.arange(0, n, .1)
        G = GFigure(xaxis=v_x,
                    yaxis=np.sin(v_x) * v_x,
                    xlabel="x",
                    title="Plot that changes over time",
                    legend=f"n = {n}")

        def f_update():
            # Modify the data in the GFigure object
            print("Updating plot...")
            nonlocal n
            n += .3
            if n > 30:
                return True  # Stop the animation
            v_x = np.arange(0, n, .1)
            G.l_subplots[0].l_curves[0].xaxis = v_x
            G.l_subplots[0].l_curves[0].yaxis = np.sin(v_x) * v_x
            G.l_subplots[0].l_curves[0].legend_str = f"n = {n}"
            return False

        G.f_update = f_update
        G.animation_interval = 50  # milliseconds

    elif ind_example == 16:
        # Figure that automatically refreshes. It avoids the need for separate
        # code for creating the figure and updating it.
        n = 2

        def make_figure():
            nonlocal n
            v_x = np.arange(0, n, .1)
            G = GFigure(xaxis=v_x,
                        yaxis=np.sin(v_x) * v_x,
                        xlabel="x",
                        title="Plot that changes over time",
                        legend=f"sin(x)*x, n = {n:.3g}")
            G.add_curve(xaxis=v_x,
                        yaxis=-np.sin(v_x) * v_x,
                        legend=f"-sin(x)*x, n = {n:.3g}")
            n += .3
            if n > 30:
                return None  # Stop the animation
            return G

        G = GFigure.make_periodically_refreshing_figure(make_figure,
                                                        interval=50)

    elif ind_example == 17:
        # Example of whiskers: (i) a curve with whiskers given explicitly;
        # (ii) whiskers computed from samples (one box per group of samples)
        # without a curve, together with the samples themselves.
        v_x = np.arange(5)
        v_y = v_x**2
        G = GFigure(
            xaxis=v_x,
            yaxis=v_y,
            whiskers=[WhiskerSpec(y - 2, y - 1, y, y + 1, y + 3) for y in v_y],
            styles='o-',
            xlabel='x',
            ylabel='f(x)',
            title='Curve with whiskers',
            legend='f(x) and whiskers')
        l_samples = [np.random.randn(30) * (ind + 1) for ind in range(3)]
        v_positions = np.arange(len(l_samples))
        G.next_subplot(xlabel='Group',
                       ylabel='Value',
                       title='Whiskers from samples',
                       xticks=v_positions,
                       xticklabels=['A', 'B', 'C'])
        G.add_whiskers_curve(l_samples,
                             xaxis=v_positions,
                             styles='#1',
                             legend='min, quartiles, median, max')
        for v_samples, position in zip(l_samples, v_positions):
            G.add_curve(xaxis=position * np.ones(len(v_samples)),
                        yaxis=v_samples,
                        styles='.#0')

    else:
        raise ValueError("Invalid example index")

    G.plot()
    plt.show()


def plot_all_example_figures():
    ind_example = 1
    while True:
        try:
            print(f"Plotting example figure {ind_example}...")
            plot_example_figure(ind_example)
            ind_example += 1
        except ValueError:
            print("No more example figures.")
            break


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("""Usage from command line: 
$ python3 gfigure.py <example_index>
            
where <example_index> is an integer or "all". See function `example_figures`."""
              )
    else:
        if sys.argv[1] == "all":
            plot_all_example_figures()
        else:
            plot_example_figure(int(sys.argv[1]))
