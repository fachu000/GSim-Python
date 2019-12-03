import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import copy
import numpy as np
"""



"""


class Curve:
    def __init__(self, xaxis=None, yaxis=[], style=None, legend_str=""):
        """
        
        xaxis : list of a numeric type or None. In the former case, its length 
            equal the length of yaxis.

        yaxis : list of a numeric type. 

        style : str used as argument to plt.plot()
        
        """

        # Input check
        if type(yaxis) != list:
            set_trace()
            raise TypeError("`yaxis` must be a list of numeric entries")
        if type(xaxis) == list:
            assert len(xaxis) == len(yaxis)
        elif xaxis is not None:
            raise TypeError(
                "`xaxis` must be a list of numeric entries or None")
        if (style is not None) and (type(style) != str):
            raise TypeError("`style` must be of type str or None")
        if type(legend_str) != str:
            raise TypeError("`legend_str` must be of type str")

        # Save
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.style = style
        self.legend_str = legend_str

    def plot(self):

        if type(self.xaxis) == list and len(self.xaxis):
            if self.style:
                plt.plot(self.xaxis, self.yaxis, self.style, label=self.legend_str)
            else:
                plt.plot(self.xaxis, self.yaxis, label=self.legend_str)
        else:
            if self.style:
                plt.plot(self.yaxis, self.style, label=self.legend_str)
            else:
                plt.plot(self.yaxis, label=self.legend_str)

    def legend_is_empty(l_curves):

        for curve in l_curves:
            if curve.legend_str != "":
                return False
        return True


    #     b_empty_legend = True
    #     for curve in l_curves:
    #         if curve.legend_str != "":
    #             b_empty_legend = False
    #             break

    #     if b_empty_legend:
    #         return tuple([])
    #     else:
    #         return tuple([curve.legend_str for curve in l_curves])


class GFigure:
    def __init__(self,
                 title="",
                 xlabel="",
                 ylabel="",
                 xaxis=[],
                 yaxis=[],
                 styles=[],
                 legend=tuple()):
        """
        Arguments of mutable types are (deep) copied so they can be modified by the 
        user after constructing the GFigure object without altering the figure.
        
        ARGUMENTS:

        title : str 

        xlabel : str

        ylabel : str

        xaxis and yaxis:
            (a) To specify only one curve:
                - `yaxis` can be a list of a numeric type or 1D np.ndarray
                - `xaxis` can be None, a list of a numeric type, or a 1D 
                np.array of the same length as `yaxis`.
            (b) To specify one or more curves:
                - `yaxis` can be:
                    -> a list of the types specified in (a)
                    -> M x N np.ndarray. Each row corresponds to a curve.
                - `xaxis` can be either as in (a), so all curves share the same 
                X-axis points, or
                    -> a list of the types specified in (a)
                    -> Mx x N np.ndarray. Each row corresponds to a curve. Mx 
                    must be either M or 1. 

        styles: specifies the style argument to plot, as in MATLAB. Possibilities:
            - str : this style is applied to all curves specified by 
             `xaxis` and `yaxis` will 
            - list of str : then style[n] is applied to the n-th curve. Its length
              must be at least the number of curves.

        legend : str, tuple of str, or list of str. If the str begins with "_", then
            that curve is not included in the legend.

        """

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.l_curves = GFigure._l_curve_from_input_args(
            xaxis, yaxis, styles, legend)

    def _l_curve_from_input_args(xaxis, yaxis, styles, legend):
        # Process the axis input.  Each entry of l_xaxis or l_yaxis is
        # a list of a numerical type. Both lists will have the same length.
        l_xaxis, l_yaxis = GFigure._list_from_axis_arguments(xaxis, yaxis)
        l_style = GFigure._list_from_style_argument(styles)

        # Process style input.
        if len(l_style) == 0:
            l_style = [None] * len(l_xaxis)
        elif len(l_style) == 1:
            l_style = l_style * len(l_xaxis)
        else:
            if len(l_style) < len(l_xaxis):
                set_trace()
            assert len(l_style) >= len(l_xaxis), "The length of `style` must be"\
                " either 1 or no less than the number of curves"

        # Process the legend
        assert ((type(legend) == tuple) or (type(legend) == list)
                or (type(legend) == str))
        if type(legend) == str:
            legend = [legend] * len(l_xaxis)
        else:  # legend is tuple or list
            if len(legend) == 0:
                legend = [""] * len(l_xaxis)
            else:
                assert type(
                    legend[0]
                ) == str, "`legend` must be an str, list of str, or tuple of str"
                assert (len(legend) == len(l_xaxis)
                        ), "len(legend) must equal 0 or the number of curves"

        b_debug = True
        if b_debug:
            conditions = [
                len(l_xaxis) == len(l_yaxis),
                len(l_xaxis) == len(l_style),
                type(l_xaxis) == list,
                type(l_yaxis) == list,
                type(l_style) == list,
                (len(l_xaxis) == 0) or (type(l_xaxis[0]) == list)
                or (l_xaxis[0] is None),
                (len(l_yaxis) == 0) or (type(l_yaxis[0]) == list)
                or (l_yaxis[0] is None),
                (len(l_style) == 0) or (type(l_style[0]) == str)
                or (l_style[0] is None),
            ]
            if not np.all(conditions):
                print(conditions)
                set_trace()

        # Construct Curve objects
        l_curve = []
        for xax, yax, stl, leg in zip(l_xaxis, l_yaxis,
                                      l_style[0:len(l_xaxis)], legend):
            l_curve.append(
                Curve(xaxis=xax, yaxis=yax, style=stl, legend_str=leg))
        return l_curve

    def _list_from_style_argument(style_arg):
        """
        Returns a list of str. 
        """
        err_msg = "Style argument must be an str "\
            "or list of str"
        if type(style_arg) == str:
            return [style_arg]
        elif type(style_arg) == list:
            for entry in style_arg:
                if type(entry) != str:
                    raise TypeError(err_msg)
            return copy.copy(style_arg)
        else:
            raise TypeError(err_msg)

    def is_number(num):
        return isinstance(num, (int, float, complex, bool))

    def _list_from_axis_arguments(xaxis_arg, yaxis_arg):
        """Processes axis arguments and returns two lists of the same length
        whose elements can be either None or lists of a numerical
        type. None means "use the default x-axis for this curve".

        """
        def unify_format(axis):
            def ndarray_to_list(arr):
                assert (type(arr) == np.ndarray)
                if arr.ndim == 1:
                    return [list(arr)]
                elif arr.ndim == 2:
                    return [[arr[row, col] for col in range(0, arr.shape[1])]
                            for row in range(0, arr.shape[0])]
                else:
                    raise TypeError

            if (type(axis) == np.ndarray):
                return ndarray_to_list(axis)
            elif (type(axis) == list):
                if len(axis) == 0:
                    return []
                if GFigure.is_number(axis[0]):
                    return [copy.copy(axis)]
                else:
                    out_list = []
                    for entry in axis:
                        if type(entry) == np.ndarray:
                            if entry.ndim == 1:
                                out_list.append(copy.copy(entry))
                            else:
                                raise Exception(
                                    "Arrays inside the list must be 1D in the current implementation"
                                )
                        elif type(entry) == list:
                            if len(entry) == 0:
                                out_list.append([])
                            elif GFigure.is_number(entry[0]):
                                out_list.append(copy.copy(entry))
                            else:
                                raise TypeError
                    return out_list
            elif axis is None:
                return [None]
            else:
                raise TypeError

        # Construct two lists of possibly different lengths.
        l_xaxis = unify_format(xaxis_arg)
        l_yaxis = unify_format(yaxis_arg)

        # Expand lists if needed to have the same length
        str_message = "Number of curves in the xaxis must be"\
            " 1 or equal to the number of curves in tye yaxis"
        if len(l_xaxis) == 0 and len(l_yaxis) > 0:
            l_xaxis = [None]
        if len(l_yaxis) > 1:
            if len(l_xaxis) == 1:
                l_xaxis = l_xaxis * len(l_yaxis)
            if len(l_xaxis) != len(l_yaxis):
                raise Exception(str_message)
        elif len(l_yaxis) == 1:
            if len(l_xaxis) != 1:
                raise Exception(str_message)

        return l_xaxis, l_yaxis

    def add_curve(self, xaxis=[], yaxis=[], styles=[], legend=tuple()):
        """

        """

        self.l_curves += GFigure._l_curve_from_input_args(
            xaxis, yaxis, styles, legend)

    def plot(self):

        F = plt.figure()

        for curve in self.l_curves:
            curve.plot()

#        plt.legend(Curve.list_to_legend(self.l_curves))
        if not Curve.legend_is_empty(self.l_curves):
            plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if self.title:
            plt.title(self.title)

        return F
