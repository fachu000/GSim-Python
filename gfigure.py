import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import copy
import numpy as np


class GFigure:
    def __init__(self,
                 title="",
                 xlabel="",
                 ylabel="",
                 xaxis=[],
                 yaxis=[],
                 legend=tuple()):
        """ARGUMENTS:

        title : str 

        xlabel : str

        ylabel : str

        xaxis : Possible types:
          - 1D list of a numeric type.
          - 2D list of a numeric type. `xaxis[i]` denotes the 1D
            list corresponding to the i-th curve.
          - M x N np.array. Each row corresponds to a curve.
          - list of 1D np.arrays. Each list entry corresponds to a curve.
        If not provided, it defaults range(0,len_of_yaxis)

        yaxis : Same type as `xaxis`. Each row corresponds to
        a curve. 

        legend : tuple or list of str

        """

        # Arguments of mutable types copied

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        # Each entry of l_xaxis or l_yaxis is a list.
        self.l_xaxis, self.l_yaxis = GFigure._list_from_axis_arguments(
            xaxis, yaxis)

        self.legend = copy.copy(legend)

    def _list_from_axis_arguments(xaxis_arg, yaxis_arg):
        # Returns a list ready for concatenation to self.l_xaxis and self.l_yaxis

        def unify_format(axis):
            def ndarray_to_list(arr):
                assert type(arr) == np.array
                if arr.ndim == 1:
                    return list(arr)
                elif arr.ndim == 2:
                    return [[arr[row, col] for col in range(0, arr.shape[1])]
                            for row in range(0, arr.shape[0])]
                else:
                    raise TypeError

            def is_number(num):
                return isinstance(num, (int, float, complex, bool))

            if type(axis) == np.array:
                return ndarray_to_list(axis)
            elif (type(axis) == list):
                if len(axis) == 0:
                    return []
                if is_number(axis[0]):
                    return [copy.copy(axis)]
                else:
                    out_list = []
                    for entry in axis:
                        if type(entry) == np.array:
                            if np.array.ndim == 1:
                                out_list.append(copy.copy(entry))
                            else:
                                raise Exception(
                                    "Arrays inside the list must be 1D in the current implementation"
                                )
                        elif type(entry) == list:
                            if len(entry) == 0:
                                out_list.append([])
                            elif is_number(entry[0]):
                                out_list.append(copy.copy(entry))
                            else:
                                raise TypeError
                    return out_list
            elif axis is None:
                return [None]
            else:
                raise TypeError


        l_xaxis = unify_format(xaxis_arg)
        l_yaxis = unify_format(yaxis_arg)
        #set_trace()

        str_message = "Number of curves in the xaxis must be 1 or equal to the number of curves in tye yaxis"
        if len(l_xaxis)==0 and len(l_yaxis)>0:
            l_xaxis = [[]] 
        if len(l_yaxis) > 1:
            if len(l_xaxis) == 1:
                l_xaxis = l_xaxis * len(l_yaxis)
            if len(l_xaxis) != len(l_yaxis):
                raise Exception(str_message)
        elif len(l_yaxis) == 1:
            if len(l_xaxis) != 1:
                raise Exception(str_message)

        return l_xaxis, l_yaxis

    def add_curve(self, *in_args):
        """
        Syntax:

        obj.add_curve(xaxis, yaxis)

        obj.add_curve(yaxis)

        """

        if len(in_args) == 1:
            xaxis = []
            yaxis = in_args[0]

        if len(in_args) == 2:
            xaxis = in_args[0]
            yaxis = in_args[1]

        l_additional_xaxis, l_additional_yaxis = GFigure._list_from_axis_arguments(
            xaxis, yaxis)

        self.l_xaxis += l_additional_xaxis
        self.l_yaxis += l_additional_yaxis


        
    def plot(self):

        F = plt.figure()
        assert (len(self.l_xaxis) == len(self.l_yaxis))

        for index in range(0, len(self.l_yaxis)):
            self.plot_curve(self.l_xaxis[index], self.l_yaxis[index])

        plt.legend(tuple(self.legend))
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        return F
        

    def plot_curve(self, l_xaxis, l_yaxis):
        """
        l_xaxis and l_yaxis are lists of a numeric type. If l_axis is empty, the default is used.
        """
        if len(l_xaxis):
            plt.plot(l_axis,l_yaxis)
        else:
            plt.plot(l_yaxis)
