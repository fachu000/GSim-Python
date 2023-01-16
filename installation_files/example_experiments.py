import numpy as np

import gsim
from gsim.gfigure import GFigure


class ExperimentSet(gsim.AbstractExperimentSet):

    def experiment_1001(l_args):
        print("This is an empty experiment.")

    def experiment_1002(l_args):
        """The full potential of GSim is exploited by returning GFigures
        rather than directly plotting them. GFigures are stored and
        can be plotted and edited afterwards without having to run
        again the experiment.

        GFigure offers a neater interface than matplotlib, whose goal
        was to resemble MATLAB's interface. 

        See gsim.gfigure.example_figures for examples on how to use
        GFigure.

        """
        print("This experiment plots a figure.")

        v_x = np.linspace(0, 10, 20)
        v_y1 = v_x**2 - v_x + 3

        # Example with a single curve, single subplot
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabola",
                    legend="P1")

        return G

    def experiment_1003(l_args):
        """ 
        In some occasions, it may be useful to access a GFigure created by a
        previously-run experiment; e.g. to combine multiple figures. 
        """

        l_G = ExperimentSet.load_GFigures(1002)
        G = GFigure()
        # Same subplot twice
        G.l_subplots = l_G[0].l_subplots + l_G[0].l_subplots
        return G
