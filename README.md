# GSim-Python

General-purpose simulation environment for Python.


To download, cd to your git repository. Once there, type this:

$ git submodule add https://github.com/fachu000/GSim-Python.git ./gsim

To install:

$ bash gsim/install.sh

Create your experiments as functions in a module in the folder
Experiments. There is an example of such a module there.

To run experiment 1002 in experiments.example_experiments, type

$ python run_experiment.py 1002

The full potential of GSim is exploited when the experiment functions
return GFigures rather than directly plotting figures. GFigures are
stored and can be plotted and edited afterwards without having to run
again the experiment.

For example, to see the figures created last time experiment 1002 was
run but without running it again, type

$ python run_experiment.py -p 1002

GFigure offers a neater interface than matplotlib, whose goal was to
resemble the interface of MATLAB. Figures are collections of
subfigures, subfigures are collections of curves. Simple to
understand. See gsim.gfigure.example_figures for a small tutorial on
how to use GFigure.

Pull requests are welcome!!
