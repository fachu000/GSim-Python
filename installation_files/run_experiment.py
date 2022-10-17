# Python 3.6
#

import sys
import os
from IPython.core.debugger import set_trace
import importlib


def initialize():
    if not os.path.exists("./gsim"):
        # If this behavior were changed, the output file storage functionality should be
        # modified accordingly.
        print(
            "Error: `run_experiment` must be invoked from the folder where it is defined"
        )
        quit()

    sys.path.insert(1, './gsim/')


initialize()

import gsim
import gsim_conf

print("Loading modules...")
module = importlib.import_module(gsim_conf.module_name)
ExperimentSet = getattr(module, "ExperimentSet")
print("done")

########################################################################

if __name__ == '__main__':

    if (len(sys.argv) < 2):
        print(
            'Usage: python3 ', sys.argv[0],
            '[option] <experiment_index> [cmdl_arg1 [cmdl_arg2 ... [cmdl_argN]]]'
        )
        print("""       <experiment_index>: identifier for the experiment 

                cmdl_argn: n-th argument to the experiment function (optional) 

                OPTIONS: 

                -p : plot only the stored results, do not run the simulations.

                -pe : plot only the stored results, do not run the simulations. Export the figures as pdf.
                -pi : load the stored figures and open a pdb prompt to inspect the GFigure objects.
        """)
        quit()

    l_args = sys.argv
    if l_args[1] == "-p":
        # Plot only
        ExperimentSet.plot_only(l_args[2])

    elif l_args[1] == "-pe":
        ExperimentSet.plot_only(l_args[2], save_pdf=True)

    elif l_args[1] == "-pi":
        ExperimentSet.plot_only(l_args[2], inspect=True)

    else:
        if (len(l_args) < 3):
            cmdl_args = ""
        else:
            cmdl_args = l_args[2:]

        ExperimentSet.run_experiment(l_args[1], cmdl_args)

    def set_permisions_recursively(folder):
        for root, dirs, files in os.walk(folder):
            for d in dirs:
                os.chmod(os.path.join(root, d), 16895)
            for f in files:
                os.chmod(os.path.join(root, f), 33279)

    try:
        set_permisions_recursively(gsim.OUTPUT_DATA_FOLDER)
    except Exception as e:
        print(e)
        print(
            f"The permisions of the files and folders in {gsim.OUTPUT_DATA_FOLDER} could not be properly set, possibly because of your operating system. Please do not use the script sync_data until this issue is fixed"
        )

###### Python API ######
""" 
To run an experiment from the iPython shell:

%load_ext autoreload
%autoreload 2
from run_experiment import run
run(1001)

"""


def run(name, *args, **kwargs):
    ExperimentSet.run_experiment(name, *args, **kwargs)
