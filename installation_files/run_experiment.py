# Python 3.6
#

import sys
import os
import importlib
from gsim import init_gsim_logger

gsim_logger = init_gsim_logger()


def initialize():

    if not os.path.exists("./gsim"):
        # If this behavior were changed, the output file storage functionality should be
        # modified accordingly.
        print(
            "Error: `run_experiment` must be invoked from the folder where it is defined"
        )
        quit()

    # Add the parent directory (rme) to the Python path so that ml_estimation
    # can be imported as a proper package with relative imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # This is the rme directory
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    sys.path.insert(1, './gsim/')


initialize()

import gsim
import gsim_conf


def process_module_name(module_name):

    if module_name.endswith('.py'):
        module_name = module_name[:-3]

    current_dir = os.path.basename(os.getcwd())
    experiment_dir = current_dir

    module_rel_path = os.path.join(*module_name.split('.')) + '.py'

    if not os.path.exists(os.path.join(module_rel_path)):
        if not os.path.exists(os.path.join('experiments', module_rel_path)):
            gsim_logger.error(
                f"The experiment module {module_name} was not found in the current directory or in the experiments subfolder."
            )
            quit()
        experiment_dir = current_dir + '.' + 'experiments'

    module_name_with_package = f"{experiment_dir}.{module_name}"
    return module_name_with_package


def load_modules(experiment_module=None):
    gsim_logger.info("Loading modules...")
    # Import the module with the proper package context so relative imports work

    module_name = experiment_module or gsim_conf.module_name
    module_name_with_package = process_module_name(module_name)
    gsim_logger.info(f"Loading experiments from {module_name_with_package}...")
    module = importlib.import_module(module_name_with_package)
    ExperimentSet = getattr(module, "ExperimentSet")
    gsim_logger.info("Finished loading modules.")
    return ExperimentSet


########################################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=
        "Run the experiment function with index <experiment_index> in the module specified in gsim_conf.py."
    )
    parser.add_argument(
        '-x',
        '--xmod',
        help='Select the experiment module (e.g. example_experiments).',
        default=None)
    parser.add_argument(
        'experiment_index',
        nargs='?',
        default=None,
        help=
        'Index (ID) of experiment to run; optional, falls back to gsim_conf.default_experiment_index if omitted.'
    )
    parser.add_argument(
        'experiment_args',
        nargs='*',
        default=[],
        help='list of arguments passed to the experiment function')
    parser.add_argument(
        '-p',
        '--plot_only',
        help=
        'plot the results stored from the last execution (the experiment will not be run).',
        action="store_true")
    parser.add_argument('-e',
                        '--export',
                        help='export the figures as PDF.',
                        action="store_true")
    parser.add_argument(
        '-i',
        '--inspect',
        help=
        'load the stored figures and open a pdb prompt to inspect the GFigure objects.',
        action="store_true")

    parser.add_argument('-g', '--gpu', help='Select the GPU.', default=None)

    args, unknown_args = parser.parse_known_args()
    ExperimentSet = load_modules(args.xmod)
    if len(unknown_args):
        print('WARNING: The following arguments were not recognized:')
        print(unknown_args)
    if len(args.experiment_args):
        if args.plot_only:
            print(
                "WARNING: the following experiment arguments were specified but will not be passed to the experiment since it will not be run."
            )
        else:
            print("The following arguments will be passed to the experiment:")
        print(args.experiment_args)

    experiment_index = args.experiment_index or gsim_conf.default_experiment_index

    if args.plot_only:
        ExperimentSet.plot_only(experiment_index,
                                save_pdf=args.export,
                                inspect=args.inspect)
    else:
        if args.gpu is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        ExperimentSet.run_experiment(experiment_index,
                                     args.experiment_args,
                                     save_pdf=args.export,
                                     inspect=args.inspect)

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
else:
    ExperimentSet = load_modules()
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
