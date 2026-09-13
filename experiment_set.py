from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import logging

from .utils import time_to_str
import os
import pickle

EXPERIMENT_FUNCTION_BASE_NAME = "experiment_"
OUTPUT_DATA_FOLDER = "./output/"

gsim_logger = logging.getLogger("gsim")

# Set by run_experiment for the duration of the experiment function, so that
# utilities called from inside it (save_to_results_text_file) know where this
# experiment's results go. None outside a run.
_current_run: tuple[type, str] | None = None  # (experiment set class, experiment id)


def is_a_gfigure(obj):
    """Returns True if `obj` is a GFigure object, False otherwise."""
    # We check it this way because importing GFigure in the same way is
    # difficult so that the package works both as a module and standalone.
    return obj.__class__.__name__ == "GFigure"


def results_folder() -> str:
    """Folder where gsim stores the running experiment's GFigures (the .pk file)."""
    if _current_run is None:
        raise RuntimeError("results_folder() called outside of run_experiment.")
    cls, _experiment_id = _current_run
    return cls.experiment_set_data_folder()


def results_file_path(file_type: str, suffix: str = "") -> str:
    """`<results_folder>/experiment_<id>[_<suffix>].<file_type>` for the running experiment."""
    if _current_run is None:
        raise RuntimeError("results_file_path() called outside of run_experiment.")
    cls, experiment_id = _current_run
    f_name = cls._experiment_id_to_f_name(experiment_id)
    if suffix:
        f_name = f"{f_name}_{suffix}"
    return f"{results_folder()}{f_name}.{file_type}"


def save_to_results_text_file(file_type: str, content: str, suffix: str = "") -> str:
    """
    Write `content` to `results_file_path(file_type, suffix)`, creating the folder,
    print "Written results to <path>", and return the path. Text only. Raises
    RuntimeError outside run_experiment.
    """
    path = results_file_path(file_type, suffix)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"Written results to {path}")
    return path


class AbstractExperimentSet:

    def _experiment_id_to_f_name(experiment_id):
        return f"{EXPERIMENT_FUNCTION_BASE_NAME}{experiment_id}"

    @classmethod
    def run_experiment(cls, experiment_id, l_args=[], save_pdf=False, inspect=False, no_plot=False):
        """ Executes the experiment function with identifier <ind_experiment>

        Args:
            experiment_id: experiment function identifier. str or numerical type.
            l_args: list of strings that can be used by the experiment function. 
                Typical usage: number of iterations.
        """

        f_name = cls._experiment_id_to_f_name(experiment_id)

        if f_name in dir(cls):
            start_time = datetime.now()
            gsim_logger.info(
                "----------------------------------------------------------------------")
            gsim_logger.info(f"Starting experiment {experiment_id} at {datetime.now()}.")
            gsim_logger.info(
                "----------------------------------------------------------------------")
            global _current_run
            _current_run = (cls, str(experiment_id))
            try:
                l_G = getattr(cls, f_name)(l_args)
            finally:
                _current_run = None
            end_time = datetime.now()
            gsim_logger.info("Elapsed time = " + time_to_str(end_time - start_time))

            # Set l_G to be a (possibly empty) list of GFigure
            if l_G is None:
                """In this case we store an emtpy list. Otherwise, it is not possible
                to know whether there are no figures because the experiment has not
                been run before or because the experiment produces no figures."""
                l_G = []
            if is_a_gfigure(l_G):
                l_G = [l_G]
            # From this point on, l_G must be a list of GFigure
            if (type(l_G) != list) or (len(l_G) > 0 and not is_a_gfigure(l_G[0])):
                raise Exception("""Function %s returns an unexpected type.
                       It must return either None, a GFigure object,
                       or a list of GFigure objects.""" % f_name)

            # Store and plot
            if len(l_G) == 0:
                gsim_logger.info("The experiment returned no GFigures.")
            else:
                cls._store_fig(l_G, experiment_id)
                if no_plot and not save_pdf:
                    gsim_logger.info("Skipping plotting because `no_plot` is True.")
                else:
                    cls._plot_list_of_GFigure(l_G,
                                              save_pdf=save_pdf,
                                              experiment_id=experiment_id,
                                              inspect=inspect,
                                              show=not no_plot)

        else:
            gsim_logger.error(
                f"Experiment not found: Class {cls.__name__} in module {cls.__module__} contains no function called {f_name}."
            )
            quit()

    @classmethod
    def _plot_list_of_GFigure(cls,
                              l_G,
                              save_pdf=False,
                              experiment_id=None,
                              inspect=False,
                              show=True):

        if inspect:
            gsim_logger.info("The GFigures are available as `l_G`.")
            gsim_logger.info("Press 'c' to continue, save, and plot. ")
            gsim_logger.info(
                "You can type `interact` to enter interactive mode and `Ctr D` to exit. ")
            from IPython.core.debugger import set_trace
            set_trace()
            cls._store_fig(l_G, experiment_id)

        if save_pdf:
            assert experiment_id

            f_name = EXPERIMENT_FUNCTION_BASE_NAME + experiment_id

            # Create the folder if it does not exist
            if not os.path.isdir(OUTPUT_DATA_FOLDER):
                os.mkdir(OUTPUT_DATA_FOLDER)
            target_folder = cls.experiment_set_data_folder()
            if not os.path.isdir(target_folder):
                os.mkdir(target_folder)

        for ind, G in enumerate(l_G):
            G.plot()
            if save_pdf:
                if len(l_G) > 1:
                    file_name = f_name + "-" + str(ind)
                else:
                    file_name = f_name
                G.export(target_folder + file_name)
        if show:
            plt.show()

    @classmethod
    def plot_only(cls, experiment_id, save_pdf=False, inspect=False, no_plot=False):

        f_name = EXPERIMENT_FUNCTION_BASE_NAME + experiment_id
        l_G = cls._load_fig(f_name)
        if l_G is None:  # There is no data for this experiment.
            gsim_logger.error("The experiment %s does not exist or has not been run before." %
                              experiment_id)
        else:
            cls._plot_list_of_GFigure(l_G,
                                      save_pdf=save_pdf,
                                      experiment_id=experiment_id,
                                      inspect=inspect,
                                      show=not no_plot)

    @classmethod
    def experiment_set_data_folder(cls):

        return OUTPUT_DATA_FOLDER + cls.__module__.split(".")[-1] + os.sep

    @classmethod
    def _store_fig(cls, l_G, experiment_id):

        # Create the folder if it does not exist
        if not os.path.isdir(OUTPUT_DATA_FOLDER):
            os.mkdir(OUTPUT_DATA_FOLDER)
        target_folder = cls.experiment_set_data_folder()
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)
        file_name = cls._experiment_id_to_f_name(experiment_id) + ".pk"

        gsim_logger.info("Storing figure as %s" % target_folder + file_name)
        pickle.dump(l_G, open(target_folder + file_name, "wb"))

    @classmethod
    def _load_fig(cls, f_name):
        """
        Returns a list of GFigure objects if the file exists. Else, it returns None.

        """

        target_folder = cls.experiment_set_data_folder()
        file_name = f_name + ".pk"
        if not os.path.isfile(target_folder + file_name):
            return None

        return pickle.load(open(target_folder + file_name, "rb"))

    @classmethod
    def load_GFigures(cls, experiment_id):
        """
        Returns a list of GFigure objects if the file containing the output of
        experiment `experiment_id` exists. Else, it returns None.
        """

        f_name = f"{EXPERIMENT_FUNCTION_BASE_NAME}{experiment_id}"
        return cls._load_fig(f_name)
