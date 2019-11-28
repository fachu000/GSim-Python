from IPython.core.debugger import set_trace
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from gsim.gfigure import GFigure
import os
import pickle

EXPERIMENT_FUNCTION_BASE_NAME = "experiment_"
OUTPUT_DATA_FOLDER = "./output/"


class AbstractExperimentSet:
    @classmethod
    def run_experiment(cls, experiment_id, l_args):
        """ Executes the experiment function with identifier <ind_experiment>

        Args:
            experiment_id: experiment function identifier
            l_args: list of strings that can be used by the experiment function. 
                Typical usage: number of iterations.
        """

        f_name = EXPERIMENT_FUNCTION_BASE_NAME + experiment_id

        if f_name in dir(cls):
            start_time = datetime.now()
            print("Running experiment %s" % experiment_id)
            G = getattr(cls, f_name)(l_args)
            end_time = datetime.now()
            AbstractExperimentSet.print_time(start_time, end_time)
            if type(G) == GFigure or (type(G) == list
                                      and type(G[0]) == GFigure):
                # set_trace()
                cls.store_fig(G, f_name)
                G.plot()
                plt.show()
            else:
                print("No figures to show")

        else:
            raise Exception("Experiment not found")

    @classmethod
    def plot_only(cls, experiment_id):

        f_name = EXPERIMENT_FUNCTION_BASE_NAME + experiment_id
        l_G = cls.load_fig(f_name)
        for G in l_G:
            G.plot()
        plt.show()

    def print_time(start_time, end_time):
        td = end_time - start_time
        hours = td.seconds // 3600
        reminder = td.seconds % 3600
        minutes = reminder // 60
        seconds = (td.seconds - hours * 3600 -
                   minutes * 60) + td.microseconds / 1e6
        time_str = ""
        if td.days:
            time_str = "%d days, " % td.days
        if hours:
            time_str = time_str + "%d hours, " % hours
        if minutes:
            time_str = time_str + "%d minutes, " % minutes
        if time_str:
            time_str = time_str + "and "

        time_str = time_str + "%.3f seconds" % seconds
        #set_trace()
        print("Elapsed time = ", time_str)

    @classmethod
    def experiment_set_data_folder(cls):

        return OUTPUT_DATA_FOLDER + cls.__module__.split(".")[-1] + os.sep

    @classmethod
    def store_fig(cls, G, f_name):

        # Create the folder if it does not exist
        if not os.path.isdir(OUTPUT_DATA_FOLDER):
            os.mkdir(OUTPUT_DATA_FOLDER)
        target_folder = cls.experiment_set_data_folder()
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)
        file_name = f_name + ".pk"

        # Unify format
        if type(G) == GFigure:
            l_G = [G]
        elif (type(G) == list) and (type(G[0]) == GFigure):
            l_G = G
        else:
            raise Exception("Function %s returns an unexpected type" % f_name)

        pickle.dump(l_G, open(target_folder + file_name, "wb"))

    @classmethod
    def load_fig(cls, f_name):
        """
        Returns a list of GFigure objects if the file exists. Else, it returns None.

        """

        target_folder = cls.experiment_set_data_folder()
        file_name = f_name + ".pk"
        if not os.path.isfile(target_folder + file_name):
            return None

        return pickle.load(open(target_folder + file_name, "rb"))
