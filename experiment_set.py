from IPython.core.debugger import set_trace
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from gsim.gfigure import GFigure

EXPERIMENT_FUNCTION_BASE_NAME = "experiment_"


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
            if type(G) == GFigure:
                G.plot()
                plt.show()                
            else:
                print("No figures to show")

        else:
            raise Exception("Experiment not found")

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
