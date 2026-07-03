"""Just decorate the functions you want to profile with @profile and set
b_profiling to True by using `set_profiling`. A profile will be printed to the
console after every request. 

Those functions must be executed after start_profiler and before
stop_profiler_and_print.

Recall that the units of the profile are printed at the beginning of its output,
e.g. "Timer unit: 1e-09 s".

Note that profiling is incompatible with the debugger, in the sense that the
execution will not stop at breakpoints when profiling is enabled.

"""

from line_profiler import LineProfiler
import threading
import logging

logger = logging.getLogger("django")

# Create a thread-local variable
local_data = threading.local()

b_profiling = False


def set_profiling(b_profiling_: bool):
    global b_profiling
    b_profiling = b_profiling_


def profile(func):
    # This is a decorator for functions that should be profiled.

    # Get the profiler for the current thread
    if not b_profiling:
        return func

    def profiled_func(*args, **kwargs):
        profiler = getattr(local_data, 'profiler', None)
        if profiler is None:
            raise ValueError(
                "You must call start_profiler before calling a function decorated with @profile"
            )
        add_func_to_profiler_if_not_added(profiler, func)
        return func(*args, **kwargs)

    return profiled_func


def add_func_to_profiler_if_not_added(profiler: LineProfiler, func):
    """ A function should not be added multiple times to the profiler, else the
    hits are reset."""
    if func not in profiler.functions:
        profiler.add_function(func)


def start_profiler():
    if not b_profiling:
        return
    logger.warning("Profiler enabled.")
    profiler = LineProfiler()
    local_data.profiler = profiler
    profiler.enable_by_count()


def stop_profiler_and_print():
    if not b_profiling:
        return
    profiler = getattr(local_data, 'profiler', None)
    if profiler is not None:
        profiler.print_stats()
        del local_data.profiler
