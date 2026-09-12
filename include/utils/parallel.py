"""Generic helper to distribute independent function calls over processes.

This module is deliberately free of any domain vocabulary (e.g. Monte Carlo):
it is a thin, reusable wrapper around `concurrent.futures.ProcessPoolExecutor`
that keeps multiprocessing details (spawn context, RNG seeding, thread
oversubscription) out of simulation code.

Example:

    def trial(ind_run, num_points):
        return np.random.randn(num_points).mean()

    if __name__ == '__main__':
        l_means = run_in_workers(trial,
                                 num_calls=100,
                                 num_workers=4,
                                 kwargs=dict(num_points=1000),
                                 b_pass_index=True,
                                 seed=0)

    `trial` must be a module-level function (picklable), and the entry point
    must be guarded by `if __name__ == '__main__':`; see `run_in_workers`'s
    docstring for the full requirements. `l_means` is a length-100 list, and
    would be bit-identical with `num_workers=1`.
"""
import multiprocessing
import os
import random
import sys

from tqdm import tqdm

_threadpool_limiter = None  # kept alive for the lifetime of a worker process


def _pool_initializer():
    """`ProcessPoolExecutor` initializer: restricts each worker to a single
    thread for BLAS/torch operations, on top of the environment variables set
    by the parent process before the pool is created."""
    global _threadpool_limiter
    try:
        import threadpoolctl
        _threadpool_limiter = threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    if 'torch' in sys.modules:
        import torch
        torch.set_num_threads(1)


def _seed_and_call(func, ind_call, seed, b_pass_index, args, kwargs):
    """Task executed by each call: seeds the RNGs deterministically as a
    function of `ind_call` and `seed`, then invokes `func`."""
    import numpy as np
    np.random.seed((seed + ind_call) % 2**32)
    random.seed(seed + ind_call)
    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(seed + ind_call)

    if b_pass_index:
        return func(ind_call, *args, **kwargs)
    return func(*args, **kwargs)


def run_in_workers(func,
                   num_calls,
                   num_workers=1,
                   *,
                   args=(),
                   kwargs=None,
                   b_pass_index=False,
                   seed=None,
                   b_progress=True):
    """Runs `num_calls` independent calls to `func`, distributed over
    processes, and returns a list of length `num_calls` whose `ind_call`-th
    entry is the return value of the `ind_call`-th call.

    Each call is `func(*args, **kwargs)`, or `func(ind_call, *args,
    **kwargs)` if `b_pass_index` is True.

    Args:

        - `num_workers`: if 1, the calls run sequentially in the calling
          process (no pool, no subprocess). If greater than 1, a persistent
          pool of `num_workers` processes is used. If None, `os.cpu_count()`
          workers are used. Any other value (0, negative) raises
          `ValueError`.

        - `seed`: the `ind_call`-th call is preceded by
          `np.random.seed((seed + ind_call) % 2**32)` and
          `random.seed(seed + ind_call)`; if `torch` has already been
          imported at that point, also `torch.manual_seed(seed + ind_call)`.
          This seeding is applied identically whether the calls run
          serially or in parallel, so the results do not depend on
          `num_workers` nor on scheduling order: `num_workers=1` and
          `num_workers=8` give bit-identical results. If `seed is None`, a
          seed is drawn as `np.random.randint(2**31)` in the calling
          process before any call is made; this consumes one draw of the
          global RNG, so an experiment that seeds `np.random` once at the
          top remains reproducible.

        - `b_progress`: if True, a `tqdm` progress bar tracks completed
          calls, in both the serial and the parallel case.

        Exceptions raised by `func` propagate to the caller unchanged (the
        pool, if any, is shut down first).

        - `KeyboardInterrupt` (e.g. Ctrl+C, or `kill -SIGINT <pid>` sent to
          the calling process from another terminal): the worker processes
          are killed immediately rather than left to drain the queue of
          already-submitted calls, so the interrupt takes effect promptly.
          Note that `kill -SIGINT` must target the PID of the process that
          called `run_in_workers` itself; it is not delivered to the worker
          processes directly (they have different PIDs), which is why this
          function kills them explicitly instead of relying on them to
          receive and handle the signal.

    Requirements:

        - `func`, `args` and `kwargs` must be picklable: module-level
          functions, static methods and bound methods of picklable objects
          are fine; lambdas and closures are not.

        - The entry-point script must guard its top-level code with `if
          __name__ == '__main__':` (as `run_experiment.py` does), since
          `num_workers > 1` spawns fresh Python processes that re-import
          that script.

        - `args` and `kwargs` are pickled once per call, so they should be
          small (e.g. a lazily-loading generator is fine, but an object
          holding a large dataset is not; this could be addressed later by
          sending such arguments once per worker through the pool
          initializer, if it ever becomes a bottleneck).

        - This helper does not move torch models across devices: a caller
          that wants to parallelize a GPU/MPS model should place it on the
          CPU first.
    """
    # Step 1: validate `num_workers` and normalize `kwargs`.
    if kwargs is None:
        kwargs = {}
    if num_workers is not None and (not isinstance(num_workers, int)
                                    or num_workers < 1):
        raise ValueError(
            f"num_workers must be a positive int or None, got {num_workers!r}")

    # Step 2: fix the seed up front so every call's RNG state is determined
    # before any call runs, regardless of `num_workers` or scheduling order.
    import numpy as np
    if seed is None:
        seed = np.random.randint(2**31)

    # Step 3: serial path. No pool, no subprocess: just call `_seed_and_call`
    # in order, in the current process.
    if num_workers == 1:
        l_results = [None] * num_calls
        it = range(num_calls)
        if b_progress:
            it = tqdm(it, total=num_calls)
        for ind_call in it:
            l_results[ind_call] = _seed_and_call(func, ind_call, seed,
                                                 b_pass_index, args, kwargs)
        return l_results

    # Parallel path (num_workers > 1 or None).

    # Step 4: cap BLAS/torch intra-op threads to 1 per worker process before
    # the pool is created, so `num_workers` processes don't each spawn
    # several threads and oversubscribe the CPU. Only variables the user
    # has not already set are touched, and they are restored afterwards.
    l_env_vars = [
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'
    ]
    d_prev_env = {}
    for var in l_env_vars:
        if var not in os.environ:
            d_prev_env[var] = None
            os.environ[var] = "1"

    from concurrent.futures import ProcessPoolExecutor, as_completed
    try:
        # Step 5: create a pool of fresh (spawned, not forked) processes,
        # each of which reapplies the thread cap via `_pool_initializer`.
        # Managed manually (rather than via `with`) so that a
        # `KeyboardInterrupt` in step 7 can kill the workers immediately
        # (step 7a) instead of going through `ProcessPoolExecutor.__exit__`,
        # which would call `shutdown(wait=True)` and let every
        # already-submitted call run to completion first.
        mp_context = multiprocessing.get_context('spawn')
        l_results = [None] * num_calls
        executor = ProcessPoolExecutor(max_workers=num_workers,
                                       mp_context=mp_context,
                                       initializer=_pool_initializer)
        try:
            # Step 6: submit one `_seed_and_call` task per call, tracking
            # which call each future belongs to so results can be placed
            # back at the right index regardless of completion order.
            d_future_to_ind = {
                executor.submit(_seed_and_call, func, ind_call, seed, b_pass_index, args, kwargs):
                ind_call
                for ind_call in range(num_calls)
            }

            # Step 7: collect results as calls finish (so the progress bar
            # advances with completions, not submissions); `future.result()`
            # re-raises any exception the call raised.
            it = as_completed(d_future_to_ind)
            if b_progress:
                it = tqdm(it, total=num_calls)
            for future in it:
                ind_call = d_future_to_ind[future]
                l_results[ind_call] = future.result()
        except KeyboardInterrupt:
            # Step 7a: kill the still-running/queued worker processes right
            # away, then let the interrupt propagate. The process list is
            # grabbed before `shutdown` is called: `shutdown` wakes up the
            # executor's internal management thread, which can clear
            # `executor._processes` to None concurrently, so reading it
            # afterwards would race and sometimes see None. `cancel_futures`
            # drops calls that have not started; `.kill()` on each worker
            # process stops the one (if any) that is currently running,
            # rather than waiting for it to finish.
            l_processes = (list(executor._processes.values())
                          if executor._processes else [])
            executor.shutdown(wait=False, cancel_futures=True)
            for process in l_processes:
                process.kill()
            raise
        else:
            executor.shutdown(wait=True)
        return l_results
    finally:
        # Step 8: restore the environment variables touched in step 4,
        # whether or not the pool completed successfully.
        for var, _ in d_prev_env.items():
            del os.environ[var]
