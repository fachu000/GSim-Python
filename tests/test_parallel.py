import multiprocessing
import os
import signal
import threading
import time

import numpy as np
import pytest

from gsim.include.utils.parallel import run_in_workers


def _return_random_vec():
    return np.random.rand(3)


def _return_index(ind_call):
    return ind_call


def _add(a, b=0):
    return a + b


def _raise_value_error():
    raise ValueError("boom")


def _sleep_a_while():
    time.sleep(5)
    return 1


def test_serial_equals_parallel():
    num_calls = 4
    l_serial = run_in_workers(_return_random_vec,
                              num_calls=num_calls,
                              num_workers=1,
                              seed=0,
                              b_progress=False)
    l_parallel = run_in_workers(_return_random_vec,
                                num_calls=num_calls,
                                num_workers=3,
                                seed=0,
                                b_progress=False)
    assert len(l_serial) == num_calls
    assert len(l_parallel) == num_calls
    for v_s, v_p in zip(l_serial, l_parallel):
        assert np.array_equal(v_s, v_p)


def test_calls_are_independent():
    num_calls = 5
    l_results = run_in_workers(_return_random_vec,
                               num_calls=num_calls,
                               num_workers=1,
                               b_progress=False)
    for ind_i in range(num_calls):
        for ind_j in range(ind_i + 1, num_calls):
            assert not np.array_equal(l_results[ind_i], l_results[ind_j])


@pytest.mark.parametrize("num_workers", [1, 3])
def test_pass_index(num_workers):
    num_calls = 6
    l_results = run_in_workers(_return_index,
                               num_calls=num_calls,
                               num_workers=num_workers,
                               b_pass_index=True,
                               b_progress=False)
    assert l_results == list(range(num_calls))


@pytest.mark.parametrize("num_workers", [1, 3])
def test_args_kwargs(num_workers):
    l_results = run_in_workers(_add,
                               num_calls=3,
                               num_workers=num_workers,
                               args=(2, ),
                               kwargs={'b': 3},
                               b_progress=False)
    assert l_results == [5, 5, 5]


@pytest.mark.parametrize("num_workers", [1, 3])
def test_exception_propagates(num_workers):
    with pytest.raises(ValueError, match="boom"):
        run_in_workers(_raise_value_error,
                       num_calls=3,
                       num_workers=num_workers,
                       b_progress=False)


def test_invalid_num_workers():
    with pytest.raises(ValueError):
        run_in_workers(_return_index,
                       num_calls=3,
                       num_workers=0,
                       b_progress=False)


def test_num_workers_none():
    l_results = run_in_workers(_return_index,
                               num_calls=3,
                               num_workers=None,
                               b_pass_index=True,
                               b_progress=False)
    assert l_results == [0, 1, 2]


def test_keyboard_interrupt_kills_workers():
    """A `KeyboardInterrupt` while calls are still running (e.g. from `kill
    -SIGINT <pid>` sent to the calling process) must terminate the worker
    processes promptly instead of waiting for them to drain the queue of
    submitted calls, and must not leave orphaned worker processes behind."""
    def _send_sigint():
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=_send_sigint, daemon=True).start()

    start = time.time()
    with pytest.raises(KeyboardInterrupt):
        run_in_workers(_sleep_a_while,
                       num_calls=4,
                       num_workers=4,
                       b_progress=False)
    elapsed = time.time() - start

    # The calls sleep for 5 s each; a prompt interrupt should return well
    # before that.
    assert elapsed < 4

    time.sleep(0.5)  # let the killed worker processes be reaped
    assert multiprocessing.active_children() == []
