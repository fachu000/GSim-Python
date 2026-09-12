import os
import pickle
from typing import Callable, TypeVar

T = TypeVar("T")


def load_or_compute(callback: Callable[[], T], path: str) -> T:
    """If `path` exists, loads and returns the pickled object stored there.
    Otherwise, calls `callback()`, pickles the result to `path` (creating
    parent directories as needed), and returns the result.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        pass

    value = callback()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f)
    return value


def load_cp_or_compute(file: str, key, fun: Callable[[], T]) -> T:
    """Checkpointed variant of `load_or_compute`: `file` holds a dict
    mapping arbitrary (picklable, hashable) keys to previously computed
    values, rather than a single value per file. If `key` is already a key
    of that dict, returns the associated value. Otherwise, calls `fun()`,
    stores the result under `key`, saves the dict back to `file` (creating
    `file` and its parent directories if they do not exist), and returns
    the result.

    Intended for sweeps with many independent, expensive-to-compute values
    (e.g. one metric per point of a parameter grid): each value is
    persisted to `file` as soon as it is computed, so an interrupted sweep
    can be resumed without recomputing what is already in `file`.
    """
    try:
        with open(file, "rb") as f:
            d_cache = pickle.load(f)
    except FileNotFoundError:
        d_cache = {}

    if key in d_cache:
        return d_cache[key]

    value = fun()
    d_cache[key] = value
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "wb") as f:
        pickle.dump(d_cache, f)
    return value
