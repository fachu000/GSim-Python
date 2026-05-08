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
