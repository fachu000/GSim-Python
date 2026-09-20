import fcntl
import logging
import os
import pickle
from pathlib import Path
from typing import Callable, TypeVar

gsim_logger = logging.getLogger("gsim")

T = TypeVar("T")


def _resolve_path(path: Path | str | None,
                  file_key: str | None,
                  file_type: str = "pk") -> str:
    """Resolves the `path`/`file_key` kwargs into a file path.

    If `path` is `None`, returns `results_file_path(file_key, file_type)` for
    the currently running experiment. Otherwise, `file_key` must be `None`, and
    `path` is returned unchanged. 
    
    Examples:
        - _resolve_path(None, "results", "md") returns
          `<results_folder>/experiment_<id>_results.md` 
          
        - _resolve_path("my_file.txt", None) returns "my_file.txt"
    
    """
    if path is None:
        # Imported lazily: `experiment_set` re-exports the functions defined in
        # this module, so importing it at module scope would create a circular
        # import. Remove once the TODO in `experiment_set.py` is resolved.
        from ...experiment_set import results_file_path
        return results_file_path(file_key, file_type)
    if file_key is not None:
        raise ValueError("file_key must be None when path is provided.")
    return path


def load_or_compute(callback: Callable[[], T],
                    path: Path | str | None = None,
                    file_key: str | None = "computed_object",
                    force_compute: bool = False,
                    verbosity: int = 1) -> T:
    """If the resolved path (see `_resolve_path`) points to an existing file,
    loads and returns the pickled object stored there. Otherwise, calls
    `callback()`, pickles the result to that path (creating parent directories
    as needed), and returns the result.

    Args:
        callback: called to compute the value when it is not already
            cached.
        path: file to load from/save to. If `None`, it is set to
            `results_file_path(file_key)`. If provided, `file_key` must be
            `None`.
        file_key: passed to `results_file_path` to obtain `path` when
            `path` is `None`. Must be `None` when `path` is provided.
        force_compute: if `True`, skips loading from the file and always
            calls `callback()`, overwriting the file with the result.
        verbosity: 0 logs nothing. 1 (default) logs write operations. 2 also
            logs cache misses.
    """
    path = _resolve_path(path, file_key)
    if not force_compute:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            if verbosity >= 2:
                gsim_logger.info(
                    f"Running computation because {path} does not exist")
    elif verbosity >= 2:
        gsim_logger.info(f"Running computation because force_compute=True "
                         f"for {path}")

    value = callback()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f)
    if verbosity >= 1:
        gsim_logger.info(f"Written computation to {path}")
    return value


def load_or_compute_checkpoint(path: Path | str | None = None,
                               file_key: str | None = "checkpoints",
                               *,
                               checkpoint_key,
                               f_compute: Callable[[], T],
                               force_compute: bool = False,
                               verbosity: int = 1) -> T:
    """Keyed variant of `load_or_compute`: the file at the resolved path
    (which is a function of `path`, `file_key`, and the current experiment)
    holds a dict mapping arbitrary (picklable, hashable) keys to previously
    computed values. That is, the dict is of the form
                    {checkpoint_key: f_compute()}.
    The behavior is as follows:

        - If `checkpoint_key` is already a key of that dict, returns the
          associated  value.

        - Otherwise, calls `f_compute()`, stores the result as the value
          corresponding to the key `checkpoint_key` in the dict, saves the dict
          back to the resolved path (creating it and its parent directories if
          they do not exist), and returns the result.

    Args:
        path: file where the checkpoint dict is stored. If `None`, it is
            set to `results_file_path(file_key)`. If provided, `file_key` must
            be `None`.
        file_key: passed to `results_file_path` to obtain `path` when
            `path` is `None`. Must be `None` when `path` is provided.
        checkpoint_key: key under which the result of `f_compute()` is
            stored in, and retrieved from, the checkpoint dict. Mandatory.
        f_compute: callable that computes the value when it is not already
            cached. Mandatory.
        force_compute: if `True`, skips the cache lookup and always calls
            `f_compute()`, overwriting the entry for `checkpoint_key` in the
            checkpoint dict with the result.
        verbosity: 0 logs nothing. 1 (default) logs write operations. 2 also
            logs cache misses.

    Intended for sweeps with many independent, expensive-to-compute values (e.g.
    one metric per point of a parameter grid): each value is persisted as soon
    as it is computed, so an interrupted sweep can be resumed without
    recomputing what was already saved.

    Several processes may share a checkpoint file (e.g. copies of an
    experiment computing disjoint sets of keys in parallel): the dict is
    re-read and merged under an exclusive lock on a sidecar `.lock` file
    right before it is written, so that a value written by another process
    while `f_compute()` was running is not lost. `f_compute()` itself runs
    outside the lock. Two processes computing the same key both write it;
    the value of the last one to finish remains.
    """
    path = _resolve_path(path, file_key)
    try:
        with open(path, "rb") as f:
            d_cache = pickle.load(f)
    except FileNotFoundError:
        d_cache = {}

    if not force_compute and checkpoint_key in d_cache:
        return d_cache[checkpoint_key]

    if verbosity >= 2:
        if force_compute:
            gsim_logger.info(
                f"Running computation because force_compute=True for key "
                f"{checkpoint_key} in {path}")
        else:
            gsim_logger.info(
                f"Running computation because the key {checkpoint_key} is "
                f"missing in {path}")

    value = f_compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as f_lock:
        fcntl.flock(f_lock, fcntl.LOCK_EX)
        try:
            with open(path, "rb") as f:
                d_cache = pickle.load(f)
        except FileNotFoundError:
            d_cache = {}
        d_cache[checkpoint_key] = value
        with open(path, "wb") as f:
            pickle.dump(d_cache, f)
        fcntl.flock(f_lock, fcntl.LOCK_UN)
    if verbosity >= 1:
        gsim_logger.info(
            f"Written computation for key {checkpoint_key} to {path}")
    return value


def save_to_results_text_file(file_type: str,
                              content: str,
                              path: Path | str | None = None,
                              file_key: str | None = "results",
                              verbosity: int = 1) -> str:
    """
    Write `content` to a file at the resolved path. It also:
        - creates the parent folder if needed,
        - logs "Written results to <path>",
        - returns the path of the file.
    The content is text only. Raises RuntimeError if `path` is `None` and this
    is called outside run_experiment.

    Args:
        file_type: extension of the file to write, e.g. "md" or "csv".
            Only used to build the path when `path` is `None`.

        content: text to write to the file.

        path: file to write to. If `None`, it is set to
            `results_file_path(file_key, file_type)`. If provided, `file_key`
            must be `None`.

        file_key: passed to `results_file_path` to obtain `path` when
            `path` is `None`. Must be `None` when `path` is provided.
        verbosity: 0 logs nothing. 1 (default) logs the write operation. This
            function has no cache to miss, so verbosity 2 behaves like 1.
    """
    path = _resolve_path(path, file_key, file_type)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    if verbosity >= 1:
        gsim_logger.info(f"Written results to {path}")
    return path
