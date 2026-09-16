import os

from gsim.include.utils.files import load_or_compute_checkpoint, load_or_compute


def test_load_or_compute_calls_callback_once(tmp_path):
    path = str(tmp_path / "sub" / "value.pk")
    l_calls = []

    def callback():
        l_calls.append(1)
        return 42

    assert load_or_compute(callback, path, file_key=None) == 42
    assert load_or_compute(callback, path, file_key=None) == 42
    assert len(l_calls) == 1
    assert os.path.exists(path)


def test_load_cp_or_compute_caches_per_key(tmp_path):
    path = str(tmp_path / "sub" / "checkpoint.pk")
    l_calls = []

    def make_fun(value):

        def fun():
            l_calls.append(value)
            return value

        return fun

    assert load_or_compute_checkpoint(
        path, file_key=None, checkpoint_key="a", f_compute=make_fun(1)) == 1
    assert load_or_compute_checkpoint(
        path, file_key=None, checkpoint_key="b", f_compute=make_fun(2)) == 2
    assert load_or_compute_checkpoint(
        path, file_key=None, checkpoint_key="a",
        f_compute=make_fun(999)) == 1

    assert l_calls == [1, 2]
    assert os.path.exists(path)


def test_load_cp_or_compute_persists_across_calls_to_a_new_file_handle(
        tmp_path):
    path = str(tmp_path / "checkpoint.pk")

    assert load_or_compute_checkpoint(
        path, file_key=None, checkpoint_key="k",
        f_compute=lambda: "first") == "first"
    # Simulates a resumed run: a fresh call with the same `checkpoint_key`
    # must read the value that a previous call already saved to `path`.
    assert load_or_compute_checkpoint(
        path, file_key=None, checkpoint_key="k",
        f_compute=lambda: "second") == "first"
