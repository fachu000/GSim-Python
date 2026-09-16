import os

import pytest

import gsim
from gsim import experiment_set as experiment_set_module


class _ThrowawayExperimentSet(gsim.AbstractExperimentSet):

    def experiment_hello(l_args):
        gsim.save_to_results_text_file('md', 'hello')
        return None


class _ThrowawayExperimentSetWithFileKey(gsim.AbstractExperimentSet):

    def experiment_hello(l_args):
        gsim.save_to_results_text_file('csv', 'a,b', file_key='predictions')
        return None


def test_save_to_results_text_file_writes_next_to_the_pk_with_default_file_key(
        tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_set_module, 'OUTPUT_DATA_FOLDER', str(tmp_path) + os.sep)

    _ThrowawayExperimentSet.run_experiment('hello', no_plot=True)

    target_folder = _ThrowawayExperimentSet.experiment_set_data_folder()
    path = os.path.join(target_folder, 'experiment_hello_results.md')
    assert os.path.exists(path)
    with open(path) as f:
        assert f.read() == 'hello'


def test_save_to_results_text_file_uses_the_given_file_key(
        tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_set_module, 'OUTPUT_DATA_FOLDER', str(tmp_path) + os.sep)

    _ThrowawayExperimentSetWithFileKey.run_experiment('hello', no_plot=True)

    target_folder = _ThrowawayExperimentSetWithFileKey.experiment_set_data_folder()
    assert os.path.exists(os.path.join(target_folder, 'experiment_hello_predictions.csv'))


def test_save_to_results_text_file_raises_outside_a_run():
    with pytest.raises(RuntimeError):
        gsim.save_to_results_text_file('md', 'hello')
