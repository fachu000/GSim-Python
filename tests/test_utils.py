import math
import os
from datetime import timedelta

import pytest

from gsim import time_to_str
from gsim.utils import xor, startTimer, printTimer, d_timers


def test_xor_basic_truth_table():
    # Truth table checks
    assert xor(False, False) is False
    assert xor(True, False) is True
    assert xor(False, True) is True
    assert xor(True, True) is False


def test_time_to_str_seconds_only():
    td = timedelta(seconds=5, microseconds=123456)
    out = time_to_str(td)
    assert out.endswith("seconds")
    # Seconds with microsecond fraction ~5.123
    assert "5.123" in out or "5.124" in out  # allow rounding variance


def test_time_to_str_with_days_hours_minutes():
    td = timedelta(days=2, hours=3, minutes=4, seconds=5)
    out = time_to_str(td)
    # Format: '2 days, 3 hours, 4 minutes, and 5.000 seconds'
    assert "2 days" in out
    assert "3 hours" in out
    assert "4 minutes" in out
    assert out.strip().endswith("seconds")


def test_timer_start_and_print(capsys):
    name = "sample"
    startTimer(name)
    # Simulate some elapsed time by modifying the stored start time backwards
    d_timers[name] = d_timers[name] - timedelta(seconds=1, microseconds=500000)
    printTimer(name)
    captured = capsys.readouterr().out
    assert name in captured
    assert "seconds" in captured


def test_experiment_id_name_generation():
    from gsim.experiment_set import AbstractExperimentSet
    # Access the helper (defined without self/cls) directly
    f = getattr(AbstractExperimentSet, "_experiment_id_to_f_name")
    assert f(10) == "experiment_10"


# Note: Additional tests for GFigure and plotting can be added by switching the matplotlib backend to 'Agg'.
