import logging
import os
import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset

from gsim.include.neural_net import NeuralNet, WeightedLoss
from gsim.include.neural_net.neural_net import TrainingHistory
from gsim.include.neural_net.normalizers import DefaultNormalizer
from gsim.include.utils.statistics import mean_and_ci


# Helper network definitions (defined at module level to be picklable)
class _SimpleNetworkTensor(NeuralNet):
    """Helper network for tensor input/output tests."""

    def __init__(self):
        super().__init__()
        self.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, M , N)
        Returns:
            output tensor of shape (batch_size, 2*M, 3*N, 4)            
        """
        return x[..., None].tile(1, 2, 3, 4)


class _SimpleNetworkListTuple(NeuralNet[list[torch.Tensor], tuple[torch.Tensor,
                                                                  ...],
                                        tuple[torch.Tensor, ...]]):
    """Helper network for list input and tuple output tests."""

    def __init__(self):
        super().__init__()
        self.initialize()

    def forward(self,
                x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input list of three tensors of shape (batch_size, M1 , N1),
               (batch_size, M1 , N2), (batch_size, M2 , N2)
        Returns:
            output: tuple of two tensors of shape (batch_size, M1, N1+N2) and (batch_size, M1+M2, N2)
        """
        x_1, x_2, x_3 = x
        out_1 = torch.concat((x_1, x_2), dim=2)
        out_2 = torch.concat((x_2, x_3), dim=1)
        return (out_1, out_2)


# Tests for uncollate_fn ######################################################
def test_uncollate_fn_when_output_is_a_tuple():
    # In this test, the output of the network is a tuple of three tensors.

    class TestNet(NeuralNet):

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([0])

    net = TestNet()

    batch_size_1 = 4
    out_item_11 = torch.randint(low=0, high=10, size=(batch_size_1, 5, 6))
    out_item_12 = torch.randint(low=0, high=10, size=(batch_size_1, 6, 7))
    out_item_13 = torch.randint(low=0, high=10, size=(batch_size_1, 7, 8))

    batch_size_2 = 3
    out_item_21 = torch.randint(low=0, high=10, size=(batch_size_2, 5, 6))
    out_item_22 = torch.randint(low=0, high=10, size=(batch_size_2, 6, 7))
    out_item_23 = torch.randint(low=0, high=10, size=(batch_size_2, 7, 8))

    batch_1 = (out_item_11, out_item_12, out_item_13)

    batch_2 = (out_item_21, out_item_22, out_item_23)

    l_batches = [batch_1, batch_2]

    l_out = net.uncollate_fn(l_batches)

    assert len(l_out) == batch_size_1 + batch_size_2
    assert torch.equal(l_out[0][0], out_item_11[0])
    assert torch.equal(l_out[0][1], out_item_12[0])
    assert torch.equal(l_out[0][2], out_item_13[0])
    assert torch.equal(l_out[1][0], out_item_11[1])
    assert torch.equal(l_out[1][1], out_item_12[1])
    assert torch.equal(l_out[1][2], out_item_13[1])
    assert torch.equal(l_out[2][0], out_item_11[2])
    assert torch.equal(l_out[2][1], out_item_12[2])
    assert torch.equal(l_out[2][2], out_item_13[2])
    assert torch.equal(l_out[3][0], out_item_11[3])
    assert torch.equal(l_out[3][1], out_item_12[3])
    assert torch.equal(l_out[3][2], out_item_13[3])
    assert torch.equal(l_out[4][0], out_item_21[0])
    assert torch.equal(l_out[4][1], out_item_22[0])
    assert torch.equal(l_out[4][2], out_item_23[0])
    assert torch.equal(l_out[5][0], out_item_21[1])
    assert torch.equal(l_out[5][1], out_item_22[1])
    assert torch.equal(l_out[5][2], out_item_23[1])
    assert torch.equal(l_out[6][0], out_item_21[2])
    assert torch.equal(l_out[6][1], out_item_22[2])
    assert torch.equal(l_out[6][2], out_item_23[2])

    print("hello")


def test_uncollate_fn_when_output_is_a_tensor():
    # In this test, the output of the network is a single tensor.
    class TestNet(NeuralNet):

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([0])

    net = TestNet()

    batch_size_1 = 4
    out_item_1 = torch.randint(low=0, high=10, size=(batch_size_1, 5, 6))

    batch_size_2 = 3
    out_item_2 = torch.randint(low=0, high=10, size=(batch_size_2, 5, 6))

    l_batches = [out_item_1, out_item_2]

    l_out = net.uncollate_fn(l_batches)

    assert len(l_out) == batch_size_1 + batch_size_2
    assert torch.equal(l_out[0], out_item_1[0])
    assert torch.equal(l_out[1], out_item_1[1])
    assert torch.equal(l_out[2], out_item_1[2])
    assert torch.equal(l_out[3], out_item_1[3])
    assert torch.equal(l_out[4], out_item_2[0])
    assert torch.equal(l_out[5], out_item_2[1])
    assert torch.equal(l_out[6], out_item_2[2])

    print("hello")


# Tests for predict ###########################################################
def test_predict_when_input_and_output_are_tensors():

    net = _SimpleNetworkTensor()

    input = torch.randint(low=0, high=10, size=(7, 5, 6))

    # Output of predict is a tensor
    output = net.predict(input, batch_size=3)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (7, 10, 18, 4)

    # Output of predict is a list
    output = net.predict(input, batch_size=3, output_class=tuple)
    assert isinstance(output, tuple)
    assert len(output) == 7
    for ind_output in range(7):
        assert output[ind_output].shape == (10, 18, 4)

    # Output of predict is a Dataset
    output = net.predict(input, batch_size=3, output_class=Dataset)
    assert isinstance(output, Dataset)
    assert len(output) == 7
    for ind_output in range(7):
        assert output[ind_output].shape == (10, 18, 4)


def test_predict_when_the_input_is_a_list_and_the_output_is_a_tuple():

    net = _SimpleNetworkListTuple()

    num_inputs = 7
    inputs = [[
        torch.randint(low=0, high=10, size=(5, 6)),
        torch.randint(low=0, high=10, size=(5, 8)),
        torch.randint(low=0, high=10, size=(9, 8))
    ] for _ in range(num_inputs)]

    # Output of predict is a list
    outputs = net.predict(inputs, batch_size=6)
    assert isinstance(outputs, list)
    assert len(outputs) == 7
    for ind_output in range(7):
        assert isinstance(outputs[ind_output], tuple)
        assert len(outputs[ind_output]) == 2
        assert outputs[ind_output][0].shape == (5, 14)
        assert outputs[ind_output][1].shape == (14, 8)

    # Output of predict is a tuple
    outputs = net.predict(inputs, batch_size=4, output_class=tuple)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 7
    for ind_output in range(7):
        assert isinstance(outputs[ind_output], tuple)
        assert len(outputs[ind_output]) == 2
        assert outputs[ind_output][0].shape == (5, 14)
        assert outputs[ind_output][1].shape == (14, 8)

    # Output of predict is a Dataset
    outputs = net.predict(inputs, batch_size=4, output_class=Dataset)
    assert isinstance(outputs, Dataset)
    assert len(outputs) == 7
    for ind_output in range(7):
        assert isinstance(outputs[ind_output], tuple)
        assert len(outputs[ind_output]) == 2
        assert outputs[ind_output][0].shape == (5, 14)
        assert outputs[ind_output][1].shape == (14, 8)

    # Use the outputs to create a dataset of pairs (input, target)
    example_dataset = NeuralNet.NeuralNetDataset(l_items=list(
        zip(inputs, outputs)))  # type: ignore
    outputs = net.predict(example_dataset, batch_size=4, no_targets=False)
    assert isinstance(outputs, Dataset)
    assert len(outputs) == 7
    for ind_output in range(7):
        prev_output = example_dataset[ind_output][1]
        current_output = outputs[ind_output]
        assert isinstance(current_output, tuple)
        assert isinstance(prev_output, tuple)
        assert len(current_output) == 2
        assert len(prev_output) == 2
        assert torch.equal(current_output[0], prev_output[0])
        assert torch.equal(current_output[1], prev_output[1])


# Tests for get_session_history_steps ########################################
def test_get_session_history_steps_example_1():
    hist = TrainingHistory()
    hist.l_step_inds_started_training = [0, 5000, 12000, 18000]
    hist.l_step_inds_checkpoints = [2000, 4000, 8000, 10000, 15000]

    result = NeuralNet.get_session_history_steps(hist)
    expected = [(0, 4001), (5000, 10001), (12000, 15001)]

    assert result == expected


def test_get_session_history_steps_example_2():
    hist = TrainingHistory()
    hist.l_step_inds_started_training = [0, 5000, 12000]
    hist.l_step_inds_checkpoints = [2000, 4000, 5000, 14000]

    result = NeuralNet.get_session_history_steps(hist)
    expected = [(0, 4001), (5000, 5001)]

    assert result == expected


def test_get_session_history_steps_single_session():
    hist = TrainingHistory()
    hist.l_step_inds_started_training = [0]
    hist.l_step_inds_checkpoints = [1000, 2000, 3000]

    result = NeuralNet.get_session_history_steps(hist)
    expected = []

    assert result == expected


def test_get_session_history_steps_no_checkpoints():
    hist = TrainingHistory()
    hist.l_step_inds_started_training = [0, 1000, 2000]
    hist.l_step_inds_checkpoints = []

    result = NeuralNet.get_session_history_steps(hist)
    expected = []

    assert result == expected


def test_get_session_history_steps_checkpoint_at_session_start():
    hist = TrainingHistory()
    hist.l_step_inds_started_training = [0, 1000, 2000]
    hist.l_step_inds_checkpoints = [500, 1000, 1500]

    result = NeuralNet.get_session_history_steps(hist)
    expected = [(0, 501), (1000, 1501)]

    assert result == expected


def test_get_session_history_steps_multiple_checkpoints_per_session():
    hist = TrainingHistory()
    hist.l_step_inds_started_training = [0, 5000, 10000]
    hist.l_step_inds_checkpoints = [1000, 2000, 3000, 4000, 6000, 7000, 8000]

    result = NeuralNet.get_session_history_steps(hist)
    expected = [(0, 4001), (5000, 8001)]

    assert result == expected


def test_get_session_history_steps_empty_history():
    hist = TrainingHistory()
    hist.l_step_inds_started_training = []
    hist.l_step_inds_checkpoints = []

    result = NeuralNet.get_session_history_steps(hist)
    expected = []

    assert result == expected


# Tests for compute_train_loss_me ############################################
def _bias_corrected_ema(values, beta):
    """Reference: per-segment, s_0=0, s_t = β s_{t-1} + (1-β) x_t,
    bias-corrected as s_t / (1 - β^t)."""
    s = 0.0
    out = []
    for t, x in enumerate(values, start=1):
        s = beta * s + (1.0 - beta) * x
        out.append(s / (1.0 - beta**t))
    return out


def test_compute_train_loss_me_single_session_matches_reference():
    hist = TrainingHistory()
    hist.l_train_loss_per_step = [1.0, 2.0, 3.0, 4.0, 5.0]
    hist.l_step_inds_started_training = [0]
    hist.last_used_training_loss_forgetting_factor = 0.9

    result = hist.compute_train_loss_me()
    expected = _bias_corrected_ema([1.0, 2.0, 3.0, 4.0, 5.0], 0.9)
    assert len(result) == 5
    for v, e in zip(result, expected):
        assert v == pytest.approx(e)


def test_compute_train_loss_me_first_step_equals_first_value():
    hist = TrainingHistory()
    hist.l_train_loss_per_step = [3.5, 1.0, 2.0]
    hist.l_step_inds_started_training = [0]
    hist.last_used_training_loss_forgetting_factor = 0.9

    assert hist.compute_train_loss_me()[0] == pytest.approx(3.5)


def test_compute_train_loss_me_session_with_no_checkpoint_starts_fresh():
    """A session start with no prior checkpoint resumes from (s=0, t=0)."""
    hist = TrainingHistory()
    hist.l_train_loss_per_step = [1.0, 1.0, 1.0, 100.0, 100.0]
    hist.l_step_inds_started_training = [0, 3]
    hist.l_step_inds_checkpoints = []
    hist.last_used_training_loss_forgetting_factor = 0.9

    # Step 3 has no prior checkpoint → resets to (0, 0); bias-corrected
    # EMA equals the first sample (100.0).
    assert hist.compute_train_loss_me()[3] == pytest.approx(100.0)


def test_compute_train_loss_me_restores_ema_at_checkpoint():
    """On session resume, (s, t) is rewound to its value at the latest
    checkpoint strictly before the session start. The abandoned tail still
    has its own EMA values (continuous from the previous step)."""
    hist = TrainingHistory()
    # Session 1: steps 0..2 (checkpoint at step 1, step 2 is abandoned tail).
    # Session 2: steps 3..4.
    hist.l_train_loss_per_step = [1.0, 1.0, 999.0, 2.0, 2.0]
    hist.l_step_inds_started_training = [0, 3]
    hist.l_step_inds_checkpoints = [1]
    beta = 0.5
    hist.last_used_training_loss_forgetting_factor = beta

    # Hand-compute:
    # step 0: s_base=(0,0). s=0.5; t=1 → 1.0
    # step 1: s_base=(0.5,1). s=0.75; t=2 → 1.0       (checkpoint stored)
    # step 2: s_base=(0.75,2). s=0.5*0.75+0.5*999=499.875; t=3
    #   → 499.875 / 0.875
    # step 3: session start; m=1 → s_base=(0.75,2). s=1.375; t=3
    #   → 1.375 / 0.875
    # step 4: s_base=(1.375,3). s=1.6875; t=4 → 1.8
    expected = [1.0, 1.0, 499.875 / 0.875, 1.375 / 0.875, 1.8]
    result = hist.compute_train_loss_me()
    assert len(result) == 5
    for v, e in zip(result, expected):
        assert v == pytest.approx(e)


def test_compute_train_loss_me_returns_list_of_floats():
    hist = TrainingHistory()
    hist.l_train_loss_per_step = list(range(10))
    hist.l_step_inds_started_training = [0]
    hist.last_used_training_loss_forgetting_factor = 0.5

    result = hist.compute_train_loss_me()
    assert isinstance(result, list)
    assert len(result) == 10
    assert all(isinstance(v, float) for v in result)


def test_compute_train_loss_me_empty_history():
    hist = TrainingHistory()
    assert hist.compute_train_loss_me(forgetting_factor=0.9) == []


def test_select_plot_steps_uniform():
    out = NeuralNet._select_plot_steps(num_steps=100,
                                       max_points=11,
                                       logx=False)
    assert len(out) == 11
    assert out[0] == 0
    assert out[-1] == 99
    assert out == sorted(out)
    # Spacing approximately uniform (within rounding)
    diffs = [b - a for a, b in zip(out, out[1:])]
    assert max(diffs) - min(diffs) <= 1


def test_select_plot_steps_returns_all_when_under_cap():
    out = NeuralNet._select_plot_steps(num_steps=5, max_points=100, logx=False)
    assert out == [0, 1, 2, 3, 4]


# Tests for resolve_fit_schedule (via fit) #####################################


class _TrainableNet(NeuralNet):
    """Minimal single-layer trainable network for fit tests."""

    def __init__(self, nn_folder=None):
        super().__init__(nn_folder=nn_folder)
        self.linear = nn.Linear(4, 1)
        self.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _TinyDataset(Dataset):
    """20 deterministic examples of (randn(4), randn(1)); 4 steps/epoch at batch_size=5."""

    def __init__(self, size=20, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(size, 4, generator=g)
        self.y = torch.randn(size, 1, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class _IterableDataset(IterableDataset):
    """Infinite stream of (randn(4), randn(1)) pairs (no __len__)."""

    def __iter__(self):
        while True:
            yield torch.randn(4), torch.randn(1)


def _mse(output, target):
    """Per-example MSE: shape (batch_size,)."""
    return ((output - target)**2).squeeze(-1)


class TestFitStepIntervals:
    """End-to-end tests for resolve_fit_schedule logic inside NeuralNet.fit."""

    BATCH_SIZE = 5
    DATASET = _TinyDataset()  # 20 examples → num_steps_per_epoch = 4
    VAL_DATASET = _TinyDataset(size=10, seed=42)

    def _net(self, nn_folder=None):
        return _TrainableNet(nn_folder=nn_folder)

    def _opt(self, net):
        return torch.optim.Adam(net.parameters(), lr=1e-3)

    def _fit(self, net, dataset, **kwargs):
        opt = self._opt(net)
        kwargs.setdefault('num_steps', 12)
        kwargs.setdefault('batch_size', self.BATCH_SIZE)
        kwargs.setdefault('shuffle', False)
        kwargs.setdefault('restore_best_checkpoint', False)
        return net.fit(dataset, opt, _mse, **kwargs)

    # train_loss_me branch =====================================================

    @staticmethod
    def _reported_train_loss_me_steps(caplog):
        """Parse 'Step N: training loss me = ...' INFO lines emitted by fit()."""
        steps = []
        for record in caplog.records:
            msg = record.getMessage()
            if 'training loss me' in msg:
                # Format: "Step {N}[ (epoch ...)]: training loss me = ..."
                head = msg.split(':', 1)[0]
                tok = head.split()
                if len(tok) >= 2 and tok[0] == 'Step':
                    steps.append(int(tok[1]))
        return steps

    def test_train_loss_me_defaults_from_report_moving(self, caplog):
        """num_steps_checkpoint resolves from num_steps_report_training_loss=5."""
        net = self._net()
        with caplog.at_level(logging.INFO, logger='gsim'):
            self._fit(net,
                      self.DATASET,
                      num_steps=15,
                      checkpoint_criterion='train_loss_me',
                      num_steps_report_training_loss=5)
        assert self._reported_train_loss_me_steps(caplog) == [5, 10]

    def test_train_loss_me_defaults_from_per_epoch(self, tmp_path, caplog):
        """num_steps_checkpoint resolves from num_steps_per_epoch=4 when nothing is set."""
        net = self._net(nn_folder=str(tmp_path))
        with caplog.at_level(logging.INFO, logger='gsim'):
            self._fit(net,
                      self.DATASET,
                      num_steps=12,
                      checkpoint_criterion='train_loss_me')
        assert self._reported_train_loss_me_steps(caplog) == [4, 8]

    def test_train_loss_me_no_length_raises(self, tmp_path):
        """IterableDataset with no num_steps_checkpoint or num_steps_report_training_loss raises."""
        net = self._net(nn_folder=str(tmp_path))
        opt = self._opt(net)
        with pytest.raises(ValueError, match="dataset has no length"):
            net.fit(_IterableDataset(),
                    opt,
                    _mse,
                    num_steps=10,
                    batch_size=self.BATCH_SIZE,
                    dataset_val=self.VAL_DATASET,
                    checkpoint_criterion='train_loss_me',
                    shuffle=False,
                    restore_best_checkpoint=False)

    def test_train_loss_me_val_iterable_no_eval_static_warns(
            self, tmp_path, caplog):
        """Warns and skips val loss when IterableDataset + val + no num_steps_eval."""
        net = self._net(nn_folder=str(tmp_path))
        opt = self._opt(net)
        with caplog.at_level(logging.WARNING, logger='gsim'):
            hist = net.fit(_IterableDataset(),
                           opt,
                           _mse,
                           num_steps=10,
                           batch_size=self.BATCH_SIZE,
                           dataset_val=self.VAL_DATASET,
                           checkpoint_criterion='train_loss_me',
                           num_steps_checkpoint=5,
                           shuffle=False,
                           restore_best_checkpoint=False)
        assert 'validation loss will not be computed' in caplog.text
        assert hist.l_val_loss == []

    def test_train_loss_me_report_moving_none_runs(self, tmp_path, caplog):
        """With num_steps_report_training_loss=None, moving metric fires only on checkpoint steps."""
        net = self._net(nn_folder=str(tmp_path))
        with caplog.at_level(logging.INFO, logger='gsim'):
            self._fit(net,
                      self.DATASET,
                      num_steps=12,
                      checkpoint_criterion='train_loss_me',
                      num_steps_checkpoint=4,
                      num_steps_report_training_loss=None)
        assert self._reported_train_loss_me_steps(caplog) == [4, 8]

    # val_loss branch ==========================================================

    def test_val_loss_eval_static_copied_to_checkpoint(self, tmp_path):
        """With only num_steps_eval=4, num_steps_checkpoint resolves to 4."""
        net = self._net(nn_folder=str(tmp_path))
        hist = self._fit(net,
                         self.DATASET,
                         num_steps=8,
                         checkpoint_criterion='val_loss',
                         dataset_val=self.VAL_DATASET,
                         num_steps_eval=4)
        val_steps = [s for s, _ in hist.l_val_loss]
        assert val_steps == [0, 4]

    def test_val_loss_checkpoint_copied_to_eval_static(self, tmp_path):
        """With only num_steps_checkpoint=4, num_steps_eval resolves to 4."""
        net = self._net(nn_folder=str(tmp_path))
        hist = self._fit(net,
                         self.DATASET,
                         num_steps=8,
                         checkpoint_criterion='val_loss',
                         dataset_val=self.VAL_DATASET,
                         num_steps_checkpoint=4)
        val_steps = [s for s, _ in hist.l_val_loss]
        assert val_steps == [0, 4]

    def test_val_loss_neither_defaults_per_epoch(self, tmp_path):
        """Both None: resolves to num_steps_per_epoch=4."""
        net = self._net(nn_folder=str(tmp_path))
        hist = self._fit(net,
                         self.DATASET,
                         num_steps=8,
                         checkpoint_criterion='val_loss',
                         dataset_val=self.VAL_DATASET)
        val_steps = [s for s, _ in hist.l_val_loss]
        assert val_steps == [0, 4]

    def test_val_loss_no_length_raises(self, tmp_path):
        """IterableDataset with both missing raises ValueError."""
        net = self._net(nn_folder=str(tmp_path))
        opt = self._opt(net)
        with pytest.raises(ValueError, match="dataset has no length"):
            net.fit(_IterableDataset(),
                    opt,
                    _mse,
                    num_steps=10,
                    batch_size=self.BATCH_SIZE,
                    dataset_val=self.VAL_DATASET,
                    checkpoint_criterion='val_loss',
                    shuffle=False,
                    restore_best_checkpoint=False)

    def test_val_loss_multiple_warning(self, tmp_path, caplog):
        """Non-multiple num_steps_checkpoint emits the 'multiple' warning."""
        net = self._net(nn_folder=str(tmp_path))
        with caplog.at_level(logging.WARNING, logger='gsim'):
            self._fit(net,
                      self.DATASET,
                      num_steps=20,
                      checkpoint_criterion='val_loss',
                      dataset_val=self.VAL_DATASET,
                      num_steps_checkpoint=10,
                      num_steps_eval=3)
        assert 'multiple' in caplog.text.lower()

    # never branch =============================================================

    def test_never_allows_all_none(self):
        """criterion='never', IterableDataset, all None: runs without error.

        dataset_val is passed explicitly so that make_validation_data does not
        require the iterable dataset to be Sized.
        """
        net = self._net()
        opt = self._opt(net)
        hist = net.fit(_IterableDataset(),
                       opt,
                       _mse,
                       num_steps=10,
                       batch_size=self.BATCH_SIZE,
                       dataset_val=self.VAL_DATASET,
                       checkpoint_criterion='never',
                       shuffle=False,
                       restore_best_checkpoint=False)
        assert len(hist.l_train_loss_per_step) == 10

    def test_never_checkpoint_set_warns_and_clears(self, caplog):
        """criterion='never' + num_steps_checkpoint set → warning issued."""
        net = self._net()
        with caplog.at_level(logging.WARNING, logger='gsim'):
            self._fit(net,
                      self.DATASET,
                      num_steps=12,
                      checkpoint_criterion='never',
                      num_steps_checkpoint=5)
        assert "checkpoint_criterion == 'never'" in caplog.text

    def test_never_with_val_defaults_eval_static_per_epoch(self):
        """criterion='never', finite dataset + val → eval_static=num_steps_per_epoch=4."""
        net = self._net()
        hist = self._fit(net,
                         self.DATASET,
                         num_steps=12,
                         checkpoint_criterion='never',
                         dataset_val=self.VAL_DATASET)
        val_steps = [s for s, _ in hist.l_val_loss]
        assert val_steps == [0, 4, 8]

    def test_never_iterable_val_no_eval_static_warns(self, caplog):
        """criterion='never', IterableDataset + val, no eval_static → warns, no val loss."""
        net = self._net()
        opt = self._opt(net)
        with caplog.at_level(logging.WARNING, logger='gsim'):
            hist = net.fit(_IterableDataset(),
                           opt,
                           _mse,
                           num_steps=10,
                           batch_size=self.BATCH_SIZE,
                           dataset_val=self.VAL_DATASET,
                           checkpoint_criterion='never',
                           shuffle=False,
                           restore_best_checkpoint=False)
        assert 'validation loss will not be computed' in caplog.text
        assert hist.l_val_loss == []

    # always branch ============================================================

    def test_always_coerces_report_moving_to_one(self, tmp_path, caplog):
        """criterion='always', num_steps_report_training_loss=5 → warns and coerces to 1."""
        net = self._net(nn_folder=str(tmp_path))
        with caplog.at_level(logging.INFO, logger='gsim'):
            self._fit(net,
                      self.DATASET,
                      num_steps=5,
                      checkpoint_criterion='always',
                      num_steps_report_training_loss=5)
        assert 'num_steps_report_training_loss=1' in caplog.text
        # Moving metric reported at every step > 0
        assert self._reported_train_loss_me_steps(caplog) == [1, 2, 3, 4]

    def test_always_coerces_checkpoint_to_one(self, tmp_path, caplog):
        """criterion='always', num_steps_checkpoint=5 → warns and coerces to 1."""
        net = self._net(nn_folder=str(tmp_path))
        with caplog.at_level(logging.WARNING, logger='gsim'):
            self._fit(net,
                      self.DATASET,
                      num_steps=5,
                      checkpoint_criterion='always',
                      num_steps_checkpoint=5)
        assert 'num_steps_checkpoint=1' in caplog.text

    def test_always_saves_every_step(self, tmp_path):
        """criterion='always': a checkpoint is saved at every step > 0."""
        net = self._net(nn_folder=str(tmp_path))
        opt = self._opt(net)
        num_steps = 5
        hist = net.fit(self.DATASET,
                       opt,
                       _mse,
                       num_steps=num_steps,
                       batch_size=self.BATCH_SIZE,
                       checkpoint_criterion='always',
                       shuffle=False,
                       restore_best_checkpoint=False)
        # Steps 1..num_steps-1 each trigger a checkpoint (step 0 is skipped
        # because b_save_checkpoint requires ind_step > 0).
        assert hist.l_step_inds_checkpoints == list(range(1, num_steps))

    # checkpoint_criterion resolution ==========================================

    def test_nn_folder_none_forces_never_warns(self, caplog):
        """nn_folder=None + criterion='val_loss' → warns and overrides to 'never'."""
        net = self._net(nn_folder=None)
        with caplog.at_level(logging.WARNING, logger='gsim'):
            self._fit(net,
                      self.DATASET,
                      num_steps=12,
                      checkpoint_criterion='val_loss',
                      dataset_val=self.VAL_DATASET,
                      num_steps_eval=4)
        assert "Setting checkpoint_criterion = 'never'" in caplog.text

    def test_nn_folder_none_silent_when_criterion_none(self, caplog):
        """nn_folder=None + criterion=None → no override warning (silently uses 'never')."""
        net = self._net(nn_folder=None)
        with caplog.at_level(logging.WARNING, logger='gsim'):
            self._fit(net,
                      self.DATASET,
                      num_steps=12,
                      checkpoint_criterion=None)
        assert "Setting checkpoint_criterion = 'never'" not in caplog.text

    def test_nn_folder_none_silent_when_criterion_never(self, caplog):
        """nn_folder=None + criterion='never' → no override warning."""
        net = self._net(nn_folder=None)
        with caplog.at_level(logging.WARNING, logger='gsim'):
            self._fit(net,
                      self.DATASET,
                      num_steps=12,
                      checkpoint_criterion='never')
        assert "Setting checkpoint_criterion = 'never'" not in caplog.text

    def test_nn_folder_set_default_val_loss_when_val(self, tmp_path):
        """nn_folder set + val data + criterion=None → defaults to 'val_loss'."""
        net = self._net(nn_folder=str(tmp_path))
        opt = self._opt(net)
        hist = net.fit(self.DATASET,
                       opt,
                       _mse,
                       num_steps=8,
                       batch_size=self.BATCH_SIZE,
                       dataset_val=self.VAL_DATASET,
                       shuffle=False,
                       restore_best_checkpoint=False)
        assert len(hist.l_val_loss) > 0

    def test_nn_folder_set_default_train_loss_me_when_no_val(self, tmp_path):
        """nn_folder set + no val + criterion=None → defaults to 'train_loss_me'."""
        net = self._net(nn_folder=str(tmp_path))
        opt = self._opt(net)
        hist = net.fit(self.DATASET,
                       opt,
                       _mse,
                       num_steps=8,
                       batch_size=self.BATCH_SIZE,
                       shuffle=False,
                       restore_best_checkpoint=False)
        assert hist.l_val_loss == []
        # Moving estimate is computed on demand; at least one checkpoint
        # implies at least one reporting step happened.
        assert len(hist.l_step_inds_checkpoints) > 0


# Tests for iterable dataset support #########################################


class TestIterableDatasetSupport:
    """Tests for fit/evaluate/predict with datasets that have no length."""

    BATCH_SIZE = 5
    VAL_DATASET = _TinyDataset(size=10, seed=42)

    def _net(self):
        return _TrainableNet()

    def _opt(self, net):
        return torch.optim.Adam(net.parameters(), lr=1e-3)

    def test_fit_num_epochs_with_iterable_raises(self):
        """num_epochs + IterableDataset raises because epoch is undefined."""
        net = self._net()
        opt = self._opt(net)
        with pytest.raises(AssertionError, match="no length"):
            net.fit(_IterableDataset(),
                    opt,
                    _mse,
                    num_epochs=2,
                    batch_size=self.BATCH_SIZE,
                    checkpoint_criterion='never',
                    shuffle=False,
                    restore_best_checkpoint=False)

    def test_fit_val_split_with_iterable_raises(self):
        """val_split != None + IterableDataset raises."""
        net = self._net()
        opt = self._opt(net)
        with pytest.raises(ValueError, match="val_split"):
            net.fit(_IterableDataset(),
                    opt,
                    _mse,
                    num_steps=10,
                    batch_size=self.BATCH_SIZE,
                    val_split=0.2,
                    checkpoint_criterion='never',
                    shuffle=False,
                    restore_best_checkpoint=False)

    def test_fit_iterable_with_static_max_num_loss_vals(self):
        """IterableDataset + static_max_num_loss_vals evaluates val loss."""
        net = self._net()
        opt = self._opt(net)
        hist = net.fit(_IterableDataset(),
                       opt,
                       _mse,
                       num_steps=10,
                       batch_size=self.BATCH_SIZE,
                       dataset_val=self.VAL_DATASET,
                       checkpoint_criterion='never',
                       num_steps_eval=5,
                       static_max_num_loss_vals=6,
                       shuffle=False,
                       restore_best_checkpoint=False)
        assert len(hist.l_val_loss) > 0

    def test_evaluate_iterable_with_max_num_loss_vals(self):
        """evaluate on IterableDataset with max_num_loss_vals returns a finite loss."""
        net = self._net()
        result = net.evaluate(_IterableDataset(),
                              batch_size=self.BATCH_SIZE,
                              f_loss=_mse,
                              max_num_loss_vals=15)
        assert np.isfinite(result["loss"])

    def test_evaluate_iterable_without_limit_raises(self):
        """evaluate on IterableDataset without max_num_loss_vals or max_hci raises."""
        net = self._net()
        with pytest.raises(ValueError):
            net.evaluate(_IterableDataset(),
                         batch_size=self.BATCH_SIZE,
                         f_loss=_mse)

    def test_predict_iterable_raises(self):
        """predict on an IterableDataset raises NotImplementedError."""
        net = self._net()
        with pytest.raises(NotImplementedError):
            net.predict(_IterableDataset(), batch_size=self.BATCH_SIZE)


# WeightedLoss ================================================================


class TestWeightedLoss:
    """Tests for loss functions returning WeightedLoss objects (cf. usage.md,
    Sec. "Weighted loss values")."""

    @staticmethod
    def _make_net():
        torch.manual_seed(0)
        return _TrainableNet()

    @staticmethod
    def _make_batch(num_rows=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(num_rows, 4, generator=g)
        y = torch.randn(num_rows, 1, generator=g)
        return x, y

    @staticmethod
    def _param_snapshot(net):
        return {
            name: param.detach().cpu().clone()
            for name, param in net.named_parameters()
        }

    @staticmethod
    def _manual_weighted_mean_grads(net, l_batches, l_weights):
        """Backpropagates the weighted mean of the per-row squared errors of
        `l_batches` on `net` and returns {param_name: grad}."""
        l_v_vals = []
        l_v_weights = []
        for (x, y), w in zip(l_batches, l_weights):
            output = net(net._move_to_device(x))
            target = net._move_to_device(y)
            l_v_vals.append(((output - target)**2).squeeze(-1))
            l_v_weights.append(w.to(l_v_vals[-1].device))
        v_vals = torch.cat(l_v_vals)
        v_weights = torch.cat(l_v_weights)
        loss = torch.sum(v_weights * v_vals) / torch.sum(v_weights)
        loss.backward()
        d_grads = {
            name: param.grad.detach().cpu()
            for name, param in net.named_parameters()
        }
        return d_grads, float(loss.detach())

    def test_ensure_assigns_unit_weights_to_plain_tensor(self):
        v_vals = torch.tensor([1.0, 2.0, 3.0])
        v_vals_out, v_weights = WeightedLoss.ensure(v_vals)
        assert torch.equal(v_vals_out, v_vals)
        assert torch.equal(v_weights, torch.ones(3))

    def test_ensure_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            WeightedLoss.ensure(
                WeightedLoss(values=torch.zeros(3), weights=torch.zeros(2)))

    def test_ensure_non_1d_raises(self):
        with pytest.raises(AssertionError):
            WeightedLoss.ensure(
                WeightedLoss(values=torch.zeros(3, 2),
                             weights=torch.zeros(3, 2)))

    def test_unit_weights_equivalent_to_plain_tensor(self):
        """A WeightedLoss with unit weights must produce exactly the same
        parameter update as a plain tensor loss."""
        x, y = self._make_batch()

        def f_plain(output, target):
            return ((output - target)**2).squeeze(-1)

        def f_weighted(output, target):
            v_vals = ((output - target)**2).squeeze(-1)
            return WeightedLoss(values=v_vals,
                                weights=torch.ones_like(v_vals))

        l_snapshots = []
        l_losses = []
        for f_loss in (f_plain, f_weighted):
            net = self._make_net()
            opt = torch.optim.SGD(net.parameters(), lr=0.1)
            loss_step, num_loss_vals = net._run_training_step(
                lambda: (x, y), f_loss, opt)
            assert num_loss_vals == 6
            l_losses.append(loss_step)
            l_snapshots.append(self._param_snapshot(net))
        assert l_losses[0] == l_losses[1]
        for name in l_snapshots[0]:
            assert torch.equal(l_snapshots[0][name], l_snapshots[1][name]), name

    def test_gradient_equals_gradient_of_weighted_mean(self):
        """The parameter update of a single-batch training step with SGD(lr=1)
        equals the gradient of the weighted mean of the loss values."""
        x, y = self._make_batch()
        v_weights = torch.tensor([1., 2., 3., 4., 5., 6.])

        def f_weighted(output, target):
            v_vals = ((output - target)**2).squeeze(-1)
            return WeightedLoss(values=v_vals,
                                weights=v_weights.to(v_vals.device))

        net = self._make_net()
        d_params_before = self._param_snapshot(net)
        opt = torch.optim.SGD(net.parameters(), lr=1.0)
        loss_step, num_loss_vals = net._run_training_step(
            lambda: (x, y), f_weighted, opt)

        net_ref = self._make_net()  # same seed -> same initial parameters
        d_grads, loss_ref = self._manual_weighted_mean_grads(
            net_ref, [(x, y)], [v_weights])

        assert num_loss_vals == 6
        assert np.isclose(loss_step, loss_ref, rtol=1e-5)
        for name, param in net.named_parameters():
            v_delta = d_params_before[name] - param.detach().cpu()
            assert torch.allclose(v_delta, d_grads[name], atol=1e-5,
                                  rtol=1e-4), name

    def test_accumulation_uses_global_weighted_mean(self):
        """With gradient accumulation, the gradient equals that of the weighted
        mean across ALL the accumulated batches, not the mean of the per-batch
        weighted means."""
        l_batches = [self._make_batch(num_rows=2, seed=1),
                     self._make_batch(num_rows=4, seed=2)]
        d_weights = {
            2: torch.tensor([1., 1.]),
            4: torch.tensor([10., 10., 10., 10.])
        }

        def f_weighted(output, target):
            v_vals = ((output - target)**2).squeeze(-1)
            return WeightedLoss(values=v_vals,
                                weights=d_weights[len(v_vals)].to(
                                    v_vals.device))

        net = self._make_net()
        d_params_before = self._param_snapshot(net)
        opt = torch.optim.SGD(net.parameters(), lr=1.0)
        batch_iter = iter(l_batches)
        # min_num_loss_vals_accumulate_grad=5 forces the step to consume both
        # batches (2 + 4 = 6 loss values).
        loss_step, num_loss_vals = net._run_training_step(
            lambda: next(batch_iter), f_weighted, opt, None, None, 5)

        net_ref = self._make_net()
        d_grads, loss_ref = self._manual_weighted_mean_grads(
            net_ref, l_batches, [d_weights[2], d_weights[4]])

        assert num_loss_vals == 6
        assert np.isclose(loss_step, loss_ref, rtol=1e-5)
        for name, param in net.named_parameters():
            v_delta = d_params_before[name] - param.detach().cpu()
            assert torch.allclose(v_delta, d_grads[name], atol=1e-5,
                                  rtol=1e-4), name

    def test_evaluate_returns_weighted_mean(self):
        """evaluate must return the weighted mean of the loss values across the
        dataset (3.4 here), not the plain mean (2.5)."""

        class _ValWeightDataset(Dataset):
            """The target of each item encodes (loss value, weight)."""

            def __init__(self):
                self.x = torch.zeros(4, 4)
                self.y = torch.tensor([[1., 1.], [2., 1.], [3., 1.], [4., 7.]])

            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        def f_loss(output, target):
            return WeightedLoss(values=target[:, 0], weights=target[:, 1])

        net = self._make_net()
        result = net.evaluate(_ValWeightDataset(), batch_size=2, f_loss=f_loss)
        expected = (1. + 2. + 3. + 4. * 7.) / 10.
        assert np.isclose(result["loss"], expected, rtol=1e-6)
        assert np.isfinite(result["hci"])

    def test_fit_with_weighted_loss(self):
        """fit runs with a WeightedLoss-returning loss function and records
        finite weighted per-step training and validation losses."""
        net = self._make_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        def f_loss(output, target):
            v_vals = ((output - target)**2).squeeze(-1)
            v_weights = 1.0 + torch.arange(
                len(v_vals), device=v_vals.device, dtype=v_vals.dtype)
            return WeightedLoss(values=v_vals, weights=v_weights)

        hist = net.fit(_TinyDataset(),
                       opt,
                       f_loss,
                       num_steps=4,
                       batch_size=5,
                       shuffle=False,
                       restore_best_checkpoint=False,
                       dataset_val=_TinyDataset(size=10, seed=42),
                       num_steps_eval=2)
        assert len(hist.l_train_loss_per_step) == 4
        assert np.all(np.isfinite(hist.l_train_loss_per_step))
        assert len(hist.l_val_loss) >= 1
        assert np.isfinite(hist.l_val_loss[-1][1])


class TestDiagnoserWithWeightedLoss:
    """The diagnoser must aggregate the loss with the same weighted mean that
    NeuralNet optimizes, so that bugs in the weights are not concealed."""

    def test_loss_mean_computes_weighted_mean(self):
        from gsim.include.neural_net.diagnoser import _loss_mean
        v_vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        v_weights = torch.tensor([1.0, 1.0, 1.0, 7.0])
        assert torch.isclose(_loss_mean(v_vals), torch.tensor(2.5))
        assert torch.isclose(
            _loss_mean(WeightedLoss(values=v_vals, weights=v_weights)),
            torch.tensor(3.4))

    def test_standard_diagnoser_logs_weighted_loss(self, caplog):
        import re

        from gsim.include.neural_net.diagnoser import StandardDiagnoser

        g = torch.Generator().manual_seed(0)
        x, y = torch.randn(4, 4, generator=g), torch.randn(4, 1, generator=g)
        v_weights = torch.tensor([1.0, 1.0, 1.0, 7.0])

        def f_weighted(output, target):
            v_vals = ((output - target)**2).squeeze(-1)
            return WeightedLoss(values=v_vals,
                                weights=v_weights.to(v_vals.device))

        torch.manual_seed(0)
        net = _TrainableNet()
        net.set_diagnoser(
            StandardDiagnoser(log_forward=True, log_backward=True))

        # Weighted mean the diagnoser should report, computed on the initial
        # parameters (check_forward runs before the optimizer step).
        with torch.no_grad():
            v_vals = ((net(net._move_to_device(x)) -
                       net._move_to_device(y))**2).squeeze(-1).cpu()
        expected = float(torch.sum(v_weights * v_vals) / torch.sum(v_weights))

        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        with caplog.at_level(logging.INFO, logger="gsim"):
            net._run_training_step(lambda: (x, y), f_weighted, opt)

        l_logged = [
            float(m.group(1)) for record in caplog.records
            if (m := re.search(r"batch_loss=([0-9.eE+-]+)",
                               record.getMessage()))
        ]
        assert len(l_logged) == 1
        assert np.isclose(l_logged[0], expected, rtol=1e-4)


class TestLossShapeValidation:
    """A loss function must return a 1D tensor of loss values (or a
    WeightedLoss with a 1D `values` field); anything else raises."""

    @staticmethod
    def _make_net():
        torch.manual_seed(0)
        return _TrainableNet()

    def test_training_with_2d_loss_raises(self):
        g = torch.Generator().manual_seed(0)
        x, y = torch.randn(6, 4, generator=g), torch.randn(6, 1, generator=g)

        def f_loss_2d(output, target):
            return (output - target)**2  # (batch_size, 1): not flattened

        net = self._make_net()
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        with pytest.raises(AssertionError, match="1D tensor"):
            net._run_training_step(lambda: (x, y), f_loss_2d, opt)

    def test_evaluate_with_scalar_loss_raises(self):

        def f_loss_scalar(output, target):
            return torch.mean((output - target)**2)  # 0-dim scalar

        net = self._make_net()
        with pytest.raises(AssertionError, match="1D tensor"):
            net.evaluate(_TinyDataset(), batch_size=5, f_loss=f_loss_scalar)


def test_mean_and_ci_unit_weights_matches_unweighted():
    """mean_and_ci with unit weights must reduce exactly to the unweighted
    case."""
    l_vals = [1.0, 2.0, 3.0, 10.0]
    mean_unweighted, hci_unweighted = mean_and_ci(l_vals)
    mean_weighted, hci_weighted = mean_and_ci(l_vals, weights=[1.0] * 4)
    assert np.isclose(mean_unweighted, mean_weighted)
    assert np.isclose(hci_unweighted, hci_weighted)


def test_mean_and_ci_weighted_mean():
    l_vals = [1.0, 2.0, 3.0, 4.0]
    l_weights = [1.0, 1.0, 1.0, 7.0]
    mean, hci = mean_and_ci(l_vals, weights=l_weights)
    assert np.isclose(mean, 3.4)
    assert np.isfinite(hci)


# Tests for import_parameters ##################################################


class _NormalizedNet(NeuralNet):
    """`_TrainableNet` with a normalizer, so that it has a normalizer file."""

    def __init__(self, nn_folder=None, normalizer_folder=None):
        super().__init__(nn_folder=nn_folder,
                         normalizer=DefaultNormalizer(mode="both",
                                                      folder=normalizer_folder))
        self.linear = nn.Linear(4, 1)
        self.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestImportParameters:

    @staticmethod
    def _make_src(folder, with_normalizer=True) -> _NormalizedNet:
        """Writes weights (and normalizer parameters) into `folder`."""
        torch.manual_seed(1)
        net = _NormalizedNet(nn_folder=str(folder))
        if with_normalizer:
            net.normalizer.fit(_TinyDataset())
            net.normalizer.save()
        net.save_weights_to_path(net.weight_file_path)
        return net

    @staticmethod
    def _make_dst(folder, normalizer_folder=None) -> _NormalizedNet:
        torch.manual_seed(2)
        return _NormalizedNet(nn_folder=str(folder),
                              normalizer_folder=normalizer_folder)

    @staticmethod
    def _equal_weights(net_a, net_b) -> bool:
        return all(
            torch.equal(a, b)
            for a, b in zip(net_a.state_dict().values(),
                            net_b.state_dict().values()))

    @staticmethod
    def _fit_normalizer_differently(net):
        net.normalizer.fit(_TinyDataset(seed=5))
        net.normalizer.save()

    def test_copies_and_reloads(self, tmp_path):
        src = self._make_src(tmp_path / "src")
        dst = self._make_dst(tmp_path / "dst")
        assert not self._equal_weights(src, dst)
        assert not dst.normalizer.are_parameters_set

        dst.import_parameters(src.nn_folder)

        assert os.path.exists(dst.weight_file_path)
        assert os.path.exists(dst.normalizer.params_file_path)
        assert self._equal_weights(src, dst)
        assert dst.normalizer.are_parameters_set
        assert torch.equal(dst.normalizer.input_batch_mean,
                           src.normalizer.input_batch_mean)

    def test_skips_when_weights_exist_and_not_overwrite(self, tmp_path):
        src = self._make_src(tmp_path / "src")
        dst = self._make_src(tmp_path / "dst")
        with torch.no_grad():
            dst.linear.weight.add_(1.0)
        dst.save_weights_to_path(dst.weight_file_path)
        v_before = dst.linear.weight.clone()

        dst.import_parameters(src.nn_folder, overwrite_if_set=False)

        assert torch.equal(dst.linear.weight, v_before)
        reloaded = _NormalizedNet(nn_folder=dst.nn_folder)
        assert torch.equal(reloaded.linear.weight, v_before)

    def test_skip_warns_when_normalizer_file_is_missing(self, tmp_path, caplog):
        src = self._make_src(tmp_path / "src")
        dst = self._make_dst(tmp_path / "dst")
        dst.save_weights_to_path(dst.weight_file_path)  # no normalizer file

        with caplog.at_level(logging.WARNING, logger="gsim"):
            dst.import_parameters(src.nn_folder, overwrite_if_set=False)

        assert "normalizer has no file" in caplog.text
        assert not os.path.exists(dst.normalizer.params_file_path)

    def test_overwrites_when_overwrite(self, tmp_path):
        src = self._make_src(tmp_path / "src")
        dst = self._make_src(tmp_path / "dst")
        with torch.no_grad():
            dst.linear.weight.add_(1.0)
        dst.save_weights_to_path(dst.weight_file_path)

        dst.import_parameters(src.nn_folder, overwrite_if_set=True)

        assert self._equal_weights(src, dst)

    def test_missing_source_weights_raise(self, tmp_path):
        (tmp_path / "src").mkdir()
        dst = self._make_dst(tmp_path / "dst")
        with pytest.raises(FileNotFoundError, match="weights.pth"):
            dst.import_parameters(str(tmp_path / "src"))

    def test_source_without_normalizer_keeps_the_existing_one(
            self, tmp_path, caplog):
        src = self._make_src(tmp_path / "src", with_normalizer=False)
        dst = self._make_dst(tmp_path / "dst")
        self._fit_normalizer_differently(dst)
        v_mean_before = dst.normalizer.input_batch_mean.clone()

        with caplog.at_level(logging.WARNING, logger="gsim"):
            dst.import_parameters(src.nn_folder)

        assert "existing normalizer" in caplog.text
        assert self._equal_weights(src, dst)
        assert torch.equal(dst.normalizer.input_batch_mean, v_mean_before)

    def test_shared_normalizer_file_is_not_overwritten(self, tmp_path):
        src = self._make_src(tmp_path / "src")
        shared = tmp_path / "shared"
        shared.mkdir()
        dst = self._make_dst(tmp_path / "dst", normalizer_folder=str(shared))
        self._fit_normalizer_differently(dst)
        v_before = dst.linear.weight.clone()

        with pytest.raises(ValueError, match="shared"):
            dst.import_parameters(src.nn_folder)

        # The checks run before any file changes.
        assert not os.path.exists(dst.weight_file_path)
        assert torch.equal(dst.linear.weight, v_before)

    def test_shared_normalizer_without_file_is_imported(self, tmp_path):
        src = self._make_src(tmp_path / "src")
        shared = tmp_path / "shared"
        shared.mkdir()
        dst = self._make_dst(tmp_path / "dst", normalizer_folder=str(shared))

        dst.import_parameters(src.nn_folder)

        assert (shared / "normalizer.pk").exists()
        assert torch.equal(dst.normalizer.input_batch_mean,
                           src.normalizer.input_batch_mean)

    def test_destination_training_state_is_removed(self, tmp_path):
        src = self._make_src(tmp_path / "src")
        dst = self._make_dst(tmp_path / "dst")
        for file_name in dst.training_state_file_names:
            (tmp_path / "dst" / file_name).write_bytes(b"stale")

        dst.import_parameters(src.nn_folder)

        for file_name in dst.training_state_file_names:
            assert not (tmp_path / "dst" / file_name).exists()

    def test_training_state_is_copied_on_request(self, tmp_path):
        src = self._make_src(tmp_path / "src")
        (tmp_path / "src" / src.optimizer_state_file_name).write_bytes(b"src")
        dst = self._make_dst(tmp_path / "dst")
        (tmp_path / "dst" / dst.hist_file_name).write_bytes(b"stale")

        dst.import_parameters(src.nn_folder, import_training_state=True)

        assert (tmp_path / "dst" /
                dst.optimizer_state_file_name).read_bytes() == b"src"
        # Not in the source, so the stale one is not left behind.
        assert not (tmp_path / "dst" / dst.hist_file_name).exists()

    def test_same_folder_raises(self, tmp_path):
        src = self._make_src(tmp_path / "src")
        with pytest.raises(ValueError, match="into itself"):
            src.import_parameters(src.nn_folder)

    def test_network_without_normalizer_imports_the_weights(self, tmp_path):
        src = self._make_src(tmp_path / "src")  # also writes normalizer.pk
        torch.manual_seed(2)
        dst = _TrainableNet(nn_folder=str(tmp_path / "dst"))
        assert dst.normalizer is None

        dst.import_parameters(src.nn_folder)

        assert self._equal_weights(src, dst)
        assert not (tmp_path / "dst" / "normalizer.pk").exists()


class TestNormalizerImportParameters:

    def test_copies_and_loads(self, tmp_path):
        src = DefaultNormalizer(mode="both", folder=str(tmp_path / "src"))
        (tmp_path / "src").mkdir()
        src.fit(_TinyDataset())
        src.save()
        dst = DefaultNormalizer(mode="both", folder=str(tmp_path / "dst"))

        dst.import_parameters(str(tmp_path / "src"))

        assert dst.are_parameters_set
        assert torch.equal(dst.input_batch_mean, src.input_batch_mean)

    def test_skips_when_set_and_not_overwrite(self, tmp_path):
        for name, seed in (("src", 0), ("dst", 5)):
            (tmp_path / name).mkdir()
            n = DefaultNormalizer(mode="both", folder=str(tmp_path / name))
            n.fit(_TinyDataset(seed=seed))
            n.save()
        dst = DefaultNormalizer(mode="both", folder=str(tmp_path / "dst"))
        dst.load_if_file_exists()
        v_before = dst.input_batch_mean.clone()

        dst.import_parameters(str(tmp_path / "src"), overwrite_if_set=False)

        assert torch.equal(dst.input_batch_mean, v_before)

    def test_missing_source_raises(self, tmp_path):
        (tmp_path / "src").mkdir()
        dst = DefaultNormalizer(mode="both", folder=str(tmp_path / "dst"))
        with pytest.raises(FileNotFoundError, match="normalizer.pk"):
            dst.import_parameters(str(tmp_path / "src"))


class TestFileNames:

    def test_paths_follow_the_names(self, tmp_path):
        net = _NormalizedNet(nn_folder=str(tmp_path))
        assert net.weight_file_path == os.path.join(str(tmp_path),
                                                    "weights.pth")
        assert net.hist_file_path == os.path.join(
            str(tmp_path), NeuralNet.DEFAULT_HIST_FILE_NAME)
        assert net.normalizer.params_file_path == os.path.join(
            str(tmp_path), "normalizer.pk")

    def test_deprecated_helpers_warn_and_agree(self, tmp_path):
        net = _NormalizedNet(nn_folder=str(tmp_path))
        folder = str(tmp_path)
        with pytest.warns(DeprecationWarning):
            assert NeuralNet.get_weight_file_path(
                folder) == net.weight_file_path
        with pytest.warns(DeprecationWarning):
            assert NeuralNet.get_optimizer_state_file_path(
                folder) == net.optimizer_state_file_path
        with pytest.warns(DeprecationWarning):
            assert net.get_lr_scheduler_state_file_path(
                folder) == net.lr_scheduler_state_file_path
        with pytest.warns(DeprecationWarning):
            assert NeuralNet.make_hist_path(folder) == net.hist_file_path
        with pytest.warns(DeprecationWarning):
            assert net.normalizer.params_file == net.normalizer.params_file_path

    def test_a_subclass_default_is_respected_everywhere(self, tmp_path):

        class _RenamedNet(_TrainableNet):
            DEFAULT_WEIGHT_FILE_NAME = "renamed.pth"

        net = _RenamedNet(nn_folder=str(tmp_path))
        assert net.weight_file_path == os.path.join(str(tmp_path),
                                                    "renamed.pth")
        with pytest.warns(DeprecationWarning):
            assert _RenamedNet.get_weight_file_path(
                str(tmp_path)) == net.weight_file_path

    def test_load_hist_from_folder_takes_a_file_name(self, tmp_path):
        net = _TrainableNet(nn_folder=str(tmp_path))
        hist = TrainingHistory()
        hist.l_train_loss_per_step = [1.0, 2.0]
        with open(tmp_path / "other_hist.pk", "wb") as f:
            pickle.dump(hist, f)

        loaded = NeuralNet.load_hist_from_folder(str(tmp_path),
                                                 file_name="other_hist.pk")

        assert loaded.l_train_loss_per_step == [1.0, 2.0]
        assert NeuralNet.load_hist_from_folder(
            str(tmp_path)).l_train_loss_per_step == []


# Tests for the step-0 checkpoint ##############################################


class TestCheckpointAtStepZero:

    BATCH_SIZE = 5
    NUM_STEPS = 12

    @staticmethod
    def _make_datasets():
        """
        Training and validation sets with the same inputs and opposite
        targets.
        -> Starting from zero weights, every training step moves the weights
           away from the validation optimum, so the validation loss only rises.
        """
        dataset_train = _TinyDataset()
        dataset_val = _TinyDataset()
        dataset_val.y = -dataset_val.y
        return dataset_train, dataset_val

    def _fit(self, net, **kwargs):
        dataset_train, dataset_val = self._make_datasets()
        opt = torch.optim.SGD(net.parameters(), lr=1e-2)
        return net.fit(dataset_train,
                       opt,
                       _mse,
                       dataset_val=dataset_val,
                       num_steps=self.NUM_STEPS,
                       batch_size=self.BATCH_SIZE,
                       shuffle=False,
                       num_steps_eval=4,
                       restore_best_checkpoint=False,
                       training_loss_forgetting_factor=0.9,
                       **kwargs)

    @staticmethod
    def _make_zero_net(folder):
        net = _TrainableNet(nn_folder=str(folder))
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
        return net

    def test_val_loss_keeps_step_zero_when_val_loss_only_rises(
            self, tmp_path, caplog):
        net = self._make_zero_net(tmp_path)
        with caplog.at_level(logging.INFO, logger='gsim'):
            hist = self._fit(net, checkpoint_criterion='val_loss')

        l_val_loss = [v for _, v in hist.l_val_loss]
        assert l_val_loss == sorted(l_val_loss) and len(set(l_val_loss)) > 1
        assert hist.l_step_inds_checkpoints == [0]

        # weights.pth holds the weights that the step-0 evaluation measured.
        saved = _TrainableNet(nn_folder=str(tmp_path))
        _, dataset_val = self._make_datasets()
        with torch.no_grad():
            val_loss = _mse(saved(dataset_val.x), dataset_val.y).mean()
        assert np.isclose(val_loss.item(), hist.l_val_loss[0][1], rtol=1e-5)

        # The moving estimate is still not reported at step 0.
        assert 0 not in TestFitStepIntervals._reported_train_loss_me_steps(
            caplog)

    def test_train_loss_me_skips_step_zero(self, tmp_path):
        net = self._make_zero_net(tmp_path)
        hist = self._fit(net,
                         checkpoint_criterion='train_loss_me',
                         num_steps_checkpoint=4)
        assert len(hist.l_step_inds_checkpoints) > 0
        assert 0 not in hist.l_step_inds_checkpoints

    def test_resumed_session_follows_the_rule_for_positive_steps(
            self, tmp_path):
        net = self._make_zero_net(tmp_path)
        self._fit(net, checkpoint_criterion='val_loss')
        hist = self._fit(net, checkpoint_criterion='val_loss')
        ind_start = hist.l_step_inds_started_training[-1]
        assert ind_start > 0
        # The second session only adds checkpoints at its own steps.
        assert all(s >= ind_start
                   for s in hist.l_step_inds_checkpoints[1:])

    @pytest.mark.parametrize("checkpoint_criterion",
                             ["val_loss", "train_loss_me", "always"])
    def test_is_checkpoint_step_unchanged_for_positive_steps(
            self, checkpoint_criterion):
        for num_steps_checkpoint in [1, 4]:
            for ind_step in range(1, 20):
                assert NeuralNet._is_checkpoint_step(
                    ind_step, num_steps_checkpoint, checkpoint_criterion) == (
                        ind_step % num_steps_checkpoint == 0)
        assert not NeuralNet._is_checkpoint_step(5, None,
                                                 checkpoint_criterion)
        assert NeuralNet._is_checkpoint_step(
            0, 4, checkpoint_criterion) == (checkpoint_criterion == "val_loss")
