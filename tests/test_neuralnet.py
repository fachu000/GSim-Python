import logging

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset

from gsim.include.neural_net import NeuralNet
from gsim.include.neural_net.neural_net import TrainingHistory


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
        return net.fit(dataset, _mse, opt, **kwargs)

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
                    _mse,
                    opt,
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
                           _mse,
                           opt,
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
                    _mse,
                    opt,
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
                       _mse,
                       opt,
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
                           _mse,
                           opt,
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
                       _mse,
                       opt,
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
                       _mse,
                       opt,
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
                       _mse,
                       opt,
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
                    _mse,
                    opt,
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
                    _mse,
                    opt,
                    num_steps=10,
                    batch_size=self.BATCH_SIZE,
                    val_split=0.2,
                    checkpoint_criterion='never',
                    shuffle=False,
                    restore_best_checkpoint=False)

    def test_fit_iterable_with_static_max_num_examples(self):
        """IterableDataset + static_max_num_examples evaluates val loss."""
        net = self._net()
        opt = self._opt(net)
        hist = net.fit(_IterableDataset(),
                       _mse,
                       opt,
                       num_steps=10,
                       batch_size=self.BATCH_SIZE,
                       dataset_val=self.VAL_DATASET,
                       checkpoint_criterion='never',
                       num_steps_eval=5,
                       static_max_num_examples=6,
                       shuffle=False,
                       restore_best_checkpoint=False)
        assert len(hist.l_val_loss) > 0

    def test_evaluate_iterable_with_max_num_examples(self):
        """evaluate on IterableDataset with max_num_examples returns a finite loss."""
        net = self._net()
        result = net.evaluate(_IterableDataset(),
                              batch_size=self.BATCH_SIZE,
                              f_loss=_mse,
                              max_num_examples=15)
        assert np.isfinite(result["loss"])

    def test_evaluate_iterable_without_limit_raises(self):
        """evaluate on IterableDataset without max_num_examples or max_hci raises."""
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
