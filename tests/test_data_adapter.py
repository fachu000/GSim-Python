import os
import tempfile

import pytest
import torch
from torch.utils.data import Dataset

from gsim.include.neural_net import NeuralNet
from gsim.include.neural_net.data_adapter import DataAdapter

# ---------------------------------------------------------------------------
# Helpers (module-level for picklability)
# ---------------------------------------------------------------------------


class _DoubleAdapter(DataAdapter):
    """Multiplies every element of the input tensor by 2."""

    def extract_feats(self, raw_input: torch.Tensor) -> torch.Tensor:
        return raw_input * 2


class _SimpleDataset(Dataset):
    """Dataset of (input, target) pairs."""

    def __init__(self, n=10, feat_dim=4, target_dim=2):
        self.inputs = torch.arange(n * feat_dim,
                                   dtype=torch.float).reshape(n, feat_dim)
        self.targets = torch.arange(n * target_dim,
                                    dtype=torch.float).reshape(n, target_dim)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class _InputOnlyDataset(Dataset):
    """Dataset that contains only inputs (no targets)."""

    def __init__(self, n=10, feat_dim=4):
        self.inputs = torch.arange(n * feat_dim,
                                   dtype=torch.float).reshape(n, feat_dim)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class _SimpleNet(NeuralNet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = torch.nn.Linear(8, 2)  # expects doubled 4-dim input
        self.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# DataAdapter
# ---------------------------------------------------------------------------


def test_data_adapter_extract_feats():
    adapter = _DoubleAdapter()
    x = torch.tensor([1.0, 2.0, 3.0])
    result = adapter.extract_feats(x)
    assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0]))


# ---------------------------------------------------------------------------
# NeuralNetDataset save / load
# ---------------------------------------------------------------------------


def test_neural_net_dataset_save_load_adapted():
    items = [torch.tensor([float(i)]) for i in range(5)]
    ds = NeuralNet.NeuralNetDataset(items, adapted=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ds.pk")
        ds.save(path)
        loaded = NeuralNet.NeuralNetDataset.load(path)
    assert loaded.adapted is True
    assert len(loaded) == 5
    assert torch.equal(loaded[2], torch.tensor([2.0]))


def test_neural_net_dataset_save_load_not_adapted():
    items = [torch.tensor([float(i)]) for i in range(3)]
    ds = NeuralNet.NeuralNetDataset(items, adapted=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ds.pk")
        ds.save(path)
        loaded = NeuralNet.NeuralNetDataset.load(path)
    assert loaded.adapted is False


# ---------------------------------------------------------------------------
# preprocess_dataset
# ---------------------------------------------------------------------------


def test_preprocess_dataset_with_targets():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    raw_ds = _SimpleDataset(n=5)
    adapted_ds = net.wrap_in_adapter(raw_ds, no_targets=False)

    assert adapted_ds.adapted is True
    assert len(adapted_ds) == 5

    for i in range(5):
        raw_input, raw_target = raw_ds[i]
        adapted_input, target = adapted_ds[i]
        assert torch.allclose(adapted_input, raw_input * 2)
        assert torch.allclose(target, raw_target)  # target unchanged


def test_preprocess_dataset_no_targets():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    raw_ds = _InputOnlyDataset(n=5)
    adapted_ds = net.wrap_in_adapter(raw_ds, no_targets=True)

    assert adapted_ds.adapted is True
    assert len(adapted_ds) == 5

    for i in range(5):
        raw_input = raw_ds[i]
        adapted_input = adapted_ds[i]
        assert torch.allclose(adapted_input, raw_input * 2)


def test_preprocess_dataset_raises_without_adapter():
    net = _SimpleNet()  # no data_adapter
    raw_ds = _SimpleDataset(n=5)
    with pytest.raises(AssertionError):
        net.wrap_in_adapter(raw_ds)


# ---------------------------------------------------------------------------
# load_or_create_preprocessed_dataset
# ---------------------------------------------------------------------------


def test_load_or_create_preprocessed_dataset_creates_and_loads():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    raw_ds = _SimpleDataset(n=5)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preprocessed.pk")

        # First call: should create and save
        ds1 = net.load_or_create_preprocessed_dataset(raw_ds, path)
        assert os.path.exists(path)
        assert ds1.adapted is True
        assert len(ds1) == 5

        # Second call: should load from file (no recomputation)
        ds2 = net.load_or_create_preprocessed_dataset(raw_ds, path)
        assert ds2.adapted is True
        assert len(ds2) == 5

        # Values should match: inputs doubled
        raw_input_0, _ = raw_ds[0]
        assert torch.allclose(ds1[0][0], raw_input_0 * 2)
        assert torch.allclose(ds2[0][0], raw_input_0 * 2)


def test_load_or_create_preprocessed_dataset_with_callback():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    call_count = [0]

    def make_dataset():
        call_count[0] += 1
        return _SimpleDataset(n=5)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preprocessed.pk")

        # First call: callback is invoked
        ds1 = net.load_or_create_preprocessed_dataset(make_dataset, path)
        assert call_count[0] == 1

        # Second call: file exists, callback must NOT be invoked
        ds2 = net.load_or_create_preprocessed_dataset(make_dataset, path)
        assert call_count[0] == 1  # still 1

        assert ds2.adapted is True


def test_load_or_create_preprocessed_dataset_no_len_raises():
    net = _SimpleNet(data_adapter=_DoubleAdapter())

    class _NoLenDataset:

        def __getitem__(self, idx):
            return torch.zeros(4)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preprocessed.pk")
        with pytest.raises(AssertionError):
            net.load_or_create_preprocessed_dataset(_NoLenDataset(), path)


# ---------------------------------------------------------------------------
# make_data_loader: auto-applies extraction when not adapted
# ---------------------------------------------------------------------------


def test_make_data_loader_applies_adapter_when_not_adapted():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    raw_ds = _SimpleDataset(n=8)

    loader = net.make_data_loader(raw_ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    input_batch, _ = batch

    # The loader should have applied extract_feats (doubling)
    expected_inputs = torch.stack([raw_ds[i][0] * 2 for i in range(8)])
    assert torch.allclose(input_batch, expected_inputs)


def test_make_data_loader_skips_adapter_when_adapted():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    raw_ds = _SimpleDataset(n=8)
    adapted_ds = net.wrap_in_adapter(raw_ds, no_targets=False)

    loader = net.make_data_loader(adapted_ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    input_batch, _ = batch

    # Already adapted: extraction must NOT be applied again
    expected_inputs = torch.stack([raw_ds[i][0] * 2 for i in range(8)])
    assert torch.allclose(input_batch, expected_inputs)
    # Verify it was NOT doubled a second time
    double_doubled = torch.stack([raw_ds[i][0] * 4 for i in range(8)])
    assert not torch.allclose(input_batch, double_doubled)


def test_make_data_loader_no_adapter():
    net = _SimpleNet()  # no data_adapter
    raw_ds = _SimpleDataset(n=8)

    loader = net.make_data_loader(raw_ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    input_batch, _ = batch

    expected_inputs = torch.stack([raw_ds[i][0] for i in range(8)])
    assert torch.allclose(input_batch, expected_inputs)
