import os
import tempfile

import pytest
import torch
from torch.utils.data import Dataset

from gsim.include.neural_net import NeuralNet
from gsim.include.neural_net.data_adapter import AdaptationSpec, DataAdapter

# ---------------------------------------------------------------------------
# Helpers (module-level for picklability)
# ---------------------------------------------------------------------------


class _DoubleAdapter(DataAdapter):
    """Doubles the input tensor in adapt_input; records specs seen."""

    def adapt_input(self, raw_input: torch.Tensor,
                    spec: AdaptationSpec) -> torch.Tensor:
        return raw_input * 2


class _SpyAdapter(DataAdapter):
    """Records the AdaptationSpec passed to each method."""

    def __init__(self):
        self.adapt_input_specs = []
        self.adapt_output_specs = []

    def adapt_input(self, raw_input, spec):
        self.adapt_input_specs.append(spec)
        return raw_input

    def adapt_output(self, output, spec):
        self.adapt_output_specs.append(spec)
        return output


class _CustomTargetAdapter(DataAdapter):
    """Doubles both input and target to verify adapt_target routing."""

    def adapt_input(self, raw_input, spec):
        return raw_input * 2

    def adapt_target(self, target, spec):
        return target * 3


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
# AdaptationSpec
# ---------------------------------------------------------------------------


def test_adaptation_spec_defaults_all_false():
    spec = AdaptationSpec()
    assert spec.preprocess_only is False
    assert spec.input_already_preprocessed is False
    assert spec.inference is False


def test_adaptation_spec_kwargs():
    spec = AdaptationSpec(preprocess_only=True,
                          input_already_preprocessed=True,
                          inference=True)
    assert spec.preprocess_only is True
    assert spec.input_already_preprocessed is True
    assert spec.inference is True


# ---------------------------------------------------------------------------
# DataAdapter default methods
# ---------------------------------------------------------------------------


def test_adapter_get_no_targets_default_inherits():
    adapter = DataAdapter()
    spec = AdaptationSpec()
    assert adapter.get_no_targets(True, spec) is True
    assert adapter.get_no_targets(False, spec) is False


def test_adapt_input_identity_by_default():
    adapter = DataAdapter()
    x = torch.tensor([1.0, 2.0])
    spec = AdaptationSpec()
    assert torch.equal(adapter.adapt_input(x, spec), x)


def test_adapt_target_identity_by_default():
    adapter = DataAdapter()
    t = torch.tensor([3.0, 4.0])
    spec = AdaptationSpec()
    assert torch.equal(adapter.adapt_target(t, spec), t)


def test_adapt_output_identity_by_default():
    adapter = DataAdapter()
    o = torch.tensor([5.0])
    spec = AdaptationSpec()
    assert torch.equal(adapter.adapt_output(o, spec), o)


def test_default_adapt_dataset_item_splits_and_routes():
    adapter = _CustomTargetAdapter()
    spec = AdaptationSpec()
    raw_input = torch.tensor([1.0, 2.0])
    target = torch.tensor([10.0])
    adapted_input, adapted_target = adapter.adapt_dataset_item(
        (raw_input, target), spec)
    assert torch.allclose(adapted_input, raw_input * 2)
    assert torch.allclose(adapted_target, target * 3)


# ---------------------------------------------------------------------------
# NeuralNetDataset save / load
# ---------------------------------------------------------------------------


def test_neural_net_dataset_save_load_preprocessed():
    items = [torch.tensor([float(i)]) for i in range(5)]
    ds = NeuralNet.NeuralNetDataset(items, preprocessed=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ds.pk")
        ds.save(path)
        loaded = NeuralNet.NeuralNetDataset.load(path)
    assert loaded.preprocessed is True
    assert len(loaded) == 5
    assert torch.equal(loaded[2], torch.tensor([2.0]))


def test_neural_net_dataset_save_load_not_preprocessed():
    items = [torch.tensor([float(i)]) for i in range(3)]
    ds = NeuralNet.NeuralNetDataset(items, preprocessed=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ds.pk")
        ds.save(path)
        loaded = NeuralNet.NeuralNetDataset.load(path)
    assert loaded.preprocessed is False


# ---------------------------------------------------------------------------
# wrap_in_adapter
# ---------------------------------------------------------------------------


def test_wrap_in_adapter_with_targets():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    raw_ds = _SimpleDataset(n=5)
    adapted_ds = net.wrap_in_adapter(raw_ds, no_targets=False)

    assert len(adapted_ds) == 5
    for i in range(5):
        raw_input, raw_target = raw_ds[i]
        adapted_input, target = adapted_ds[i]
        assert torch.allclose(adapted_input, raw_input * 2)
        assert torch.allclose(target, raw_target)  # target unchanged


def test_wrap_in_adapter_no_targets():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    raw_ds = _InputOnlyDataset(n=5)
    adapted_ds = net.wrap_in_adapter(raw_ds, no_targets=True)

    assert len(adapted_ds) == 5
    for i in range(5):
        raw_input = raw_ds[i]
        adapted_input = adapted_ds[i]
        assert torch.allclose(adapted_input, raw_input * 2)


def test_wrap_in_adapter_raises_without_adapter():
    net = _SimpleNet()  # no data_adapter
    raw_ds = _SimpleDataset(n=5)
    with pytest.raises(AssertionError):
        net.wrap_in_adapter(raw_ds)


def test_wrap_in_adapter_sets_precompute_only_spec():
    specs_seen = []

    class _SpyPrecompute(DataAdapter):

        def adapt_input(self, raw_input, spec):
            specs_seen.append(spec)
            return raw_input

    net = _SimpleNet(data_adapter=_SpyPrecompute())
    raw_ds = _InputOnlyDataset(n=3)
    adapted_ds = net.wrap_in_adapter(raw_ds,
                                     preprocess_only=True,
                                     no_targets=True)
    _ = [adapted_ds[i] for i in range(3)]
    assert all(s.preprocess_only is True for s in specs_seen)


def test_wrap_in_adapter_sets_input_already_preprocessed():
    specs_seen = []

    class _SpyPreprocessed(DataAdapter):

        def adapt_dataset_item(self, item, spec):
            specs_seen.append(spec)
            return item

    items = [torch.zeros(4) for _ in range(3)]
    preprocessed_ds = NeuralNet.NeuralNetDataset([(x, x) for x in items],
                                                 preprocessed=True)

    net = _SimpleNet(data_adapter=_SpyPreprocessed())
    adapted_ds = net.wrap_in_adapter(preprocessed_ds)
    _ = [adapted_ds[i] for i in range(3)]
    assert all(s.input_already_preprocessed is True for s in specs_seen)


def test_wrap_in_adapter_raw_dataset_not_preprocessed():
    specs_seen = []

    class _SpyRaw(DataAdapter):

        def adapt_dataset_item(self, item, spec):
            specs_seen.append(spec)
            return item

    net = _SimpleNet(data_adapter=_SpyRaw())
    raw_ds = _SimpleDataset(n=3)
    adapted_ds = net.wrap_in_adapter(raw_ds)
    _ = [adapted_ds[i] for i in range(3)]
    assert all(s.input_already_preprocessed is False for s in specs_seen)


def test_adapted_dataset_no_targets_property_default():
    # Default get_no_targets inherits inner dataset's no_targets
    net = _SimpleNet(data_adapter=_DoubleAdapter())

    adapted_no = net.wrap_in_adapter(_InputOnlyDataset(n=3), no_targets=True)
    assert adapted_no.no_targets is True

    adapted_yes = net.wrap_in_adapter(_SimpleDataset(n=3), no_targets=False)
    assert adapted_yes.no_targets is False


def test_adapter_can_flip_no_targets():
    """An adapter that synthesizes targets changes the adapted dataset's
    no_targets from True to False."""

    class _PairAdapter(DataAdapter):
        """Wraps a single input x into a (x, x) pair."""

        def adapt_input(self, raw_input, spec):
            return raw_input, raw_input  # produces (input, target)

        def get_no_targets(self, inner_dataset_has_no_targets, spec):
            return False  # always produces targets

    net = _SimpleNet(data_adapter=_PairAdapter())
    raw_ds = _InputOnlyDataset(n=4)
    adapted_ds = net.wrap_in_adapter(raw_ds, no_targets=True)

    assert adapted_ds.no_targets is False
    # Each item is now a (input, input) pair
    item = adapted_ds[0]
    assert isinstance(item, tuple) and len(item) == 2


def test_make_data_loader_uses_effective_no_targets():
    """make_data_loader passes the post-adapter no_targets to collate_fn so
    the batch structure matches what the adapter produces."""

    class _PairAdapter(DataAdapter):

        def adapt_input(self, raw_input, spec):
            return raw_input, raw_input

        def get_no_targets(self, inner_dataset_has_no_targets, spec):
            return False

    net = _SimpleNet(data_adapter=_PairAdapter())
    raw_ds = _InputOnlyDataset(n=4, feat_dim=4)
    # Raw dataset has no_targets=True, but the adapter produces pairs
    loader = net.make_data_loader(raw_ds,
                                  batch_size=4,
                                  shuffle=False,
                                  no_targets=True)
    batch = next(iter(loader))
    # Batch must be a 2-tuple (inputs, targets) since effective_no_targets=False
    assert isinstance(batch, (list, tuple)) and len(batch) == 2


# ---------------------------------------------------------------------------
# load_or_create_preprocessed_dataset
# ---------------------------------------------------------------------------


def test_load_or_create_preprocessed_dataset_creates_and_loads():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    raw_ds = _SimpleDataset(n=5)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preprocessed.pk")

        ds1 = net.load_or_create_preprocessed_dataset(raw_ds, path)
        assert os.path.exists(path)
        assert ds1.preprocessed is True
        assert len(ds1) == 5

        ds2 = net.load_or_create_preprocessed_dataset(raw_ds, path)
        assert ds2.preprocessed is True
        assert len(ds2) == 5

        raw_input_0, _ = raw_ds[0]
        assert torch.allclose(ds1[0][0], raw_input_0 * 2)
        assert torch.allclose(ds2[0][0], raw_input_0 * 2)


def test_load_or_create_preprocessed_dataset_uses_precompute_only_spec():
    specs_seen = []

    class _SpyAdapter(DataAdapter):

        def adapt_input(self, raw_input, spec):
            specs_seen.append(spec)
            return raw_input

    net = _SimpleNet(data_adapter=_SpyAdapter())
    raw_ds = _InputOnlyDataset(n=3)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preprocessed.pk")
        net.load_or_create_preprocessed_dataset(raw_ds, path, no_targets=True)
    assert all(s.preprocess_only is True for s in specs_seen)
    assert all(s.inference is False for s in specs_seen)


def test_load_or_create_preprocessed_dataset_with_callback():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    call_count = [0]

    def make_dataset():
        call_count[0] += 1
        return _SimpleDataset(n=5)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preprocessed.pk")

        ds1 = net.load_or_create_preprocessed_dataset(make_dataset, path)
        assert call_count[0] == 1

        ds2 = net.load_or_create_preprocessed_dataset(make_dataset, path)
        assert call_count[0] == 1  # not called again

        assert ds2.preprocessed is True


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
# make_data_loader: adapter always applied, inference flag propagation
# ---------------------------------------------------------------------------


def test_make_data_loader_applies_adapter():
    net = _SimpleNet(data_adapter=_DoubleAdapter())
    raw_ds = _SimpleDataset(n=8)

    loader = net.make_data_loader(raw_ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    input_batch, _ = batch

    expected = torch.stack([raw_ds[i][0] * 2 for i in range(8)])
    assert torch.allclose(input_batch, expected)


def test_make_data_loader_no_adapter():
    net = _SimpleNet()  # no data_adapter
    raw_ds = _SimpleDataset(n=8)

    loader = net.make_data_loader(raw_ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    input_batch, _ = batch

    expected = torch.stack([raw_ds[i][0] for i in range(8)])
    assert torch.allclose(input_batch, expected)


def test_make_data_loader_passes_inference_flag_true():
    specs_seen = []

    class _InferenceSpy(DataAdapter):

        def adapt_dataset_item(self, item, spec):
            specs_seen.append(spec)
            return item

    net = _SimpleNet(data_adapter=_InferenceSpy())
    raw_ds = _SimpleDataset(n=4)
    loader = net.make_data_loader(raw_ds,
                                  batch_size=4,
                                  shuffle=False,
                                  inference=True)
    _ = next(iter(loader))
    assert all(s.inference is True for s in specs_seen)


def test_make_data_loader_passes_inference_flag_false():
    specs_seen = []

    class _InferenceSpy(DataAdapter):

        def adapt_dataset_item(self, item, spec):
            specs_seen.append(spec)
            return item

    net = _SimpleNet(data_adapter=_InferenceSpy())
    raw_ds = _SimpleDataset(n=4)
    loader = net.make_data_loader(raw_ds,
                                  batch_size=4,
                                  shuffle=False,
                                  inference=False)
    _ = next(iter(loader))
    assert all(s.inference is False for s in specs_seen)


def test_make_data_loader_passes_input_already_preprocessed():
    specs_seen = []

    class _PreprocessedSpy(DataAdapter):

        def adapt_dataset_item(self, item, spec):
            specs_seen.append(spec)
            return item

    items = [(torch.zeros(4), torch.zeros(2)) for _ in range(4)]
    preprocessed_ds = NeuralNet.NeuralNetDataset(items, preprocessed=True)

    net = _SimpleNet(data_adapter=_PreprocessedSpy())
    loader = net.make_data_loader(preprocessed_ds, batch_size=4, shuffle=False)
    _ = next(iter(loader))
    assert all(s.input_already_preprocessed is True for s in specs_seen)


# ---------------------------------------------------------------------------
# predict: adapt_output is called
# ---------------------------------------------------------------------------


def test_predict_calls_adapt_output():
    """adapt_output must be called once per individual output (not per batch)
    and must receive a single output, not a batch."""
    outputs_seen = []
    output_specs = []

    class _OutputSpy(DataAdapter):

        def adapt_output(self, output, spec):
            outputs_seen.append(output)
            output_specs.append(spec)
            return output

    class _Net4(NeuralNet):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.linear = torch.nn.Linear(4, 2)
            self.initialize()

        def forward(self, x):
            return self.linear(x)

    n_inputs = 8
    batch_size = 4  # 2 batches — wrong code would call adapt_output 2 times
    net = _Net4(data_adapter=_OutputSpy())
    inputs = torch.randn(n_inputs, 4)
    net.predict(inputs, batch_size=batch_size)

    # Called once per individual output, not once per batch
    assert len(outputs_seen) == n_inputs
    assert all(s.inference is True for s in output_specs)
    # Each call receives a single output of shape (2,), not a batch (4, 2)
    assert all(o.shape == torch.Size([2]) for o in outputs_seen)
