from collections.abc import Sized

from torch.utils.data import Dataset, IterableDataset

from .data_adapter import AdaptationSpec, DataAdapter


class AdaptedDataset(Dataset):

    def __init__(self, inner, adapter: DataAdapter, spec: AdaptationSpec,
                 inner_dataset_has_no_targets: bool):
        self._inner = inner
        self._adapter = adapter
        self.adaptation_spec = spec
        self._inner_dataset_has_no_targets = inner_dataset_has_no_targets

    @property
    def no_targets(self) -> bool:
        return self._adapter.get_no_targets(self._inner_dataset_has_no_targets,
                                            self.adaptation_spec)

    def _adapt_item(self, item):
        if self._inner_dataset_has_no_targets:
            return self._adapter.adapt_input(item, self.adaptation_spec)
        return self._adapter.adapt_dataset_item(item, self.adaptation_spec)


class AdaptedSizedDataset(AdaptedDataset):

    def __getitem__(self, idx):
        return self._adapt_item(self._inner[idx])

    def __len__(self):
        return len(self._inner)  # type: ignore


class AdaptedIterableDataset(AdaptedDataset, IterableDataset):

    def __iter__(self):
        for item in self._inner:
            yield self._adapt_item(item)


def make_adapted_dataset(dataset: Dataset, adapter: DataAdapter,
                         spec: AdaptationSpec,
                         inner_dataset_has_no_targets: bool) -> Dataset:
    if isinstance(dataset, Sized):
        wrapper_class = AdaptedSizedDataset
    else:
        assert isinstance(dataset, IterableDataset), (
            "Wrapped datasets must be Sized or IterableDataset.")
        wrapper_class = AdaptedIterableDataset

    return wrapper_class(dataset, adapter, spec, inner_dataset_has_no_targets)
