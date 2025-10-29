import functools
import logging
import os
import pickle
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Callable, Generic, TypeVar, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import (DataLoader, Dataset, Subset, default_collate,
                              random_split)
from tqdm import tqdm

from .defs import InputType, OutputType, TargetType, LossFunType
from gsim.gfigure import Subplot


class Normalizer(ABC, Generic[InputType, OutputType, TargetType]):

    l_params_to_save = [
    ]  # List the attributes to be saved/loaded in this list

    def __init__(self, nn_folder: str | None = None):
        """       

        General scheme of normalization and unnormalization
        ---------------------------------------------------

        Training: 
        - - - - -

                             ┌───┐      ┌─────────┐  output batch ┌──────┐
           ┌-> input batch-> │ 1 │ ---> | forward | ---┬--------->| loss |-----> normalized loss
           |                 └───┘      └─────────┘    |    ┌─--->|      | 
           |                                           |    |     └──────┘
        Dataset                                        |    |
           |                                           |    |     ┌───┐    
           |                                           └--------->| 3 |---┐   ┌──────┐
           |                                                |     └───┘   └-->|      |
           |                              ┌───┐             |                 | loss |---> unnormalized loss
           └-> target batch --------------| 2 |-------------|     ┌───┐   ┌─->|      |
                                          └───┘             └-----| 4 |---┘   └──────┘
                                                                  └───┘
        
        Prediction:
        - - - - - -
                           ┌───┐      ┌─────────┐  output batch ┌───┐        unnormalized 
            input batch -> │ 1 │ ---> | forward | ------------->| 3 |----->  outputs
                           └───┘      └─────────┘               └───┘
        
        Legend:        
        - - - -
        
        1. normalize_input_batch      |
        2. normalize_targets_batch    | -> NeuralNet.collate_fn
        
        3. unnormalize_output_batch   |
        4. unnormalize_targets_batch  | -> NeuralNet.make_unnormalized_loss

        
        Remarks:
        - - - - 

            - The unnormalized loss is computed only if requested (eval_unnormalized_loss). 

            - The normalization (1 and 2) is applied in the DataLoader. One
              could think of alternatively normalizing the entire dataset before
              training, but that would require storage space equal to the size
              of the dataset. For this reason, normalization is carried out in
              the DataLoader. 

            - Steps 1 and 2 are performed in the CPU. The forward and upper loss
              computations are carried out by the GPU. Thus, while the GPU
              processes one batch, the CPU can normalize the next batches.

        """
        self.nn_folder = nn_folder

    @abstractmethod
    def fit(self, dataset: Dataset):
        pass

    def save(self):
        # Override if necessary
        if self.nn_folder is None:
            return

        d_params = {
            param: getattr(self, param)
            for param in self.l_params_to_save
        }

        assert self.params_file is not None
        if os.path.exists(self.nn_folder):
            with open(self.params_file, "wb") as f:
                pickle.dump(d_params, f)

    def load(self):
        # Override if necessary
        if self.nn_folder is None:
            return
        assert self.params_file is not None
        with open(self.params_file, "rb") as f:
            d_params = pickle.load(f)
            for param in self.l_params_to_save:
                assert param in d_params, f"{param} not found in {self.params_file}."
                setattr(self, param, d_params[param])

    @property
    def params_file(self):
        if self.nn_folder is None:
            return None
        return os.path.join(self.nn_folder, "normalizer.pk")

    @abstractmethod
    def normalize_input_batch(self, input_batch: InputType) -> InputType:
        """
        Args:

            `input_batch`: batch_size x ... (input tensor)

        Returns:

            normalized input tensor of same shape as `input_batch`.

        """
        pass

    @abstractmethod
    def normalize_targets_batch(self, targets_batch: TargetType) -> TargetType:
        """
        Args:

            `targets_batch`: batch_size x ... (target tensor)

        Returns:
            normalized targets tensor of same shape as `targets_batch`.
        """
        pass

    @abstractmethod
    def unnormalize_output_batch(self, output_batch: OutputType) -> OutputType:
        """
        Args:

            `output_batch`: batch_size x ... (output tensor)

        Returns:
            unnormalized output tensor of same shape as `output_batch`.

        """
        pass

    @abstractmethod
    def unnormalize_targets_batch(self,
                                  targets_batch: TargetType) -> TargetType:
        """
        Args:

            `targets_batch`: batch_size x ... (target tensor)

        Returns:
            unnormalized targets tensor of same shape as `targets_batch`.

        """
        pass

    def normalize_example_batch(
            self, l_batch: tuple[InputType,
                                 TargetType]) -> tuple[InputType, TargetType]:
        """
        In datasets formed by pairs (input, targets), batch is a
        list of length 2. The first element of this list is a batch
        of input, and the second element is a batch of targets.
        """

        input_batch, targets_batch = l_batch
        return (self.normalize_input_batch(input_batch),
                self.normalize_targets_batch(targets_batch))


class DefaultNormalizer(Normalizer[InputType, OutputType, TargetType]):

    l_params_to_save = [
        "input_batch_mean", "input_batch_std", "targets_batch_mean",
        "targets_batch_std"
    ]

    def __init__(self, mode: str, **kwargs):
        """
        This normalizer is to be used with datasets of pairs (input, targets). 

        This class separately normalizes each entry of the input and targets.
        For example, if the input are an M x N matrix, then normalization
        will subtract a mean M x N matrix obtained by averaging the input
        matrices in the dataset and will divide by the standard deviation M x N
        matrix obtained likewise. 

        Args:

            `mode`: can be
                - "none": no normalization
                - "input": normalize only the input
                - "targets": normalize only the targets
                - "both": normalize both input and targets

            `nn_folder`: if not None, the statistics of the normalization are
            loaded from this folder. When training, the statistics are saved in
            this folder.
        
        """

        super().__init__(**kwargs)
        assert mode in ["input", "targets", "both"
                        ], 'mode must be one of "input", "targets", or "both".'
        self.mode = mode

        self.input_batch_mean: torch.Tensor | None = None  # same shape as the input
        self.input_batch_std: torch.Tensor | None = None  # same shape as the input
        self.targets_batch_mean: torch.Tensor | None = None  # same shape as the targets
        self.targets_batch_std: torch.Tensor | None = None  # same shape as the targets

    def fit(self, dataset: Dataset):

        assert isinstance(dataset, Dataset)

        # Type check to ensure dataset has __len__ method
        if not hasattr(dataset, "__len__"):
            raise NotImplementedError("Dataset must implement __len__ method")

        num_examples = len(dataset)  # type: ignore
        assert num_examples > 0

        # Check that the dataset contains pairs t_input, t_target
        example = dataset[0]
        if not (isinstance(example, (tuple, list)) and len(example) == 2):
            raise NotImplementedError(
                "Dataset must return pairs (t_input, t_target) when indexed.")

        # Initializations
        input_batch, targets_batch = example
        if self.mode in ["input", "both"]:
            self.input_batch_mean = torch.zeros_like(input_batch)
        if self.mode in ["targets", "both"]:
            self.targets_batch_mean = torch.zeros_like(targets_batch)
        t_input_var = torch.zeros_like(input_batch)
        targets_batch_var = torch.zeros_like(targets_batch)

        with torch.no_grad():
            # Estimate the mean matrices
            for ind_example in range(num_examples):
                input_batch, targets_batch = dataset[ind_example]
                if self.mode in ["input", "both"]:
                    self.input_batch_mean += input_batch / num_examples
                if self.mode in ["targets", "both"]:
                    self.targets_batch_mean += targets_batch / num_examples

            # Estimate the standard deviation matrices
            for ind_example in range(num_examples):
                input_batch, targets_batch = dataset[ind_example]
                if self.mode in ["input", "both"]:
                    t_input_var += (input_batch -
                                    self.input_batch_mean)**2 / num_examples
                if self.mode in ["targets", "both"]:
                    targets_batch_var += (
                        targets_batch -
                        self.targets_batch_mean)**2 / num_examples

            if self.mode in ["input", "both"]:
                self.input_batch_std = torch.sqrt(t_input_var)
                # Avoid division by zero
                self.input_batch_std[self.input_batch_std == 0] = 1.0
            if self.mode in ["targets", "both"]:
                self.targets_batch_std = torch.sqrt(targets_batch_var)
                # Avoid division by zero
                self.targets_batch_std[self.targets_batch_std == 0] = 1.0

    def normalize_input_batch(self, input_batch: InputType) -> InputType:
        """
        Args:

            `input_batch`: shape (batch_size, ...)

        """
        if isinstance(input_batch, (list, tuple)):
            raise NotImplementedError()

        if self.mode in ["input", "both"]:
            assert self.input_batch_mean is not None and self.input_batch_std is not None, "The normalizer has not been fitted or loaded from a file."
            return (input_batch - self.input_batch_mean[None, ...]
                    ) / self.input_batch_std[None, ...]
        return input_batch

    def normalize_targets_batch(self, targets_batch: TargetType) -> TargetType:
        """
        Args:

            `targets_batch`: shape (batch_size, *targets_shape)

        """
        if isinstance(targets_batch, (list, tuple)):
            raise NotImplementedError()

        if self.mode in ["targets", "both"]:
            assert self.targets_batch_mean is not None and self.targets_batch_std is not None, "The normalizer has not been fitted or loaded from a file."
            return (targets_batch - self.targets_batch_mean[None, ...]
                    ) / self.targets_batch_std[None, ...]
        return targets_batch

    def unnormalize_targets_batch(self,
                                  targets_batch: TargetType) -> TargetType:
        """
        Args:

            `targets_batch`: shape (batch_size, *targets_shape)

        """
        if isinstance(targets_batch, (list, tuple)):
            raise NotImplementedError()

        if self.mode in ["targets", "both"]:
            assert self.targets_batch_mean is not None and self.targets_batch_std is not None, "The normalizer has not been fitted or loaded from a file."
            return targets_batch.to("cpu") * self.targets_batch_std[
                None, ...] + self.targets_batch_mean[None, ...]
        return targets_batch

    def unnormalize_output_batch(self, output_batch: OutputType) -> OutputType:
        """
        Args:

            `output_batch`: shape (batch_size, *targets_shape)

        """
        if isinstance(output_batch, (list, tuple)):
            raise NotImplementedError()
        return self.unnormalize_targets_batch(output_batch)  # type: ignore


# ============================================================================
# Feature-wise Normalizers
# ============================================================================


class FeatNormalizer(ABC):
    """
    Abstract base class for feature-wise normalizers.
    
    These normalizers process a single feature (column) at a time. They maintain
    an internal state that is updated via the `fit` method, which is called with
    batches of values for that feature.
    """

    l_params_to_save = []  # List the attributes to be saved/loaded

    @abstractmethod
    def fit(self, values: torch.Tensor):
        """
        Update the internal state with a batch of values for this feature.
        
        Args:
            values: 1D tensor or array-like containing values for this feature
        """
        pass

    @abstractmethod
    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values.
        
        Args:
            values: 1D tensor containing values to normalize
            
        Returns:
            Normalized values as a tensor
        """
        pass

    @abstractmethod
    def unnormalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize the given values.
        
        Args:
            values: 1D tensor containing normalized values
            
        Returns:
            Unnormalized values as a tensor
        """
        pass


class IdentityFeatNormalizer(FeatNormalizer):
    """
    A feature normalizer that does not perform any normalization.
    """

    def fit(self, values: torch.Tensor):
        """No-op for identity normalization."""
        pass

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Return values unchanged."""
        return values

    def unnormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Return values unchanged."""
        return values


class StdFeatNormalizer(FeatNormalizer):
    """
    A feature normalizer that normalizes to zero mean and unit standard deviation.
    
    This normalizer accumulates values during the fit phase to compute mean and
    standard deviation, then applies the transformation: (x - mean) / std
    """

    l_params_to_save = ["mean", "std"]

    def __init__(self):
        self.mean: float | None = None
        self.std: float | None = None
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._count: int = 0

    def fit(self, values: torch.Tensor):
        """
        Accumulate statistics from a batch of values.
        
        Args:
            values: 1D tensor containing values for this feature
        """
        if len(values) == 0:
            return

        values_np = values.cpu().numpy() if isinstance(
            values, torch.Tensor) else np.array(values)

        self._sum += np.sum(values_np)
        self._sum_sq += np.sum(values_np**2)
        self._count += len(values_np)

        # Update mean and std
        self.mean = self._sum / self._count
        variance = (self._sum_sq / self._count) - (self.mean**2)
        self.std = np.sqrt(max(variance, 0.0))  # Ensure non-negative variance

        # Avoid division by zero
        if self.std == 0.0:
            self.std = 1.0

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Normalize values to zero mean and unit standard deviation.
        
        Args:
            values: 1D tensor containing values to normalize
            
        Returns:
            Normalized values
        """
        assert self.mean is not None and self.std is not None, \
            "StdFeatNormalizer must be fitted before normalization"
        return (values - self.mean) / self.std

    def unnormalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize values back to original scale.
        
        Args:
            values: 1D tensor containing normalized values
            
        Returns:
            Unnormalized values
        """
        assert self.mean is not None and self.std is not None, \
            "StdFeatNormalizer must be fitted before unnormalization"
        return values * self.std + self.mean


class IntervalFeatNormalizer(FeatNormalizer):
    """
    A feature normalizer that scales values to a specified interval.
    
    This normalizer tracks the minimum and maximum values seen during fitting
    and scales values to the specified target interval.
    """

    l_params_to_save = ["min_val", "max_val", "interval"]

    def __init__(self, interval: tuple[float, float] = (-1.0, 1.0)):
        """
        Args:
            interval: Target interval as (min, max) tuple
        """
        self.interval = interval
        self.min_val: float | None = None
        self.max_val: float | None = None

    def fit(self, values: torch.Tensor):
        """
        Update min and max values from a batch.
        
        Args:
            values: 1D tensor containing values for this feature
        """
        if len(values) == 0:
            return

        values_np = values.cpu().numpy() if isinstance(
            values, torch.Tensor) else np.array(values)

        batch_min = np.min(values_np)
        batch_max = np.max(values_np)

        if self.min_val is None:
            self.min_val = batch_min
        else:
            self.min_val = min(self.min_val, batch_min)

        if self.max_val is None:
            self.max_val = batch_max
        else:
            self.max_val = max(self.max_val, batch_max)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Scale values to the target interval.
        
        Args:
            values: 1D tensor containing values to normalize
            
        Returns:
            Normalized values in the target interval
        """
        assert self.min_val is not None and self.max_val is not None, \
            "IntervalFeatNormalizer must be fitted before normalization"

        # Avoid division by zero
        if self.max_val == self.min_val:
            # All values are the same, map to middle of interval
            return torch.full_like(values,
                                   (self.interval[0] + self.interval[1]) / 2.0)

        # Scale from [min_val, max_val] to [interval[0], interval[1]]
        normalized = (values - self.min_val) / (self.max_val - self.min_val)
        normalized = normalized * (self.interval[1] -
                                   self.interval[0]) + self.interval[0]
        return normalized

    def unnormalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Scale values back from the target interval to original scale.
        
        Args:
            values: 1D tensor containing normalized values
            
        Returns:
            Unnormalized values
        """
        assert self.min_val is not None and self.max_val is not None, \
            "IntervalFeatNormalizer must be fitted before unnormalization"

        # Avoid division by zero
        if self.max_val == self.min_val:
            return torch.full_like(values, self.min_val)

        # Scale from [interval[0], interval[1]] to [min_val, max_val]
        unnormalized = (values - self.interval[0]) / (self.interval[1] -
                                                      self.interval[0])
        unnormalized = unnormalized * (self.max_val -
                                       self.min_val) + self.min_val
        return unnormalized


class MultiFeatNormalizer(Normalizer[InputType, OutputType, TargetType]):
    """
    A normalizer that applies different FeatNormalizers to each feature (column).
    
    This normalizer processes batches of shape (batch_size, num_feat) and applies
    a separate normalizer to each column.
    
    Example:
        normalizer = MultiFeatNormalizer(
            input_normalizers=[
                StdFeatNormalizer(),
                IntervalFeatNormalizer(interval=(-1, 1)),
                IdentityFeatNormalizer()
            ],            
            targets_normalizers=[...],
            output_normalizers=[...]
        )

    The `input_normalizers` and `targets_normalizers` are fit to the dataset. 
        
    If `output_normalizers` is not provided, the target normalizers are used
    for unnormalizing outputs.        
    """

    l_params_to_save = ["input_normalizers_state", "targets_normalizers_state"]

    def __init__(self,
                 input_normalizers: list[FeatNormalizer] | None = None,
                 targets_normalizers: list[FeatNormalizer] | None = None,
                 output_normalizers: list[FeatNormalizer] | None = None,
                 batch_size: int = 32,
                 **kwargs):
        """
        Args:
            input_normalizers: List of FeatNormalizers, one per input feature

            targets_normalizers: List of FeatNormalizers, one per target feature
            
            output_normalizers: List of FeatNormalizers, one per output feature.
            If None or empty, the target normalizers are used for unnormalizing
            outputs.
            
            batch_size: Batch size for fitting normalizers
        """
        super().__init__(**kwargs)
        self.input_normalizers = input_normalizers or []
        self.targets_normalizers = targets_normalizers or []
        self.output_normalizers = output_normalizers or []
        self.batch_size = batch_size

    @property
    def input_normalizers_state(self) -> list[dict]:
        """Get the state of all input normalizers for saving."""
        return [{
            param: getattr(norm, param)
            for param in norm.l_params_to_save
        } for norm in self.input_normalizers]

    @input_normalizers_state.setter
    def input_normalizers_state(self, state: list[dict]):
        """Restore the state of all input normalizers from loaded data."""
        for norm, norm_state in zip(self.input_normalizers, state):
            for param, value in norm_state.items():
                setattr(norm, param, value)

    @property
    def targets_normalizers_state(self) -> list[dict]:
        """Get the state of all target normalizers for saving."""
        return [{
            param: getattr(norm, param)
            for param in norm.l_params_to_save
        } for norm in self.targets_normalizers]

    @targets_normalizers_state.setter
    def targets_normalizers_state(self, state: list[dict]):
        """Restore the state of all target normalizers from loaded data."""
        for norm, norm_state in zip(self.targets_normalizers, state):
            for param, value in norm_state.items():
                setattr(norm, param, value)

    def fit(self, dataset: Dataset):
        """
        Fit all FeatNormalizers using the dataset.
        
        This method iterates through the dataset in batches and fits each
        FeatNormalizer with the values from its corresponding column.
        
        Args:
            dataset: Dataset returning pairs (input, targets)
        """
        assert isinstance(dataset, Dataset)

        if not hasattr(dataset, "__len__"):
            raise NotImplementedError("Dataset must implement __len__ method")

        num_examples = len(dataset)  # type: ignore
        assert num_examples > 0

        # Check that the dataset contains pairs
        example = dataset[0]
        if not (isinstance(example, (tuple, list)) and len(example) == 2):
            raise NotImplementedError(
                "Dataset must return pairs (input, targets) when indexed.")

        input_example, targets_example = example

        # Verify shapes match the number of normalizers
        if len(self.input_normalizers) > 0:
            if isinstance(input_example, torch.Tensor):
                num_input_feats = input_example.shape[-1] if input_example.dim(
                ) > 0 else 1
                assert num_input_feats == len(self.input_normalizers), \
                    f"Number of input normalizers ({len(self.input_normalizers)}) must match " \
                    f"number of input features ({num_input_feats})"

        if len(self.targets_normalizers) > 0:
            if isinstance(targets_example, torch.Tensor):
                num_target_feats = targets_example.shape[
                    -1] if targets_example.dim() > 0 else 1
                assert num_target_feats == len(self.targets_normalizers), \
                    f"Number of target normalizers ({len(self.targets_normalizers)}) must match " \
                    f"number of target features ({num_target_feats})"

        # Fit normalizers by iterating through the dataset in batches
        with torch.no_grad():
            for batch_start in range(0, num_examples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_examples)

                # Collect batch
                input_list = []
                targets_list = []
                for idx in range(batch_start, batch_end):
                    input_ex, targets_ex = dataset[idx]
                    input_list.append(input_ex)
                    targets_list.append(targets_ex)

                # Stack to form batches
                if isinstance(input_list[0], torch.Tensor):
                    input_batch = torch.stack(input_list)
                    if input_batch.dim() == 1:
                        input_batch = input_batch.unsqueeze(1)

                    # Fit input normalizers column by column
                    if len(self.input_normalizers) > 0:
                        for feat_idx, normalizer in enumerate(
                                self.input_normalizers):
                            normalizer.fit(input_batch[:, feat_idx])

                if isinstance(targets_list[0], torch.Tensor):
                    targets_batch = torch.stack(targets_list)
                    if targets_batch.dim() == 1:
                        targets_batch = targets_batch.unsqueeze(1)

                    # Fit target normalizers column by column
                    if len(self.targets_normalizers) > 0:
                        for feat_idx, normalizer in enumerate(
                                self.targets_normalizers):
                            normalizer.fit(targets_batch[:, feat_idx])

    def _apply_to_batch(self, batch: torch.Tensor,
                        normalizers: list[FeatNormalizer],
                        method_name: str) -> torch.Tensor:
        """
        Helper method to apply a transformation to a batch using a list of normalizers.
        
        Args:
            batch: Tensor of shape (batch_size, num_feats)
            normalizers: List of FeatNormalizers, one per feature
            method_name: Name of the method to call on each normalizer ('normalize' or 'unnormalize')
            
        Returns:
            Transformed batch of the same shape
        """
        if len(normalizers) == 0:
            return batch

        if isinstance(batch, (list, tuple)):
            raise NotImplementedError()

        if not isinstance(batch, torch.Tensor):
            return batch

        # Apply each normalizer to its corresponding column
        transformed_columns = []
        for feat_idx, normalizer in enumerate(normalizers):
            col = batch[:, feat_idx]
            transformed_col = getattr(normalizer, method_name)(col)
            transformed_columns.append(transformed_col)

        return torch.stack(transformed_columns, dim=1)

    def normalize_input_batch(self, input_batch: InputType) -> InputType:
        """
        Normalize input batch by applying each FeatNormalizer to its column.
        
        Args:
            input_batch: Tensor of shape (batch_size, num_input_feats)
            
        Returns:
            Normalized input batch of the same shape
        """
        return self._apply_to_batch(input_batch, self.input_normalizers,
                                    'normalize')  # type: ignore

    def normalize_targets_batch(self, targets_batch: TargetType) -> TargetType:
        """
        Normalize targets batch by applying each FeatNormalizer to its column.
        
        Args:
            targets_batch: Tensor of shape (batch_size, num_target_feats)
            
        Returns:
            Normalized targets batch of the same shape
        """
        return self._apply_to_batch(targets_batch, self.targets_normalizers,
                                    'normalize')  # type: ignore

    def unnormalize_output_batch(self, output_batch: OutputType) -> OutputType:
        """
        Unnormalize output batch by applying each FeatNormalizer to its column.
        
        Args:
            output_batch: Tensor of shape (batch_size, num_output_feats)
            
        Returns:
            Unnormalized output batch of the same shape
        """
        if len(self.output_normalizers) == 0:
            # If no output normalizers, use target normalizers (as in DefaultNormalizer)
            return self.unnormalize_targets_batch(output_batch)  # type: ignore

        return self._apply_to_batch(output_batch, self.output_normalizers,
                                    'unnormalize')  # type: ignore

    def unnormalize_targets_batch(self,
                                  targets_batch: TargetType) -> TargetType:
        """
        Unnormalize targets batch by applying each FeatNormalizer to its column.
        
        Args:
            targets_batch: Tensor of shape (batch_size, num_target_feats)
            
        Returns:
            Unnormalized targets batch of the same shape
        """
        return self._apply_to_batch(targets_batch, self.targets_normalizers,
                                    'unnormalize')  # type: ignore
