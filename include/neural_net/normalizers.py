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
