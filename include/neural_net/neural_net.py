import time
import functools
import logging
import os
import pickle
import multiprocessing
from filelock import FileLock
from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Any, Callable, Generic, List, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import (DataLoader, Dataset, Subset, default_collate,
                              random_split)
from torch.optim.lr_scheduler import _LRScheduler, LRScheduler
from tqdm import tqdm

from gsim.include.utils.statistics import mean_and_ci

from .normalizers import Normalizer, DefaultNormalizer

from .defs import InputType, OutputType, TargetType, LossFunType
from gsim.gfigure import Subplot

try:
    from ....gsim import GFigure
except ImportError:
    from gsim import GFigure
"""
This module provides a utility layer on top of PyTorch to facilitate the
development and deployment of neural networks. 

For examples, install gsim and run the experiments in
`experiments/neuralnet_experiments.py`.

----

Terminology:

    - input batch: argument of `self.forward`. It can be:
        - a tensor of shape (batch_size, ...)
        - a list/tuple of tensors, each of shape (batch_size, ...)

    - output batch: what `self.forward` returns. It can be:
        - a tensor of shape (batch_size, ...)
        - a list/tuple of tensors, each of shape (batch_size, ...)

    - target batch: the expected output. It can be:
        - a tensor of shape (batch_size, ...)
        - a list/tuple of tensors, each of shape (batch_size, ...)

From these batch definitions, one can define:

    - input: each of the items of an input batch. Thus, 
        - if the input batch is a (batch_size, N_1,...,N_D) tensor, then
            the inputs are tensors of shape (N_1,...,N_D). The entries of these
            tensors are referred to as "features".
        - if the input batch is a list/tuple of tensors, each of shape
            (batch_size, N_1,...,N_D), then the inputs are lists/tuples of
            tensors, each of shape (N_1,...,N_D). The entries of these tensors
            are referred to as "features". 
        
    - output: Defined likewise. The entries of the output are referred to
        as "predictions".
        
    - target: defined similarly. The entries of the target are referred to as
        "targets entries".
        
Notes: 
    - In the past, the term "features" was used to refer to "input". However,
      since this is plural, we did not have a way of referring to multiple
      inputs besides `feature batch`. The term `feature batch` only applied to
      the case where the inputs formed a batch. So, if one wanted to refer to a
      collection of inputs (e.g. a dataset), one would have to say `a collection
      of num_feat features`, which was confusing since it seemed to refer to the
      entries of an input. 


"""

gsim_logger = logging.getLogger("gsim")


class Diagnoser(ABC):
    """Abstract base class for neural network diagnosers.
    
    The methods check_forward and check_backward are invoked right after
    the forward and backward passes, respectively. They allow the user to 
    perform custom checks while a batch is being processed.

    """

    @abstractmethod
    def check_forward(self, model: 'NeuralNet', loss: torch.Tensor,
                      data: tuple[InputType, TargetType], f_loss: LossFunType):
        """
        This function is invoked right after a forward pass. 

        Args:
            
            `model`: instance of NeuralNet

            `loss`: computed loss tensor. The result of running
            model._get_loss(data, f_loss)

            `data`: typ. a tuple of two elements. The first is an input batch
            and the second a target batch. 

            `f_loss`: loss function
        
        """
        pass

    @abstractmethod
    def check_backward(self, model: 'NeuralNet', loss: torch.Tensor,
                       data: tuple[InputType,
                                   TargetType], f_loss: LossFunType):
        """
        This function is invoked right after the backward pass.

        Args: same as in check_forward.
        """
        pass


class TrainingHistory():

    def __init__(self):
        # The length of these lists equals the number of steps.
        self.l_train_loss_per_step = []  # Average loss for each batch
        self.l_batch_length_per_step = []  # Needed to compute averages
        self.l_lr = []

        # List of indices where a training session started/resumed
        self.l_step_inds_started_training = []

        # List of indices where a checkpoint was saved. The current weight file
        # corresponds to the last index in this list.
        self.l_step_inds_checkpoints = []

        # The following are lists of (ind_step, value)
        self.l_train_loss_me = []
        self.l_train_loss = []
        self.l_val_loss = []
        self.l_unnormalized_train_loss = []
        self.l_unnormalized_val_loss = []

    @property
    def ind_first_step_current_session(self):
        if len(self.l_step_inds_started_training) == 0:
            return 0
        return self.l_step_inds_started_training[-1]


class NeuralNet(nn.Module, Generic[InputType, OutputType, TargetType], ABC):
    """
    Type arguments:

    - InputType: the type of the inputs and input batches.

    - OutputType: the type of the outputs and output batches.

    - TargetType: the type of the targets and target batches.

    Note: The above syntax can be understood more easily in other languages. For
    example, in Typescript, one would write

    abstract class NeuralNet<InputType, OutputType> extends nn.Module { ... }
    
    """

    _initialized = False

    def __init__(self,
                 *args,
                 nn_folder=None,
                 normalizer: Union[None, Normalizer, str] = None,
                 device_type: Union[None, str] = None,
                 num_workers: int = 0,
                 **kwargs):
        """
        
        Args: 

            `nn_folder`: if not None, the weights of the network are loaded from
            this folder. When training, if validation data is provided, the
            weights that minimize the validation loss are saved in this folder
            together with training metrics. If validation data is not provided,
            the weights that minimize the training loss are saved.
        
            `normalizer`: can be
                - None: no normalization
                - "input": normalize only the input
                - "targets": normalize only the targets
                - "both": normalize both input and targets            
                - an instance of Normalizer: use the provided normalizer
            The options "input", "targets", and "both" can be selected only
            when the dataset comprises pairs of (input, targets). For other
            dataset forms, writing a custom Normalizer is required. 

        """

        super().__init__(*args, **kwargs)
        if device_type is not None:
            self.device_type = device_type
        else:
            self.device_type = (
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.num_workers = num_workers
        gsim_logger.info(f"Using {self.device_type} device")
        if nn_folder is None:
            gsim_logger.warning("* " * 50)
            gsim_logger.warning(
                "*   WARNING: No folder has been specified. The weights of the network will not be saved when training."
            )
            gsim_logger.warning("* " * 50)
        self.nn_folder = nn_folder

        # Set the normalizer to None or to an instance of Normalizer
        if normalizer is None:
            self.normalizer = None
        elif isinstance(normalizer, str):
            self.normalizer = DefaultNormalizer(mode=normalizer,
                                                nn_folder=self.nn_folder)
        elif isinstance(normalizer, Normalizer):
            self.normalizer = normalizer
        else:
            raise ValueError("Invalid normalizer type.")

        # Other initializations
        self._diagnoser: Union[None, Diagnoser] = None

    def initialize(self):
        """
        Any subclass of NeuralNet must call this function at the end of its
        constructor.
        """
        self._initialized = True

        if self.nn_folder is not None:
            # Create the folder if it does not exist
            os.makedirs(self.nn_folder, exist_ok=True)

            if self.normalizer is not None:
                normalizer = self.normalizer
                normalizer.load_if_file_exists()

            if os.path.exists(self.weight_file_path):
                self.load_weights_from_path(self.weight_file_path)
                gsim_logger.info(
                    f"Weights loaded from {self.weight_file_path}")
            else:
                gsim_logger.warning(
                    f"Warning: {os.path.abspath(self.weight_file_path)} does not exist. The network weights will be initialized."
                )

        self.to(
            device=self.device_type, non_blocking=self.device_type
            != "mps")  # bug https://github.com/pytorch/pytorch/issues/139550

    @abstractmethod
    def forward(self, x: InputType) -> OutputType:
        # This method must be overridden by subclasses
        raise NotImplementedError

    def _assert_initialized(self):
        assert self._initialized, "The network has not been initialized. A subclass of NeuralNet must call self.initialize() at the end of its constructor."

    @staticmethod
    def collate_fn(*args, no_targets=False, **kwargs):
        # Override if needed
        return default_collate(*args, **kwargs)

    def uncollate_fn(self, l_batches: list[OutputType]) -> list[OutputType]:
        """            
        Args:
            'l_batches': a list of output batches. Recall from the terminology
            above that output batches are of type OutpuType and, thus, they can
            be tensors, lists, or tuples. 

        Returns:
            A list of outputs. Thus, 

                - If the output batches are (batch_size, N_1,...,N_D)
                    tensors, then the function returns a list of N tensors of
                    shape (N_1,...,N_D), where N is the sum of the batch sizes.

                - If the output batches are lists/tuples of (batch_size,
                    N_1,...,N_D) tensors, then the function returns a list of N
                    lists/tuples of tensors of shape (N_1,...,N_D).
                    
        """

        if isinstance(l_batches[0], torch.Tensor):
            return [
                l_batches[ind_batch][ind_output]
                for ind_batch in range(len(l_batches))
                for ind_output in range(len(l_batches[ind_batch]))
            ]

        elif isinstance(l_batches[0], (list, tuple)):
            # If e.g.
            #
            #  l_batches = [ (T1,T2,T3), (T4,T5,T6), ... ]
            #
            # then the output is
            #
            #  [ (T1[0], T2[0], T3[0]), (T1[1], T2[1], T3[1]), ..., (T1[B1-1], T2[B1-1], T3[B1-1]),
            #   (T4[0], T5[0], T6[0]), (T4[1], T5[1], T6[1]), ..., (T4[B2-1], T5[B2-1], T6[B2-1]),
            #  ... ]
            #
            return [
                type(l_batches[0])(
                    l_batches[ind_batch][ind_output_tensor][ind_output]
                    for ind_output_tensor in range(len(l_batches[ind_batch])))
                for ind_batch in range(len(l_batches))
                for ind_output in range(len(l_batches[ind_batch][0]))
            ]

        else:
            raise TypeError(f"Unsupported batch type: {type(l_batches[0])}")

    def collate_and_normalize(self,
                              l_batch: list[tuple[InputType, TargetType]]
                              | list[InputType],
                              no_targets=False):
        """
        Args:
            
            l_batch' is a list of batch_size pairs (inputs, targets) or
            only inputs.

            'no_targets' (bool): If True, the batch contains only inputs.
            Else, it contains both inputs and targets.

        """

        l_batch = self.collate_fn(l_batch, no_targets=no_targets)

        # After collation, l_batch is (input_batch, targets_batch)
        if self.normalizer is not None:
            if no_targets:
                a = self.normalizer.normalize_input_batch(l_batch)
                l_batch = a
            else:
                l_batch = self.normalizer.normalize_example_batch(
                    l_batch)  # type: ignore
        return l_batch

    def make_unnormalized_loss(self, f_loss: LossFunType) -> LossFunType:
        normalizer = self.normalizer
        assert normalizer is not None
        return lambda output_batch, target_batch: f_loss(
            normalizer.unnormalize_output_batch(output_batch),
            normalizer.unnormalize_targets_batch(target_batch),
        )

    def _get_loss(self, data: tuple[InputType, TargetType],
                  f_loss: LossFunType):
        """
        Args:

            `data`: typ. a tuple of two elements. The first is an input batch
            and the second a target batch. 

        If `unnormalize` is True, the unnormalized loss is returned. This is
        just the result of
             f_loss(
                 unnormalize(self(input_batch)),
                 unnormalize(target_batch)
                 ).
        """

        assert f_loss is not None, "f_loss must be provided unless you override _get_loss."
        input_batch, targets_batch = data
        input_batch = self._move_to_device(input_batch)
        targets_batch = self._move_to_device(targets_batch)

        output_batch = self(input_batch)
        loss = f_loss(output_batch, targets_batch)

        if isinstance(targets_batch, torch.Tensor):
            assert loss.shape[0] == targets_batch.shape[
                0] and loss.ndim == 1, "f_loss must return a vector of length batch_size."
        return loss

    def _run_training_step(self,
                           batch,
                           f_loss: LossFunType,
                           optimizer,
                           lr_scheduler=None,
                           max_grad_norm=None):
        """
        Args:

            `optimizer` 

            `f_loss`: LossFunType

            `lr_scheduler`: if provided, its step() method is invoked after
            the optimizer step.

            `max_grad_norm`: if provided, gradients are clipped to have maximum
            norm `max_grad_norm` during training.

        Returns:
            The vector of losses for the batch.
        
        """

        # Forward pass
        v_loss = self._get_loss(batch, f_loss)  # vector of length batch_size
        if self._diagnoser is not None:
            self._diagnoser.check_forward(self, v_loss, batch, f_loss)

        # Backward pass
        self.zero_grad()
        torch.mean(v_loss).backward()
        if self._diagnoser is not None:
            self._diagnoser.check_backward(self, v_loss, batch, f_loss)

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        # Weight update
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        return v_loss.detach().cpu().numpy()

    def _eval_static_metric(
        self,
        dataloader,
        f_loss: LossFunType,
        max_hci=None,
        alpha: float = 0.05,
    ):
        """
        Averages f_loss across the dataset.

        Args: 

            `f_loss`: LossFunType

            `max_hci`: if not None, the computation stops when the half-width of
            the confidence interval (CI) is below this threshold. 

            `alpha`: significance level for the CI (e.g. 0.05 for 95% CI)

        Returns:
            `metric`: the estimated value of the metric.

            `hci`: half-width of the confidence interval (CI) for the metric.
        
        """

        l_loss_this_epoch = []
        for ind_data, data in enumerate(dataloader):

            with torch.no_grad():
                # Forward pass
                loss = self._get_loss(data,
                                      f_loss)  # vector of length batch_size

            l_loss_this_epoch += loss.detach().cpu().numpy().tolist()

            # Stop if enough examples have been processed
            if max_hci is not None and len(l_loss_this_epoch) >= 2:
                mean, hci = mean_and_ci(l_loss_this_epoch, alpha=alpha)
                if hci <= max_hci:
                    gsim_logger.info(
                        f"    Target accuracy reached during metric evaluation after {ind_data+1} out of {len(dataloader)} batches (used {len(l_loss_this_epoch)} examples)."
                    )
                    return mean, hci

        if len(l_loss_this_epoch) == 0:
            return np.nan, np.nan
        elif len(l_loss_this_epoch) == 1:
            return l_loss_this_epoch[0], np.nan
        else:
            mean, hci = mean_and_ci(l_loss_this_epoch, alpha=alpha)
            if max_hci is not None:
                gsim_logger.info(
                    f"    Target accuracy not reached during metric evaluation (used all {len(dataloader)} batches, {len(l_loss_this_epoch)} examples)."
                )
            return mean, hci

    def evaluate(self,
                 dataset,
                 batch_size,
                 f_loss: LossFunType,
                 unnormalized=True,
                 max_hci=None):
        """
        Args:

            `unnormalized`: If True, the unnormalized loss is returned. If no
            Normalizer is set, then the loss is already unnormalized.

        Returns a dict with key-values:

        "loss": the result of averaging `f_loss` across `dataset`.

        "hci": half-width of the confidence interval (CI) for the loss.
        """
        self._assert_initialized()

        if not unnormalized and self.normalizer is None:
            raise ValueError(
                "Cannot return normalized loss if a normalizer is not set.")

        if unnormalized and self.normalizer is not None:
            f_loss = self.make_unnormalized_loss(f_loss)

        dataloader = self.make_data_loader(dataset, batch_size)
        self.eval()
        loss, hci = self._eval_static_metric(dataloader,
                                             f_loss=f_loss,
                                             max_hci=max_hci)
        return {"loss": loss, "hci": hci}

    class NeuralNetDataset(Dataset):

        def __init__(self, l_items: \
                     list[InputType] | torch.Tensor
                     | tuple[InputType] | list[InputType]):
            self.l_items = l_items

        def __len__(self):
            return len(self.l_items)  # type: ignore

        def __getitem__(self, idx):
            return self.l_items[idx]

    def predict(self,
                data: Union[torch.Tensor, tuple[InputType], list[InputType],
                            Dataset],
                batch_size=32,
                unnormalize=True,
                dataset_includes_targets=False,
                output_class: None | type[torch.Tensor] | type[list]
                | type[tuple] | type[Dataset] = None):
        """

        Note: The terminology in __init__ is used below.

        Args:
            'data': contains a collection of N inputs. It can be:

                - A tensor of shape (N, ...).

                - A tuple/list of length N. Note that, since each item is an
                  input, it can be itself a tensor, a tuple, or a list.

                - A Dataset. 
                    - If `dataset_includes_targets` is False, the Dataset
                      contains N inputs, i.e., dataset[n] is the n-th input.
                    - If `dataset_includes_targets` is True, the Dataset
                      contains N pairs (input, target), i.e., dataset[n][0] is
                      the n-th input. 

            `unnormalize`: if True, the outputs are unnormalized before being
            returned.

        Returns:
            The outputs in an object of class 'output_class'. If
        'output_class' is None, it is set to type('data').

            - If `output_class==torch.Tensor`, then the output is an (N, ...)
              tensor provided that the output of the network is a tensor. If the
              output of the network is a list/tuple of tensors, an exception is
              raised.

            - If `output_class==list` or `output_class==tuple`, then the output
              is a list/tuple with the N outputs. Note that, since each item is
              an output, it can be itself a tensor, a tuple, or a list.

            - If `output_class==Dataset`, then the output is a Dataset with N
              elements, where each element is the output for the corresponding
              input.

        """

        def make_output(l_out, output_class):
            """
            Args:

                'l_out': list of N outputs. Each output can be a tensor, a
                list, or a tuple.

            Returns:
                An object of class 'output_class' containing the outputs.
            
            """

            # Set the default output class
            if output_class is None:
                output_class = Dataset if isinstance(data,
                                                     Dataset) else type(data)

            if output_class == torch.Tensor:
                assert isinstance(
                    l_out[0], torch.Tensor
                ), "If output_class is torch.Tensor, the output of the network must be a tensor."
                return torch.stack(l_out, dim=0)
            elif output_class == tuple:
                return tuple(l_out)
            elif output_class == list:
                return l_out
            elif output_class == Dataset:
                return NeuralNet.NeuralNetDataset(l_out)
            else:
                raise TypeError(f"Unsupported data type: {output_class}")

        self._assert_initialized()

        if not unnormalize and self.normalizer is None:
            raise ValueError(
                "Cannot return normalized outputs if a normalizer is not set.")
        if not isinstance(data, Dataset):
            dataset_includes_targets = False
            dataset = NeuralNet.NeuralNetDataset(data)
        else:
            dataset = data
            if len(dataset) > 0:  # type: ignore
                if dataset_includes_targets:
                    assert (len(dataset[0]) == 2)  # type: ignore

        data_loader = self.make_data_loader(
            dataset,
            batch_size=batch_size,
            no_targets=not dataset_includes_targets)
        l_out = []
        self.eval()
        for batch in data_loader:
            # Ignore the targets if present
            input_batch = batch[0] if dataset_includes_targets else batch

            # Run the forward pass
            input_batch = self._move_to_device(input_batch)
            output_batch = self._move_to_cpu(self(input_batch))
            if unnormalize and self.normalizer is not None:
                output_batch = self.normalizer.unnormalize_output_batch(
                    output_batch)

            # l_out is a list of batches
            l_out.append(output_batch)
        return make_output(self.uncollate_fn(l_out), output_class)

    @property
    def weight_file_path(self):
        assert self.nn_folder is not None
        return self.get_weight_file_path(self.nn_folder)

    @staticmethod
    def make_hist_path(nn_folder):
        assert nn_folder is not None
        return os.path.join(nn_folder, "hist.pk")

    @staticmethod
    def get_weight_file_path(folder):
        return os.path.join(folder, "weights.pth")

    @staticmethod
    def get_best_val_weight_file_path(folder):
        return os.path.join(folder, "weights-best_val.pth")

    @staticmethod
    def get_optimizer_state_file_path(folder):
        return os.path.join(folder, "optimizer.pth")

    def get_lr_scheduler_state_file_path(self, folder):
        return os.path.join(folder, "lr_scheduler.pth")

    def load_weights_from_path(self, path):
        checkpoint = torch.load(path,
                                weights_only=True,
                                map_location=self.device_type)
        self.load_state_dict(checkpoint["weights"])
        self.to(
            device=self.device_type, non_blocking=self.device_type
            != "mps")  # bug https://github.com/pytorch/pytorch/issues/139550
        #load_optimizer_state(initial_optimizer_state_file)

    def save_weights_to_path(self, path):
        gsim_logger.info(f"   💾 Saving weights to {path}")
        torch.save({"weights": self.state_dict()}, path)

    def make_data_loader(self,
                         dataset,
                         batch_size,
                         shuffle=None,
                         no_targets=False):
        """
        Args:
        
            'no_targets' (bool): If True, the batch contains only inputs.
            Else, it contains both inputs and targets.

        """
        # MPS requires 'fork' multiprocessing context to work with num_workers > 0
        # See: https://github.com/pytorch/pytorch/issues/87688
        mp_context = 'fork' if (self.num_workers
                                and self.device_type == "mps") else None

        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers,
                          pin_memory=(self.device_type == "cuda"),
                          multiprocessing_context=mp_context,
                          persistent_workers=self.num_workers > 0,
                          collate_fn=functools.partial(
                              self.collate_and_normalize,
                              no_targets=no_targets))

    def save_hist(self, d_hist):
        if self.nn_folder is not None:
            os.makedirs(self.nn_folder, exist_ok=True)
            lock = FileLock(self.make_hist_path(self.nn_folder) + ".lock")
            # Prevent read/write conflicts, which can occur when we plot the
            # training history dynamically.
            with lock:
                with open(self.make_hist_path(self.nn_folder), "wb") as f:
                    pickle.dump(d_hist, f)

    def load_hist(self) -> TrainingHistory:
        return self.load_hist_from_folder(self.nn_folder)

    @staticmethod
    def load_hist_from_folder(nn_folder) -> TrainingHistory:
        if nn_folder is not None and os.path.exists(
                NeuralNet.make_hist_path(nn_folder)):
            lock = FileLock(NeuralNet.make_hist_path(nn_folder) + ".lock")
            with lock:
                with open(NeuralNet.make_hist_path(nn_folder), "rb") as f:
                    hist = pickle.load(f)
            assert isinstance(
                hist, TrainingHistory
            ), "The training history file has an old format. Please delete it and try again."

        else:
            hist = TrainingHistory()
        return hist

    def fit(self,
            dataset: Dataset,
            f_loss: Callable,
            optimizer,
            lr_scheduler: _LRScheduler | LRScheduler | None = None,
            num_epochs=None,
            num_steps=None,
            dataset_val=None,
            val_split=0.0,
            batch_size=32,
            batch_size_eval=None,
            shuffle=True,
            num_patience_evals=None,
            num_steps_eval_static: int | None = None,
            num_steps_eval_moving: int | None = None,
            num_steps_checkpoint: int | None = None,
            checkpoint_criterion: str | None = None,
            restore_best_checkpoint=None,
            keep_best_val_weights=False,
            static_max_hci=None,
            eval_unnormalized_losses=False,
            unnormalized_max_hci=None,
            obtain_static_training_loss=False,
            max_grad_norm: float | None = None,
            live_plot=False,
            live_plot_interval=1000) -> TrainingHistory:
        """ 
        Starts a training session.

        If 
            - self.nn_folder exists

            - self.nn_folder/optimizer.pth exists,
        
        this function will attempt to load this state into the optimizer. To
        reset the optimizer state, just erase this file before invoking fit. 

        NOTES:
            - If you would like to reset the optimizer, erase/rename
              optimizer.pth. 

            - If you would like to fit the normalizer again, erase/rename
              normalizer.pk.
        
            - If you change the dataset, erase/rename the hist.pk file. This is
              because the losses change. If you do not do this, a checkpoint
              will not be saved until the values of the new (e.g. validation)
              loss are lower than the values of the old (validation) loss.

        Args:
            `dataset` (Dataset): The training dataset.

            `f_loss` (Callable): The loss function f_loss(output_batch,
            target_batch). It should return a vector of shape (batch_size,).

            `optimizer`: The optimizer to use.

            `lr_scheduler` (_LRScheduler | LRScheduler | None): The learning
            rate scheduler.

            `num_epochs` (int | None): Number of additional epochs to train.
            Exactly one of `num_epochs` and `num_steps` must be provided.

            `num_steps` (int | None): Number of additional steps (backward
            passes) to perform. Exactly one of `num_epochs` and `num_steps` must
            be provided.

            `dataset_val` (Dataset | None): The validation dataset. At most one
            of `val_split` and `dataset_val` can be provided.

            `val_split` (float): Fraction of the training data to use for
            validation. Default is 0.0.

            `batch_size` (int): Batch size for training. The default is 32.

            `batch_size_eval` (int | None): Batch size used for evaluating
            metrics/losses. If None, `batch_size` is used.

            `shuffle` (bool): Whether to shuffle the training data. Default is
            True.                        

            `num_patience_evals` (int | None): If provided and the validation
            loss does not improve its minimum in this session for
            `num_patience_evals` evaluations, training will be stopped.

            `num_steps_eval_static` (int | None): Number of steps between static
            metric evaluations. A static metric evaluation means that the
            network weights are the same across batches, i.e., there is no
            gradient noise. 

            `num_steps_eval_moving` (int | None): Number of steps between moving
            metric evaluations. A moving metric evaluation means that the
            network weights are updated between batches, i.e., there is gradient
            noise.

            `num_steps_checkpoint` (int | None): Number of steps between
            checkpoints.

            `checkpoint_criterion` (str | None): Criterion for saving
            checkpoints. Can be:

                - "val_loss": A checkpoint is saved only if the validation loss
                  has improved (i.e., is lower) compared to the validation loss
                  at the previous checkpoint. This is the default if validation
                  data is provided.
                - "train_loss_me": A checkpoint is saved only if the moving
                  estimate of the training loss has improved. This is the
                  default if no validation data is provided.
                - "always": A checkpoint is always saved at every checkpoint
                  interval (num_steps_checkpoint), regardless of whether the
                  loss has improved.
                - "never": Checkpoints are never saved during training.

            `restore_best_checkpoint` (bool | None): Whether to restore the best
            checkpoint at the end of training. If None, it defaults to True if
            `self.nn_folder` is not None.

            `keep_best_val_weights` (bool): In addition to checkpointing, one
            can save the weights that achieve the best validation loss in
            `self.nn_folder/weights-best_val.pth` by using this option. Every
            time the validation loss is evaluated and it improves, the weights
            are saved to this file, but not the optimizer and lr_scheduler
            states, so it is not a checkpoint. This option is ignored if no
            validation data is provided or if `self.nn_folder` is None.

            `static_max_hci` (float | None): Maximum half-width of the
            confidence interval for static metric evaluations. If the half-width
            is below this threshold, the metric evaluation stops early, which
            saves computation time.

            `eval_unnormalized_losses` (bool): Whether to evaluate unnormalized
            losses. Default is False.

            `unnormalized_max_hci` (float | None): Maximum half-width of the
            confidence interval for unnormalized loss evaluations.

            `obtain_static_training_loss` (bool): If True, the training loss is
            computed for fixed network weights at the end of each epoch (or
            static eval interval). Otherwise, only the moving estimate of the
            training loss is computed.

            `max_grad_norm` (float | None): If provided, gradients are clipped
            to have maximum norm `max_grad_norm` during training.

            `live_plot` (bool): If True, a live plot of the training history is
            shown during training.

            `live_plot_interval` (int): Number of ms between updates of the
            live plot.

        Returns:
            TrainingHistory: An object containing the training history.
        """

        def make_validation_data(dataset: Dataset, dataset_val, val_split):
            assert val_split == 0.0 or dataset_val is None
            if dataset_val is None:
                # The data is deterministically split into training and validation
                # sets so that we can resume training.
                assert isinstance(dataset, Sized)
                num_examples_val = int(val_split * len(dataset))
                dataset_train = Subset(dataset,
                                       range(len(dataset) - num_examples_val))
                dataset_val = Subset(
                    dataset,
                    range(len(dataset) - num_examples_val, len(dataset)))
            else:
                dataset_train = dataset
                num_examples_val = len(dataset_val)
            return dataset_train, dataset_val

        def check_checkpoint_args(checkpoint_criterion, val,
                                  num_steps_checkpoint, num_steps_eval_static,
                                  num_steps_eval_moving):
            if checkpoint_criterion == "val_loss":
                assert val, "Validation data must be provided to use val_loss as checkpoint criterion."
                assert num_steps_checkpoint >= num_steps_eval_static, \
                    "num_steps_checkpoint must be at least num_steps_eval_static when using val_loss as checkpoint criterion."
                if num_steps_checkpoint % num_steps_eval_static != 0:
                    gsim_logger.warning(
                        "It is recommended that num_steps_checkpoint be a multiple of num_steps_eval_static when using val_loss as checkpoint criterion. Otherwise, the reference validation loss may be stale."
                    )
            elif checkpoint_criterion == "train_loss_me":
                assert num_steps_checkpoint >= num_steps_eval_moving, \
                    "num_steps_checkpoint must be at least num_steps_eval_moving when using train_loss_me as checkpoint criterion."
                if num_steps_checkpoint % num_steps_eval_moving != 0:
                    gsim_logger.warning(
                        "It is recommended that num_steps_checkpoint be a multiple of num_steps_eval_moving when using train_loss_me as checkpoint criterion. Otherwise, the reference training loss may be stale."
                    )

        def fit_normalizer_if_needed():
            assert self.normalizer is not None
            if not self.normalizer.are_parameters_set:
                gsim_logger.info("Fitting the normalizer...")
                self.normalizer.fit(dataset_train)
                self.normalizer.save()
            else:
                gsim_logger.info(
                    "The normalizer will not be fitted since its parameters have been loaded. "
                    " If you want to fit it again, delete/rename normalizer.pk."
                )

        def make_training_loss_history(hist: TrainingHistory):
            """
            Returns the subset of training loss values and batch lengths
            included in the history of this session, that is, the segments
            corresponding to restored checkpoints. In other words, these would
            be the values if there had been no interruptions in training.
            """
            l_intervals = self.get_session_history_steps(hist)

            def get_points_in_intervals(l_in):
                l_out = []
                for (step_start, step_end) in l_intervals:
                    for (ind_step, loss) in enumerate(l_in):
                        if step_start <= ind_step < step_end:
                            l_out += [loss]
                return l_out

            l_train_loss_per_step = get_points_in_intervals(
                hist.l_train_loss_per_step)
            l_batch_length_per_step = get_points_in_intervals(
                hist.l_batch_length_per_step)
            return l_train_loss_per_step, l_batch_length_per_step

        def get_log_loss_str(l_loss, hci=None):
            l_vals = [t[1] for t in l_loss]
            str_val = f"{l_vals[-1]:.4f}"
            if hci is not None:
                str_val += f" ± {hci:.4f}"
            if l_vals[-1] == min(l_vals):
                return f"{str_val} (best ⭐)"
            else:
                return f"{str_val} (best {min(l_vals):.4f})"

        def get_log_step_str(ind_step):
            if ind_step % num_steps_per_epoch == 0:
                str_epoch = f"{ind_step // num_steps_per_epoch}"
            else:
                str_epoch = f"{ind_step / num_steps_per_epoch:.2f}"
            return f"Step {ind_step} (epoch {str_epoch})"

        def eval_examples_per_second(hist: TrainingHistory):
            step_ind_start = hist.ind_first_step_current_session
            examples_this_session = sum(
                hist.l_batch_length_per_step[step_ind_start:])
            return examples_this_session / total_time_training

        def eval_moving_metrics(ind_step, hist: TrainingHistory):
            """
            Moving metrics are obtained by averaging across the dataset, but the
            values of the parameters differ across batches. Thus, there is
            stochastic gradient noise.
            """

            # A moving estimate of the training loss needs to be stored because
            # we may need to store a checkpoint based on it. However, we can
            # extend the functionality of this function to compute other moving
            # metrics if needed (e.g. an exponential moving average).
            #
            # In the past, we only considered the training loss since the
            # beginning of the current training session. However, previous
            # values from other restored sessions need to be considered as well.
            # Else, the moving estimates may be too optimistic for a few steps
            # right after starting a session. As a result, checkpoints may not
            # be saved afterwards.
            #
            assert len(hist.l_train_loss_per_step) == ind_step + 1, \
                "l_step_inds_started_training has not been computed for this step."

            # Reconstruct the training loss per step for this session plus the
            # historic values.
            l_train_loss_per_step = l_train_loss_per_step_hist + hist.l_train_loss_per_step[
                hist.ind_first_step_current_session:]
            l_batch_length_per_step = l_batch_length_per_step_hist + hist.l_batch_length_per_step[
                hist.ind_first_step_current_session:]

            # We go at most num_steps_per_epoch steps back to compute the
            # moving estimate of the training loss.
            first_step_ind = max(
                0,
                len(l_train_loss_per_step) - num_steps_per_epoch)
            train_loss_me = np.average(
                l_train_loss_per_step[first_step_ind:],
                weights=l_batch_length_per_step[first_step_ind:])
            hist.l_train_loss_me += [(ind_step, train_loss_me)]
            gsim_logger.info(
                f"{get_log_step_str(ind_step)}: "
                f"training loss me = {get_log_loss_str(hist.l_train_loss_me)}, "
                f"lr = {hist.l_lr[-1]:.2g}, "
                f"{int(eval_examples_per_second(hist))} examples/s")

        def eval_static_metrics(ind_step, hist: TrainingHistory,
                                dataloader_train_eval, dataloader_val):
            """
            
            Static metrics are obtained by averaging a function across the
            dataset, but the values of the network weights are the same for all
            batches.
            
            """

            def save_weights_if_best_val(hist: TrainingHistory):
                """
                If the validation loss is the best so far, save the weights to
                a separate file.                
                """
                if self.nn_folder is None:
                    return
                l_intervals = self.get_session_history_steps(
                    hist, include_current_session=True)
                # Get the validation loss values for this session plus the historic ones.
                l_val_loss = []
                for (ind_step, val_loss) in hist.l_val_loss:
                    for (step_start, step_end) in l_intervals:
                        if step_start <= ind_step < step_end:
                            l_val_loss += [val_loss]
                            continue
                if len(l_val_loss) == 0:
                    return
                if l_val_loss[-1] == min(l_val_loss):
                    path_best_val_weights = self.get_best_val_weight_file_path(
                        self.nn_folder)
                    gsim_logger.info(f"│ 🎉 val_loss reached a minimum.")
                    self.save_weights_to_path(path_best_val_weights)

            gsim_logger.info(f"┌{'─' * 100}┐")
            gsim_logger.info(
                f"│ Evaluating static metrics at {get_log_step_str(ind_step)}")

            l_str_log = []
            self.eval()
            if obtain_static_training_loss:
                gsim_logger.info(
                    "│ Computing the static estimate of the training loss...")
                m, hci = self._eval_static_metric(dataloader_train_eval,
                                                  f_loss,
                                                  max_hci=static_max_hci)
                hist.l_train_loss += [(ind_step, m)]
                l_str_log.append("train loss = " +
                                 get_log_loss_str(hist.l_train_loss, hci))
            if dataloader_val:
                gsim_logger.info("│ Computing the validation loss...")
                m, hci = self._eval_static_metric(dataloader_val,
                                                  f_loss,
                                                  max_hci=static_max_hci)
                hist.l_val_loss += [(ind_step, m)]
                l_str_log.append("val loss = " +
                                 get_log_loss_str(hist.l_val_loss, hci))
                if keep_best_val_weights:
                    save_weights_if_best_val(hist)
            if eval_unnormalized_losses and self.normalizer is not None:
                gsim_logger.info(
                    "│ Computing the static estimate of the unnormalized training loss..."
                )
                m, hci = self._eval_static_metric(
                    dataloader_train_eval,
                    self.make_unnormalized_loss(f_loss),
                    max_hci=unnormalized_max_hci)
                hist.l_unnormalized_train_loss += [(ind_step, m)]
                l_str_log.append(
                    "unnormalized train loss = " +
                    get_log_loss_str(hist.l_unnormalized_train_loss, hci))
            if eval_unnormalized_losses and self.normalizer is not None and val:
                gsim_logger.info(
                    "│ Computing the static estimate of the unnormalized validation loss..."
                )
                m, hci = self._eval_static_metric(
                    dataloader_val,
                    self.make_unnormalized_loss(f_loss),
                    max_hci=unnormalized_max_hci)
                hist.l_unnormalized_val_loss += [(ind_step, m)]
                l_str_log.append(
                    "unnormalized val loss = " +
                    get_log_loss_str(hist.l_unnormalized_val_loss, hci))
            gsim_logger.info(f"│ ")
            gsim_logger.info(f"│ Results: ")
            for s in l_str_log:
                gsim_logger.info(f"│ {s}")
            gsim_logger.info(f"└{'─' * 100}┘")

        def save_checkpoint_if_needed(ind_step, hist: TrainingHistory):

            def has_metric_improved_since_prev_checkpoint(l_metric):
                current_criterion_value = l_metric[-1][1]
                # Now let us find the most recent value of the criterion metric
                # at the time of the last checkpoint.
                if len(hist.l_step_inds_checkpoints) == 0:
                    prev_checkpoint_criterion_value = float('inf')
                else:
                    ind_last_checkpoint = hist.l_step_inds_checkpoints[-1]
                    criterion_values_until_last_checkpoint = [
                        v for (s, v) in l_metric if s <= ind_last_checkpoint
                    ]
                    # If there are no such values, set prev_checkpoint_criterion_value
                    # to infinity so that we save a checkpoint.
                    prev_checkpoint_criterion_value = criterion_values_until_last_checkpoint[
                        -1] if len(criterion_values_until_last_checkpoint
                                   ) > 0 else float('inf')
                return current_criterion_value < prev_checkpoint_criterion_value

            if ind_step in hist.l_step_inds_checkpoints:
                return  # Checkpoint already saved at this step.

            if self.nn_folder is None:
                return

            if checkpoint_criterion == "val_loss":
                assert val, "Validation data must be provided to use val_loss as checkpoint criterion."
                assert len(hist.l_val_loss) > 0, \
                    "Validation loss has not been evaluated yet. This should not happen, as num_steps_checkpoint >= num_steps_eval_static."
                is_value_fresh = hist.l_val_loss[-1][0] == ind_step
                if not is_value_fresh:
                    gsim_logger.warning(
                        "The checkpoint criterion is `val_loss`, but the validation loss has not been evaluated at this step. Using the last available value. To avoid this issue, set num_steps_checkpoint to be a multiple of num_steps_eval_static."
                    )
                if has_metric_improved_since_prev_checkpoint(hist.l_val_loss):
                    gsim_logger.info(
                        f"Step {ind_step}: val_loss improved, saving checkpoint."
                    )
                    save_checkpoint()
            elif checkpoint_criterion == "train_loss_me":
                assert len(hist.l_train_loss_me) > 0, \
                    "Training moving estimate loss has not been evaluated yet. This should not happen, as num_steps_checkpoint >= num_steps_eval_moving."
                is_value_fresh = hist.l_train_loss_me[-1][0] == ind_step
                if not is_value_fresh:
                    gsim_logger.warning(
                        "The checkpoint criterion is `train_loss_me`, but the training moving estimate loss has not been evaluated at this step. Using the last available value. To avoid this issue, set num_steps_checkpoint to be a multiple of num_steps_eval_moving."
                    )
                if has_metric_improved_since_prev_checkpoint(
                        hist.l_train_loss_me):
                    gsim_logger.info(
                        f"Step {ind_step}: train_loss_me improved, saving checkpoint."
                    )
                    save_checkpoint()
            elif checkpoint_criterion == "never":
                pass
            elif checkpoint_criterion == "always":
                gsim_logger.info(
                    f"Step {ind_step}: saving checkpoint (always).")
                save_checkpoint()
            else:
                raise ValueError(
                    f"Invalid checkpoint_criterion: {checkpoint_criterion}")

        def save_checkpoint():
            if self.nn_folder is None:
                return
            self.save_weights_to_path(self.get_weight_file_path(
                self.nn_folder))
            save_optimizer_state(
                self.get_optimizer_state_file_path(self.nn_folder))
            if lr_scheduler is not None:
                save_lr_scheduler_state(
                    self.get_lr_scheduler_state_file_path(self.nn_folder))

            hist.l_step_inds_checkpoints.append(ind_step)
            self.save_hist(hist)

        def load_checkpoint():
            assert self.nn_folder is not None
            self.load_weights_from_path(
                self.get_weight_file_path(self.nn_folder))
            load_optimizer_state(
                self.get_optimizer_state_file_path(self.nn_folder))
            if lr_scheduler is not None:
                load_lr_scheduler_state(
                    self.get_lr_scheduler_state_file_path(self.nn_folder))

        def save_optimizer_state(path):
            torch.save({"state": optimizer.state_dict()}, path)

        def load_optimizer_state(path):
            try:
                checkpoint = torch.load(path,
                                        weights_only=True,
                                        map_location=self.device_type)
                optimizer.load_state_dict(checkpoint["state"])
            except Exception as e:
                gsim_logger.warning(
                    f"No optimizer state file found at {path}. Using default initialization."
                )

        def save_lr_scheduler_state(path):
            assert lr_scheduler is not None
            torch.save({"state": lr_scheduler.state_dict()}, path)

        def load_lr_scheduler_state(path):
            assert lr_scheduler is not None
            try:
                checkpoint = torch.load(path,
                                        weights_only=True,
                                        map_location=self.device_type)
                lr_scheduler.load_state_dict(checkpoint["state"])
            except Exception as e:
                gsim_logger.warning(
                    f"LR scheduler state was not found at {path}. Using default initialization."
                )

        def is_patience_exhausted(hist: TrainingHistory) -> bool:
            if num_patience_evals is None:
                return False
            if not val:
                l_ref_tvals = hist.l_train_loss if obtain_static_training_loss else hist.l_train_loss_me
            else:
                l_ref_tvals = hist.l_val_loss
            l_vals = [t[1] for t in l_ref_tvals]
            # If the global minimum of l_vals has not improved in the last
            # num_patience_evals evaluations, return True.
            if len(l_vals) < num_patience_evals + 1:
                return False
            return min(l_vals[-num_patience_evals:]) > min(l_vals)

        # Preparations
        self._assert_initialized()
        torch.cuda.empty_cache()

        # Validation data
        dataset_train, dataset_val = make_validation_data(
            dataset, dataset_val, val_split)
        val = dataset_val is not None and len(dataset_val) > 0

        # Input processing
        batch_size_eval = batch_size_eval if batch_size_eval else batch_size
        num_steps_per_epoch = int(
            np.ceil(len(dataset) /  # type: ignore
                    batch_size))
        assert (num_epochs is None) ^ (num_steps is None), \
            "Exactly one of num_epochs and num_steps must be provided."
        if num_steps is None:
            assert num_epochs is not None
            num_steps = num_epochs * num_steps_per_epoch
        if restore_best_checkpoint is None:
            restore_best_checkpoint = (self.nn_folder is not None)
        if checkpoint_criterion is None:
            checkpoint_criterion = "val_loss" if val else "train_loss_me"
        if num_steps_eval_moving is None:
            num_steps_eval_moving = num_steps_per_epoch
        if num_steps_eval_static is None:
            num_steps_eval_static = max(num_steps_eval_moving,
                                        num_steps_per_epoch)
        if num_steps_checkpoint is None:
            if checkpoint_criterion == "val_loss":
                num_steps_checkpoint = num_steps_eval_static
            elif checkpoint_criterion == "train_loss_me":
                num_steps_checkpoint = num_steps_eval_moving
            else:
                num_steps_checkpoint = num_steps_per_epoch
        check_checkpoint_args(checkpoint_criterion, val, num_steps_checkpoint,
                              num_steps_eval_static, num_steps_eval_moving)

        # Fit the normalizer
        if self.normalizer is not None:
            fit_normalizer_if_needed()

        # Instantiate the data loaders
        dataloader_train = self.make_data_loader(dataset_train, batch_size,
                                                 shuffle)
        dataloader_train_eval = self.make_data_loader(dataset_train,
                                                      batch_size_eval, shuffle)
        dataloader_val = self.make_data_loader(dataset_val, batch_size,
                                               shuffle) if val else None

        # History initialization
        hist = self.load_hist()
        ind_step = len(hist.l_train_loss_per_step)
        hist.l_step_inds_started_training += [ind_step]
        total_time_training = 0.0
        l_train_loss_per_step_hist, l_batch_length_per_step_hist = \
            make_training_loss_history(hist)

        # Try to load the optimizer state if available in self.nn_folder
        if self.nn_folder is not None:
            load_optimizer_state(
                self.get_optimizer_state_file_path(self.nn_folder))
            if lr_scheduler is not None:
                load_lr_scheduler_state(
                    self.get_lr_scheduler_state_file_path(self.nn_folder))

        # Live plotting
        lpprocess = None
        if live_plot and self.nn_folder is not None:
            lpprocess = NeuralNet.live_plot(self.nn_folder,
                                            interval=live_plot_interval,
                                            background=True)

        done = False
        while not done:
            for batch in dataloader_train:

                # Training step
                self.train()
                time_start_step = time.perf_counter()
                v_loss_train_this_step = self._run_training_step(
                    batch, f_loss, optimizer, lr_scheduler, max_grad_norm)
                total_time_training += time.perf_counter() - time_start_step
                hist.l_train_loss_per_step += [v_loss_train_this_step.mean()]
                hist.l_batch_length_per_step += [len(v_loss_train_this_step)]
                hist.l_lr.append(optimizer.param_groups[0]["lr"])

                # Moving-metric evaluation
                if ind_step % num_steps_eval_moving == 0:
                    eval_moving_metrics(ind_step, hist)
                    self.save_hist(hist)

                # Static-metric evaluation
                if ind_step % num_steps_eval_static == 0:
                    eval_static_metrics(ind_step, hist, dataloader_train_eval,
                                        dataloader_val)
                    self.save_hist(hist)

                # Checkpointing
                if ind_step > 0 and ind_step % num_steps_checkpoint == 0:
                    save_checkpoint_if_needed(ind_step, hist)

                # Patience
                if is_patience_exhausted(hist):
                    gsim_logger.info("Patience exhausted. Stopping training.")
                    done = True
                    break

                ind_step += 1
                if ind_step >= hist.ind_first_step_current_session + num_steps:
                    done = True
                    break

        if restore_best_checkpoint and hist.l_step_inds_checkpoints:
            load_checkpoint()

        # Terminate the plotting process if running
        if lpprocess is not None:
            lpprocess.terminate()
            lpprocess.join()

        return hist

    @staticmethod
    def live_plot(nn_folder: str,
                  interval=1000,
                  background: bool = False) -> multiprocessing.Process | None:
        """
        It starts a figure that is periodically refreshed to show the latest
        training history stored in `nn_folder`.

        Args:
            `nn_folder`: folder where the neural network training history is
            stored.
            
            `interval`: refresh interval in milliseconds.

            `background`: If True, the live plot is started in a separate
            process and a handle to this process is returned. 
        """

        def launch_in_background() -> 'multiprocessing.Process':
            """
            It starts a separate process that does the plotting.
            """

            # Start the live plotting in a separate process
            plot_process = multiprocessing.Process(target=NeuralNet.live_plot,
                                                   kwargs={
                                                       "nn_folder": nn_folder,
                                                       "interval": interval,
                                                       "background": False
                                                   })
            plot_process.start()
            return plot_process

        if background:
            return launch_in_background()

        def make_figure():
            hist = NeuralNet.load_hist_from_folder(nn_folder)
            return NeuralNet.plot_training_history(hist)[0]

        G = GFigure.make_periodically_refreshing_figure(
            f_make_figure=make_figure, interval=interval)
        if G is not None:
            G.plot()
            G.show()

    def set_diagnoser(self, diagnoser: Diagnoser | None):
        """
        If provided, the Diagnoser is used to analyze the network right after
        every forward and backward pass. To disable diagnosing, just set it to
        None. 
        """
        self._diagnoser = diagnoser

    def _move_to_device(self, obj: Union[torch.Tensor, list, tuple]):
        if isinstance(obj, torch.Tensor):
            return obj.float().to(
                self.device_type, non_blocking=self.device_type !=
                "mps")  # bug https://github.com/pytorch/pytorch/issues/139550
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(item) for item in obj)
        else:
            raise TypeError("Unsupported type.")

    @staticmethod
    def _move_to_cpu(obj: Union[torch.Tensor, list, tuple]):
        if isinstance(obj, torch.Tensor):
            return obj.detach().to(
                "cpu", non_blocking=False
            )  # bug https://github.com/pytorch/pytorch/issues/139550
        elif isinstance(obj, (list, tuple)):
            return type(obj)(NeuralNet._move_to_cpu(item) for item in obj)
        else:
            raise TypeError("Unsupported type.")

    @staticmethod
    def get_session_history_steps(
            hist: TrainingHistory,
            include_current_session=False) -> List[Tuple[int, int]]:
        """
        Returns a list of (start_step, end_step) tuples that define the the
        intervals of steps that belong to the history of the current training
        session. This is used to compute the moving estimate of the training
        loss. 

        `start_step` corresponds to the beginning of a training session, whereas
        `end_step - 1` corresponds to the last checkpoint in that session. 

        The current session is not included unless `include_current_session` is
        True, in which case the last interval contains all steps in the current
        session. 

        The tuples are non-overlapping and sorted by start_step.

        For example, if 

            hist.l_step_inds_started_training = [0,              5000, 12000,
            18000] hist.l_step_inds_checkpoints =      [   2000, 4000,
            8000, 10000,        15000       ]

            This means that at 5000, the checkpoint at 4000 was restored and at
            12000 the checkpoint at 10000 was restored. The current session
            starts at 18000, but it is not included in the output.
    
            Then, the output will be [(0, 4001), (5000, 10001), (12000, 15001)].

        Another example: if 

            hist.l_step_inds_started_training = [0,             5000, 12000 ]
            hist.l_step_inds_checkpoints =      [   2000, 4000, 5000, 14000]

            This means that at 5000, the checkpoint at 4000 was restored and a
            new checkpoint was saved. At 12000, the checkpoint at 5000 was
            restored. The current session started at 12000 and a new checkpoint
            was saved, but this is not included in the output.

            Then, the output will be [(0, 4001), (5000, 5001)].
            
        """
        l_sessions = []

        for i in range(len(hist.l_step_inds_started_training) - 1):
            start_step = hist.l_step_inds_started_training[i]
            next_session_start = hist.l_step_inds_started_training[i + 1]

            last_checkpoint = None
            for checkpoint in hist.l_step_inds_checkpoints:
                if start_step <= checkpoint < next_session_start:
                    last_checkpoint = checkpoint

            if last_checkpoint is not None:
                l_sessions.append((start_step, last_checkpoint + 1))

        # Include current session if requested
        if include_current_session and len(
                hist.l_step_inds_started_training) > 0:
            l_sessions.append((hist.ind_first_step_current_session,
                               len(hist.l_train_loss_per_step)))

        return l_sessions

    @staticmethod
    def plot_training_history(hist: TrainingHistory, first_step_to_plot=0):

        def split_data_by_session_history(l_x, l_y, l_session_steps,
                                          current_session_start):
            """
            Splits data points into solid (in session history) and dotted (out
            of session history) point lists.
            
            NaN values are inserted at transitions between in and out session
            history to prevent matplotlib from connecting non-contiguous
            segments.
            
            Returns: (l_x_solid, l_y_solid, l_x_dotted, l_y_dotted)
            """

            def build_invalid_intervals(l_valid_intervals, min_x, max_x):
                l_invalid_intervals = []

                # Before first valid interval
                if l_valid_intervals and min_x < l_valid_intervals[0][0]:
                    l_invalid_intervals.append(
                        (min_x, l_valid_intervals[0][0]))

                # Gaps between valid intervals
                for i in range(len(l_valid_intervals) - 1):
                    gap_start = l_valid_intervals[i][1]
                    gap_end = l_valid_intervals[i + 1][0]
                    if gap_start < gap_end:
                        l_invalid_intervals.append((gap_start, gap_end))

                # After last valid interval
                if l_valid_intervals and max_x >= l_valid_intervals[-1][1]:
                    l_invalid_intervals.append(
                        (l_valid_intervals[-1][1], max_x + 1))

                return l_invalid_intervals

            l_x_solid = []
            l_y_solid = []
            l_x_dotted = []
            l_y_dotted = []

            if len(l_x) == 0:
                return l_x_solid, l_y_solid, l_x_dotted, l_y_dotted

            # Build list of all valid intervals (historical sessions + current session)
            l_valid_intervals = list(l_session_steps)
            l_valid_intervals.append((current_session_start, max(l_x) + 1))

            # Sort intervals by start
            l_valid_intervals.sort(key=lambda interval: interval[0])

            # Build invalid intervals (gaps between valid intervals and before/after)
            min_x = min(l_x)
            max_x = max(l_x)
            l_invalid_intervals = build_invalid_intervals(
                l_valid_intervals, min_x, max_x)

            # Extract points in each valid interval
            for i, (start, end) in enumerate(l_valid_intervals):
                if i > 0:
                    # Add NaN separator between intervals
                    l_x_solid.append(l_x[0])  # Arbitrary x value
                    l_y_solid.append(np.nan)

                for x, y in zip(l_x, l_y):
                    if start <= x < end:
                        l_x_solid.append(x)
                        l_y_solid.append(y)

            # Extract points in each invalid interval
            for i, (start, end) in enumerate(l_invalid_intervals):
                if i > 0:
                    # Add NaN separator between intervals
                    l_x_dotted.append(l_x[0])  # Arbitrary x value
                    l_y_dotted.append(np.nan)

                for x, y in zip(l_x, l_y):
                    if start <= x < end:
                        l_x_dotted.append(x)
                        l_y_dotted.append(y)

            return l_x_solid, l_y_solid, l_x_dotted, l_y_dotted

        def plot_keys(l_keys, margin_coef=0.1):
            max_y_value = -np.inf
            min_y_value = np.inf
            max_x_value = -np.inf
            s1 = Subplot(xlabel="Step", ylabel="Loss")

            l_session_steps = NeuralNet.get_session_history_steps(hist)
            current_session_start = hist.l_step_inds_started_training[
                -1] if hist.l_step_inds_started_training else 0

            for ind_key, key in enumerate(l_keys):
                lt_step_values = getattr(hist, key)
                if len(lt_step_values) == 0:
                    # Add a placeholder to be modified later in case of dynamic plotting
                    s1.add_curve(yaxis=[np.nan], legend="_")
                    continue
                assert isinstance(
                    lt_step_values[0],
                    tuple), "Not implemented for non-tuple values."
                l_x = [t[0] for t in lt_step_values]
                l_y = [t[1] for t in lt_step_values]

                l_x_solid, l_y_solid, l_x_dotted, l_y_dotted = split_data_by_session_history(
                    l_x, l_y, l_session_steps, current_session_start)

                if l_x_dotted:
                    s1.add_curve(xaxis=l_x_dotted,
                                 yaxis=l_y_dotted,
                                 legend="_",
                                 styles=f":#{ind_key}")
                if l_x_solid:
                    s1.add_curve(xaxis=l_x_solid,
                                 yaxis=l_y_solid,
                                 legend=key,
                                 styles=f"-#{ind_key}")

                if len(l_y) > first_step_to_plot:
                    l_vals = l_y[first_step_to_plot:]
                    max_y_value = max(max_y_value, np.nanmax(l_vals))
                    min_y_value = min(min_y_value, np.nanmin(l_vals))
                if len(l_x) > 0:
                    max_x_value = max(max_x_value, l_x[-1])
            if max_y_value != -np.inf and min_y_value != np.inf:
                margin = margin_coef * (max_y_value - min_y_value)
                s1.ylim = (min_y_value - margin, max_y_value + margin)
            if max_x_value != -np.inf:
                s1.xlim = (first_step_to_plot, max_x_value)
            return s1

        def plot_loss_and_learning_rate():

            def plot_restored_checkpoints_and_session_starts(
                    subplot: Subplot, hist: TrainingHistory):
                l_step_inds_started_training = hist.l_step_inds_started_training[
                    1:]  # Exclude the first
                if not len(l_step_inds_started_training):
                    # Add placeholders to be modified later in case of dynamic
                    # plotting
                    subplot.add_vertical_lines(x_positions=[np.nan],
                                               style="k",
                                               legend_str="_")
                    subplot.add_vertical_lines(x_positions=[np.nan],
                                               style="r",
                                               legend_str="_")
                    return

                l_session_steps = NeuralNet.get_session_history_steps(hist)
                l_restored_checkpoints = [
                    end_step - 1 for _, end_step in l_session_steps
                ]

                subplot.add_vertical_lines(x_positions=l_restored_checkpoints,
                                           style="k",
                                           legend_str="Restored checkpoints")
                subplot.add_vertical_lines(
                    x_positions=l_step_inds_started_training,
                    style="r",
                    legend_str="Session starts")

            s1 = plot_keys(["l_train_loss_me", "l_train_loss", "l_val_loss"])
            plot_restored_checkpoints_and_session_starts(s1, hist)
            s2 = Subplot(xlabel="Step", ylabel="Learning rate", sharex=True)
            s2.add_curve(yaxis=hist.l_lr if len(hist.l_lr) > 0 else [np.nan])
            G = GFigure()
            G.l_subplots = [s1, s2]
            return G

        def plot_unnormalized_loss():
            l_keys = ["l_unnormalized_train_loss", "l_unnormalized_val_loss"]
            l_keys_to_plot = []
            for key in l_keys:
                lt_step_vals = getattr(hist, key)
                if len(lt_step_vals):
                    l_keys_to_plot.append(key)
            if not len(l_keys_to_plot):
                return None
            G = GFigure()
            G.l_subplots = [plot_keys(l_keys_to_plot)]
            return G

        l_G = []

        G1 = plot_loss_and_learning_rate()
        l_G.append(G1)

        G2 = plot_unnormalized_loss()
        if G2 is not None:
            l_G.append(G2)

        return l_G

    def print_num_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Total number of parameters: {total_params}')
