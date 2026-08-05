import bisect
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

from ...include.utils.statistics import mean_and_ci

from .datasets import (AdaptedDataset, AdaptedIterableDataset,
                       AdaptedSizedDataset, make_adapted_dataset)
from .normalizers import Normalizer, DefaultNormalizer
from .data_adapter import AdaptationSpec, DataAdapter


def _seed_worker(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)


from .defs import InputType, OutputType, TargetType, LossFunType, WeightedLoss
from ...gfigure import Subplot

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
    def check_forward(self, model: 'NeuralNet',
                      loss: 'torch.Tensor | WeightedLoss',
                      data: tuple[InputType, TargetType], f_loss: LossFunType):
        """
        This function is invoked right after a forward pass.

        Args:

            `model`: instance of NeuralNet

            `loss`: the computed loss. The result of running
            model._get_loss(data, f_loss), i.e., a tensor of loss values or a
            WeightedLoss.

            `data`: typ. a tuple of two elements. The first is an input batch
            and the second a target batch.

            `f_loss`: loss function

        """
        pass

    @abstractmethod
    def check_backward(self, model: 'NeuralNet',
                       loss: 'torch.Tensor | WeightedLoss',
                       data: tuple[InputType,
                                   TargetType], f_loss: LossFunType):
        """
        This function is invoked right after the backward pass.

        Args: same as in check_forward.
        """
        pass


def _get_me_coefficient_from_optimizer(optimizer) -> float:
    """Return the β₁ coefficient shared by all param_groups of `optimizer`.

    Raises ValueError if the optimizer does not expose β₁, or if param_groups
    disagree on its value (in which case pass `training_loss_forgetting_factor` explicitly to
    `NeuralNet.fit`).
    """
    values = []
    for group in optimizer.param_groups:
        betas = group.get("betas")
        if betas is None:
            raise ValueError(
                "Optimizer does not expose a beta_1 coefficient. "
                "Pass training_loss_forgetting_factor explicitly to fit().")
        values.append(betas[0])
    if len(set(values)) > 1:
        raise ValueError(
            "Optimizer param_groups have differing beta_1 values. "
            "Pass training_loss_forgetting_factor explicitly to fit().")
    return float(values[0])


class TrainingHistory():

    def __init__(self):
        # The length of these lists equals the number of steps.
        self.l_train_loss_per_step = []  # Average loss for each batch

        # Number of loss values (i.e., entries of the vector returned by the
        # loss function) that were used at each step. Needed to compute
        # averages. See `usage.md`, Sec. "Multiple loss values per example".
        self.l_num_loss_vals_per_step = []
        self.l_lr = []

        # List of indices where a training session started/resumed
        self.l_step_inds_started_training = []

        # List of indices where a checkpoint was saved. The current weight file
        # corresponds to the last index in this list.
        self.l_step_inds_checkpoints = []

        # The following are lists of (ind_step, value)
        self.l_train_loss = []
        self.l_val_loss = []
        self.l_unnormalized_train_loss = []
        self.l_unnormalized_val_loss = []

        # Forgetting factor used to compute the moving estimate of the training
        # loss. Set by `fit`. The moving estimate is recomputed on demand from
        # `l_train_loss_per_step` via `TrainingHistory.compute_train_loss_me`.
        self.last_used_training_loss_forgetting_factor: float | None = None

        # Step indices at which the moving estimate of the training loss was
        # reported. Used by `fit` to compute the "best so far" log string.
        # On `fit` resume, indices that fell into the abandoned tail of the
        # most recent session are dropped via
        # `drop_reported_train_loss_me_steps_since_last_restored_checkpoint`.
        self.l_reported_train_loss_me_steps: list[int] = []

    def __setstate__(self, state):
        # Backwards compatibility: `l_batch_length_per_step` was renamed to
        # `l_num_loss_vals_per_step`. Histories pickled before the rename carry
        # the old attribute name, so map it onto the new one on load.
        if ("l_num_loss_vals_per_step" not in state
                and "l_batch_length_per_step" in state):
            state["l_num_loss_vals_per_step"] = state.pop(
                "l_batch_length_per_step")
        self.__dict__.update(state)

    @property
    def ind_first_step_current_session(self):
        if len(self.l_step_inds_started_training) == 0:
            return 0
        return self.l_step_inds_started_training[-1]

    def compute_train_loss_me(self,
                              forgetting_factor: float | None = None
                              ) -> list[float]:
        """
        Returns the bias-corrected EMA of the per-step training loss as a list
        of length `len(self.l_train_loss_per_step)`. The entry with index `i`
        holds the moving estimate at step `i`.

        The list is built from two parallel sequences of the same length:
        `l_train_loss_ema` (raw EMA) and `l_num_batches_ema` (effective count)
        defined recursively as
            - if i ∈ l_step_inds_started_training and m is the largest
              checkpoint index strictly smaller than i (taking m = ∅ if no such
              checkpoint exists, with ema_m = 0 and num_batches_m = 0):
                  ema_i          = β · ema_m + (1 - β) · x_i num_batches_i  =
                  num_batches_m + 1
            - otherwise:
                  ema_i          = β · ema_{i-1} + (1 - β) · x_i num_batches_i
                  = num_batches_{i-1} + 1
        with x_i = self.l_train_loss_per_step[i]. The returned value at index i
        is the bias-corrected estimate
            ema_i / (1 - β^{num_batches_i}).

        If `forgetting_factor` is None, it is read from
        `self.last_used_training_loss_forgetting_factor`. Returns the empty list
        if there are no per-step values or no forgetting factor.
        """
        if forgetting_factor is None:
            forgetting_factor = self.last_used_training_loss_forgetting_factor
        num_steps = len(self.l_train_loss_per_step)
        if num_steps == 0 or forgetting_factor is None:
            return []

        beta = float(forgetting_factor)
        started = set(self.l_step_inds_started_training)
        l_ckpts = self.l_step_inds_checkpoints  # already sorted ascending

        l_train_loss_ema: list[float] = [np.nan] * num_steps
        l_num_batches_ema: list[int] = [-1] * num_steps
        s_at_ckpt: dict[int, float] = {}
        t_at_ckpt: dict[int, int] = {}
        next_ckpt_idx = 0  # index of the next entry of l_ckpts to consume

        for i in range(num_steps):
            if i in started:
                idx = bisect.bisect_left(l_ckpts, i)
                if idx > 0:
                    m = l_ckpts[idx - 1]
                    s_base = s_at_ckpt[m]
                    t_base = t_at_ckpt[m]
                else:
                    s_base = 0.0
                    t_base = 0
            else:
                s_base = l_train_loss_ema[i - 1] if i > 0 else 0.0
                t_base = l_num_batches_ema[i - 1] if i > 0 else 0
            x = float(self.l_train_loss_per_step[i])
            l_train_loss_ema[i] = beta * s_base + (1.0 - beta) * x
            l_num_batches_ema[i] = t_base + 1
            if next_ckpt_idx < len(l_ckpts) and l_ckpts[next_ckpt_idx] == i:
                s_at_ckpt[i] = l_train_loss_ema[i]
                t_at_ckpt[i] = l_num_batches_ema[i]
                next_ckpt_idx += 1

        return [
            s / (1.0 - beta**t) if (1.0 - beta**t) > 0 else s
            for s, t in zip(l_train_loss_ema, l_num_batches_ema)
        ]

    def drop_reported_train_loss_me_steps_since_last_restored_checkpoint(
            self) -> None:
        """
        Drops entries of `l_reported_train_loss_me_steps` whose step lies
        strictly after the most recent checkpoint, i.e. reports that came
        from the abandoned tail discarded by the latest checkpoint
        restoration. If no checkpoint exists, no entries are dropped.

        This relies on being called once per `fit` resume so that, at any
        time, at most one such abandoned tail can be present.
        """
        if not self.l_step_inds_checkpoints:
            return
        last_restored = self.l_step_inds_checkpoints[-1]
        self.l_reported_train_loss_me_steps = [
            s for s in self.l_reported_train_loss_me_steps
            if s <= last_restored
        ]


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
                 data_adapter: Union[None, DataAdapter] = None,
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

        def set_normalizer_if_needed(normalizer):
            # Set the normalizer to None or to an instance of Normalizer
            if normalizer is None:
                return None
            elif isinstance(normalizer, str):
                return DefaultNormalizer(mode=normalizer)
            elif isinstance(normalizer, Normalizer):
                return normalizer
            else:
                raise ValueError("Invalid normalizer type.")

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

        self.normalizer = set_normalizer_if_needed(normalizer)
        if self.normalizer is not None and hasattr(
                self.normalizer, "folder") and self.normalizer.folder is None:
            self.normalizer.folder = self.nn_folder

        self.data_adapter = data_adapter

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

    def collate_fn(self, *args, no_targets=False, **kwargs):
        """
        If `no_targets` is True, then the `l_batch` argument and the returned
        batch contain only inputs. Else, they contain both inputs and targets.
        """
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
            
            l_batch' is a list of batch_size pairs (inputs, targets) or only
            inputs.

            'no_targets' (bool): If True, the `l_batch` and the returned batch
            contain only inputs. Else, they contain both inputs and targets.

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
        Computes the loss for a batch of data. The implementation below assumes
        that data is a tuple (input_batch, target_batch). For other setups,
        override this function.         
        
        Args:

            `data`: one of the items returned by the DataLoader. In supervised
            learning, it is typically a tuple of two elements: the first is an
            input batch and the second a target batch.
            
            `f_loss`: the loss function that the user passes to `fit` or
            `evaluate`. `f_loss` is typically a square loss, l1 loss,
            cross-entropy loss, etc. In complex setups, one may override
            `_get_loss` and ignore `f_loss`. 

        If `unnormalize` is True, the unnormalized loss is returned. This is
        just the result of
             f_loss(
                 unnormalize(self(input_batch)), unnormalize(target_batch) ).

        Returns:
            This function returns what `f_loss` returns, i.e., either a vector
            of `num_loss_vals` entries or a `WeightedLoss` whose `values` field
            is a vector of `num_loss_vals` entries (see `usage.md`, Sec.
            "Weighted loss values"). 
            
            Usually, each example (i.e., each of the `batch_size` input-target
            pairs in a batch) produces one loss value. In that case,
            `num_loss_vals` equals the batch size. The reason why the loss
            values are returned as a vector rather than in an aggregate scalar
            is so that each loss value can be weighted properly when batches
            have different sizes. Note that this can happen even without
            gradient accumulation, e.g., when the length of the dataset is not
            an integer multiple of the batch size, so that the last batch is
            smaller.

            However, each example may produce multiple loss values. In that
            case, `num_loss_vals` equals the total number of loss values
            produced by all the examples in the batch in `data`.

            See `usage.md`, Sec. "Multiple loss values per example".
        """

        assert f_loss is not None, "f_loss must be provided unless you override _get_loss."
        input_batch, targets_batch = data
        input_batch = self._move_to_device(input_batch)
        targets_batch = self._move_to_device(targets_batch)

        output_batch = self(input_batch)
        loss = f_loss(output_batch, targets_batch)

        v_loss_vals = loss.values if isinstance(loss, WeightedLoss) else loss
        assert v_loss_vals.ndim == 1, (
            "f_loss must return a 1D tensor of loss values (or a WeightedLoss "
            "whose `values` field is a 1D tensor), but the returned loss "
            f"values have shape {tuple(v_loss_vals.shape)}. If each entry of "
            "the returned tensor is a loss value, just flatten it (e.g. with "
            ".squeeze(-1) or .flatten()). See usage.md.")
        if isinstance(targets_batch, torch.Tensor):
            batch_size = targets_batch.shape[0]
            if v_loss_vals.shape[0] < batch_size:
                gsim_logger.warning(
                    "f_loss returns fewer loss values than batch elements. "
                    "This may introduce bias in the training loss estimate "
                    "if all batches do not have the same number of loss values. "
                    "See usage.md.")
        return loss

    def _run_training_step(self,
                           get_batch,
                           f_loss: LossFunType,
                           optimizer,
                           lr_scheduler=None,
                           max_grad_norm=None,
                           min_num_loss_vals_accumulate_grad=None):
        """
        Performs a single training step, i.e., a single update of the network
        weights. When `min_num_loss_vals_accumulate_grad` is provided, the
        gradients of multiple batches are accumulated before the weight update,
        which emulates training with a larger batch without the associated
        memory cost (gradient accumulation).

        Args:

            `get_batch`: a callable that returns a batch each time it is called.
            It is invoked once per training step when
            `min_num_loss_vals_accumulate_grad` is None, and possibly multiple
            times otherwise (see below).

            `f_loss`: LossFunType

            `optimizer`

            `lr_scheduler`: if provided, its step() method is invoked after the
            optimizer step.

            `max_grad_norm`: if provided, gradients are clipped to have maximum
            norm `max_grad_norm` during training.

            `min_num_loss_vals_accumulate_grad`: if None (default), a single
            batch is used per training step (no accumulation) and the behavior
            is that of a standard training step. If an int is provided, batches
            are fetched via `get_batch` and their gradients accumulated until
            the total number of loss values seen in the step reaches this value;
            only then is the optimizer step performed. The resulting gradient
            equals the gradient of the (possibly weighted) mean loss across all
            the loss values used in the step, exactly as if they formed a single
            batch. See `usage.md`, Sec. "Multiple loss values per example".

        Returns:
            `(loss_this_step, num_loss_vals)`: the weighted mean of the loss
            values across all the batches used in the step (the plain mean when
            `f_loss` returns unweighted tensors; cf. `usage.md`, Sec. "Weighted
            loss values") and the number of loss values used in the step.

        """

        self.zero_grad()

        sum_weighted_loss = 0.0
        sum_weights = 0.0
        num_loss_vals = 0
        while True:
            # Forward pass
            batch = get_batch()
            loss = self._get_loss(batch, f_loss)
            v_loss, v_weights = WeightedLoss.ensure(
                loss)  # vectors of length num_loss_vals of this batch
            if self._diagnoser is not None:
                self._diagnoser.check_forward(self, loss, batch, f_loss)

            # Backward pass. The weighted sum (rather than the weighted mean) of
            # the loss values is used so that, after dividing by the total sum
            # of the weights below, the accumulated gradient equals the gradient
            # of the weighted mean loss over all the loss values used in the
            # step, regardless of how many batches were accumulated or how many
            # loss values each contained. For plain (unweighted) losses, the
            # weights are all ones and this reduces to the mean over all the
            # loss values.
            weighted_loss_sum = torch.sum(v_weights * v_loss)
            weighted_loss_sum.backward()
            if self._diagnoser is not None:
                self._diagnoser.check_backward(self, loss, batch, f_loss)

            sum_weighted_loss += float(weighted_loss_sum.detach())
            sum_weights += float(v_weights.sum())
            num_loss_vals += len(v_loss)

            if (min_num_loss_vals_accumulate_grad is None
                    or num_loss_vals >= min_num_loss_vals_accumulate_grad):
                break

        # Turn the accumulated sum of gradients into the weighted mean over all
        # loss values.
        assert sum_weights > 0, (
            "The weights returned by f_loss in this training step sum to 0, so "
            "the weighted mean loss is undefined.")
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.grad /= sum_weights

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        # Weight update
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        return sum_weighted_loss / sum_weights, num_loss_vals

    def _eval_static_metric(
        self,
        dataloader,
        f_loss: LossFunType,
        max_hci=None,
        max_num_loss_vals=None,
        alpha: float = 0.05,
    ):
        """
        Averages f_loss across the dataset. If `f_loss` returns `WeightedLoss`
        objects, the weighted mean of the loss values is computed instead of the
        plain mean; cf. `usage.md`, Sec. "Weighted loss values".

        Args:

            `f_loss`: LossFunType

            `max_hci`: if not None, the computation stops when the half-width of
            the confidence interval (CI) is below this threshold.

            `max_num_loss_vals`: if not None, the computation stops after this
            many loss values (i.e., entries of the vector returned by
            `_get_loss`) have been processed. If None and the dataset has a
            length, the whole dataset is processed once. If both
            `max_num_loss_vals` and `max_hci` are None and the dataset has no
            length, a ValueError is raised.

            `alpha`: significance level for the CI (e.g. 0.05 for 95% CI)

        Returns:
            `metric`: the estimated value of the metric.

            `hci`: half-width of the confidence interval (CI) for the metric.

        """
        dataset = dataloader.dataset
        if (max_num_loss_vals is None and max_hci is None
                and not isinstance(dataset, Sized)):
            raise ValueError(
                "Evaluation of static metrics requires either a dataset with length, "
                "(static_)max_num_loss_vals, or (static_)max_hci.")

        # For a dataset with length, `max_num_loss_vals=None` means "process the
        # whole dataset once", which is achieved by exhausting the dataloader
        # (the loop below stops on StopIteration). No count limit is imposed in
        # that case, so it works regardless of how many loss values each example
        # produces.
        l_loss_vals = []
        l_weights = []
        num_batches = 0
        data_iter = iter(dataloader)
        while True:
            if max_num_loss_vals is not None and len(
                    l_loss_vals) >= max_num_loss_vals:
                break
            try:
                data = next(data_iter)
            except StopIteration:
                break
            num_batches += 1

            with torch.no_grad():
                v_loss, v_weights = WeightedLoss.ensure(
                    self._get_loss(data, f_loss)
                )  # vectors of length num_loss_vals of the batch

            l_loss_vals += v_loss.detach().cpu().numpy().tolist()
            l_weights += v_weights.cpu().numpy().tolist()

            if max_hci is not None and len(l_loss_vals) >= 2:
                mean, hci = mean_and_ci(l_loss_vals,
                                        alpha=alpha,
                                        weights=l_weights)
                if hci <= max_hci:
                    gsim_logger.info(
                        f"    Target accuracy reached during metric evaluation after {num_batches} batches (used {len(l_loss_vals)} loss values)."
                    )
                    return mean, hci

        if len(l_loss_vals) == 0:
            return np.nan, np.nan
        elif len(l_loss_vals) == 1:
            return l_loss_vals[0], np.nan
        else:
            mean, hci = mean_and_ci(l_loss_vals,
                                    alpha=alpha,
                                    weights=l_weights)
            if max_hci is not None:
                gsim_logger.info(
                    f"    Target accuracy not reached during metric evaluation (used {num_batches} batches, {len(l_loss_vals)} loss values)."
                )
            return mean, hci

    def evaluate(self,
                 dataset,
                 batch_size,
                 f_loss: LossFunType,
                 no_targets=False,
                 unnormalized=True,
                 max_hci=None,
                 max_num_loss_vals=None):
        """
        Args:

            `no_targets` (bool): If True, the dataset contains only inputs.
            Else, it contains pairs (input, target). This is needed e.g. for
            unsupervised learning.

            `unnormalized`: If True, the unnormalized loss is returned. If no
            Normalizer is set, then the loss is already unnormalized.

            `max_num_loss_vals` (int | None): Maximum number of loss values
            (i.e., entries of the vector returned by `_get_loss`) used to
            estimate the loss. If the dataset has a length and this is None, the
            whole dataset is processed once. For datasets without a length,
            either `max_num_loss_vals` or `max_hci` must be provided.

        Returns a dict with key-values:

        "loss": the result of averaging `f_loss` across `dataset`. If `f_loss`
        returns `WeightedLoss` objects, this is the weighted mean of the loss
        values; cf. `usage.md`, Sec. "Weighted loss values".

        "hci": half-width of the confidence interval (CI) for the loss.
        """
        self._assert_initialized()

        if not unnormalized and self.normalizer is None:
            raise ValueError(
                "Cannot return normalized loss if a normalizer is not set.")

        if unnormalized and self.normalizer is not None:
            f_loss = self.make_unnormalized_loss(f_loss)

        dataloader = self.make_data_loader(dataset,
                                           batch_size,
                                           no_targets=no_targets)
        self.eval()
        loss, hci = self._eval_static_metric(
            dataloader,
            f_loss=f_loss,
            max_hci=max_hci,
            max_num_loss_vals=max_num_loss_vals)
        return {"loss": loss, "hci": hci}

    class NeuralNetDataset(Dataset):

        def __init__(self, l_items: \
                     list[InputType] | torch.Tensor
                     | tuple[InputType] | list[InputType],
                     preprocessed: bool = False):
            self.l_items = l_items
            self.preprocessed = preprocessed

        def __len__(self):
            return len(self.l_items)  # type: ignore

        def __getitem__(self, idx):
            return self.l_items[idx]

        def save(self, path: str):
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "l_items": self.l_items,
                        "preprocessed": self.preprocessed
                    }, f)

        @classmethod
        def load(cls, path: str) -> 'NeuralNet.NeuralNetDataset':
            with open(path, "rb") as f:
                d = pickle.load(f)
            return cls(l_items=d["l_items"], preprocessed=d["preprocessed"])

    def wrap_in_adapter(self,
                        dataset: Dataset,
                        preprocess_only: bool = False,
                        inference: bool = False,
                        no_targets: bool = False) -> AdaptedDataset:
        """Return a lazy wrapper dataset that applies the data adapter to each
        item.

        Args:
            dataset: The dataset to wrap. preprocess_only: True when called from
                ``load_or_create_preprocessed_dataset``.
            inference: True when called from ``predict``. no_targets: If True,
            the dataset contains only inputs (no targets).

        Returns:
            A dataset `D` that applies the adapter on-the-fly. The outputs of
            this method comprise only inputs if `D.no_targets` is True and
            pairs (input, target) otherwise.
        """
        assert self.data_adapter is not None, \
            "wrap_in_adapter requires data_adapter to be set."

        spec = AdaptationSpec(
            preprocess_only=preprocess_only,
            input_already_preprocessed=getattr(dataset, "preprocessed", False),
            inference=inference,
        )

        return make_adapted_dataset(dataset, self.data_adapter, spec,
                                    no_targets)

    def load_or_create_preprocessed_dataset(
            self,
            dataset_or_callback,
            path: str,
            no_targets: bool = False) -> 'NeuralNet.NeuralNetDataset':
        """Load a preprocessed dataset from disk or create and save it.

        Args:
            dataset_or_callback: Either a Dataset or a callable that returns a
                Dataset. The callable form avoids loading the dataset into
                memory when the preprocessed file already exists.
            path: File path for saving/loading the preprocessed dataset.
            no_targets: Passed to `preprocess_dataset`.

        Returns:
            A NeuralNetDataset with preprocessed=True.
        """
        if os.path.exists(path):
            gsim_logger.info(f"Loading preprocessed dataset from {path}")
            ds_out = NeuralNet.NeuralNetDataset.load(path)
            assert getattr(ds_out, "preprocessed", False) is True, \
                "The loaded dataset is not preprocessed. This may be because `path` points to a file that was not created by `load_or_create_preprocessed_dataset`. To fix this, delete the file at `path` and run again."
            return ds_out

        dataset = dataset_or_callback() if callable(
            dataset_or_callback) else dataset_or_callback

        assert isinstance(dataset, Dataset), \
            "Only torch Dataset instances can be preprocessed and saved to disk."
        assert isinstance(dataset, Sized), \
            ("Only datasets with a finite length can be preprocessed and saved to disk.")

        gsim_logger.info(f"Preprocessing dataset and saving to {path}...")
        preprocessed_dataset = self.wrap_in_adapter(dataset,
                                                    preprocess_only=True,
                                                    no_targets=no_targets)
        assert isinstance(preprocessed_dataset, AdaptedSizedDataset)
        l_items = [
            preprocessed_dataset[i] for i in range(len(preprocessed_dataset))
        ]
        nn_dataset = NeuralNet.NeuralNetDataset(l_items, preprocessed=True)
        nn_dataset.save(path)
        return nn_dataset

    def predict(self,
                data: Union[torch.Tensor, tuple[InputType], list[InputType],
                            Dataset],
                batch_size=32,
                unnormalize=True,
                no_targets=None,
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

            `unnormalize`: if True, the outputs are unnormalized before being
            returned.
            
            `no_targets`:
                - If `no_targets` is True, the data contains only inputs. For
                  example, if `data` is a dataset, then data[n] is the n-th input.
                - If `no_targets` is False, the data contains N pairs (input,
                  target). If `data` is a dataset, data[n][0] is the n-th input.
                By default, `no_targets` is set to True, since this is the most
                common case for prediction.
                  

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

        no_targets = True if no_targets is None else no_targets

        if not unnormalize and self.normalizer is None:
            raise ValueError(
                "Cannot return normalized outputs if a normalizer is not set.")
        if not isinstance(data, Dataset):
            dataset = NeuralNet.NeuralNetDataset(data)
        else:
            dataset = data
            if not isinstance(dataset, Sized):
                raise NotImplementedError(
                    "predict does not support datasets without a length (e.g. IterableDataset)."
                )
            if len(dataset) > 0:  # type: ignore
                if not no_targets:
                    assert (len(dataset[0]) == 2)  # type: ignore

        data_loader = self.make_data_loader(dataset,
                                            batch_size=batch_size,
                                            no_targets=no_targets,
                                            inference=True)
        effective_no_targets = getattr(data_loader.dataset, 'no_targets',
                                       no_targets)

        l_out = []
        self.eval()
        for batch in data_loader:
            # Ignore the targets if present
            input_batch = batch[0] if not effective_no_targets else batch

            # Run the forward pass
            input_batch = self._move_to_device(input_batch)
            output_batch = self._move_to_cpu(self(input_batch))
            if unnormalize and self.normalizer is not None:
                output_batch = self.normalizer.unnormalize_output_batch(
                    output_batch)
            # l_out is a list of batches
            l_out.append(output_batch)

        l_uncollated = self.uncollate_fn(l_out)

        if self.data_adapter is not None:
            l_uncollated = [
                self.data_adapter.adapt_output(x,
                                               AdaptationSpec(inference=True))
                for x in l_uncollated
            ]
        return make_output(l_uncollated, output_class)

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
                         dataset: Dataset,
                         batch_size,
                         shuffle=None,
                         no_targets=False,
                         inference=False):
        """
        Args:
            no_targets (bool): If True, the batch contains only inputs.
            inference (bool): If True, sets ``AdaptationSpec.inference=True``
                in the adapter wrapper (used by ``predict``).
        """
        if self.data_adapter is not None:
            adapted_dataset: Any = self.wrap_in_adapter(dataset,
                                                        inference=inference,
                                                        no_targets=no_targets)
            effective_no_targets = adapted_dataset.no_targets
            dataset = adapted_dataset
        else:
            effective_no_targets = no_targets

        # IterableDataset does not support shuffling inside DataLoader.
        if not isinstance(dataset, Sized) and shuffle:
            shuffle = False

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
                          worker_init_fn=_seed_worker,
                          collate_fn=functools.partial(
                              self.collate_and_normalize,
                              no_targets=effective_no_targets))

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
            optimizer: torch.optim.Optimizer,
            f_loss: LossFunType | None = None,
            lr_scheduler: _LRScheduler | LRScheduler | None = None,
            no_targets=False,
            num_epochs=None,
            num_steps=None,
            dataset_val=None,
            val_split=None,
            batch_size=32,
            batch_size_eval=None,
            shuffle=True,
            num_patience_evals=None,
            num_steps_eval: int | None = None,
            num_steps_report_training_loss: int | None = None,
            training_loss_forgetting_factor: float | None = None,
            num_steps_checkpoint: int | None = None,
            checkpoint_criterion: str | None = None,
            min_num_steps_reliable_train_loss_me: int | None = None,
            restore_best_checkpoint=None,
            keep_best_val_weights=False,
            static_max_hci=None,
            static_max_num_loss_vals=None,
            eval_unnormalized_losses=False,
            unnormalized_max_hci=None,
            obtain_static_training_loss=False,
            max_grad_norm: float | None = None,
            min_num_loss_vals_accumulate_grad: int | None = None,
            live_plot=False,
            live_plot_interval=1000,
            num_significant_figures=4) -> TrainingHistory:
        """ 
        Starts a training session. A session comprises a sequence of training
        steps. The state is saved at checkpoints, which take place at a subset
        of these steps. The session ends when `fit` returns or when it is
        interrupted. The weights and optimizer state are saved only at
        checkpoints, whereas the loss values are saved at every step. This
        allows one (i) to recover from divergence by restoring the last
        checkpoint, and (ii) to visualize how the losses evolved after the last
        checkpoint. For example, consider the following sequence:
        
        S S S S C S S S S S C S S S S S EOS S S S C S S S C S S S EOS S S S EOS
        * * * *   * * * * *                 * * *   * * * 
        
        where S stands for a training step, C for a checkpoint, and EOS for the
        end of a session. When plotting, the loss values at the steps marked
        with * are plotted with a solid line, whereas the loss values at the
        remaining steps are plotted with a dashed line. 

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
            
            `optimizer`: The optimizer to use. 

            `f_loss` (LossFunType | None): The loss function
            f_loss(output_batch, target_batch). It is expected to return either

                - a 1D tensor of shape (num_loss_values,), where num_loss_values
                  is typically equal to the batch size (but can be greater; cf.
                  `usage.md`, Sec. "Multiple loss values per example"), or

                - an object of class `WeightedLoss` whose `values` field is a 1D
                  tensor of shape (num_loss_values,).

            The reason why the loss values are returned as a vector rather than
            as a single aggregate scalar is so that each loss value can be
            weighted properly when batches have different sizes, which can
            happen even without gradient accumulation; cf. `usage.md`, Secs.
            "Multiple loss values per example" and "Weighted loss values". When
            a `WeightedLoss` is returned, all loss aggregations (training steps
            and static metric evaluations) compute the weighted mean of the loss
            values instead of the plain mean.

            `f_loss` may be None only when `_get_loss` is overridden. 

            `no_targets` (bool): If True, the datasets (training and validation)
            contain only inputs. Else, they contain pairs (input, target).
            Default is False. This is needed e.g. for unsupervised learning.

            `lr_scheduler` (_LRScheduler | LRScheduler | None): The learning
            rate scheduler.

            `num_epochs` (int | None): Number of additional epochs to train.
            Exactly one of `num_epochs` and `num_steps` must be provided.

            `num_steps` (int | None): Number of additional steps (backward
            passes) to perform. Exactly one of `num_epochs` and `num_steps` must
            be provided.

            `dataset_val` (Dataset | None): The validation dataset. At most one
            of `val_split` and `dataset_val` can be provided.

            `val_split` (float | None): Fraction of the training data to use for
            validation. Default is None, which means validation is only
            performed if `dataset_val` is provided. Must be None for datasets
            without a length (e.g. IterableDataset).

            `batch_size` (int): Batch size for training. The default is 32.

            `batch_size_eval` (int | None): Batch size used for evaluating
            metrics/losses. If None, `batch_size` is used.

            `shuffle` (bool): Whether to shuffle the training data. Default is
            True.                        

            `num_patience_evals` (int | None): If provided and the validation
            loss does not improve its minimum in this session for
            `num_patience_evals` evaluations, training will be stopped.

            `num_steps_eval` (int | None): Number of steps between static metric
            evaluations. Here, "static" means that the network weights are the
            same across batches, i.e., there is no gradient noise. The
            validation loss is an example of a static metric. 

            `num_steps_report_training_loss` (int | None): Every this many
            steps, the moving estimate of the training loss is printed. This
            estimate is obtained as an exponential moving average of the
            per-step (batch) training loss (cf.
            `training_loss_forgetting_factor` below). It is called a `moving`
            estimate because the weights of the network are different at each
            step (gradient noise). This estimate is conceptually the loss seen
            by the optimizer. It is recommended to adjust this parameter to
            print the moving estimate every few seconds.

            `training_loss_forgetting_factor` (float | None): The coefficient
            used for the exponential moving average of the training loss:
                train_loss_me = training_loss_forgetting_factor * train_loss_me
                               + (1 - training_loss_forgetting_factor) *
                                 loss_this_step
            If `training_loss_forgetting_factor==None`, it is read from the
            optimizer as β₁ (e.g. `betas[0]` for Adam/AdamW). All param_groups
            must agree on β₁; otherwise a ValueError is raised asking for an
            explicit value. If the optimizer does not expose β₁ (e.g. plain
            SGD), a ValueError is also raised. The coefficient can differ
            between sessions.

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

            `min_num_steps_reliable_train_loss_me` (int | None): Step index
            below which the moving estimate of the training loss is considered
            too noisy to be used as a "best so far" criterion. When set:

                - The "(best …)" annotation in the per-step training-loss-me log
                  is suppressed while `ind_step` is below this threshold, and
                  earlier reports are excluded from the running minimum once the
                  threshold has been crossed.
                - When `checkpoint_criterion == "train_loss_me"`, no checkpoint
                  is saved before this step.

                Known limitation: the threshold is applied to the absolute step
                index, regardless of whether training has been resumed from a
                checkpoint. 

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

            `static_max_num_loss_vals` (int | None): Maximum number of loss
            values (each batch element contributes at least one loss value; cf.
            `usage.md`) used for static metric evaluations (static training
            loss, validation loss, and unnormalized variants). When None, the
            whole dataset is processed once (if the dataset is finite). For
            datasets without a length, either this or `static_max_hci` must be
            provided.

            `eval_unnormalized_losses` (bool): Whether to evaluate unnormalized
            losses. Default is False.

            `unnormalized_max_hci` (float | None): Maximum half-width of the
            confidence interval for unnormalized loss evaluations.

            `obtain_static_training_loss` (bool): If True, the training loss is
            computed for fixed network weights every `num_steps_eval` steps.
            This allows one to see the training loss without the gradient noise.
            Default is False.

            `max_grad_norm` (float | None): If provided, gradients are clipped
            to have maximum norm `max_grad_norm` during training.

            `min_num_loss_vals_accumulate_grad` (int | None): If None (default),
            each training step uses a single batch, as usual. If an int is
            provided, gradient accumulation is enabled: at each training step,
            batches are drawn and their gradients accumulated until the total
            number of loss values (each batch element (example) contributes at
            least one loss value; cf. usage.md) reaches this value, and only
            then is the weight update performed. This emulates training with a
            larger batch (of at least this many loss values) without the
            associated memory cost. The batch size of each individual
            forward/backward pass is still given by `batch_size`. Note that each
            step thus consumes (generally) multiple batches, so `num_steps`
            counts weight updates, not batches.

            `live_plot` (bool): If True, a live plot of the training history is
            shown during training.

            `live_plot_interval` (int): Number of ms between updates of the live
            plot.

            `num_significant_figures` (int): Number of significant figures used
            when printing loss values and other relevant floats. Default is 4.

        Returns:
            TrainingHistory: An object containing the training history.
        """

        def make_validation_data(dataset: Dataset, dataset_val,
                                 val_split) -> tuple[Dataset, Dataset | None]:
            assert val_split is None or dataset_val is None, \
                "At most one of val_split and dataset_val can be provided."
            if val_split is None:
                dataset_train = dataset
            else:
                # Deterministically split into training and validation sets so
                # that the partition if the same if training is resumed.
                assert isinstance(dataset, Sized), \
                    "val_split requires a dataset with length; use dataset_val instead."
                num_examples_val = int(val_split * len(dataset))
                dataset_train = Subset(dataset,
                                       range(len(dataset) - num_examples_val))
                dataset_val = Subset(
                    dataset,
                    range(len(dataset) - num_examples_val, len(dataset)))
            return dataset_train, dataset_val

        def resolve_fit_schedule(checkpoint_criterion, val,
                                 num_steps_per_epoch, num_steps_checkpoint,
                                 num_steps_eval,
                                 num_steps_report_training_loss):
            """Resolves checkpoint_criterion and the three step-interval
            arguments.

            Returns (checkpoint_criterion, num_steps_checkpoint,
                     num_steps_eval, num_steps_report_training_loss).
            Any of the three `num_steps_` constants may remain None on exit.
            
            The main rule that this function needs to ensure is the following:
            
            RULE 1: If nn_folder is not None and checkpoint_criterion is not
                "never", then the returned constants should allow saving
                checkpoints. This requires:
        
                - `checkpoint_criterion` to be defined     
        
                - if `checkpoint_criterion` is "train_loss_me" or == "val_loss",
                  then `num_steps_checkpoint` is set. 
        
                - If checkpoint_criterion is "val_loss", then
                  `num_steps_eval` must be set. 
            
            RULE 2: If checkpoint_criterion is "always", then
                `num_steps_checkpoint` must be 1, and `num_steps_report_training_loss`
                must be 1.
                
            RULE 3: If checkpoint_criterion is "never", then
                `num_steps_checkpoint` must be None.
                
            Other than that, this function tries to infer reasonable defaults
            for the returned values based on the provided arguments. 
            
            """
            # --- Step 0: resolve checkpoint_criterion ---
            if self.nn_folder is None:
                if checkpoint_criterion is not None and checkpoint_criterion != "never":
                    gsim_logger.warning(
                        f"Setting checkpoint_criterion = 'never' because no nn_folder was provided "
                        f"(was '{checkpoint_criterion}').")
                checkpoint_criterion = "never"
            elif checkpoint_criterion is None:
                checkpoint_criterion = "val_loss" if val else "train_loss_me"

            # --- Step 1: resolve step intervals ---
            if checkpoint_criterion == "train_loss_me":
                if num_steps_checkpoint is None:
                    if num_steps_report_training_loss is not None:
                        num_steps_checkpoint = num_steps_report_training_loss
                    elif num_steps_per_epoch is not None:
                        num_steps_checkpoint = num_steps_per_epoch
                    else:
                        raise ValueError(
                            "At least `num_steps_report_training_loss` or `num_steps_checkpoint` need to be "
                            "provided because checkpoint_criterion is 'train_loss_me' and the dataset "
                            "has no length.")
                if num_steps_eval is None and val:
                    if num_steps_per_epoch is not None:
                        num_steps_eval = num_steps_per_epoch
                    else:
                        gsim_logger.warning(
                            "Validation data was provided but `num_steps_eval` was not set, "
                            "so the validation loss will not be computed.")

            elif checkpoint_criterion == "val_loss":
                assert val, "Validation data must be provided to use val_loss as checkpoint criterion."
                if num_steps_eval is None and num_steps_checkpoint is not None:
                    num_steps_eval = num_steps_checkpoint
                elif num_steps_checkpoint is None and num_steps_eval is not None:
                    num_steps_checkpoint = num_steps_eval
                elif num_steps_eval is None and num_steps_checkpoint is None:
                    if num_steps_per_epoch is not None:
                        num_steps_eval = num_steps_per_epoch
                        num_steps_checkpoint = num_steps_per_epoch
                    else:
                        raise ValueError(
                            "At least `num_steps_eval` or `num_steps_checkpoint` need to be "
                            "provided because `checkpoint_criterion` is 'val_loss' and the dataset "
                            "has no length.")
                # num_steps_report_training_loss left as-is
                assert num_steps_checkpoint >= num_steps_eval, \
                    "num_steps_checkpoint must be at least num_steps_eval when using val_loss as checkpoint criterion."
                if num_steps_checkpoint % num_steps_eval != 0:
                    gsim_logger.warning(
                        "It is recommended that num_steps_checkpoint be a multiple of "
                        "num_steps_eval when using 'val_loss' as checkpoint criterion. "
                        "Otherwise, the reference validation loss may be stale."
                    )

            elif checkpoint_criterion in ("never", "always"):
                if num_steps_eval is None and val:
                    if num_steps_per_epoch is not None:
                        num_steps_eval = num_steps_per_epoch
                    else:
                        gsim_logger.warning(
                            "Validation data was provided but `num_steps_eval` was not set, "
                            "so the validation loss will not be computed.")
                if checkpoint_criterion == "always":
                    if num_steps_report_training_loss is not None and num_steps_report_training_loss != 1:
                        gsim_logger.warning(
                            "When checkpoint_criterion == 'always', only num_steps_report_training_loss=1 "
                            "is allowed.")
                    num_steps_report_training_loss = 1
                    if num_steps_checkpoint is not None and num_steps_checkpoint != 1:
                        gsim_logger.warning(
                            "When checkpoint_criterion == 'always', only num_steps_checkpoint=1 "
                            "is allowed.")
                    num_steps_checkpoint = 1
                else:  # "never"
                    if num_steps_checkpoint is not None:
                        gsim_logger.warning(
                            "When checkpoint_criterion == 'never', only num_steps_checkpoint=None "
                            "is allowed.")
                    num_steps_checkpoint = None

            else:
                raise ValueError(
                    f"Invalid checkpoint_criterion: {checkpoint_criterion}")

            return (checkpoint_criterion, num_steps_checkpoint, num_steps_eval,
                    num_steps_report_training_loss)

        def fit_normalizer_if_needed():
            assert self.normalizer is not None
            if not self.normalizer.are_parameters_set:
                gsim_logger.info("Fitting the normalizer...")
                fit_dataset = dataset_train
                if self.data_adapter is not None:
                    fit_dataset = self.wrap_in_adapter(dataset_train,
                                                       no_targets=no_targets)
                self.normalizer.fit(fit_dataset)
                self.normalizer.save()
            else:
                gsim_logger.info(
                    "The normalizer will not be fitted since its parameters have been loaded. "
                    "If you want to fit it again, delete/rename normalizer.pk."
                )

        def get_log_loss_str(l_loss,
                             hci=None,
                             min_step_for_declaring_best=None):
            l_vals = [t[1] for t in l_loss]
            str_val = f"{l_vals[-1]:.{num_significant_figures}g}"
            if hci is not None:
                str_val += f" ± {hci:.{num_significant_figures}g}"
            if min_step_for_declaring_best is not None:
                if l_loss[-1][0] < min_step_for_declaring_best:
                    return str_val
                l_vals = [
                    v for (s, v) in l_loss if s >= min_step_for_declaring_best
                ]
            if l_vals[-1] == min(l_vals):
                return f"{str_val} (best ⭐)"
            else:
                return f"{str_val} (best {min(l_vals):.{num_significant_figures}g})"

        def get_log_step_str(ind_step):
            if num_steps_per_epoch is None:
                return f"Step {ind_step}"
            if ind_step % num_steps_per_epoch == 0:
                str_epoch = f"{ind_step // num_steps_per_epoch}"
            else:
                str_epoch = f"{ind_step / num_steps_per_epoch:.2f}"
            return f"Step {ind_step} (epoch {str_epoch})"

        def eval_loss_vals_per_second(hist: TrainingHistory):
            step_ind_start = hist.ind_first_step_current_session
            loss_vals_this_session = sum(
                hist.l_num_loss_vals_per_step[step_ind_start:])
            return loss_vals_this_session / total_time_training

        def report_training_loss_me(ind_step, hist: TrainingHistory):
            hist.l_reported_train_loss_me_steps.append(ind_step)
            ema = hist.compute_train_loss_me(training_loss_forgetting_factor)
            l_reported = [(s, ema[s])
                          for s in hist.l_reported_train_loss_me_steps]
            gsim_logger.info(
                f"{get_log_step_str(ind_step)}: "
                f"training loss me = {get_log_loss_str(l_reported, min_step_for_declaring_best=min_num_steps_reliable_train_loss_me)}, "
                f"lr = {hist.l_lr[-1]:.2g}, "
                f"{int(eval_loss_vals_per_second(hist))} loss vals/s")

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
                m, hci = self._eval_static_metric(
                    dataloader_train_eval,
                    f_loss,
                    max_hci=static_max_hci,
                    max_num_loss_vals=static_max_num_loss_vals)
                hist.l_train_loss += [(ind_step, m)]
                l_str_log.append("train loss = " +
                                 get_log_loss_str(hist.l_train_loss, hci))
            if dataloader_val is not None:
                gsim_logger.info("│ Computing the validation loss...")
                m, hci = self._eval_static_metric(
                    dataloader_val,
                    f_loss,
                    max_hci=static_max_hci,
                    max_num_loss_vals=static_max_num_loss_vals)
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
                    max_hci=unnormalized_max_hci,
                    max_num_loss_vals=static_max_num_loss_vals)
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
                    max_hci=unnormalized_max_hci,
                    max_num_loss_vals=static_max_num_loss_vals)
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
                    "Validation loss has not been evaluated yet. This should not happen, as num_steps_checkpoint >= num_steps_eval."
                is_value_fresh = hist.l_val_loss[-1][0] == ind_step
                if not is_value_fresh:
                    gsim_logger.warning(
                        "The checkpoint criterion is `val_loss`, but the validation loss has not been evaluated at this step. Using the last available value. To avoid this issue, set num_steps_checkpoint to be a multiple of num_steps_eval."
                    )
                if has_metric_improved_since_prev_checkpoint(hist.l_val_loss):
                    gsim_logger.info(
                        f"Step {ind_step}: val_loss improved, saving checkpoint."
                    )
                    save_checkpoint()
            elif checkpoint_criterion == "train_loss_me":
                if (min_num_steps_reliable_train_loss_me is not None
                        and ind_step < min_num_steps_reliable_train_loss_me):
                    # The moving estimate of the training loss is too noisy to
                    # be used as a checkpoint criterion, so we skip
                    # checkpointing until we have enough data points.
                    return
                # The following list should have been populated
                assert len(hist.l_train_loss_per_step) > 0
                l_ema = hist.compute_train_loss_me(
                    training_loss_forgetting_factor)
                current_value = l_ema[ind_step]
                if len(hist.l_step_inds_checkpoints) == 0:
                    prev_value = float('inf')
                else:
                    prev_value = l_ema[hist.l_step_inds_checkpoints[-1]]
                if current_value < prev_value:
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
            """
                Returns `True` iff the optimizer state was successfully loaded. 
            """
            try:
                checkpoint = torch.load(path,
                                        weights_only=True,
                                        map_location=self.device_type)
                optimizer.load_state_dict(checkpoint["state"])
                return True
            except Exception as e:
                gsim_logger.warning(
                    f"No optimizer state file found at {path}. Using default initialization."
                )
                return False

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
                if obtain_static_training_loss:
                    l_ref_tvals = hist.l_train_loss
                else:
                    # The moving estimate of the training loss is not stored as
                    # an attribute; it is recomputed on demand and reported at
                    # `l_reported_train_loss_me_steps` (see
                    # `report_training_loss_me`).
                    ema = hist.compute_train_loss_me(
                        training_loss_forgetting_factor)
                    l_ref_tvals = [(s, ema[s])
                                   for s in hist.l_reported_train_loss_me_steps]
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

        # Input checks for no-length datasets
        dataset_has_length = isinstance(dataset, Sized)
        if not dataset_has_length:
            if val_split is not None:
                raise ValueError(
                    "val_split cannot be used with a dataset that has no length; "
                    "use dataset_val instead.")

        # Validation data
        dataset_train, dataset_val = make_validation_data(
            dataset, dataset_val, val_split)
        val = dataset_val is not None and (not isinstance(dataset_val, Sized)
                                           or len(dataset_val) > 0)

        # Input processing
        batch_size_eval = batch_size_eval if batch_size_eval else batch_size
        num_steps_per_epoch = int(np.ceil(len(dataset_train) /
                                          batch_size)) if isinstance(
                                              dataset_train, Sized) else None

        assert (num_epochs is None) ^ (num_steps is None), \
            "Exactly one of num_epochs and num_steps must be provided."
        if num_steps is None:
            assert num_epochs is not None
            assert num_steps_per_epoch is not None, \
                "num_epochs cannot be used with a dataset that has no length; use num_steps instead."
            num_steps = num_epochs * num_steps_per_epoch
        if restore_best_checkpoint is None:
            restore_best_checkpoint = (self.nn_folder is not None)
        if training_loss_forgetting_factor is None:
            training_loss_forgetting_factor = _get_me_coefficient_from_optimizer(
                optimizer)
        else:
            assert 0 <= training_loss_forgetting_factor < 1, "training_loss_forgetting_factor must be in [0, 1)."

        (checkpoint_criterion, num_steps_checkpoint, num_steps_eval,
         num_steps_report_training_loss) = resolve_fit_schedule(
             checkpoint_criterion, val, num_steps_per_epoch,
             num_steps_checkpoint, num_steps_eval,
             num_steps_report_training_loss)

        # Fit the normalizer
        if self.normalizer is not None:
            fit_normalizer_if_needed()

        # Instantiate the data loaders
        dataloader_train = self.make_data_loader(dataset_train,
                                                 batch_size,
                                                 shuffle,
                                                 no_targets=no_targets)
        dataloader_train_eval = self.make_data_loader(dataset_train,
                                                      batch_size_eval,
                                                      shuffle,
                                                      no_targets=no_targets)
        if val:
            assert dataset_val is not None
            dataloader_val = self.make_data_loader(dataset_val,
                                                   batch_size,
                                                   shuffle,
                                                   no_targets=no_targets)
        else:
            dataloader_val = None

        # History initialization
        hist = self.load_hist()
        ind_step = len(hist.l_train_loss_per_step)
        hist.drop_reported_train_loss_me_steps_since_last_restored_checkpoint()
        hist.l_step_inds_started_training += [ind_step]
        hist.last_used_training_loss_forgetting_factor = training_loss_forgetting_factor
        total_time_training = 0.0

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

        # Batch provider for the training loop. A single call returns the next
        # batch of `dataloader_train`, transparently starting a new epoch (which
        # possibly reshuffles the data) when the current one is exhausted. This
        # lets a training step draw one or several batches (see gradient
        # accumulation in `_run_training_step`) without complicating the loop
        # below.
        data_iter = iter(dataloader_train)

        def get_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader_train)
                return next(data_iter)

        # Training loop
        done = False
        while not done:
            while True:

                if ind_step >= hist.ind_first_step_current_session + num_steps:
                    done = True
                    break

                # Training step
                self.train()
                time_start_step = time.perf_counter()
                loss_train_this_step, num_loss_vals_this_step = self._run_training_step(
                    get_batch, f_loss, optimizer, lr_scheduler, max_grad_norm,
                    min_num_loss_vals_accumulate_grad)
                total_time_training += time.perf_counter() - time_start_step
                hist.l_train_loss_per_step += [loss_train_this_step]
                hist.l_num_loss_vals_per_step += [num_loss_vals_this_step]
                hist.l_lr.append(optimizer.param_groups[0]["lr"])

                b_save_checkpoint = (num_steps_checkpoint is not None
                                     and ind_step > 0
                                     and ind_step % num_steps_checkpoint == 0)

                # Moving-metric reporting
                if (num_steps_report_training_loss is not None and ind_step
                        and ind_step % num_steps_report_training_loss
                        == 0) or b_save_checkpoint:
                    # Not reported when ind_step == 0 because that potentially
                    # results in a very noisy value which may ruin reporting the
                    # best value so far.
                    report_training_loss_me(ind_step, hist)
                    self.save_hist(hist)

                # Static-metric evaluation
                if (num_steps_eval is not None
                        and ind_step % num_steps_eval == 0):
                    eval_static_metrics(ind_step, hist, dataloader_train_eval,
                                        dataloader_val)
                    self.save_hist(hist)

                # Checkpointing
                if b_save_checkpoint:
                    save_checkpoint_if_needed(ind_step, hist)

                # Patience
                if is_patience_exhausted(hist):
                    gsim_logger.info("Patience exhausted. Stopping training.")
                    done = True
                    break

                ind_step += 1

        if restore_best_checkpoint and hist.l_step_inds_checkpoints:
            gsim_logger.info(
                "Restoring the best checkpoint at the end of training.")
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
        non_blocking = self.device_type != "mps"
        if isinstance(obj, torch.Tensor):
            return obj.float().to(
                self.device_type, non_blocking=non_blocking
            )  # bug https://github.com/pytorch/pytorch/issues/139550
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(item) for item in obj)
        elif hasattr(obj, "to_device"):
            # For custom objects, implement a to_device method that moves all
            # tensors inside the object to the device.
            return obj.to_device(self.device_type, non_blocking=non_blocking)
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
        elif hasattr(obj, "to_device"):
            # For custom objects, implement a to_device method that moves all
            # tensors inside the object to the device.
            return obj.to_device("cpu", non_blocking=False)
        else:
            raise TypeError("Unsupported type.")

    @staticmethod
    def _select_plot_steps(num_steps: int, max_points: int,
                           logx: bool) -> list[int]:
        """
        Returns up to `max_points` integer steps in [0, num_steps - 1],
        spaced uniformly (or log-uniformly if `logx`).
        """
        if num_steps <= 0 or max_points <= 0:
            return []
        if num_steps <= max_points:
            return list(range(num_steps))
        if logx:
            hi = num_steps - 1
            if hi < 1:
                return list(range(num_steps))
            pts = np.logspace(0.0, np.log10(hi), max_points)
        else:
            pts = np.linspace(0, num_steps - 1, max_points)
        return sorted(set(int(round(p)) for p in pts))

    @staticmethod
    def get_session_history_steps(
            hist: TrainingHistory,
            include_current_session=False) -> List[Tuple[int, int]]:
        """
        Returns a list of (start_step, end_step) tuples that define the the
        intervals of steps that belong to the history of the current training
        session (see the docstring of `fit` for more information).

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
    def plot_training_history(hist: TrainingHistory,
                              first_step_to_plot=None,
                              logx=False,
                              logy=False,
                              max_train_loss_me_points: int = 500,
                              plot_num_loss_vals_per_step: bool = False):
        """
        Plots the training history of a neural network.

        Generates one or more GFigure objects visualizing the training history,
        including loss curves and learning rate evolution. Distinguishes between
        steps that are part of the training history (solid lines) and steps
        outside the history due to checkpoint restoration (dotted lines).

        Args:
            hist (TrainingHistory): The training history object containing loss
                values, learning rates, and checkpoint information.
            first_step_to_plot (int, optional): The first step index to include
                in the plot. Steps before this index are excluded from the view.
                If None, then the first step in the history is used.
            logx (bool, optional): If True, the x-axis uses a logarithmic scale.
                Defaults to False.
            logy (bool, optional): If True, the y-axis uses a logarithmic scale.
                Defaults to False.
            plot_num_loss_vals_per_step (bool, optional): If True, an extra
                subplot showing the number of loss values (cf. usage.md) used at
                each training step (`hist.l_num_loss_vals_per_step`) is added to
                the first figure. This is useful e.g. to inspect gradient
                accumulation, where the number of loss values per step can
                exceed the batch size. Defaults to False.

        Returns:
            list[GFigure]: A list of GFigure objects. The first figure shows
            loss
                curves (train_loss_me, train_loss, val_loss) and learning rate
                evolution. Additional figures show unnormalized losses if
                available.
        """

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

        def plot_keys(l_label_data, margin_coef=0.1):
            """`l_label_data`: list of (label, lt_step_values) pairs."""

            def get_first_step_to_plot():
                if first_step_to_plot is not None:
                    return first_step_to_plot
                min_step = np.inf
                for _, lt_step_values in l_label_data:
                    if len(lt_step_values) == 0:
                        continue
                    step_values = [t[0] for t in lt_step_values]
                    min_step = min(min_step, min(step_values))
                return int(min_step) if min_step != np.inf else 0

            def get_axis_limits():
                max_y_value = -np.inf
                min_y_value = np.inf
                max_x_value = -np.inf
                min_x_value = get_first_step_to_plot()
                for _, lt_step_values in l_label_data:
                    # Keep those values within x range
                    lt_step_values = [(t[0], t[1]) for t in lt_step_values
                                      if t[0] >= min_x_value]
                    if len(lt_step_values) == 0:
                        continue

                    l_steps = [t[0] for t in lt_step_values]
                    l_vals = [t[1] for t in lt_step_values]
                    max_y_value = max(max_y_value, np.nanmax(l_vals))
                    min_y_value = min(min_y_value, np.nanmin(l_vals))
                    max_x_value = max(max_x_value, l_steps[-1])
                return min_x_value, max_x_value, min_y_value, max_y_value

            s1 = Subplot(xlabel="Step", ylabel="Loss", logx=logx, logy=logy)

            l_session_steps = NeuralNet.get_session_history_steps(hist)
            current_session_start = hist.l_step_inds_started_training[
                -1] if hist.l_step_inds_started_training else 0

            for ind_key, (label, lt_step_values) in enumerate(l_label_data):
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
                                 legend=label,
                                 styles=f"-#{ind_key}")

            # Set axis limits
            min_x_value, max_x_value, min_y_value, max_y_value = get_axis_limits(
            )
            if max_y_value != -np.inf and min_y_value != np.inf:
                margin = margin_coef * (max_y_value - min_y_value)
                if not logy:
                    min_y_value = min_y_value - margin
                else:
                    min_y_value = min_y_value - margin if (
                        min_y_value - margin > 0) else min_y_value
                s1.ylim = (min_y_value, max_y_value + margin)
            if max_x_value != -np.inf:
                if logx:
                    min_x_value = max(
                        min_x_value,
                        1)  # log scale cannot include non-positive values
                s1.xlim = (min_x_value, max_x_value)
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
                    style="r--",
                    legend_str="Session starts")

            def make_train_loss_me_points():
                """
                Returns a list of (step, train_loss_me) points to plot, where
                train_loss_me is the moving-metric estimate of the training loss
                at that step. The points are selected to be at most
                `max_train_loss_me_points` and spaced uniformly or log-uniformly
                according to `logx`.
                """
                l_query_steps = NeuralNet._select_plot_steps(
                    num_steps=len(hist.l_train_loss_per_step),
                    max_points=max_train_loss_me_points,
                    logx=logx,
                )
                l_ema = hist.compute_train_loss_me()
                return [(s, l_ema[s]) for s in l_query_steps]

            l_train_loss_me = make_train_loss_me_points()
            s1 = plot_keys([("l_train_loss_me", l_train_loss_me),
                            ("l_train_loss", hist.l_train_loss),
                            ("l_val_loss", hist.l_val_loss)])
            plot_restored_checkpoints_and_session_starts(s1, hist)
            s2 = Subplot(xlabel="Step",
                         ylabel="Learning rate",
                         sharex=True,
                         logx=logx)
            s2.xlim = s1.xlim
            s2.add_curve(yaxis=hist.l_lr if len(hist.l_lr) > 0 else [np.nan])
            G = GFigure()
            G.l_subplots = [s1, s2]
            return G

        def plot_unnormalized_loss():
            l_keys = ["l_unnormalized_train_loss", "l_unnormalized_val_loss"]
            l_label_data = []
            for key in l_keys:
                lt_step_vals = getattr(hist, key)
                if len(lt_step_vals):
                    l_label_data.append((key, lt_step_vals))
            if not len(l_label_data):
                return None
            G = GFigure()
            G.l_subplots = [plot_keys(l_label_data)]
            return G

        def plot_num_loss_vals_per_step_subplot(G: GFigure):
            """Appends a subplot showing the number of loss values used at each
            training step to the figure `G`."""
            s = Subplot(xlabel="Step",
                        ylabel="Loss vals per step",
                        sharex=True,
                        logx=logx)
            s.xlim = G.l_subplots[0].xlim
            l_num_loss_vals = hist.l_num_loss_vals_per_step
            s.add_curve(xaxis=list(range(len(l_num_loss_vals))),
                        yaxis=l_num_loss_vals
                        if len(l_num_loss_vals) > 0 else [np.nan])
            G.l_subplots.append(s)

        l_G = []

        G1 = plot_loss_and_learning_rate()
        if plot_num_loss_vals_per_step:
            plot_num_loss_vals_per_step_subplot(G1)
        l_G.append(G1)

        G2 = plot_unnormalized_loss()
        if G2 is not None:
            l_G.append(G2)

        return l_G

    def print_num_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Total number of parameters: {total_params}')
