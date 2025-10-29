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


class LossLandscapeConfig():

    def __init__(self,
                 epoch_inds=[],
                 neg_gradient_step_scales=[],
                 max_num_directions=None):
        """
        Args:

            `epoch_inds`: for each epoch index in this list, a figure with the
            loss landscape is produced. 

            `neg_gradient_step_scales`: for each item i in this iterable, the
            loss function is plotted at w - i*\nabla, where w is the vector of
            weights and \nabla the gradient estimate obtained from one of the
            batches. 

            `max_num_directions`: if not None, then the loss landscape is
            plotted for the first `max_num_directions` directions, which correspond
            to the first `max_num_directions` batches.
          
        """
        self.epoch_inds = epoch_inds
        self.neg_gradient_step_scales = neg_gradient_step_scales
        self.max_num_directions = max_num_directions


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
            gsim_logger.warning(
                "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"
            )
            gsim_logger.warning(
                "*   WARNING: The weights of the network are not being saved.")
            gsim_logger.warning(
                "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"
            )
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

    def initialize(self):
        self._initialized = True

        if self.nn_folder is not None:
            # Create the folder if it does not exist
            os.makedirs(self.nn_folder, exist_ok=True)

            if os.path.exists(self.weight_file_path):
                self.load_weights_from_path(self.weight_file_path)
                gsim_logger.info(
                    f"Weights loaded from {self.weight_file_path}")
                if self.normalizer is not None:
                    normalizer = self.normalizer
                    normalizer.load()
                    gsim_logger.info(
                        f"Normalizer loaded from {normalizer.params_file}")
            else:
                gsim_logger.warning(
                    f"Warning: {os.path.abspath(self.weight_file_path)} does not exist. The network will be initialized."
                )

        self.to(device=self.device_type)

    @abstractmethod
    def forward(self, x: InputType) -> OutputType:
        # This method must be overridden by subclasses
        raise NotImplementedError

    def _assert_initialized(self):
        assert self._initialized, "The network has not been initialized. A subclass of NeuralNet must call self.initialize() at the end of its constructor."

    @staticmethod
    def collate_fn(*args, **kwargs):
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

        l_batch = self.collate_fn(l_batch)

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

    def _run_epoch(self, dataloader, f_loss: LossFunType, optimizer=None):
        """
        Args:

            `optimizer`: if None, the weights are not updated. This is useful to
            evaluate the loss. 

            `f_loss`: LossFunType
        
        """

        l_loss_this_epoch = []
        iterator = tqdm(dataloader) if optimizer else dataloader
        for data in iterator:

            if optimizer:
                loss = self._get_loss(data,
                                      f_loss)  # vector of length batch_size
                torch.mean(loss).backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    loss = self._get_loss(
                        data, f_loss)  # vector of length batch_size

            l_loss_this_epoch.append(loss.detach())

        return float(torch.cat(l_loss_this_epoch).mean().cpu().numpy()) if len(
            l_loss_this_epoch) else np.nan

    def evaluate(self,
                 dataset,
                 batch_size,
                 f_loss: LossFunType,
                 unnormalized=True):
        """
        Args:

            `unnormalized`: If True, the unnormalized loss is returned. If no
            Normalizer is set, then the loss is already unnormalized.

        Returns a dict with key-values:

        "loss": the result of averaging `f_loss` across `dataset`.
        """
        self._assert_initialized()

        if not unnormalized and self.normalizer is None:
            raise ValueError(
                "Cannot return normalized loss if a normalizer is not set.")

        if unnormalized and self.normalizer is not None:
            f_loss = self.make_unnormalized_loss(f_loss)

        dataloader = self.make_data_loader(dataset, batch_size)
        self.eval()
        loss = self._run_epoch(dataloader, f_loss=f_loss)
        return {"loss": loss}

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

    @property
    def hist_path(self):
        assert self.nn_folder is not None
        return os.path.join(self.nn_folder, "hist.pk")

    @staticmethod
    def get_weight_file_path(folder):
        return os.path.join(folder, "weights.pth")

    @staticmethod
    def get_optimizer_state_file_path(folder):
        return os.path.join(folder, "optimizer.pth")

    def load_weights_from_path(self, path):
        checkpoint = torch.load(path,
                                weights_only=True,
                                map_location=self.device_type)
        self.load_state_dict(checkpoint["weights"])
        self.to(device=self.device_type)
        #load_optimizer_state(initial_optimizer_state_file)

    def save_weights_to_path(self, path):
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
                          collate_fn=functools.partial(
                              self.collate_and_normalize,
                              no_targets=no_targets))

    def fit(self,
            dataset: Dataset,
            optimizer,
            num_epochs,
            f_loss: Callable,
            dataset_val=None,
            batch_size=32,
            batch_size_eval=None,
            shuffle=True,
            val_split=0.0,
            best_weights=True,
            patience=None,
            lr_patience=None,
            lr_decay=.8,
            restart_optimizer_when_reducing_lr=False,
            eval_unnormalized_loss=False,
            llc=LossLandscapeConfig()):
        """ 
        Starts a training session.

        If 
            - self.nn_folder exists

            - self.nn_folder/optimizer.pth exists,
        
        this function will attempt to load this state into the optimizer. To
        reset the optimizer state, just erase this file before invoking fit. 

        Args:

         `f_loss`: f_loss(output_batch,target_batch) 

          `batch_size_eval` is the batch size used to evaluate the loss. If
          None, `batch_size` is used also for evaluation.

          `llc`: instance of LossLandscapeConfig.
        
          At most one of `val_split` and `dataset_val` can be provided. If one
          is provided, we say that `val` is True. 

          `patience`: if provided and the validation loss does not improve its
          minimum in this session for `patience` epochs, training will be
          stopped.

          `restart_optimizer_when_reducing_lr`: if True, the state of the
          optimizer is reset to its state at the beginning of the session (the
          one in self.nn_folder/optimizer.pth if this file exists and can be
          loaded, or the state of a new optimizer otherwise) every time the
          learning rate is reduced. This may help escape local minima.

        Returns a dict with keys and values given by:
         
          'train_loss_me': list of length num_epochs with the values of the
          moving estimate of the training loss at each epoch. The moving
          estimate is obtained by averaging the training loss after each batch
          update. Thus, it is an average of loss values obtained for different
          network weights.

          'train_loss': list of length num_epochs with the values of the
          training loss computed at the end of each epoch.

          'val_loss': same as before but for the validation loss. Only if `val`
          is true. 

          'lr': list of length num_epochs with the learning rate at each epoch.

          'l_loss_landscapes': list of figures with loss landscapes.
                      
           'unnormalized_train_loss': list of length num_epochs with the values
           of the training loss computed at the end of each epoch after
           unnormalizing the targets and the outputs. These values are only
           computed if `eval_unnormalized_loss` is True; else they are np.nan.

           'unnormalized_val_loss': list of length num_epochs with the values of
           the validation loss computed at the end of each epoch after
           unnormalizing the targets and the outputs. Only if `val` is True.
           These values are only computed if `eval_unnormalized_loss` is True;
           else they are np.nan.

         These losses are just informative, the network is trained on the
         normalized loss if a normalizer that normalizes the targets is
         specified in the constructor. 

        If `best_weights` is False, then the weights of the network at the end
        of the execution of this function equal the weights at the last epoch.
        Otherwise: 
         
          - if `val` is True, then the weights of the epoch with the
        best validation loss are returned; 
        
          - if `val` is False, then the weights of the epoch with the
        best training loss are returned; 

        """

        def get_landscape_plot(dataloader_train, dataloader_train_eval):
            """
            Returns:

                GFigure with a figure in which the loss is plotted vs. the
                distance traveled along negative gradient estimates. There is
                one curve for each batch, since each one provides a gradient. 
            
            """

            self.save_weights_to_path(llp_weight_file)
            self.train()
            ll_loss = [
            ]  # One list per considered batch. Each inner list contains the loss for each distance.
            for input_batch, targets_batch in dataloader_train:

                # 1. Compute the gradient
                input_batch = input_batch.float().to(self.device_type)
                targets_batch = targets_batch.float().to(self.device_type)

                output_batch = self(input_batch.float())
                loss = f_loss(output_batch.float(), targets_batch.float())
                torch.mean(loss).backward()

                # 2. Compute the loss for gradient displacement
                l_loss = []
                for scale in llc.neg_gradient_step_scales:
                    for param in self.parameters():
                        param.data -= scale * param.grad

                    with torch.no_grad():  # Disable gradient computation
                        output_batch = self(input_batch.float())[:, 0]
                        self.train()
                        loss = self._run_epoch(dataloader_train_eval, f_loss)
                    l_loss.append(loss)

                    # Restore the weights (can alt. be combined with next iteration)
                    for param in self.parameters():
                        param.data += scale * param.grad

                self.zero_grad()

                ll_loss.append(l_loss)
                if llc.max_num_directions is not None and len(
                        ll_loss) >= llc.max_num_directions:
                    break

            self.load_weights_from_path(llp_weight_file)
            return GFigure(xaxis=llc.neg_gradient_step_scales,
                           yaxis=np.array(ll_loss),
                           xlabel="Step size along the negative gradient",
                           ylabel="Loss",
                           title=f"Loss landscape for epoch {ind_epoch}")

        def get_temp_file_path():
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
            return temp_file_path

        def get_temp_folder_path():
            """Returns a temporary folder path."""
            return tempfile.mkdtemp()

        def get_train_val_state_folder_paths():
            """
            Returns the folders where the weights and optimizer state need to be
            stored at the epochs with the best training loss and best validation
            loss.

                best_train_loss_folder, best_val_loss_folder
            
            """

            if self.nn_folder is None:
                best_train_loss_folder = get_temp_folder_path()
                best_val_loss_folder = get_temp_folder_path()
            else:
                os.makedirs(self.nn_folder, exist_ok=True)
                if val:
                    best_train_loss_folder = get_temp_folder_path()
                    best_val_loss_folder = self.nn_folder
                else:
                    best_train_loss_folder = self.nn_folder
                    best_val_loss_folder = get_temp_folder_path()
            return best_train_loss_folder, best_val_loss_folder

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
                    f"Optimizer state was not loaded from {path}. Using default initialization."
                )

        def decrease_lr(optimizer, lr_decay):
            """Resets the optimizer state and decreases the learning rate by a factor of `lr_decay`."""

            # Store the lr values
            l_lr = [
                optimizer.param_groups[ind_group]["lr"]
                for ind_group in range(len(optimizer.param_groups))
            ]
            if restart_optimizer_when_reducing_lr:
                load_optimizer_state(initial_optimizer_state_file)
            for ind_group in range(len(optimizer.param_groups)):
                optimizer.param_groups[ind_group][
                    "lr"] = l_lr[ind_group] * lr_decay

        def save_hist(d_hist):
            if self.nn_folder is not None:
                os.makedirs(self.nn_folder, exist_ok=True)
                pickle.dump(d_hist, open(self.hist_path, "wb"))

        def load_hist():
            if self.nn_folder is not None and os.path.exists(self.hist_path):
                d_hist = pickle.load(open(self.hist_path, "rb"))

                # Backwards compatibility: populate the unnormalized loss if needed
                if "unnormalized_train_loss" not in d_hist:
                    d_hist["unnormalized_train_loss"] = [np.nan] * len(
                        d_hist["train_loss"])
                if "unnormalized_val_loss" not in d_hist:
                    d_hist["unnormalized_val_loss"] = [np.nan] * len(
                        d_hist["val_loss"])

            else:
                d_hist = {
                    'train_loss_me': [],
                    'train_loss': [],
                    'val_loss': [],
                    "lr": [],
                    "l_loss_landscapes": [],
                    "ind_epoch": 0,
                    "unnormalized_train_loss": [],
                    "unnormalized_val_loss": [],
                }
            return d_hist

        def compute_unnormalized_losses(dataloader_train_eval, dataloader_val,
                                        f_loss):

            if self.normalizer is None or not eval_unnormalized_loss:
                unnormalized_loss_train_this_epoch = np.nan
                unnormalized_loss_val_this_epoch = np.nan
            else:
                self.train()
                unnormalized_loss_train_this_epoch = self._run_epoch(
                    dataloader_train_eval, self.make_unnormalized_loss(f_loss))

                self.eval()
                unnormalized_loss_val_this_epoch = self._run_epoch(
                    dataloader_val, self.make_unnormalized_loss(
                        f_loss)) if dataloader_val else np.nan

            l_unnormalized_loss_train.append(
                unnormalized_loss_train_this_epoch)
            l_unnormalized_loss_val.append(unnormalized_loss_val_this_epoch)

        self._assert_initialized()
        torch.cuda.empty_cache()

        batch_size_eval = batch_size_eval if batch_size_eval else batch_size

        assert val_split == 0.0 or dataset_val is None
        if dataset_val is None:
            # The data is deterministically split into training and validation
            # sets so that we can resume training.
            assert isinstance(dataset, Sized)
            num_examples_val = int(val_split * len(dataset))
            dataset_train = Subset(dataset,
                                   range(len(dataset) - num_examples_val))
            dataset_val = Subset(
                dataset, range(len(dataset) - num_examples_val, len(dataset)))
        else:
            dataset_train = dataset
            num_examples_val = len(dataset_val)
        val = num_examples_val > 0
        if patience is not None and val is False:
            gsim_logger.warning(
                "patience is set but no validation data is provided. Ignoring patience."
            )
            patience = None

        # Fit the normalizer
        if self.normalizer is not None:
            gsim_logger.info("Fitting the normalizer...")
            self.normalizer.fit(dataset_train)
            self.normalizer.save()

        # Instantiate the data loaders
        dataloader_train = self.make_data_loader(dataset_train, batch_size,
                                                 shuffle)
        dataloader_train_eval = self.make_data_loader(dataset_train,
                                                      batch_size_eval, shuffle)
        dataloader_val = self.make_data_loader(dataset_val, batch_size,
                                               shuffle) if val else None

        d_hist = load_hist()
        l_loss_train_me = d_hist['train_loss_me']
        l_loss_train = d_hist['train_loss']
        l_loss_val = d_hist['val_loss']
        l_lr = d_hist['lr']
        l_llplots = d_hist['l_loss_landscapes']  # loss landscape plots
        ind_epoch_start = d_hist['ind_epoch']
        l_unnormalized_loss_train = d_hist['unnormalized_train_loss']
        l_unnormalized_loss_val = d_hist['unnormalized_val_loss']

        llp_weight_file = get_temp_file_path(
        )  # file to restore the weights when the loss landscape needs to be plotted

        best_train_loss = torch.inf
        #num_epochs_left_to_expire_patience = patience
        num_epochs_left_to_expire_lr_patience = lr_patience if isinstance(
            lr_patience, int) else 0
        best_train_loss_state_folder, best_val_loss_state_folder = get_train_val_state_folder_paths(
        )

        # Try to load the optimizer state if available in self.nn_folder
        if self.nn_folder is not None:
            load_optimizer_state(
                self.get_optimizer_state_file_path(self.nn_folder))

        if restart_optimizer_when_reducing_lr:
            # Regardless of whether the optimizer state could not be loaded in
            # the previous step, we save the initial optimizer state so that we
            # can reset it later.
            initial_optimizer_state_file = get_temp_file_path()
            save_optimizer_state(initial_optimizer_state_file)

        for ind_epoch in range(ind_epoch_start, ind_epoch_start + num_epochs):
            self.train()
            loss_train_me_this_epoch = self._run_epoch(dataloader_train,
                                                       f_loss, optimizer)
            loss_train_this_epoch = self._run_epoch(dataloader_train_eval,
                                                    f_loss)
            self.eval()
            loss_val_this_epoch = self._run_epoch(
                dataloader_val, f_loss) if dataloader_val else np.nan

            compute_unnormalized_losses(dataloader_train_eval, dataloader_val,
                                        f_loss)

            gsim_logger.info(
                f"Epoch {ind_epoch-ind_epoch_start}/{num_epochs}: train loss me = {loss_train_me_this_epoch:.4f}, train loss = {loss_train_this_epoch:.4f}, val loss = {loss_val_this_epoch:.4f}, lr = {optimizer.param_groups[0]['lr']:.2e}"
            )

            l_loss_train_me.append(loss_train_me_this_epoch)
            l_loss_train.append(loss_train_this_epoch)
            l_loss_val.append(loss_val_this_epoch)
            l_lr.append(optimizer.param_groups[0]["lr"])

            ind_epoch_best_loss_val = np.argmin(
                [v if not np.isnan(v) else np.inf for v in l_loss_val])
            if ind_epoch_best_loss_val == ind_epoch:
                self.save_weights_to_path(
                    self.get_weight_file_path(best_val_loss_state_folder))
                save_optimizer_state(
                    self.get_optimizer_state_file_path(
                        best_val_loss_state_folder))

            if patience:
                ind_epoch_best_loss_val_this_session = np.argmin([
                    v if not np.isnan(v) else np.inf
                    for v in l_loss_val[ind_epoch_start:]
                ]) + ind_epoch_start

                if ind_epoch_best_loss_val_this_session + patience < ind_epoch:
                    gsim_logger.info("Patience expired.")
                    break

            if lr_patience or not val:
                # The weights should also be stored when val_split==0 since they
                # need to be returned at the end.
                if loss_train_this_epoch < best_train_loss:
                    best_train_loss = loss_train_this_epoch
                    self.save_weights_to_path(
                        self.get_weight_file_path(
                            best_train_loss_state_folder))
                    save_optimizer_state(
                        self.get_optimizer_state_file_path(
                            best_train_loss_state_folder))
                    num_epochs_left_to_expire_lr_patience = lr_patience if isinstance(
                        lr_patience, int) else 0
                else:
                    if lr_patience:
                        num_epochs_left_to_expire_lr_patience -= 1
                        if num_epochs_left_to_expire_lr_patience == 0:
                            self.load_weights_from_path(
                                self.get_weight_file_path(
                                    best_train_loss_state_folder))
                            load_optimizer_state(
                                self.get_optimizer_state_file_path(
                                    best_train_loss_state_folder))
                            decrease_lr(optimizer, lr_decay)
                            num_epochs_left_to_expire_lr_patience = lr_patience

            # Loss landscapes
            if ind_epoch - ind_epoch_start in llc.epoch_inds:
                l_llplots.append(
                    get_landscape_plot(dataloader_train,
                                       dataloader_train_eval))

            d_hist = {
                'train_loss_me': l_loss_train_me,
                'train_loss': l_loss_train,
                'val_loss': l_loss_val,
                "lr": l_lr,
                "l_loss_landscapes": l_llplots,
                "ind_epoch": ind_epoch,
                "unnormalized_train_loss": l_unnormalized_loss_train,
                "unnormalized_val_loss": l_unnormalized_loss_val,
            }
            save_hist(d_hist)

        if best_weights and num_epochs > 0:
            if val:
                best_val_loss_weight_file = self.get_weight_file_path(
                    best_val_loss_state_folder)
                if os.path.exists(best_val_loss_weight_file):
                    self.load_weights_from_path(best_val_loss_weight_file)
            else:
                best_train_loss_weight_file = self.get_weight_file_path(
                    best_train_loss_state_folder)
                if os.path.exists(best_train_loss_weight_file):
                    self.load_weights_from_path(best_train_loss_weight_file)

        return d_hist

    def _move_to_device(self, obj: Union[torch.Tensor, list, tuple]):
        if isinstance(obj, torch.Tensor):
            return obj.float().to(self.device_type)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(item) for item in obj)
        else:
            raise TypeError("Unsupported type.")

    @staticmethod
    def _move_to_cpu(obj: Union[torch.Tensor, list, tuple]):
        if isinstance(obj, torch.Tensor):
            return obj.detach().to("cpu")
        elif isinstance(obj, (list, tuple)):
            return type(obj)(NeuralNet._move_to_cpu(item) for item in obj)
        else:
            raise TypeError("Unsupported type.")

    @staticmethod
    def plot_training_history(d_hist, first_epoch_to_plot=0):

        def plot_keys(l_keys, margin_coef=0.1):
            max_y_value = -np.inf
            min_y_value = np.inf
            s1 = Subplot(xlabel="Epoch",
                         ylabel="Loss",
                         xlim=(first_epoch_to_plot, None))
            for key in l_keys:
                s1.add_curve(yaxis=d_hist[key], legend=key)
                if len(d_hist[key]) > first_epoch_to_plot:
                    l_vals = d_hist[key][first_epoch_to_plot:]
                    if not all(np.isnan(l_vals)):
                        max_y_value = max(max_y_value, np.nanmax(l_vals))
                        min_y_value = min(min_y_value, np.nanmin(l_vals))
            if max_y_value != -np.inf and min_y_value != np.inf:
                margin = margin_coef * (max_y_value - min_y_value)
                s1.ylim = (min_y_value - margin, max_y_value + margin)
            return s1

        def plot_loss_and_learning_rate():
            s1 = plot_keys(["train_loss_me", "train_loss", "val_loss"])
            s2 = Subplot(xlabel="Epoch", ylabel="Learning rate", sharex=True)
            s2.add_curve(yaxis=d_hist["lr"], legend="Learning rate")
            G = GFigure()
            G.l_subplots = [s1, s2]
            return G

        def plot_unnormalized_loss():
            l_keys = ["unnormalized_train_loss", "unnormalized_val_loss"]
            l_keys_to_plot = []
            for key in l_keys:
                l_vals = d_hist[key]
                if not all(np.isnan(l_vals)):
                    l_keys_to_plot.append(key)
            if not len(l_keys_to_plot):
                return None
            G = GFigure()
            G.l_subplots = [plot_keys(l_keys)]
            return G

        l_G = []

        G1 = plot_loss_and_learning_rate()
        l_G.append(G1)

        G2 = plot_unnormalized_loss()
        if G2 is not None:
            l_G.append(G2)

        l_G += d_hist["l_loss_landscapes"]

        return l_G

    def print_num_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Total number of parameters: {total_params}')
