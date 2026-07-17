from typing import Callable, Generic, NamedTuple, TypeVar, Union

import torch

# Type variables: InputType refers both to the input and input batch types.
# Likewise for OutputType and TargetType. We do not use separate type variables
# for batches and inputs/outputs/targets due to limitations in the Python typing
# system.

RawInputType = TypeVar("RawInputType", torch.Tensor, list[torch.Tensor],
                       tuple[torch.Tensor, ...])
InputType = TypeVar("InputType", torch.Tensor, list[torch.Tensor],
                    tuple[torch.Tensor, ...])
OutputType = TypeVar("OutputType", torch.Tensor, list[torch.Tensor],
                     tuple[torch.Tensor, ...])
TargetType = TypeVar("TargetType", torch.Tensor, list[torch.Tensor],
                     tuple[torch.Tensor, ...])


class WeightedLoss(NamedTuple):
    """
    Return type for loss functions that assign a weight to each loss value.

    Attributes:

        `values`: (num_loss_vals,) tensor with the loss values.

        `weights`: (num_loss_vals,) tensor with the weight of each loss value.
        Weights must be non-negative constants; they are detached before use,
        so no gradient flows through them.

    When a loss function returns a WeightedLoss, every loss aggregation in
    NeuralNet (training steps, including gradient accumulation, and static
    metric evaluations such as the validation loss) computes the weighted mean

        (sum_i weights[i] * values[i]) / (sum_i weights[i])

    instead of the plain mean. Returning a 1D tensor is equivalent to returning
    a WeightedLoss with unit weights. See `usage.md`, Sec. "Weighted loss
    values", and `experiment_1011` in `neuralnet_experiments.py`.
    """
    values: torch.Tensor
    weights: torch.Tensor

    @staticmethod
    def ensure(loss: Union[torch.Tensor, "WeightedLoss"]) -> "WeightedLoss":
        """
        Returns `loss` as a WeightedLoss. If `loss` is a plain tensor, unit
        weights are assigned. If it is already a WeightedLoss, it is validated
        (`values` and `weights` must be 1D tensors of the same length) and its
        weights detached.
        """
        if isinstance(loss, WeightedLoss):
            assert loss.values.ndim == 1, (
                f"WeightedLoss.values must be a 1D tensor, but its shape is "
                f"{tuple(loss.values.shape)}.")
            assert loss.values.shape == loss.weights.shape, (
                f"WeightedLoss.values (shape {tuple(loss.values.shape)}) and "
                f"WeightedLoss.weights (shape {tuple(loss.weights.shape)}) "
                "must have the same shape.")
            return WeightedLoss(values=loss.values,
                                weights=loss.weights.detach())
        return WeightedLoss(values=loss, weights=torch.ones_like(loss))


LossFunType = Callable[[OutputType, TargetType], Union[torch.Tensor,
                                                       WeightedLoss]]
