from typing import Callable, Generic, TypeVar, Union

import torch

# Type variables: InputType refers both to the input and input batch types.
# Likewise for OutputType and TargetType. We do not use separate type variables
# for batches and inputs/outputs/targets due to limitations in the Python typing
# system.

InputType = TypeVar("InputType", torch.Tensor, list[torch.Tensor],
                    tuple[torch.Tensor, ...])
OutputType = TypeVar("OutputType", torch.Tensor, list[torch.Tensor],
                     tuple[torch.Tensor, ...])
TargetType = TypeVar("TargetType", torch.Tensor, list[torch.Tensor],
                     tuple[torch.Tensor, ...])

LossFunType = Callable[[OutputType, TargetType], torch.Tensor]
