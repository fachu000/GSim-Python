import pytest
import torch
import numpy as np
from torch.utils.data import Dataset

from gsim.include.neural_net import DefaultNormalizer


class SimpleDataset(Dataset):
    """Simple dataset for testing that returns pairs of (input, target)."""

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def test_default_normalizer_both_mode():
    """
    Test DefaultNormalizer with mode='both'.
    
    Fits the normalizer on random data, applies normalization, and verifies
    that the normalized data has zero mean and unit standard deviation.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate random data
    num_samples = 100
    input_shape = (3, 4)  # Example: 3x4 matrices
    target_shape = (2, )  # Example: 2-dimensional vectors

    inputs = torch.randn(num_samples, *input_shape)
    targets = torch.randn(num_samples, *target_shape)

    # Create dataset
    dataset = SimpleDataset(inputs, targets)

    # Create and fit normalizer
    normalizer = DefaultNormalizer(mode="both")
    normalizer.fit(dataset)

    # Normalize all data
    normalized_inputs = []
    normalized_targets = []
    for i in range(num_samples):
        inp, tgt = dataset[i]
        # Add batch dimension
        norm_inp = normalizer.normalize_input_batch(inp.unsqueeze(0))
        norm_tgt = normalizer.normalize_targets_batch(tgt.unsqueeze(0))
        # Remove batch dimension
        normalized_inputs.append(norm_inp.squeeze(0))
        normalized_targets.append(norm_tgt.squeeze(0))

    normalized_inputs = torch.stack(normalized_inputs)
    normalized_targets = torch.stack(normalized_targets)

    # Verify normalized inputs have zero mean and unit std
    # Note: Use unbiased=False because the normalizer uses population std (divides by N)
    input_mean = normalized_inputs.mean(dim=0)
    input_std = normalized_inputs.std(dim=0, unbiased=False)

    assert torch.allclose(input_mean, torch.zeros_like(input_mean), atol=1e-6), \
        f"Normalized input mean should be ~0, got {input_mean}"
    assert torch.allclose(input_std, torch.ones_like(input_std), atol=1e-6), \
        f"Normalized input std should be ~1, got {input_std}"

    # Verify normalized targets have zero mean and unit std
    target_mean = normalized_targets.mean(dim=0)
    target_std = normalized_targets.std(dim=0, unbiased=False)

    assert torch.allclose(target_mean, torch.zeros_like(target_mean), atol=1e-6), \
        f"Normalized target mean should be ~0, got {target_mean}"
    assert torch.allclose(target_std, torch.ones_like(target_std), atol=1e-6), \
        f"Normalized target std should be ~1, got {target_std}"
