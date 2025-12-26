import pytest
import torch
import numpy as np
from torch.utils.data import Dataset

from gsim.include.neural_net import DefaultNormalizer
from gsim.include.neural_net.normalizers import (
    FeatNormalizer,
    IdentityFeatNormalizer,
    StdFeatNormalizer,
    IntervalFeatNormalizer,
    ScaleToUnitPowerFeatNormalizer,
    MultiFeatNormalizer,
)


class SimpleDataset(Dataset):
    """Simple dataset for testing that returns pairs of (input, target)."""

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class TestFeatNormalizers:
    """Test suite for FeatNormalizer classes."""

    def test_identity_normalizer(self):
        """Test that IdentityFeatNormalizer doesn't change values."""
        normalizer = IdentityFeatNormalizer()

        # Create test data
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        # Fit (should be no-op)
        normalizer.fit(values)

        # Normalize and unnormalize should return same values
        normalized = normalizer.normalize(values)
        unnormalized = normalizer.unnormalize(values)

        assert torch.allclose(normalized, values)
        assert torch.allclose(unnormalized, values)

    def test_std_normalizer_basic(self):
        """Test StdFeatNormalizer with basic data."""
        normalizer = StdFeatNormalizer()

        # Create test data with known mean and std
        values = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0])
        # mean = 4.0, std = sqrt(8) = 2.828...

        normalizer.fit(values)

        # Check mean and std
        assert normalizer.mean is not None
        assert normalizer.std is not None
        assert abs(normalizer.mean - 4.0) < 1e-5
        assert abs(normalizer.std - np.sqrt(8.0)) < 1e-5

        # Normalize
        normalized = normalizer.normalize(values)

        # Normalized data should have mean ~0 and std ~1
        # Note: Use unbiased=False to match the population std computed by the normalizer
        assert abs(normalized.mean().item()) < 1e-5
        assert abs(normalized.std(unbiased=False).item() - 1.0) < 1e-5

        # Unnormalize should return original values
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(unnormalized, values, rtol=1e-4, atol=1e-6)

    def test_std_normalizer_incremental_fit(self):
        """Test that StdFeatNormalizer correctly accumulates statistics."""
        normalizer = StdFeatNormalizer()

        # Fit in batches
        batch1 = torch.tensor([1.0, 2.0, 3.0])
        batch2 = torch.tensor([4.0, 5.0, 6.0])

        normalizer.fit(batch1)
        normalizer.fit(batch2)

        # Should have computed mean and std of all values [1,2,3,4,5,6]
        # mean = 3.5
        # variance = E[X²] - E[X]² = ((1² + 2² + 3² + 4² + 5² + 6²) / 6) - 3.5²
        #          = (91 / 6) - 12.25 = 15.1667 - 12.25 = 2.9167
        # std = sqrt(variance) = sqrt(2.9167) ≈ 1.708
        assert normalizer.mean is not None
        assert normalizer.std is not None
        assert abs(normalizer.mean - 3.5) < 1e-5
        expected_std = np.sqrt(np.var([1, 2, 3, 4, 5, 6]))
        assert abs(normalizer.std - expected_std) < 1e-5

    def test_std_normalizer_constant_values(self):
        """Test StdFeatNormalizer with constant values (zero std)."""
        normalizer = StdFeatNormalizer()

        # All same values
        values = torch.tensor([5.0, 5.0, 5.0, 5.0])
        normalizer.fit(values)

        # Should set std to 1.0 to avoid division by zero
        assert normalizer.std == 1.0
        assert normalizer.mean == 5.0

    def test_interval_normalizer_basic(self):
        """Test IntervalFeatNormalizer with basic data."""
        normalizer = IntervalFeatNormalizer(interval=(-1.0, 1.0))

        # Create test data
        values = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0])
        # min = 0, max = 8

        normalizer.fit(values)

        # Check min and max
        assert normalizer.min_val == 0.0
        assert normalizer.max_val == 8.0

        # Normalize
        normalized = normalizer.normalize(values)

        # Should be in range [-1, 1]
        assert normalized.min().item() >= -1.0 - 1e-5
        assert normalized.max().item() <= 1.0 + 1e-5
        assert abs(normalized.min().item() - (-1.0)) < 1e-5
        assert abs(normalized.max().item() - 1.0) < 1e-5

        # Unnormalize should return original values
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(unnormalized, values, rtol=1e-4)

    def test_interval_normalizer_incremental_fit(self):
        """Test that IntervalFeatNormalizer correctly tracks min/max."""
        normalizer = IntervalFeatNormalizer(interval=(0.0, 1.0))

        # Fit in batches
        batch1 = torch.tensor([3.0, 5.0, 7.0])
        batch2 = torch.tensor([1.0, 9.0, 4.0])

        normalizer.fit(batch1)
        normalizer.fit(batch2)

        # Should have min=1.0, max=9.0
        assert normalizer.min_val == 1.0
        assert normalizer.max_val == 9.0

    def test_interval_normalizer_constant_values(self):
        """Test IntervalFeatNormalizer with constant values."""
        normalizer = IntervalFeatNormalizer(interval=(-1.0, 1.0))

        # All same values
        values = torch.tensor([5.0, 5.0, 5.0, 5.0])
        normalizer.fit(values)

        # Should map to middle of interval
        normalized = normalizer.normalize(values)
        expected = torch.full_like(values, 0.0)  # middle of [-1, 1]
        assert torch.allclose(normalized, expected)

    def test_interval_normalizer_custom_interval(self):
        """Test IntervalFeatNormalizer with custom interval."""
        normalizer = IntervalFeatNormalizer(interval=(10.0, 20.0))

        values = torch.tensor([0.0, 5.0, 10.0])
        normalizer.fit(values)

        normalized = normalizer.normalize(values)

        # Check bounds
        assert abs(normalized.min().item() - 10.0) < 1e-5
        assert abs(normalized.max().item() - 20.0) < 1e-5

    def test_scale_to_unit_power_normalizer_basic(self):
        """Test ScaleToUnitPowerFeatNormalizer with basic data."""
        normalizer = ScaleToUnitPowerFeatNormalizer()

        # Create test data with known power
        # values = [1, 2, 3, 4, 5]
        # E[X²] = (1 + 4 + 9 + 16 + 25) / 5 = 55 / 5 = 11
        # scale_factor = 1 / sqrt(11) ≈ 0.30151
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        normalizer.fit(values)

        # Check scale factor
        expected_scale = 1.0 / np.sqrt(11.0)
        assert normalizer.scale_factor is not None
        assert abs(normalizer.scale_factor - expected_scale) < 1e-5

        # Normalize
        normalized = normalizer.normalize(values)

        # Check that normalized values have unit power: E[normalized²] = 1
        power = (normalized**2).mean().item()
        assert abs(power - 1.0) < 1e-5

        # Unnormalize should return original values
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(unnormalized, values, rtol=1e-4, atol=1e-6)

    def test_scale_to_unit_power_normalizer_incremental_fit(self):
        """Test that ScaleToUnitPowerFeatNormalizer correctly accumulates statistics."""
        normalizer = ScaleToUnitPowerFeatNormalizer()

        # Fit in batches
        batch1 = torch.tensor([1.0, 2.0, 3.0])
        batch2 = torch.tensor([4.0, 5.0, 6.0])

        normalizer.fit(batch1)
        normalizer.fit(batch2)

        # Should have computed E[X²] over all values [1,2,3,4,5,6]
        # E[X²] = (1 + 4 + 9 + 16 + 25 + 36) / 6 = 91 / 6 ≈ 15.1667
        # scale_factor = 1 / sqrt(91/6) ≈ 0.2567
        expected_mean_sq = 91.0 / 6.0
        expected_scale = 1.0 / np.sqrt(expected_mean_sq)

        assert normalizer.scale_factor is not None
        assert abs(normalizer.scale_factor - expected_scale) < 1e-5

        # Verify unit power
        all_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        normalized = normalizer.normalize(all_values)
        power = (normalized**2).mean().item()
        assert abs(power - 1.0) < 1e-5

    def test_scale_to_unit_power_normalizer_zero_values(self):
        """Test ScaleToUnitPowerFeatNormalizer with all zero values."""
        normalizer = ScaleToUnitPowerFeatNormalizer()

        # All zero values
        values = torch.tensor([0.0, 0.0, 0.0, 0.0])
        normalizer.fit(values)

        # Should set scale_factor to 1.0 to avoid division by zero
        assert normalizer.scale_factor == 1.0

        # Normalization should keep zeros as zeros
        normalized = normalizer.normalize(values)
        assert torch.allclose(normalized, values)

    def test_scale_to_unit_power_normalizer_unit_values(self):
        """Test ScaleToUnitPowerFeatNormalizer when E[X²] = 1."""
        normalizer = ScaleToUnitPowerFeatNormalizer()

        # Create values where E[X²] is already 1
        # For example: values that already have unit power
        # sqrt(0.5²+0.5²+0.5²+0.5²)/4 would give E[X²] = 0.25, so let's use ±1
        # E[1² + (-1)²] / 2 = 2/2 = 1
        values = torch.tensor([1.0, -1.0, 1.0, -1.0])

        normalizer.fit(values)

        # scale_factor should be 1/sqrt(1) = 1
        assert abs(normalizer.scale_factor - 1.0) < 1e-5

        # Normalized values should be the same
        normalized = normalizer.normalize(values)
        assert torch.allclose(normalized, values, rtol=1e-5)

    def test_scale_to_unit_power_normalizer_negative_values(self):
        """Test ScaleToUnitPowerFeatNormalizer with negative values."""
        normalizer = ScaleToUnitPowerFeatNormalizer()

        # Mix of positive and negative values
        values = torch.tensor([-3.0, -1.0, 0.0, 2.0, 4.0])
        # E[X²] = (9 + 1 + 0 + 4 + 16) / 5 = 30 / 5 = 6
        # scale_factor = 1 / sqrt(6) ≈ 0.4082

        normalizer.fit(values)

        expected_scale = 1.0 / np.sqrt(6.0)
        assert abs(normalizer.scale_factor - expected_scale) < 1e-5

        # Normalize and check power
        normalized = normalizer.normalize(values)
        power = (normalized**2).mean().item()
        assert abs(power - 1.0) < 1e-5

        # Unnormalize
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(unnormalized, values, rtol=1e-4)

    def test_scale_to_unit_power_normalizer_large_values(self):
        """Test ScaleToUnitPowerFeatNormalizer with large values."""
        normalizer = ScaleToUnitPowerFeatNormalizer()

        # Large values to test numerical stability
        values = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])

        normalizer.fit(values)
        normalized = normalizer.normalize(values)

        # Should still have unit power
        power = (normalized**2).mean().item()
        assert abs(power - 1.0) < 1e-5

        # Round-trip
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(unnormalized, values, rtol=1e-4)


class TestMultiFeatNormalizer:
    """Test suite for MultiFeatNormalizer."""

    def test_multi_feat_normalizer_basic(self):
        """Test MultiFeatNormalizer with mixed normalizers."""
        # Create dataset with 3 input features and 2 target features
        num_samples = 100
        inputs = torch.randn(num_samples, 3) * 10 + 5  # Random data
        targets = torch.randn(num_samples, 2) * 5 + 10

        dataset = SimpleDataset(inputs, targets)

        # Create normalizer with different normalizers per feature
        normalizer = MultiFeatNormalizer(
            input_normalizers=[
                StdFeatNormalizer(),
                IntervalFeatNormalizer(interval=(-1, 1)),
                IdentityFeatNormalizer(),
            ],
            targets_normalizers=[
                StdFeatNormalizer(),
                IntervalFeatNormalizer(interval=(0, 1)),
            ],
            batch_size=10,
        )

        # Fit the normalizer
        normalizer.fit(dataset)

        # Test normalization on a batch
        test_inputs = inputs[:5]
        test_targets = targets[:5]

        normalized_inputs = normalizer.normalize_input_batch(test_inputs)
        normalized_targets = normalizer.normalize_targets_batch(test_targets)

        # Check shapes are preserved
        assert normalized_inputs.shape == test_inputs.shape
        assert normalized_targets.shape == test_targets.shape

        # Check first column of inputs is standardized (approximately)
        assert abs(
            normalized_inputs[:,
                              0].mean().item()) < 1.0  # Should be close to 0

        # Check second column of inputs is in [-1, 1]
        assert normalized_inputs[:, 1].min().item() >= -1.0 - 1e-3
        assert normalized_inputs[:, 1].max().item() <= 1.0 + 1e-3

        # Check third column is unchanged (identity normalizer)
        assert torch.allclose(normalized_inputs[:, 2], test_inputs[:, 2])

        # Check second target column is in [0, 1]
        assert normalized_targets[:, 1].min().item() >= 0.0 - 1e-3
        assert normalized_targets[:, 1].max().item() <= 1.0 + 1e-3

        # Test unnormalization
        # Note: unnormalize_output_batch uses target normalizers when output_normalizers is empty
        unnormalized_targets = normalizer.unnormalize_targets_batch(
            normalized_targets)

        assert torch.allclose(unnormalized_targets, test_targets, rtol=1e-3)

    def test_multi_feat_normalizer_empty_normalizers(self):
        """Test MultiFeatNormalizer with no normalizers."""
        num_samples = 20
        inputs = torch.randn(num_samples, 3)
        targets = torch.randn(num_samples, 2)

        dataset = SimpleDataset(inputs, targets)

        # Create normalizer with no normalizers
        normalizer = MultiFeatNormalizer()
        normalizer.fit(dataset)

        # Should return inputs/targets unchanged
        normalized_inputs = normalizer.normalize_input_batch(inputs)
        normalized_targets = normalizer.normalize_targets_batch(targets)

        assert torch.allclose(normalized_inputs, inputs)
        assert torch.allclose(normalized_targets, targets)

    def test_multi_feat_normalizer_batch_processing(self):
        """Test that MultiFeatNormalizer processes data in batches correctly."""
        num_samples = 50
        inputs = torch.arange(num_samples, dtype=torch.float32).unsqueeze(1)
        targets = torch.arange(num_samples,
                               dtype=torch.float32).unsqueeze(1) * 2

        dataset = SimpleDataset(inputs, targets)

        # Create normalizer with small batch size
        normalizer = MultiFeatNormalizer(
            input_normalizers=[StdFeatNormalizer()],
            targets_normalizers=[StdFeatNormalizer()],
            batch_size=7,  # Not a divisor of 50
        )

        normalizer.fit(dataset)

        # Check that statistics are correct (computed over all data)
        expected_input_mean = (num_samples - 1) / 2.0
        expected_target_mean = (num_samples - 1)

        assert abs(normalizer.input_normalizers[0].mean -
                   expected_input_mean) < 1e-3
        assert abs(normalizer.targets_normalizers[0].mean -
                   expected_target_mean) < 1e-3

    def test_multi_feat_normalizer_save_load(self):
        """Test save/load functionality of MultiFeatNormalizer."""
        import tempfile
        import shutil

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Create and fit normalizer
            num_samples = 50
            inputs = torch.randn(num_samples, 2)
            targets = torch.randn(num_samples, 2)
            dataset = SimpleDataset(inputs, targets)

            normalizer1 = MultiFeatNormalizer(
                input_normalizers=[
                    StdFeatNormalizer(),
                    IntervalFeatNormalizer(interval=(-1, 1)),
                ],
                targets_normalizers=[
                    StdFeatNormalizer(),
                    IdentityFeatNormalizer(),
                ],
                nn_folder=temp_dir,
            )

            normalizer1.fit(dataset)

            # Get state via properties
            input_state = normalizer1.input_normalizers_state
            targets_state = normalizer1.targets_normalizers_state

            # Save
            normalizer1.save()

            # Get parameters before loading
            input_mean_before = normalizer1.input_normalizers[0].mean
            input_std_before = normalizer1.input_normalizers[0].std
            input_min_before = normalizer1.input_normalizers[1].min_val
            input_max_before = normalizer1.input_normalizers[1].max_val

            # Create new normalizer and load
            normalizer2 = MultiFeatNormalizer(
                input_normalizers=[
                    StdFeatNormalizer(),
                    IntervalFeatNormalizer(interval=(-1, 1)),
                ],
                targets_normalizers=[
                    StdFeatNormalizer(),
                    IdentityFeatNormalizer(),
                ],
                nn_folder=temp_dir,
            )

            normalizer2.load_if_file_exists()

            # Check that parameters match via the property setters
            assert normalizer2.input_normalizers[0].mean == input_mean_before
            assert normalizer2.input_normalizers[0].std == input_std_before
            assert normalizer2.input_normalizers[1].min_val == input_min_before
            assert normalizer2.input_normalizers[1].max_val == input_max_before

            # Test that normalization produces same results
            test_batch = inputs[:5]
            normalized1 = normalizer1.normalize_input_batch(test_batch)
            normalized2 = normalizer2.normalize_input_batch(test_batch)

            assert torch.allclose(normalized1, normalized2)

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)

    def test_multi_feat_normalizer_shape_validation(self):
        """Test that MultiFeatNormalizer validates feature count."""
        num_samples = 20
        inputs = torch.randn(num_samples, 3)
        targets = torch.randn(num_samples, 2)

        dataset = SimpleDataset(inputs, targets)

        # Create normalizer with wrong number of normalizers
        normalizer = MultiFeatNormalizer(
            input_normalizers=[
                StdFeatNormalizer(),
                StdFeatNormalizer(),
                # Missing third normalizer
            ],
            targets_normalizers=[
                StdFeatNormalizer(),
                StdFeatNormalizer(),
            ],
        )

        # Should raise assertion error
        with pytest.raises(AssertionError):
            normalizer.fit(dataset)
