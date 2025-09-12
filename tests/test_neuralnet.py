import pytest
import torch
from torch.utils.data import Dataset

from gsim.include.neural_net import NeuralNet


def test_uncollate_fn_when_output_is_a_tuple():
    # In this test, the output of the network is a tuple of three tensors.
    net = NeuralNet()  # pyright: ignore[reportAbstractUsage]

    batch_size_1 = 4
    out_item_11 = torch.randint(low=0, high=10, size=(batch_size_1, 5, 6))
    out_item_12 = torch.randint(low=0, high=10, size=(batch_size_1, 6, 7))
    out_item_13 = torch.randint(low=0, high=10, size=(batch_size_1, 7, 8))

    batch_size_2 = 3
    out_item_21 = torch.randint(low=0, high=10, size=(batch_size_2, 5, 6))
    out_item_22 = torch.randint(low=0, high=10, size=(batch_size_2, 6, 7))
    out_item_23 = torch.randint(low=0, high=10, size=(batch_size_2, 7, 8))

    batch_1 = (out_item_11, out_item_12, out_item_13)

    batch_2 = (out_item_21, out_item_22, out_item_23)

    l_batches = [batch_1, batch_2]

    l_out = net.uncollate_fn(l_batches)

    assert len(l_out) == batch_size_1 + batch_size_2
    assert torch.equal(l_out[0][0], out_item_11[0])
    assert torch.equal(l_out[0][1], out_item_12[0])
    assert torch.equal(l_out[0][2], out_item_13[0])
    assert torch.equal(l_out[1][0], out_item_11[1])
    assert torch.equal(l_out[1][1], out_item_12[1])
    assert torch.equal(l_out[1][2], out_item_13[1])
    assert torch.equal(l_out[2][0], out_item_11[2])
    assert torch.equal(l_out[2][1], out_item_12[2])
    assert torch.equal(l_out[2][2], out_item_13[2])
    assert torch.equal(l_out[3][0], out_item_11[3])
    assert torch.equal(l_out[3][1], out_item_12[3])
    assert torch.equal(l_out[3][2], out_item_13[3])
    assert torch.equal(l_out[4][0], out_item_21[0])
    assert torch.equal(l_out[4][1], out_item_22[0])
    assert torch.equal(l_out[4][2], out_item_23[0])
    assert torch.equal(l_out[5][0], out_item_21[1])
    assert torch.equal(l_out[5][1], out_item_22[1])
    assert torch.equal(l_out[5][2], out_item_23[1])
    assert torch.equal(l_out[6][0], out_item_21[2])
    assert torch.equal(l_out[6][1], out_item_22[2])
    assert torch.equal(l_out[6][2], out_item_23[2])

    print("hello")


def test_uncollate_fn_when_output_is_a_tensor():
    # In this test, the output of the network is a single tensor.
    net = NeuralNet()  # pyright: ignore[reportAbstractUsage]

    batch_size_1 = 4
    out_item_1 = torch.randint(low=0, high=10, size=(batch_size_1, 5, 6))

    batch_size_2 = 3
    out_item_2 = torch.randint(low=0, high=10, size=(batch_size_2, 5, 6))

    l_batches = [out_item_1, out_item_2]

    l_out = net.uncollate_fn(l_batches)

    assert len(l_out) == batch_size_1 + batch_size_2
    assert torch.equal(l_out[0], out_item_1[0])
    assert torch.equal(l_out[1], out_item_1[1])
    assert torch.equal(l_out[2], out_item_1[2])
    assert torch.equal(l_out[3], out_item_1[3])
    assert torch.equal(l_out[4], out_item_2[0])
    assert torch.equal(l_out[5], out_item_2[1])
    assert torch.equal(l_out[6], out_item_2[2])

    print("hello")


def test_predict_when_input_and_output_are_tensors():

    class TestNet(NeuralNet):

        def __init__(self):
            super().__init__()
            self.initialize()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: input tensor of shape (batch_size, M , N)
            Returns:
                output tensor of shape (batch_size, 2*M, 3*N, 4)            
            """
            return x[..., None].tile(1, 2, 3, 4)

    net = TestNet()

    input = torch.randint(low=0, high=10, size=(7, 5, 6))

    # Output of predict is a tensor
    output = net.predict(input, batch_size=3)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (7, 10, 18, 4)

    # Output of predict is a list
    output = net.predict(input, batch_size=3, output_class=tuple)
    assert isinstance(output, tuple)
    assert len(output) == 7
    for ind_output in range(7):
        assert output[ind_output].shape == (10, 18, 4)

    # Output of predict is a Dataset
    output = net.predict(input, batch_size=3, output_class=Dataset)
    assert isinstance(output, Dataset)
    assert len(output) == 7
    for ind_output in range(7):
        assert output[ind_output].shape == (10, 18, 4)


def test_predict_when_the_input_is_a_list_and_the_output_is_a_tuple():

    class TestNet(NeuralNet[list[torch.Tensor], tuple[torch.Tensor, ...]]):

        def __init__(self):
            super().__init__()
            self.initialize()

        def forward(
                self,
                x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: input list of three tensors of shape (batch_size, M1 , N1),
                   (batch_size, M1 , N2), (batch_size, M2 , N2)
            Returns:
                output: tuple of two tensors of shape (batch_size, M1, N1+N2) and (batch_size, M1+M2, N2)
            """
            x_1, x_2, x_3 = x
            out_1 = torch.concat((x_1, x_2), dim=2)
            out_2 = torch.concat((x_2, x_3), dim=1)
            return (out_1, out_2)

    net = TestNet()

    num_inputs = 7
    input = [[
        torch.randint(low=0, high=10, size=(5, 6)),
        torch.randint(low=0, high=10, size=(5, 8)),
        torch.randint(low=0, high=10, size=(9, 8))
    ] for _ in range(num_inputs)]

    # Output of predict is a list
    output = net.predict(input, batch_size=6)
    assert isinstance(output, list)
    assert len(output) == 7
    for ind_output in range(7):
        assert isinstance(output[ind_output], tuple)
        assert len(output[ind_output]) == 2
        assert output[ind_output][0].shape == (5, 14)
        assert output[ind_output][1].shape == (14, 8)

    # Output of predict is a tuple
    output = net.predict(input, batch_size=4, output_class=tuple)
    assert isinstance(output, tuple)
    assert len(output) == 7
    for ind_output in range(7):
        assert isinstance(output[ind_output], tuple)
        assert len(output[ind_output]) == 2
        assert output[ind_output][0].shape == (5, 14)
        assert output[ind_output][1].shape == (14, 8)

    # Output of predict is a Dataset
    output = net.predict(input, batch_size=4, output_class=Dataset)
    assert isinstance(output, Dataset)
    assert len(output) == 7
    for ind_output in range(7):
        assert isinstance(output[ind_output], tuple)
        assert len(output[ind_output]) == 2
        assert output[ind_output][0].shape == (5, 14)
        assert output[ind_output][1].shape == (14, 8)
