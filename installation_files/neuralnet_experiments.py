from random import random
import numpy as np
import os
import pickle
import tempfile
import logging

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset, random_split
except ImportError:
    raise ValueError(
        "PyTorch is not installed. This experiment requires PyTorch.")

from tqdm import tqdm

import gsim
from gsim.gfigure import GFigure
from gsim.include.neural_net import NeuralNet


class ExperimentSet(gsim.AbstractExperimentSet):

    # Simple experiment where a neural network is trained and tested
    def experiment_1001(l_args):

        class MyDataset(Dataset):

            def __init__(self, num_examples):
                self.num_examples = num_examples
                self.m_feat = torch.randn(num_examples, 2)
                self.m_targets = torch.sum(
                    self.m_feat, dim=1,
                    keepdim=True) + 0.5 * torch.randn(num_examples, 1)

            def __len__(self):
                return self.num_examples

            def __getitem__(self, ind):
                return self.m_feat[ind], self.m_targets[ind]

        class MyNet(NeuralNet):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fc = nn.Linear(2, 1)
                self.initialize()

            def forward(self, x):
                return self.fc(x)

        dataset = MyDataset(1000)
        net = MyNet()

        f_loss = lambda m_pred, m_targets: torch.mean(
            (m_targets - m_pred)**2, dim=1)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        d_training_history = net.fit(dataset,
                                     optimizer,
                                     val_split=0.2,
                                     lr_patience=20,
                                     num_epochs=200,
                                     batch_size=200,
                                     f_loss=f_loss)
        d_metrics = net.evaluate(dataset, batch_size=32, f_loss=f_loss)
        print(d_metrics)
        return net.plot_training_history(d_training_history)

    # Experiment that illustrates how to use normalization with NeuralNet
    def experiment_1002(l_args):

        class MyDataset(Dataset):

            def __init__(self, num_examples):
                self.num_examples = num_examples
                self.m_feat = 300 + 5 * torch.randn(num_examples, 20)
                self.m_targets = MyDataset.target_fun(
                    self.m_feat) + 100 * torch.randn(num_examples, 1)

            @staticmethod
            def target_fun(m_feat: torch.Tensor) -> torch.Tensor:
                # m_feat is num_examples x 20
                m_feat = (m_feat[:, :10] -
                          300)**2 / 10 + 0.5 * m_feat[:, 10:] + 20
                return torch.sum(m_feat, dim=1, keepdim=True)

            def __len__(self):
                return self.num_examples

            def __getitem__(self, ind):
                return self.m_feat[ind], self.m_targets[ind]

        def plot_data_distribution(dataset: MyDataset) -> GFigure:
            G = GFigure(xlabel="Feature value",
                        ylabel="Histogram",
                        num_subplot_columns=1)
            G.add_histogram_curve(
                data=dataset.m_feat.numpy().flatten(),
                hist_args={
                    'bins': 50,
                    'density': True
                },
            )
            G.next_subplot(xlabel="Target value", ylabel="Histogram")
            G.add_histogram_curve(
                data=dataset.m_targets.numpy().flatten(),
                hist_args={
                    'bins': 50,
                    'density': True
                },
            )
            print(f"Target variance = {dataset.m_targets.var().item()}")

            return G

        class MyNet(NeuralNet):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fc1 = nn.Linear(20, 50)
                self.fc2 = nn.Linear(50, 1)
                self.initialize()

            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x

        torch.manual_seed(0)
        np.random.seed(0)

        dataset = MyDataset(1000)

        d_unnormalized = {
            "name": "Unnormalized network",
            "model": MyNet(),
            "lr": 0.01
        }

        d_normalized = {
            "name": "Normalized network",
            "model": MyNet(normalizer="both"),
            "lr": 0.5
        }

        l_nets = [d_unnormalized, d_normalized]

        f_loss = lambda m_pred, m_targets: torch.mean(
            (m_targets - m_pred)**2, dim=1)

        l_G = []
        for d_net in l_nets:
            print(f"Training {d_net['name']}")
            net: NeuralNet = d_net["model"]
            optimizer = torch.optim.Adam(net.parameters(), lr=d_net["lr"])

            d_training_history = net.fit(
                dataset,
                optimizer,
                val_split=0.2,
                lr_patience=20,
                num_epochs=200,
                batch_size=200,
                f_loss=f_loss,
                eval_unnormalized_loss=True,
            )
            d_net["metrics"] = net.evaluate(dataset,
                                            batch_size=32,
                                            f_loss=f_loss)

            l_G_now: list[GFigure] = net.plot_training_history(
                d_training_history)
            for G in l_G_now:
                main_subplot = G.l_subplots[0]
                if main_subplot:
                    main_subplot.title = d_net["name"]
            l_G += l_G_now

        gsim.gfigure.title_to_caption = False

        l_G += [plot_data_distribution(dataset)]
        for d_net in l_nets:
            print(f"Metrics for {d_net['name']}:")
            print(d_net["metrics"])
        return l_G

    # Simple experiment where a neural network is trained to learn a 1D
    #function, then it is saved to disk, and finally it is loaded.
    def experiment_1003(l_args):

        class MyDataset(Dataset):

            def __init__(self, num_examples, fun):
                self.num_examples = num_examples
                self.m_feat = 10 + 20 * torch.rand(num_examples, 1)
                self.m_targets = fun(self.m_feat) + torch.randn(
                    num_examples, 1)

            def __len__(self):
                return self.num_examples

            def __getitem__(self, ind):
                return self.m_feat[ind], self.m_targets[ind]

        class MyNet(NeuralNet):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fc1 = nn.Linear(1, 30)
                self.fc2 = nn.Linear(30, 1)
                self.initialize()

            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x

        import tempfile, shutil
        folder = tempfile.mkdtemp(prefix="neural_net_")

        fun = lambda m_x: 5 * torch.sin(2 * np.pi * m_x / 20)

        train_dataset = MyDataset(1000, fun=fun)
        test_dataset = MyDataset(100, fun=fun)
        net = MyNet(normalizer="both", nn_folder=folder)

        f_loss = lambda m_pred, m_targets: torch.mean(
            (m_targets - m_pred)**2, dim=1)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        d_training_history = net.fit(train_dataset,
                                     optimizer,
                                     val_split=0.2,
                                     lr_patience=20,
                                     num_epochs=200,
                                     batch_size=200,
                                     f_loss=f_loss)
        d_metrics = net.evaluate(train_dataset, batch_size=32, f_loss=f_loss)
        print(d_metrics)
        l_G = net.plot_training_history(d_training_history)

        # We load the network from disk and test it
        net = MyNet(normalizer="both", nn_folder=folder)

        def plot_data():
            preds = [float(p) for p in net.predict(test_dataset)[:, 0]]
            feat = [float(data[0]) for data in test_dataset]
            true_target = [float(data[1]) for data in test_dataset]

            G = GFigure(xlabel="Feature", ylabel="Target")
            G.add_curve(feat, true_target, legend="True target", styles="r.")
            G.add_curve(feat, preds, legend="Prediction", styles="kx")
            return G

        l_G += [plot_data()]

        return l_G
