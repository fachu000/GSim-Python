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
from gsim.include.neural_net import NeuralNet, WarmupCosineMinLRScheduler
from gsim.include.neural_net.normalizers import (
    MultiFeatNormalizer,
    StdFeatNormalizer,
    IntervalFeatNormalizer,
    IdentityFeatNormalizer,
)


class ExperimentSet(gsim.AbstractExperimentSet):

    # Simple experiment where a neural network is trained and tested
    def experiment_1001(l_args):

        class ExampleDataset(Dataset):

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

        class ExampleNet(NeuralNet):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fc = nn.Linear(2, 1)
                self.initialize()

            def forward(self, x):
                return self.fc(x)

        dataset = ExampleDataset(1000)
        net = ExampleNet()

        f_loss = lambda m_pred, m_targets: torch.mean(
            (m_targets - m_pred)**2, dim=1)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        training_history = net.fit(
            dataset,
            f_loss,
            optimizer,
            val_split=0.2,
            num_epochs=200,
            batch_size=200,
        )
        d_metrics = net.evaluate(dataset, batch_size=32, f_loss=f_loss)
        print(d_metrics)
        return net.plot_training_history(training_history)

    # Simple experiment to illustrate learning rate scheduler usage
    def experiment_1002(l_args):

        class ExampleDataset(Dataset):

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

        class ExampleNet(NeuralNet):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fc = nn.Linear(2, 1)
                self.initialize()

            def forward(self, x):
                return self.fc(x)

        torch.manual_seed(0)
        np.random.seed(0)
        dataset = ExampleDataset(1000)
        net = ExampleNet()

        f_loss = lambda m_pred, m_targets: torch.mean(
            (m_targets - m_pred)**2, dim=1)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        # There is a great difference in training speed when using the scheduler.
        # Try setting it to None and comparing the results.
        lr_scheduler = WarmupCosineMinLRScheduler(
            optimizer,
            warmup_steps=15,
            total_steps=300,
            min_lr=1e-4,
        )
        d_training_history = net.fit(
            dataset,
            f_loss,
            optimizer,
            lr_scheduler=lr_scheduler,
            val_split=0.2,
            num_epochs=200,
            batch_size=200,
        )
        d_metrics = net.evaluate(dataset, batch_size=32, f_loss=f_loss)
        print(d_metrics)
        return net.plot_training_history(d_training_history)

    # WIP: set the parameters and target function properly.
    # Experiment that illustrates how to use normalization with NeuralNet
    def experiment_1003(l_args):

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
                self.fc1 = nn.Linear(20, 100)
                self.fc2 = nn.Linear(100, 100)
                self.fc3 = nn.Linear(100, 1)
                self.initialize()

            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                x = torch.relu(x)
                x = self.fc3(x)
                return x

        def compare_unnormalized_losses():
            lt_vals = l_nets[0]["history"].l_train_loss_me
            last_val = lt_vals[-1][1]
            G = GFigure(xlabel="Epoch",
                        ylabel="Loss",
                        ylim=(.8 * last_val, 1.5 * last_val))
            G.add_curve(
                xaxis=[t[0] for t in lt_vals],
                yaxis=[t[1] for t in lt_vals],
                legend=f"Unnormalized training loss of {l_nets[0]['name']}",
                styles="b-")
            lt_vals = l_nets[0]["history"].l_val_loss
            G.add_curve(
                xaxis=[t[0] for t in lt_vals],
                yaxis=[t[1] for t in lt_vals],
                legend=f"Unnormalized validation loss of {l_nets[0]['name']}",
                styles="b--")
            G.next_subplot(xlabel="Epoch", ylabel="Loss")
            lt_vals = l_nets[1]["history"].l_unnormalized_train_loss
            G.add_curve(
                xaxis=[t[0] for t in lt_vals],
                yaxis=[t[1] for t in lt_vals],
                legend=f"Unnormalized training loss of {l_nets[1]['name']}",
                styles="r-")
            lt_vals = l_nets[1]["history"].l_unnormalized_val_loss
            G.add_curve(
                xaxis=[t[0] for t in lt_vals],
                yaxis=[t[1] for t in lt_vals],
                legend=f"Unnormalized validation loss of {l_nets[1]['name']}",
                styles="r--")
            return G

        torch.manual_seed(0)
        np.random.seed(0)

        dataset = MyDataset(10000)

        num_epochs = 50

        d_unnormalized = {
            "name": "Unnormalized network",
            "model": MyNet(),
            "lr": 0.01,
            "weight_decay": 0,
            "num_epochs": num_epochs
        }

        d_normalized = {
            "name": "Normalized network",
            "model": MyNet(normalizer="both"),
            "lr": 5e-3,  #0.5
            "weight_decay": 1,
            "num_epochs": num_epochs
        }

        l_nets = [d_unnormalized, d_normalized]

        f_loss = lambda m_pred, m_targets: torch.mean(
            (m_targets - m_pred)**2, dim=1)

        l_G = []
        for d_net in l_nets:
            print(f"Training {d_net['name']}")
            net: NeuralNet = d_net["model"]
            optimizer = torch.optim.AdamW(net.parameters(),
                                          lr=d_net["lr"],
                                          weight_decay=d_net["weight_decay"])

            d_training_history = net.fit(
                dataset,
                f_loss,
                optimizer,
                val_split=0.2,
                num_epochs=d_net["num_epochs"],
                batch_size=200,
                eval_unnormalized_losses=True,
                num_steps_eval_static=302,
                num_steps_eval_moving=305,
            )
            d_net["metrics"] = net.evaluate(dataset,
                                            batch_size=32,
                                            f_loss=f_loss)
            d_net["history"] = d_training_history

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

        G = compare_unnormalized_losses()
        l_G += [G]
        return l_G

    # Simple experiment where a neural network is trained to learn a 1D
    # function in multiple sessions. Demonstrates saving/loading.
    def experiment_1004(l_args):

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

            def __init__(self, dim_hidden=30, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fc1 = nn.Linear(1, dim_hidden)
                self.fc2 = nn.Linear(dim_hidden, 1)
                self.initialize()

            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x

        def train(net: NeuralNet, num_steps):
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
            lr_scheduler = WarmupCosineMinLRScheduler(
                optimizer,
                warmup_steps=100,
                total_steps=180,
                min_lr=5e-5,
            )
            return net.fit(train_dataset,
                           f_loss,
                           optimizer,
                           lr_scheduler=lr_scheduler,
                           dataset_val=val_dataset,
                           num_steps=num_steps,
                           batch_size=200,
                           num_steps_eval_moving=1,
                           num_steps_eval_static=1,
                           num_steps_checkpoint=20)

        def plot_data():
            preds = [
                float(p[0]) for p in net.predict(test_dataset,
                                                 dataset_includes_targets=True)
            ]
            feat = [float(data[0]) for data in test_dataset]
            true_target = [float(data[1]) for data in test_dataset]

            G = GFigure(xlabel="Feature", ylabel="Target")
            G.add_curve(feat, true_target, legend="True target", styles="r.")
            G.add_curve(feat, preds, legend="Prediction", styles="kx")
            return G

        import tempfile
        folder = tempfile.mkdtemp(prefix="neural_net_")

        fun = lambda m_x: 5 * torch.sin(2 * np.pi * m_x / 20)

        torch.manual_seed(0)
        np.random.seed(0)
        train_dataset = MyDataset(10, fun=fun)
        val_dataset = MyDataset(1000, fun=fun)
        test_dataset = MyDataset(100, fun=fun)
        dim_hidden = 3000
        net = MyNet(dim_hidden=dim_hidden, normalizer="both", nn_folder=folder)

        f_loss = lambda m_pred, m_targets: torch.mean(
            (m_targets - m_pred)**2, dim=1)
        training_history = train(net, num_steps=35)
        d_metrics = net.evaluate(train_dataset, batch_size=32, f_loss=f_loss)
        print(d_metrics)

        # We load the network from disk and keep training it
        net = MyNet(dim_hidden=dim_hidden, normalizer="both", nn_folder=folder)
        training_history = train(net, num_steps=60)
        training_history = train(net, num_steps=100)
        training_history = train(net, num_steps=50)
        # Delete the lr_scheduler file to restart the learning rate schedule
        os.remove(os.path.join(folder, "lr_scheduler.pth"))
        training_history = train(net, num_steps=50)
        l_G = net.plot_training_history(training_history)
        l_G += [plot_data()]

        return l_G

    # Experiment demonstrating MultiFeatNormalizer with linear regression on
    # feature vectors
    def experiment_1005(l_args):
        """
        This experiment creates a dataset with feature vectors where each
        feature has different characteristics (scale, range, etc.). It then
        performs linear regression using MultiFeatNormalizer to normalize
        different features appropriately, comparing normalized vs unnormalized
        approaches.
        """

        class FeatureVectorDataset(Dataset):
            """Dataset with feature vectors having different characteristics."""

            def __init__(self, num_examples):
                self.num_examples = num_examples

                # Create features with different scales and ranges:
                # Feature 0: Small values around 0 (already normalized-ish)
                feat_0 = torch.randn(num_examples, 1)

                # Feature 1: Large values with high mean
                feat_1 = 500 + 100 * torch.randn(num_examples, 1)

                # Feature 2: Values in [0, 100] range
                feat_2 = 100 * torch.rand(num_examples, 1)

                # Feature 3: Binary-like feature
                feat_3 = torch.randint(0, 2, (num_examples, 1)).float()

                # Concatenate all features
                self.m_feat = torch.cat([feat_0, feat_1, feat_2, feat_3],
                                        dim=1)

                # Linear target function: y = 2*x0 + 0.5*x1 + 3*x2 - 10*x3 + noise
                self.m_targets = (2 * self.m_feat[:, 0:1] +
                                  0.5 * self.m_feat[:, 1:2] +
                                  3 * self.m_feat[:, 2:3] -
                                  10 * self.m_feat[:, 3:4] +
                                  5 * torch.randn(num_examples, 1))

            def __len__(self):
                return self.num_examples

            def __getitem__(self, ind):
                return self.m_feat[ind], self.m_targets[ind]

        class LinearRegressionNet(NeuralNet):
            """Simple linear regression network (no hidden layers)."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fc = nn.Linear(4, 1)  # 4 input features, 1 output
                self.initialize()

            def forward(self, x):
                return self.fc(x)

        normalizer = MultiFeatNormalizer(
            input_normalizers=[
                IdentityFeatNormalizer(),  # Feature 0: already well-scaled
                StdFeatNormalizer(
                ),  # Feature 1: standardize (remove mean, divide by std)
                IntervalFeatNormalizer(
                    interval=(0, 1)),  # Feature 2: scale to [0,1]
                IdentityFeatNormalizer(),  # Feature 3: binary, keep as is
            ],
            targets_normalizers=[
                StdFeatNormalizer(),  # Standardize target
            ],
            batch_size=32)

        # Create dataset
        torch.manual_seed(42)
        np.random.seed(42)
        dataset = FeatureVectorDataset(5000)

        net: NeuralNet = LinearRegressionNet(normalizer=normalizer)

        d_training_history = net.fit(
            dataset,
            optimizer=torch.optim.Adam(net.parameters(), lr=0.01),
            val_split=0.2,
            num_epochs=100,
            batch_size=128,
            f_loss=lambda m_pred, m_targets: torch.mean(
                (m_targets - m_pred)**2, dim=1),
            eval_unnormalized_losses=True,
        )

        return net.plot_training_history(d_training_history)

    # Experiment to illustrate live plotting
    # WIP: improve the network or make the function simpler.
    def experiment_1006(l_args):

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

        class MyNet(NeuralNet):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fc1 = nn.Linear(20, 100)
                self.fc2 = nn.Linear(100, 100)
                self.fc3 = nn.Linear(100, 1)
                self.initialize()

            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                x = torch.relu(x)
                x = self.fc3(x)
                return x

        torch.manual_seed(0)
        np.random.seed(0)
        import tempfile
        folder = tempfile.mkdtemp(prefix="neural_net_")
        folder = "/tmp/neural_net_live_plot/"

        dataset = MyDataset(500000)

        f_loss = lambda m_pred, m_targets: torch.mean(
            (m_targets - m_pred)**2, dim=1)

        net: NeuralNet = MyNet(normalizer="both", nn_folder=folder)

        # Training
        optimizer = torch.optim.AdamW(net.parameters(),
                                      lr=5e-3,
                                      weight_decay=1e-2)
        lr_scheduler = WarmupCosineMinLRScheduler(
            optimizer,
            warmup_steps=10000,
            total_steps=50000,
            min_lr=1e-4,
        )
        d_training_history = net.fit(dataset,
                                     f_loss,
                                     optimizer,
                                     lr_scheduler=lr_scheduler,
                                     val_split=0.2,
                                     num_steps=50000,
                                     batch_size=64,
                                     eval_unnormalized_losses=False,
                                     num_steps_eval_static=2000,
                                     num_steps_eval_moving=128,
                                     keep_best_val_weights=True,
                                     static_max_hci=0.01,
                                     live_plot=True)

        return net.plot_training_history(d_training_history)
