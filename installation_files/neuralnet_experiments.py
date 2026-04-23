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
from gsim.include.neural_net.data_adapter import AdaptationSpec, DataAdapter
from gsim.include.neural_net.normalizers import (
    MultiFeatNormalizer,
    StdFeatNormalizer,
    IntervalFeatNormalizer,
    IdentityFeatNormalizer,
)

np.random.seed(0)
torch.manual_seed(0)


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

        dataset = ExampleDataset(10000)
        net = ExampleNet()

        f_loss = lambda m_pred, m_targets: torch.mean(
            (m_targets - m_pred)**2, dim=1)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        training_history = net.fit(
            dataset,
            f_loss,
            optimizer,
            val_split=0.4,
            num_steps=1000,
            batch_size=200,
        )
        d_metrics = net.evaluate(dataset, batch_size=32, f_loss=f_loss)
        print(d_metrics)
        return net.plot_training_history(training_history,
                                         logx=True,
                                         logy=True)

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
            total_steps=500,
            min_lr=1e-4,
        )
        d_training_history = net.fit(
            dataset,
            f_loss,
            optimizer,
            lr_scheduler=lr_scheduler,
            val_split=0.2,
            num_steps=500,
            batch_size=200,
        )
        d_metrics = net.evaluate(dataset, batch_size=32, f_loss=f_loss)
        print(d_metrics)
        return net.plot_training_history(d_training_history,
                                         logx=True,
                                         logy=True)

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
                num_steps_report_moving=305,
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
                           num_steps_report_moving=1,
                           num_steps_eval_static=1,
                           num_steps_checkpoint=20)

        def plot_data():
            preds = [
                float(p[0])
                for p in net.predict(test_dataset, no_targets=False)
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

    # Experiment illustrating DataAdapter.adapt_input.
    #
    # Scenario: the raw dataset contains 1-D signals (time series). Feature
    # extraction is performed by SignalAdapter, which computes the mean and
    # standard deviation of each signal, reducing the 50-sample input to a
    # 2-element feature vector.  The network is then trained to predict the
    # target from those two features.
    #
    # The experiment highlights two usage modes:
    #   (a) On-the-fly extraction: pass the raw dataset directly to fit().
    #       NeuralNet applies adapt_input automatically in the data loader.
    #   (b) Pre-computed extraction: call load_or_create_preprocessed_dataset()
    #       to materialise and cache the feature vectors on disk, then pass the
    #       cached dataset to fit().  Useful when adapt_input is expensive.
    def experiment_1006(l_args):

        # ------------------------------------------------------------------
        # Dataset: each item is a (signal, target) pair.
        #   signal  : Tensor of shape (50,)  — a noisy sinusoid
        #   target  : Tensor of shape (1,)   — amplitude of the sinusoid
        # ------------------------------------------------------------------
        class RawSignalDataset(Dataset):

            def __init__(self, num_examples, seed=0):
                rng = torch.Generator()
                rng.manual_seed(seed)
                t = torch.linspace(0, 2 * np.pi, 50)
                # Random amplitude in [1, 5]
                amplitudes = 1 + 4 * torch.rand(num_examples, 1, generator=rng)
                signals = amplitudes * torch.sin(t).unsqueeze(0)
                signals += 0.2 * torch.randn(num_examples, 50, generator=rng)
                self.signals = signals  # (N, 50)
                self.targets = amplitudes  # (N, 1)

            def __len__(self):
                return len(self.signals)

            def __getitem__(self, idx):
                return self.signals[idx], self.targets[idx]

        # ------------------------------------------------------------------
        # DataAdapter: compress each 50-sample signal to [mean, std].
        # ------------------------------------------------------------------
        class SignalAdapter(DataAdapter):

            def adapt_input(self, signal: torch.Tensor, spec) -> torch.Tensor:
                """Return [mean, std] of the signal — shape (2,)."""
                return torch.stack([signal.mean(), signal.std()])

        # ------------------------------------------------------------------
        # Network: takes 2 extracted features, predicts amplitude.
        # ------------------------------------------------------------------
        class SignalNet(NeuralNet):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fc1 = nn.Linear(2, 16)
                self.fc2 = nn.Linear(16, 1)
                self.initialize()

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        f_loss = lambda pred, target: (pred - target).pow(2).mean(dim=1)

        torch.manual_seed(0)
        np.random.seed(0)

        dataset_train = RawSignalDataset(2000, seed=0)
        dataset_val = RawSignalDataset(500, seed=1)

        # ------------------------------------------------------------------
        # Mode (a): on-the-fly extraction.
        # The adapter is set on the network; make_data_loader and
        # fit_normalizer_if_needed apply adapt_input automatically.
        # ------------------------------------------------------------------
        print("Mode (a): on-the-fly feature extraction")
        net_a = SignalNet(data_adapter=SignalAdapter(), normalizer="both")
        hist_a = net_a.fit(
            dataset_train,
            f_loss,
            optimizer=torch.optim.Adam(net_a.parameters(), lr=1e-3),
            dataset_val=dataset_val,
            num_epochs=30,
            batch_size=64,
        )
        print("  val loss:", net_a.evaluate(dataset_val, 64, f_loss)["loss"])

        # ------------------------------------------------------------------
        # Mode (b): pre-computed extraction with caching.
        # load_or_create_preprocessed_dataset materialises all feature vectors
        # once and saves them to disk.  Subsequent runs load from disk and skip
        # extraction entirely.
        # ------------------------------------------------------------------
        print("Mode (b): pre-computed feature extraction with caching")
        cache_dir = tempfile.mkdtemp(prefix="gsim_adapter_")
        net_b = SignalNet(data_adapter=SignalAdapter(), normalizer="both")

        adapted_train = net_b.load_or_create_preprocessed_dataset(
            dataset_train,
            path=os.path.join(cache_dir, "train_adapted.pk"),
        )
        adapted_val = net_b.load_or_create_preprocessed_dataset(
            dataset_val,
            path=os.path.join(cache_dir, "val_adapted.pk"),
        )

        # The adapted datasets already have preprocessed=True, so the adapter
        # will receive spec.input_already_preprocessed=True.
        hist_b = net_b.fit(
            adapted_train,
            f_loss,
            optimizer=torch.optim.Adam(net_b.parameters(), lr=1e-3),
            dataset_val=adapted_val,
            num_epochs=30,
            batch_size=64,
        )
        print("  val loss:", net_b.evaluate(adapted_val, 64, f_loss)["loss"])

        # ------------------------------------------------------------------
        # Mode (b) with callback: the raw dataset is only instantiated if the
        # cached file does not yet exist.
        # ------------------------------------------------------------------
        net_b2 = SignalNet(data_adapter=SignalAdapter(), normalizer="both")
        adapted_train2 = net_b2.load_or_create_preprocessed_dataset(
            lambda: RawSignalDataset(2000, seed=0),  # callback form
            path=os.path.join(cache_dir, "train_adapted.pk"),
        )
        # File already exists from above, so RawSignalDataset is never built.
        assert adapted_train2.preprocessed is True

        # ------------------------------------------------------------------
        # Plot training curves for both modes
        # ------------------------------------------------------------------
        l_G = net_a.plot_training_history(hist_a)
        if l_G:
            l_G[0].l_subplots[0].title = "On-the-fly extraction"

        l_G_b = net_b.plot_training_history(hist_b)
        if l_G_b:
            l_G_b[0].l_subplots[0].title = "Pre-computed extraction"

        return l_G + l_G_b

    # Experiment to illustrate live plotting
    # WIP: improve the network or make the function simpler.
    def experiment_1007(l_args):

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
                                     num_steps_report_moving=128,
                                     keep_best_val_weights=True,
                                     static_max_hci=0.01,
                                     live_plot=True)

        return net.plot_training_history(d_training_history)

    # Experiment illustrating self-supervised learning via DataAdapter.
    #
    # Scenario: clean 1-D sinusoidal signals are stored without labels.  A
    # DenoisingAdapter synthesises (noisy, clean) training pairs on the fly
    # inside adapt_input, so the raw dataset can be declared no_targets=True
    # and fit() requires no changes.  At inference the adapter skips pair
    # formation and the network denoises user-supplied noisy signals.
    #
    # Key DataAdapter API illustrated:
    #   - adapt_input   : synthesises pairs at training time, identity at inference
    #   - get_no_targets: returns False during training so collate_fn sees
    #                     (inputs, targets); returns True at inference
    def experiment_1008(l_args):

        noise_std = 0.5
        signal_len = 50
        n_train = 2000
        n_val = 500

        # ------------------------------------------------------------------
        # Raw dataset: clean sinusoids, NO targets.
        # Each item is a Tensor of shape (signal_len,).
        # ------------------------------------------------------------------
        class CleanSignalDataset(Dataset):

            def __init__(self, n, seed=0):
                rng = torch.Generator()
                rng.manual_seed(seed)
                t = torch.linspace(0, 2 * np.pi, signal_len)
                amplitudes = 1 + 3 * torch.rand(n, generator=rng)
                freqs = 1 + 2 * torch.rand(n, generator=rng)
                self.signals = amplitudes.unsqueeze(1) * torch.sin(
                    freqs.unsqueeze(1) * t.unsqueeze(0))  # (n, signal_len)

            def __len__(self):
                return len(self.signals)

            def __getitem__(self, idx):
                return self.signals[idx]  # plain tensor, no target

        # ------------------------------------------------------------------
        # DenoisingAdapter: synthesises (noisy, clean) pairs at training time.
        # ------------------------------------------------------------------
        class DenoisingAdapter(DataAdapter):

            def adapt_input(self, signal, spec: AdaptationSpec):
                if spec.inference or spec.preprocess_only:
                    # At inference the caller supplies the noisy signal; at
                    # preprocess time we cache clean signals (noise is
                    # stochastic and must not be frozen).
                    return signal
                # Training / evaluation: add noise and form a supervised pair.
                noise = noise_std * torch.randn_like(signal)
                return signal + noise, signal  # (noisy, clean)

            def get_no_targets(self, inner_dataset_has_no_targets, spec):
                # The adapted dataset has targets during training/eval but
                # not at inference or preprocess time.
                if spec.inference or spec.preprocess_only:
                    return True
                return False

        # ------------------------------------------------------------------
        # Network: MLP 50 -> 128 -> 128 -> 50.
        # ------------------------------------------------------------------
        class DenoisingNet(NeuralNet):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fc1 = nn.Linear(signal_len, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, signal_len)
                self.initialize()

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        f_loss = lambda pred, clean: (pred - clean).pow(2).mean(dim=1)

        torch.manual_seed(0)
        np.random.seed(0)

        dataset_train = CleanSignalDataset(n_train, seed=0)
        dataset_val = CleanSignalDataset(n_val, seed=1)

        # ------------------------------------------------------------------
        # Training.  fit() is called with no_targets=True because the raw
        # dataset has no labels.  DenoisingAdapter.adapt_input synthesises
        # pairs on the fly; get_no_targets signals this to the pipeline.
        # ------------------------------------------------------------------
        net = DenoisingNet(data_adapter=DenoisingAdapter(), normalizer="both")
        hist = net.fit(
            dataset_train,
            f_loss,
            optimizer=torch.optim.Adam(net.parameters(), lr=1e-3),
            dataset_val=dataset_val,
            num_epochs=30,
            batch_size=64,
            no_targets=True,
        )

        # ------------------------------------------------------------------
        # Inference.  Supply noisy signals directly; adapt_input returns them
        # unchanged (spec.inference=True) and the network denoises them.
        # ------------------------------------------------------------------
        rng_inf = torch.Generator()
        rng_inf.manual_seed(42)
        clean_test = dataset_val[0]  # shape (50,)
        noisy_test = clean_test + noise_std * torch.randn(signal_len,
                                                          generator=rng_inf)

        recon = net.predict(noisy_test.unsqueeze(0),
                            output_class=torch.Tensor).squeeze(0)

        mse_noisy = (noisy_test - clean_test).pow(2).mean().item()
        mse_recon = (recon.cpu() - clean_test).pow(2).mean().item()
        print(f"MSE (noisy input vs clean): {mse_noisy:.4f}")
        print(f"MSE (reconstruction vs clean): {mse_recon:.4f}")
        assert mse_recon < mse_noisy, \
            "Reconstruction should be closer to clean than the noisy input."

        # ------------------------------------------------------------------
        # Plots
        # ------------------------------------------------------------------
        l_G = net.plot_training_history(hist)
        if l_G:
            l_G[0].l_subplots[0].title = "Self-supervised denoising — training"

        t = np.linspace(0, 2 * np.pi, signal_len)
        G_signals = GFigure(
            xaxis=t,
            yaxis=clean_test.numpy(),
            legend="clean",
            styles="-",
            xlabel="t",
            ylabel="amplitude",
            title="Denoising: clean / noisy / reconstruction",
        )
        G_signals.add_curve(xaxis=t,
                            yaxis=noisy_test.numpy(),
                            legend="noisy",
                            styles="--")
        G_signals.add_curve(xaxis=t,
                            yaxis=recon.detach().numpy(),
                            legend="reconstruction",
                            styles="-.")

        return l_G + [G_signals]
