import os
import pickle
import tempfile
import numpy as np
import torch
from torch import nn
from gsim import GFigure
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, Subset


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


class NeuralNet(nn.Module):

    _initialized = False

    def __init__(self, *args, nn_folder=None, device_type=None, **kwargs):
        """
        
        Args: 

            `nn_folder`: if not None, the weights of the network are loaded from
            this folder. When training, if validation data is provided, the
            weights that minimize the validation loss are saved in this folder
            together with training metrics. If validation data is not provided,
            the weights that minimize the training loss are saved.
        
        
        """

        super().__init__(*args, **kwargs)
        if device_type is not None:
            self.device_type = device_type
        else:
            self.device_type = (
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using {self.device_type} device")
        if nn_folder is None:
            print()
            print(
                "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"
            )
            print(
                "*   WARNING: The weights of the network are not being saved.")
            print(
                "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"
            )
            print()
        self.nn_folder = nn_folder

    def initialize(self):
        self._initialized = True

        if self.nn_folder is not None and os.path.exists(
                self.weight_file_path):
            self.load_weights_from_path(self.weight_file_path)

    def _assert_initialized(self):
        assert self._initialized, "The network has not been initialized. A subclass of NeuralNet must call self.initialize() at the end of its constructor."

    def _get_loss(self, data, f_loss):
        assert f_loss is not None, "f_loss must be provided unless you override _get_loss."
        m_feat_batch, v_targets_batch = data
        m_feat_batch = m_feat_batch.float().to(self.device_type)
        v_targets_batch = v_targets_batch.float().to(self.device_type)

        v_targets_batch_pred = self(m_feat_batch.float())
        #assert v_targets_batch_pred.shape == v_targets_batch.shape
        loss = f_loss(v_targets_batch_pred.float(), v_targets_batch.float())

        assert loss.shape[0] == m_feat_batch.shape[
            0] and loss.ndim == 1, "f_loss must return a vector of length batch_size."
        return loss

    def _run_epoch(self, dataloader, f_loss, optimizer=None):
        """
        Args:

            `optimizer`: if None, the weights are not updated. This is useful to
            evaluate the loss. 

            `f_loss`: f_loss(pred,targets) where pred.shape[0] =
            targets.shape[0] = batch_size is a vector of length batch_size.
        
        """

        l_loss_this_epoch = []
        iterator = tqdm(dataloader) if optimizer else dataloader
        for data in iterator:

            if optimizer:
                loss = self._get_loss(data, f_loss)
                torch.mean(loss).backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    loss = self._get_loss(data, f_loss)

            l_loss_this_epoch.append(loss.detach())

        return torch.cat(l_loss_this_epoch).mean().cpu().numpy() if len(
            l_loss_this_epoch) else np.nan

    def evaluate(self, dataset, batch_size, f_loss):
        """
        Returns a dict with key-values:

        "loss": the result of averaging `f_loss` across `dataset`.
        """
        self._assert_initialized()

        dataloader = DataLoader(dataset, batch_size=batch_size)
        self.eval()
        loss = self._run_epoch(dataloader, f_loss=f_loss)
        return {"loss": loss}

    def predict(self, dataset, batch_size=32):
        """
        Returns a tensor with the result of performing a forward pass on the
        points in `dataset`. 
        """
        self._assert_initialized()

        dataloader = DataLoader(dataset, batch_size=batch_size)
        l_out = []
        self.eval()
        for m_feat_batch, _ in dataloader:
            m_targets_batch_pred = self(m_feat_batch.float())
            l_out.append(m_targets_batch_pred.detach().numpy())

        return np.concatenate(l_out, axis=0)

    @property
    def weight_file_path(self):
        assert self.nn_folder is not None
        return os.path.join(self.nn_folder, "weights.pth")

    @property
    def hist_path(self):
        assert self.nn_folder is not None
        return os.path.join(self.nn_folder, "hist.pk")

    def load_weights_from_path(self, path):
        checkpoint = torch.load(path,
                                weights_only=True,
                                map_location=self.device_type)
        self.load_state_dict(checkpoint["weights"])
        self.to(device=self.device_type)
        #load_optimizer_state(initial_optimizer_state_file)

    def save_weights_to_path(self, path):
        torch.save({"weights": self.state_dict()}, path)

    def fit(self,
            dataset: Dataset,
            optimizer,
            num_epochs,
            f_loss=None,
            dataset_val=None,
            batch_size=32,
            batch_size_eval=None,
            shuffle=True,
            val_split=0.0,
            best_weights=True,
            patience=None,
            lr_patience=None,
            lr_decay=.8,
            llc=LossLandscapeConfig()):
        """ 
        Args:

         `f_loss`: f_loss(pred,targets) where pred.shape[0] =
            targets.shape[0] = batch_size is a vector of length batch_size.

          `batch_size_eval` is the batch size used to evaluate the loss. If
          None, `batch_size` is used also for evaluation.

          `llc`: instance of LossLandscapeConfig.
        
          At most one of `val_split` and `dataset_val` can be provided. If one
          is provided, we say that `val` is True. 

        Returns a dict with keys and values given by:
         
          'train_loss_me': list of length num_epochs with the values of the
          moving estimate of the training loss at each epoch. The moving
          estimate is obtained by averaging the training loss after each batch
          update. Thus, it is an average of loss values obtained for different
          network weights.

          'train_loss': list of length num_epochs with the values of the
          training loss computed at the end of each epoch.

          'val_loss': same as before but for the validation loss. Only if
          `val` is true. 

          'lr': list of length num_epochs with the learning rate at each epoch.

          'l_loss_landscapes': list of figures with loss landscapes.

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
            for m_feat_batch, targets_batch in dataloader_train:

                # 1. Compute the gradient
                m_feat_batch = m_feat_batch.float().to(self.device_type)
                targets_batch = targets_batch.float().to(self.device_type)

                targets_batch_pred = self(m_feat_batch.float())
                loss = f_loss(targets_batch_pred.float(),
                              targets_batch.float())
                torch.mean(loss).backward()

                # 2. Compute the loss for gradient displacement
                l_loss = []
                for scale in llc.neg_gradient_step_scales:
                    for param in self.parameters():
                        param.data -= scale * param.grad

                    with torch.no_grad():  # Disable gradient computation
                        targets_batch_pred = self(m_feat_batch.float())[:, 0]
                        # loss = f_loss(v_targets_batch.float(),
                        #               v_targets_batch_pred.float())
                        self.train()
                        loss = self._run_epoch(dataloader_train_eval, f_loss)
                        # loss = self.evaluate(dataset_train, batch_size_eval,
                        #                      f_loss)['loss']
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

        def get_weight_file_paths():

            if self.nn_folder is None:
                best_train_loss_file = get_temp_file_path()
                best_val_loss_file = get_temp_file_path()
            else:
                os.makedirs(self.nn_folder, exist_ok=True)
                if val:
                    best_train_loss_file = get_temp_file_path()
                    best_val_loss_file = self.weight_file_path
                else:
                    best_train_loss_file = self.weight_file_path
                    best_val_loss_file = get_temp_file_path()
            return best_train_loss_file, best_val_loss_file

        def save_optimizer_state(path):
            torch.save({"state": optimizer.state_dict()}, path)

        def load_optimizer_state(path):
            checkpoint = torch.load(path,
                                    weights_only=True,
                                    map_location=self.device_type)
            optimizer.load_state_dict(checkpoint["state"])

        def decrease_lr(optimizer, lr_decay):
            """Resets the optimizer state and decreases the learning rate by a factor of `lr_decay`."""

            # Store the lr values
            l_lr = [
                optimizer.param_groups[ind_group]["lr"]
                for ind_group in range(len(optimizer.param_groups))
            ]
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
            else:
                d_hist = {
                    'train_loss_me': [],
                    'train_loss': [],
                    'val_loss': [],
                    "lr": [],
                    "l_loss_landscapes": [],
                    "ind_epoch": 0
                }
            return d_hist

        self._assert_initialized()
        torch.cuda.empty_cache()

        batch_size_eval = batch_size_eval if batch_size_eval else batch_size

        self.to(device=self.device_type)

        assert val_split == 0.0 or dataset_val is None
        if dataset_val is None:
            # The data is deterministically split into training and validation
            # sets so that we can resume training.
            num_examples_val = int(val_split * len(dataset))
            dataset_train = Subset(dataset,
                                   range(len(dataset) - num_examples_val))
            dataset_val = Subset(
                dataset, range(len(dataset) - num_examples_val, len(dataset)))
        else:
            dataset_train = dataset
            num_examples_val = len(dataset_val)
        val = num_examples_val > 0

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=batch_size,
                                      shuffle=shuffle)
        dataloader_train_eval = DataLoader(dataset_train,
                                           batch_size=batch_size_eval,
                                           shuffle=shuffle)
        if val:
            dataloader_val = DataLoader(dataset_val,
                                        batch_size=batch_size,
                                        shuffle=shuffle)

        d_hist = load_hist()
        l_loss_train_me = d_hist['train_loss_me']
        l_loss_train = d_hist['train_loss']
        l_loss_val = d_hist['val_loss']
        l_lr = d_hist['lr']
        l_llplots = d_hist['l_loss_landscapes']  # loss landscape plots
        ind_epoch_start = d_hist['ind_epoch']

        llp_weight_file = get_temp_file_path(
        )  # file to restore the weights when the loss landscape needs to be plotted

        best_train_loss = torch.inf
        #num_epochs_left_to_expire_patience = patience
        num_epochs_left_to_expire_lr_patience = lr_patience
        best_train_loss_file, best_val_loss_file = get_weight_file_paths()

        initial_optimizer_state_file = get_temp_file_path()
        save_optimizer_state(initial_optimizer_state_file)

        for ind_epoch in range(ind_epoch_start, ind_epoch_start + num_epochs):
            self.train()
            loss_train_me_this_epoch = self._run_epoch(dataloader_train,
                                                       f_loss, optimizer)
            loss_train_this_epoch = self._run_epoch(dataloader_train_eval,
                                                    f_loss)
            self.eval()
            loss_val_this_epoch = self._run_epoch(dataloader_val,
                                                  f_loss) if val else np.nan

            print(
                f"Epoch {ind_epoch-ind_epoch_start}/{num_epochs}: train loss me = {loss_train_me_this_epoch:.2f}, train loss = {loss_train_this_epoch:.2f}, val loss = {loss_val_this_epoch:.2f}, lr = {optimizer.param_groups[0]['lr']:.2e}"
            )

            l_loss_train_me.append(loss_train_me_this_epoch)
            l_loss_train.append(loss_train_this_epoch)
            l_loss_val.append(loss_val_this_epoch)
            l_lr.append(optimizer.param_groups[0]["lr"])

            ind_epoch_best_loss_val = np.argmin(
                [v if not np.isnan(v) else np.inf for v in l_loss_val])
            if ind_epoch_best_loss_val == ind_epoch:
                self.save_weights_to_path(best_val_loss_file)

            if patience:
                if np.max([ind_epoch_best_loss_val, ind_epoch_start
                           ]) + patience < ind_epoch:
                    print("Patience expired.")
                    break

            if lr_patience or val:
                # The weights should also be stored when val_split==0 since they
                # need to be returned at the end.
                if loss_train_this_epoch < best_train_loss:
                    best_train_loss = loss_train_this_epoch
                    self.save_weights_to_path(best_train_loss_file)
                    num_epochs_left_to_expire_lr_patience = lr_patience
                else:
                    if lr_patience:
                        num_epochs_left_to_expire_lr_patience -= 1
                        if num_epochs_left_to_expire_lr_patience == 0:
                            decrease_lr(optimizer, lr_decay)
                            self.load_weights_from_path(best_train_loss_file)
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
                "ind_epoch": ind_epoch
            }
            save_hist(d_hist)

        if best_weights and num_epochs > 0:
            if val:
                if os.path.exists(best_val_loss_file):
                    self.load_weights_from_path(best_val_loss_file)
            else:
                if os.path.exists(best_train_loss_file):
                    self.load_weights_from_path(best_train_loss_file)

        return d_hist

    @staticmethod
    def plot_training_history(d_metrics_train, first_epoch_to_plot=0):
        G = GFigure()
        max_y_value = -np.inf
        min_y_value = np.inf
        G.next_subplot(xlabel="Epoch",
                       ylabel="Loss",
                       xlim=(first_epoch_to_plot, None))
        for key in d_metrics_train.keys():
            if key not in ["lr", "l_loss_landscapes", "ind_epoch"]:
                G.add_curve(yaxis=d_metrics_train[key], legend=key)
                if len(d_metrics_train[key]) > first_epoch_to_plot:
                    max_y_value = max(
                        max_y_value,
                        np.max(d_metrics_train[key][first_epoch_to_plot:]))
                    min_y_value = min(
                        min_y_value,
                        np.min(d_metrics_train[key][first_epoch_to_plot:]))
        if max_y_value != -np.inf and min_y_value != np.inf:
            margin = 0.1 * (max_y_value - min_y_value)
            G.l_subplots[-1].ylim = (min_y_value - margin,
                                     max_y_value + margin)
        G.next_subplot(xlabel="Epoch", ylabel="Learning rate", sharex=True)
        G.add_curve(yaxis=d_metrics_train["lr"], legend="Learning rate")
        return [G] + d_metrics_train["l_loss_landscapes"]

    def print_num_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Total number of parameters: {total_params}')


if __name__ == "__main__":
    """
    To run this test code, execute:

    python -m gsim.include.neural_net

    from the root folder of the repository.
    
    """

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
                                 f_loss=f_loss,
                                 llc=LossLandscapeConfig(
                                     epoch_inds=[199, 399],
                                     neg_gradient_step_scales=np.linspace(
                                         -0.2, 0.3, 13)))
    d_metrics = net.evaluate(dataset, batch_size=32, f_loss=f_loss)
    print(d_metrics)
    l_G = net.plot_training_history(d_training_history)
    [G.plot() for G in l_G]
    GFigure.show()
