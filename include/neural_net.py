import os
import pickle
import tempfile
import numpy as np
import torch
from torch import nn
from gsim.gfigure import GFigure
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split


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

    def __init__(self, *args, nn_folder=None, **kwargs):
        """
        
        Args: 

            `nn_folder`: if not None, the weights of the network are loaded from
            this folder. When training, if validation data is provided, the
            weights that minimize the validation loss are saved in this folder
            together with training metrics. If validation data is not provided,
            the weights that minimize the training loss are saved.
        
        
        """

        super().__init__(*args, **kwargs)
        self.device_type = ("cuda" if torch.cuda.is_available() else "mps"
                            if torch.backends.mps.is_available() else "cpu")
        print(f"Using {self.device_type} device")
        self.nn_folder = nn_folder

    def initialize(self):
        self._initialized = True

        if self.nn_folder is not None and os.path.exists(
                self.weight_file_path):
            self.load_weights_from_path(self.weight_file_path)

    def _assert_initialized(self):
        assert self._initialized, "The network has not been initialized. A subclass of NeuralNet must call self.initialize() at the end of its constructor."

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
        for m_feat_batch, v_targets_batch in iterator:
            m_feat_batch = m_feat_batch.float().to(self.device_type)
            v_targets_batch = v_targets_batch.float().to(self.device_type)

            if optimizer:
                v_targets_batch_pred = self(m_feat_batch.float())
                loss = f_loss(v_targets_batch_pred.float(),
                              v_targets_batch.float())
                torch.mean(loss).backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    v_targets_batch_pred = self(m_feat_batch.float())
                    loss = f_loss(v_targets_batch_pred.float(),
                                  v_targets_batch.float())

            l_loss_this_epoch.append(loss.cpu().detach().numpy())

        return np.mean(np.concatenate(
            l_loss_this_epoch, axis=0)) if len(l_loss_this_epoch) else np.nan

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
        checkpoint = torch.load(path, weights_only=True)
        self.load_state_dict(checkpoint["weights"])
        #load_optimizer_state(initial_optimizer_state_file)

    def save_weights_to_path(self, path):
        torch.save({"weights": self.state_dict()}, path)

    def fit(self,
            dataset: Dataset,
            optimizer,
            num_epochs,
            f_loss,
            batch_size=32,
            batch_size_eval=None,
            shuffle=True,
            val_split=0.0,
            best_weights=True,
            patience=None,
            lr_patience=None,
            lr_decay=None,
            llc=LossLandscapeConfig()):
        """ 
        Args:

         `f_loss`: f_loss(pred,targets) where pred.shape[0] =
            targets.shape[0] = batch_size is a vector of length batch_size.

          `batch_size_eval` is the batch size used to evaluate the loss. If
          None, `batch_size` is used also for evaluation.

          `llc`: instance of LossLandscapeConfig.
        

        Returns a dict with keys and values given by:
         
          'train_loss_me': list of length num_epochs with the values of the
          moving estimate of the training loss at each epoch. The moving
          estimate is obtained by averaging the training loss after each batch
          update. Thus, it is an average of loss values obtained for different
          network weights.

          'train_loss': list of length num_epochs with the values of the
          training loss computed at the end of each epoch.

          'val_loss': same as before but for the validation loss. Only if
          val_split != 0. 

          'lr': list of length num_epochs with the learning rate at each epoch.

          'l_loss_landscapes': list of figures with loss landscapes.

        If `best_weights` is False, then the weights of the network at the end
        of the execution of this function equal the weights at the last epoch.
        Otherwise: 
         
          - if val_split != 0, then the weights of the epoch with the
        best validation loss are returned; 
        
          - if val_split == 0, then the weights of the epoch with the
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
                if val_split != 0:
                    best_train_loss_file = get_temp_file_path()
                    best_val_loss_file = self.weight_file_path
                else:
                    best_train_loss_file = self.weight_file_path
                    best_val_loss_file = get_temp_file_path()
            return best_train_loss_file, best_val_loss_file

        def save_optimizer_state(path):
            torch.save({"state": optimizer.state_dict()}, path)

        def load_optimizer_state(path):
            checkpoint = torch.load(path, weights_only=True)
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

        batch_size_eval = batch_size_eval if batch_size_eval else batch_size

        self.to(device=self.device_type)

        dataset_train, dataset_val = random_split(dataset,
                                                  [1 - val_split, val_split])

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=batch_size,
                                      shuffle=shuffle)
        dataloader_train_eval = DataLoader(dataset_train,
                                           batch_size=batch_size_eval,
                                           shuffle=shuffle)
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
            loss_val_this_epoch = self._run_epoch(dataloader_val, f_loss)

            print(
                f"Epoch {ind_epoch-ind_epoch_start}/{num_epochs}: train loss me = {loss_train_me_this_epoch:.2f}, train loss = {loss_train_this_epoch:.2f}, val loss = {loss_val_this_epoch:.2f}, lr = {optimizer.param_groups[0]['lr']:.2e}"
            )

            l_loss_train_me.append(loss_train_me_this_epoch)
            l_loss_train.append(loss_train_this_epoch)
            l_loss_val.append(loss_val_this_epoch)
            l_lr.append(optimizer.param_groups[0]["lr"])

            ind_epoch_best_loss_val = np.argmin(l_loss_val)
            if ind_epoch_best_loss_val == ind_epoch:
                self.save_weights_to_path(best_val_loss_file)

            if patience:
                if ind_epoch_best_loss_val + patience < ind_epoch:
                    print("Patience expired.")
                    break

            if lr_patience:
                if loss_train_this_epoch < best_train_loss:
                    best_train_loss = loss_train_this_epoch
                    self.save_weights_to_path(best_train_loss_file)
                    num_epochs_left_to_expire_lr_patience = lr_patience
                else:
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
            if val_split != 0:
                self.load_weights_from_path(best_val_loss_file)
            else:
                self.load_weights_from_path(best_train_loss_file)

        return d_hist

    @staticmethod
    def plot_training_history(d_metrics_train):
        G = GFigure()
        G.next_subplot(xlabel="Epoch", ylabel="Loss")
        for key in d_metrics_train.keys():
            if key not in ["lr", "l_loss_landscapes", "ind_epoch"]:
                G.add_curve(yaxis=d_metrics_train[key], legend=key)
        G.next_subplot(xlabel="Epoch", ylabel="Learning rate")
        G.add_curve(yaxis=d_metrics_train["lr"], legend="Learning rate")
        return [G] + d_metrics_train["l_loss_landscapes"]
