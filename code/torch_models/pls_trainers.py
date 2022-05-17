import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

import time
from typing import Dict

from torch_models.pls_ae import PLS_AE
from torch_models.pls_ccm import PLS_CCM


class PLS_AETrainer:
    def __init__(self,
                 n_epochs: int,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler=None,
                 recov_x_coef: float = 1,
                 recov_y_coef: float = 1,
                 consist_coef: float = 1,
                 additional_coef: float = 1,
                 log_every_n_steps: int = 1,
                 dir_to_save_model: str = "./",
                 verbose: bool = True):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.n_epochs = n_epochs
        self.recov_x_coef = recov_x_coef
        self.recov_y_coef = recov_y_coef
        self.consist_coef = consist_coef
        self.additional_coef = additional_coef
        self.log_every_n_step = log_every_n_steps
        self.dir_to_save_model = dir_to_save_model
        self.verbose = verbose

    @staticmethod
    def plot_intermediate_results(epoch_history: np.array,
                                  train_losses: np.array,
                                  val_losses: np.array):
        _, losses_count = epoch_history.shape
        assert losses_count >= 3

        losses_titles = ['consistency', 'recov_x', 'recov_y']

        def get_loss_title(ind):
            return f"additional-{ind - 2}" if ind >= 3 else losses_titles[i]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
        clear_output(True)

        for i in range(losses_count):
            ax[0].plot(epoch_history[:, i], label=f'train {get_loss_title(i)} loss')

        for i in range(train_losses.shape[-1]):
            ax[1].plot(train_losses[:, i], label=f'{get_loss_title(i)} train history')
            ax[1].plot(val_losses[:, i], label=f'{get_loss_title(i)} valid history')

        for j in (0, 1):
            if j == 1:
                ax[j].set_title('All losses')
                ax[j].set_xlabel('Epoch')

            else:
                ax[j].set_title('Train losses')
                ax[j].set_xlabel('Batch')

            ax[j].set_ylabel('Loss')
            ax[j].set_yscale('log')
            ax[j].legend()

        plt.show()

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def fit_epoch(self,
                  model: PLS_AE,
                  train_loader: DataLoader,
                  train_losses: np.array,
                  val_losses: np.array,
                  plot_results: bool = True):
        model.train()
        has_additional_loss = hasattr(model, "calc_additional_loss")
        consist_loss, extra_loss, recov_x_loss, recov_y_loss = 0, 0, 0, 0
        history = np.array([])

        for i, (x_batch, y_batch) in enumerate(train_loader):
            self.optimizer.zero_grad()

            x_batch = x_batch.to(model.device)
            y_batch = y_batch.to(model.device)
            batch_size = x_batch.shape[0]

            x_latent, y_latent = model.get_latent(x_batch, y_batch)
            x_recov, y_recov = model.decode(x_latent, y_latent)

            batch_consist_loss = model.calc_consistency_loss(x_latent, y_latent) * self.consist_coef
            batch_recov_x_loss = model.calc_recovering_loss(x_batch, x_recov) * self.recov_x_coef
            batch_recov_y_loss = model.calc_recovering_loss(y_batch, y_recov) * self.recov_y_coef
            batch_additional_loss = 0

            loss = batch_consist_loss + batch_recov_x_loss + batch_recov_y_loss

            if has_additional_loss:
                batch_additional_loss = model.calc_additional_loss(x_batch, y_batch) * \
                                        self.additional_coef
                loss += batch_additional_loss

            loss.backward()
            self.optimizer.step()

            consist_loss += batch_consist_loss.item() / len(train_loader) / batch_size
            recov_x_loss += batch_recov_x_loss.item() / len(train_loader) / batch_size
            recov_y_loss += batch_recov_y_loss.item() / len(train_loader) / batch_size

            batch_losses = [batch_consist_loss.item(),
                            batch_recov_x_loss.item(),
                            batch_recov_y_loss.item()]

            if has_additional_loss:
                extra_loss += batch_additional_loss.item() / len(train_loader) / batch_size
                batch_losses.append(batch_additional_loss.item())

            if history.size > 0:
                history = np.append(history,
                                    np.array([batch_losses]),
                                    axis=0)
            else:
                history = np.array([batch_losses])

            if plot_results and (i + 1) % 5 == 0:
                PLS_AETrainer.plot_intermediate_results(history,
                                                        train_losses,
                                                        val_losses)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if has_additional_loss:
            return consist_loss, recov_x_loss, recov_y_loss, extra_loss
        else:
            return consist_loss, recov_x_loss, recov_y_loss

    def eval_epoch(self,
                   model: PLS_AE,
                   val_loader: DataLoader):
        model.eval()
        consist_loss, extra_loss, recov_x_loss, recov_y_loss = 0, 0, 0, 0
        has_additional_loss = hasattr(model, "calc_additional_loss")

        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(model.device)
            y_batch = y_batch.to(model.device)
            batch_size = x_batch.shape[0]

            with torch.no_grad():
                x_latent, y_latent = model.get_latent(x_batch, y_batch)
                x_recov, y_recov = model.decode(x_latent, y_latent)

            batch_consist_loss = model.calc_consistency_loss(x_latent, y_latent) * self.consist_coef
            batch_recov_x_loss = model.calc_recovering_loss(x_batch, x_recov) * self.recov_x_coef
            batch_recov_y_loss = model.calc_recovering_loss(y_batch, y_recov) * self.recov_y_coef
            batch_additional_loss = 0

            loss = batch_consist_loss + batch_recov_x_loss + batch_recov_y_loss

            if has_additional_loss:
                batch_additional_loss = model.calc_additional_loss(x_batch, y_batch) * self.additional_coef
                loss += batch_additional_loss

            consist_loss += batch_consist_loss.item() / len(val_loader) / batch_size
            recov_x_loss += batch_recov_x_loss.item() / len(val_loader) / batch_size
            recov_y_loss += batch_recov_y_loss.item() / len(val_loader) / batch_size

            if has_additional_loss:
                extra_loss += batch_additional_loss.item() / len(val_loader) / batch_size

        if has_additional_loss:
            return consist_loss, recov_x_loss, recov_y_loss, extra_loss
        else:
            return consist_loss, recov_x_loss, recov_y_loss

    def fit(self,
            model: PLS_AE,
            train_loader: DataLoader,
            val_loader: DataLoader,
            model_filename: str = "pls_ae",
            plot_results: bool = True):
        train_losses, val_losses = np.array([]), np.array([])
        min_losses_sum = float('inf')

        for epoch in range(self.n_epochs):
            start_time = time.time()

            ep_train_losses = self.fit_epoch(model, train_loader, train_losses,
                                             val_losses, plot_results)
            ep_val_losses = self.eval_epoch(model, val_loader)

            end_time = time.time()
            epoch_mins, epoch_secs = PLS_AETrainer.epoch_time(start_time, end_time)

            if min_losses_sum > sum(ep_val_losses):
                min_losses_sum = sum(ep_val_losses)
                model.save(self.dir_to_save_model + f"{model_filename}")

            if train_losses.size > 0:
                train_losses = np.append(train_losses, np.array([ep_train_losses]), axis=0)
                val_losses = np.append(val_losses, np.array([ep_val_losses]), axis=0)
            else:
                train_losses = np.array([ep_train_losses])
                val_losses = np.array([ep_val_losses])

            if self.verbose and (epoch + 1) % self.log_every_n_step:
                print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {sum(ep_train_losses):.3f}')
                print(f'\tVal. Loss: {sum(ep_val_losses):.3f}')

        return train_losses, val_losses


class PLS_CCMTrainer(PLS_AETrainer):
    def fit_epoch(self,
                  model: PLS_CCM,
                  source: torch.Tensor,
                  target: torch.Tensor,
                  train_loader: torch.utils.data.DataLoader,
                  train_losses: np.array,
                  val_losses: np.array,
                  plot_results: bool = True):
        model.train()

        ep_losses = super().fit_epoch(model, train_loader, train_losses, val_losses,
                                      plot_results)
        ep_losses = list(ep_losses)

        source = source.to(model.device)
        target = target.to(model.device)

        self.optimizer.zero_grad()
        ccm_loss = model.calc_ccm_loss(source, target)
        ccm_loss.backward()
        self.optimizer.step()

        ep_losses.append(ccm_loss.item())

        return ep_losses

    def eval_epoch(self,
                   model: PLS_CCM,
                   source: torch.Tensor,
                   target: torch.Tensor,
                   val_loader: torch.utils.data.DataLoader):
        model.eval()

        ep_losses = super().eval_epoch(model, val_loader)
        ep_losses = list(ep_losses)

        source = source.to(model.device)
        target = target.to(model.device)

        with torch.no_grad():
            ccm_loss = model.calc_ccm_loss(source, target)

        ep_losses.append(ccm_loss.item())

        return ep_losses

    def fit(self,
            model: PLS_CCM,
            source: Dict[str, torch.Tensor],
            target: Dict[str, torch.Tensor],
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            model_filename: str = "pls_ccm",
            plot_results: bool = True):
        train_losses, val_losses = np.array([]), np.array([])
        min_losses_sum = float('inf')

        for epoch in range(self.n_epochs):
            start_time = time.time()

            ep_train_losses = self.fit_epoch(model,
                                             source.get('train'), target.get('train'),
                                             train_loader, train_losses,
                                             val_losses, plot_results)
            ep_val_losses = self.eval_epoch(model,
                                            source.get('val'), target.get('val'),
                                            val_loader)

            end_time = time.time()
            epoch_mins, epoch_secs = PLS_AETrainer.epoch_time(start_time, end_time)

            if min_losses_sum > sum(ep_val_losses):
                min_losses_sum = sum(ep_val_losses)
                model.save(self.dir_to_save_model + f"{model_filename}")

            if train_losses.size > 0:
                train_losses = np.append(train_losses, np.array([ep_train_losses]), axis=0)
                val_losses = np.append(val_losses, np.array([ep_val_losses]), axis=0)
            else:
                train_losses = np.array([ep_train_losses])
                val_losses = np.array([ep_val_losses])

            if self.verbose and (epoch + 1) % self.log_every_n_step:
                print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {sum(ep_train_losses):.3f}')
                print(f'\tVal. Loss: {sum(ep_val_losses):.3f}')

        return train_losses, val_losses
