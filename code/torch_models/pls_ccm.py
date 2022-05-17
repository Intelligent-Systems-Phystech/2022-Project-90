import torch
import torch.nn.functional as F

from torch_models.autoencoders import AutoEncoder
from torch_models.pls_ae import PLS_AE
from torch_models.ccm_metrics import AbstractCCMMetric
from typing import Dict


class PLS_CCM(PLS_AE):
    def __init__(self,
                 x_autoenc: AutoEncoder,
                 y_autoenc: AutoEncoder,
                 source: Dict[str, torch.Tensor],
                 target: Dict[str, torch.Tensor],
                 ccm_measure: AbstractCCMMetric):
        # input = [n_samples, n_features]
        # target = [n_samples, n_targets]
        super().__init__(x_autoenc, y_autoenc)

        self.ccm_measure = ccm_measure
        self.ccm_train = ccm_measure(source.get('train'), target.get('train'))
        self.ccm_valid = ccm_measure(source.get('val'), target.get('val'))

    def calc_ccm_loss(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        x_latent, y_latent = self.get_latent(x_batch, y_batch)
        latent_ccm = self.ccm_measure(x_latent, y_latent)

        if self.is_training:
            return F.mse_loss(latent_ccm, self.ccm_train)
        else:
            return F.mse_loss(latent_ccm, self.ccm_valid)
