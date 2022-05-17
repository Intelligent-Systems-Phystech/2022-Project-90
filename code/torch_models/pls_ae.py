import torch
import torch.nn.functional as F

from typing import Optional
from torch_models.autoencoders import AutoEncoder
from torch_models.prediction_models import PredictionModelInterface


class PLS_AE:
    def __init__(self,
                 x_autoenc: AutoEncoder,
                 y_autoenc: AutoEncoder):
        assert x_autoenc.device == y_autoenc.device

        self.x_autoenc = x_autoenc
        self.y_autoenc = y_autoenc
        self.device = x_autoenc.device

    @property
    def is_training(self):
        assert self.x_autoenc.training == self.y_autoenc.training

        return self.x_autoenc.training

    def train(self):
        self.x_autoenc.train()
        self.y_autoenc.train()

    def eval(self):
        self.x_autoenc.eval()
        self.y_autoenc.eval()

    def _call_autoencoder_method(self,
                                 method_name: str,
                                 input: Optional[torch.Tensor] = None,
                                 target: Optional[torch.Tensor] = None):
        if not input is None and not target is None:
            assert input.shape[0] == target.shape[0]

        output = []
        x_method = getattr(self.x_autoenc, method_name)
        y_method = getattr(self.y_autoenc, method_name)

        if not input is None:
            output.append(x_method(input))

        if not target is None:
            output.append(y_method(target))

        if len(output) == 0:
            return None
        elif len(output) == 1:
            return output[0]
        else:
            return output[0], output[1]

    def encode(self,
               input: Optional[torch.Tensor] = None,
               target: Optional[torch.Tensor] = None):
        return self._call_autoencoder_method("encode", input, target)

    def get_latent(self,
                   input: Optional[torch.Tensor] = None,
                   target: Optional[torch.Tensor] = None):
        return self._call_autoencoder_method("get_latent", input, target)

    def decode(self,
               x_latent: Optional[torch.Tensor] = None,
               y_latent: Optional[torch.Tensor] = None):
        return self._call_autoencoder_method("decode", x_latent, y_latent)

    def predict(self,
                input: torch.Tensor,
                target: torch.Tensor,
                model: PredictionModelInterface):
        latent_input, latent_target = self.get_latent(input, target)
        preds = model.predict(latent_input, latent_target)

        return self.decode(y_latent=preds).detach()

    def save(self, filename: str = "pls_ae"):
        torch.save(self.x_autoenc.state_dict(), f"{filename}-x_autoenc")
        torch.save(self.y_autoenc.state_dict(), f"{filename}-y_autoenc")

    def load(self, filename: str = "pls_ae"):
        self.x_autoenc.load_state_dict(torch.load(f"{filename}-x_autoenc"))
        self.y_autoenc.load_state_dict(torch.load(f"{filename}-y_autoenc"))

    @staticmethod
    def calc_consistency_loss(x_latent: torch.Tensor, y_latent: torch.Tensor):
        assert x_latent.size() == y_latent.size()
        assert x_latent.ndim == 2

        # batch_size = x_hidden.shape[0]
        x_latent_centered = x_latent - x_latent.mean(dim=1, keepdim=True)
        y_latent_centered = y_latent - y_latent.mean(dim=1, keepdim=True)

        cov_mat = x_latent_centered.T @ y_latent_centered
        features_cov = cov_mat.diag()
        trace = features_cov.sum()

        return 1.0 / (1.0 + torch.square(trace) / 10.0)

    @staticmethod
    def calc_recovering_loss(input: torch.Tensor, recov_input: torch.Tensor):
        return F.mse_loss(input, recov_input, reduction='mean')


class PLS_NormalVAE(PLS_AE):
    @staticmethod
    def asymmetrical_kl_loss(first_mu: torch.Tensor, first_logsigma: torch.Tensor,
                             second_mu: torch.Tensor, second_logsigma: torch.Tensor):
        first_sigma_sq = torch.exp(2 * first_logsigma)
        second_sigma_sq = torch.exp(2 * second_logsigma)

        kl_loss_summand = first_sigma_sq + torch.square(first_mu - second_mu)
        kl_loss_summand /= (2 * second_sigma_sq)

        return first_logsigma - second_logsigma - 0.5 + kl_loss_summand

    @staticmethod
    def kl_loss(first_mu: torch.Tensor, first_logsigma: torch.Tensor,
                second_mu: torch.Tensor, second_logsigma: torch.Tensor):
        return PLS_NormalVAE.asymmetrical_kl_loss(first_mu, first_logsigma,
                                                  second_mu, second_logsigma) + \
               PLS_NormalVAE.asymmetrical_kl_loss(second_mu, second_logsigma,
                                                  first_mu, first_logsigma)

    def calc_additional_loss(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        x_params, y_params = self.encode(x_batch, y_batch)

        x_mu, x_logsigma = x_params.split(split_size=1, dim=1)
        y_mu, y_logsigma = y_params.split(split_size=1, dim=1)

        return PLS_NormalVAE.kl_loss(x_mu, x_logsigma, y_mu, y_logsigma)
