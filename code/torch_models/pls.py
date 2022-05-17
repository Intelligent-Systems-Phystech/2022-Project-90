import torch

from sklearn.cross_decomposition import PLSCanonical
from typing import Optional

from torch_models.prediction_models import PredictionModelInterface


# The wrapper over PLSCanonical in sklearn
class PLS:
    def __init__(self,
                 n_components: int = 2,
                 scale: bool = True,
                 max_iter: int = 500,
                 tol: float = 1e-6,
                 copy: bool = True):
        self.pls = PLSCanonical(n_components=n_components,
                                scale=scale,
                                max_iter=max_iter,
                                tol=tol,
                                copy=copy)

    def fit(self, X: torch.tensor, Y: torch.tensor):
        self.pls.fit(X.numpy(), Y.numpy())

    def fit_transform(self, X: torch.tensor, Y: Optional[torch.tensor] = None):
        if Y is None:
            return self.pls.fit_transform(X.numpy())
        else:
            return self.pls.fit_transform(X.numpy(), Y.numpy())

    def encode(self,
               input: Optional[torch.Tensor] = None,
               target: Optional[torch.Tensor] = None):
        if not input is None and not target is None:
            assert input.shape[0] == target.shape[0]

        output = []

        if not input is None:
            input = input.detach().numpy()
            output.append(torch.Tensor(input @ self.pls.x_rotations_))

        if not target is None:
            target = target.detach().numpy()
            output.append(torch.Tensor(target @ self.pls.y_rotations_))

        if len(output) == 0:
            return None
        elif len(output) == 1:
            return output[0]
        else:
            return output[0], output[1]

    def decode(self,
               input: Optional[torch.Tensor] = None,
               target: Optional[torch.Tensor] = None):
        if not input is None and not target is None:
            assert input.shape[0] == target.shape[0]

        output = []

        if not input is None:
            input = input.detach().numpy()
            output.append(torch.Tensor(input @ self.pls.x_loadings_.T))

        if not target is None:
            target = target.detach().numpy()
            output.append(torch.Tensor(target @ self.pls.y_loadings_.T))

        if len(output) == 0:
            return None
        elif len(output) == 1:
            return output[0]
        else:
            return output[0], output[1]

    def predict(self, input: torch.Tensor, target: torch.Tensor,
                model: PredictionModelInterface):
        latent_input, latent_target = self.encode(input, target)
        preds = model.predict(latent_input, latent_target)
        output = self.decode(target=preds)

        return output
