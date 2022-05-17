import torch
import pandas as pd
import numpy as np
import pyEDM as edm


class PredictionModelInterface:
    def __init__(self): pass

    def predict(self, input, target): raise NotImplementedError


class SMapLegacy(PredictionModelInterface):
    def __init__(self,
                 libsize: int,
                 predsize: int,
                 theta: float,
                 solver=None):
        super().__init__()

        self.theta = theta
        self.solver = solver
        self.libsize = libsize
        self.predsize = predsize

    def predict(self, input: torch.Tensor, target: torch.Tensor):
        assert input.ndim == target.ndim == 2
        assert input.size()[0] == target.size()[0]

        n_objects, n_features = input.size()
        _, n_targets = target.size()

        device = input.device
        input = input.detach().numpy()
        target = target.detach().numpy()
        multi_data = np.hstack((input, target))

        multi_data = pd.DataFrame(multi_data)
        multi_data.insert(loc=0, column="time", value=range(1, n_objects + 1))
        multi_data.columns = list(map(str, multi_data.columns))

        lib = f"1 {self.libsize}"
        pred = f"{self.libsize + 1} {self.libsize + self.predsize}"

        predictions = []

        for i in range(n_targets):
            target_ind = str(i + n_features)

            target_preds = edm.SMap(dataFrame=multi_data,
                                    lib=lib,
                                    pred=pred,
                                    embedded=True,
                                    target=target_ind,
                                    columns=list(map(str, range(n_features))) + [target_ind],
                                    theta=self.theta,
                                    solver=self.solver,
                                    showPlot=False);

            predictions.append(target_preds['predictions']['Predictions'].values[1:])

        return torch.Tensor(np.array(predictions).T).to(device)
