import torch
import torch.nn.functional as F


class AbstractCCMMetric:
    def __init__(self): pass

    def __call__(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        raise NotImplementedError

    def _check_arguments(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        assert x_elems.size()[0] == y_elems.size()[0]
        assert x_elems.ndim == y_elems.ndim == 2


class CanonicalCCM(AbstractCCMMetric):
    @staticmethod
    def pearson_corr_coef(x_arr: torch.Tensor, y_arr: torch.Tensor):
        assert x_arr.size() == y_arr.size() and x_arr.ndim == 1

        x_centered = x_arr - x_arr.mean()
        y_centered = y_arr - y_arr.mean()

        cov = torch.dot(x_centered, y_centered)
        x_std = torch.sqrt(torch.dot(x_centered, x_centered))
        y_std = torch.sqrt(torch.dot(y_centered, y_centered))

        if cov.item() == 0:
            return cov
        else:
            return cov / x_std / y_std

    def __call__(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        # x_elems = [n_samples, n_features]
        # y_elems = [n_samples, n_targets]
        self._check_arguments(x_elems, y_elems)

        manifold_x_elems, last_x_elem = x_elems[:-1, :], x_elems[-1, :]
        manifold_y_elems, last_y_elem = y_elems[:-1, :], y_elems[-1, :]

        distances = torch.sqrt(torch.sum(torch.square(manifold_x_elems - last_x_elem), dim=1))
        coefs = F.softmax(-distances, dim=0)

        y_hat = torch.sum(torch.multiply(manifold_y_elems, coefs.unsqueeze(1)), dim=0)

        return CanonicalCCM.pearson_corr_coef(y_hat, last_y_elem)


class ModifiedCCM(CanonicalCCM):
    def __init__(self, min_manifold_size: int = 100):
        super().__init__()
        self.min_manifold_size = min_manifold_size

    def __call__(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        # x_elems = [n_samples, n_features]
        # y_elems = [n_samples, n_targets]
        self._check_arguments(x_elems, y_elems)

        smaller_manifold_size = min(x_elems.size()[0] // 3, self.min_manifold_size)

        return super().__call__(x_elems, y_elems) - \
               super().__call__(x_elems[:smaller_manifold_size, :],
                                y_elems[:smaller_manifold_size, :])


class LipschitzCCM(AbstractCCMMetric):
    def __init__(self, k_neighbours: int):
        super().__init__()
        self.k_neighbours = k_neighbours

    def __call__(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        self._check_arguments(x_elems, y_elems)

        manifold_x_elems, last_x_elem = x_elems[:-1, :], x_elems[-1, :]
        manifold_y_elems, last_y_elem = y_elems[:-1, :], y_elems[-1, :]

        x_distances = torch.sqrt(torch.sum(torch.square(manifold_x_elems - last_x_elem), dim=1))
        y_distances = torch.sqrt(torch.sum(torch.square(manifold_y_elems - last_y_elem), dim=1))

        x_distances = torch.sort(x_distances).values[:self.k_neighbours]
        y_distances = torch.sort(y_distances).values[:self.k_neighbours]

        return torch.mean(x_distances) / torch.mean(y_distances)

    def _check_arguments(self, x_elems: torch.Tensor, y_elems: torch.Tensor):
        super()._check_arguments(x_elems, y_elems)
        assert x_elems.size()[0] > self.k_neighbours
