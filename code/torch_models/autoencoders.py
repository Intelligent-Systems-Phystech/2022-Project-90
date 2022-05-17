import torch
from torch import nn
from torch import distributions as D


class AutoEncoder(nn.Module):
    def __init__(self,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 device: torch.device = torch.device('cpu')):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def encode(self, input: torch.Tensor):
        assert input.device == self.device

        return self.encoder(input)

    def decode(self, latent: torch.Tensor):
        assert latent.device == self.device

        return self.decoder(latent)

    def get_latent(self, input):
        return self.encode(input)

    def forward(self, input):
        return self.decode(self.get_latent(input))


class NormalVAE(AutoEncoder):
    def __init__(self,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 hidden_size: int,
                 device: torch.device = torch.device('cpu')):
        super().__init__(encoder, decoder, device)

        self.hidden_size = hidden_size
        self.distribution = D.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def get_latent(self, input):
        distr_params = self.encode(input)
        batch_size = input.size()[0]

        assert distr_params.size() == (batch_size, 2)

        mu, logsigma = distr_params.split(split_size=1, dim=1)
        sample = self.distribution.sample((batch_size, self.hidden_size)).to(self.device)

        return mu + sample * torch.exp(logsigma)
